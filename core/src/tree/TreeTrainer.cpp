#include "tree/TreeTrainer.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace drf {

TreeTrainer::TreeTrainer(
    std::unique_ptr<RelabelingStrategy> relabeling_strategy,
    std::unique_ptr<SplittingRuleFactory> splitting_rule_factory,
    std::unique_ptr<OptimizedPredictionStrategy> prediction_strategy) :
    relabeling_strategy(std::move(relabeling_strategy)),
    splitting_rule_factory(std::move(splitting_rule_factory)),
    prediction_strategy(std::move(prediction_strategy)) {}

std::unique_ptr<Tree> TreeTrainer::train(
    const Data& data,
    RandomSampler& sampler,
    const std::vector<size_t>& clusters,
    const TreeOptions& options) const {
  std::vector<size_t> split_samples;
  std::vector<size_t> estimation_samples;

  if (options.get_honesty()) {
    std::vector<size_t> tree_growing_clusters;
    std::vector<size_t> leaf_clusters;
    sampler.subsample(clusters,
                      options.get_honesty_fraction(),
                      tree_growing_clusters,
                      leaf_clusters);
    sampler.sample_from_clusters(tree_growing_clusters, split_samples);
    sampler.sample_from_clusters(leaf_clusters, estimation_samples);
  } else {
    sampler.sample_from_clusters(clusters, split_samples);
  }

  std::vector<size_t> drawn_samples;
  sampler.get_samples_in_clusters(clusters, drawn_samples);
  return train_internal(data,
                        sampler,
                        split_samples,
                        estimation_samples,
                        std::vector<double>(),
                        std::vector<double>(),
                        drawn_samples,
                        options);
}

std::unique_ptr<Tree> TreeTrainer::train_survey(
    const Data& data,
    RandomSampler& sampler,
    size_t tree_index,
    const TreeOptions& options) const {
  if (!data.has_survey_data()) {
    throw std::logic_error(
        "Survey tree training requires pi, psu_id, and multipliers.");
  }

  const size_t num_rows = data.get_num_rows();
  const double q = options.get_honesty_probability();
  std::vector<size_t> split_samples;
  std::vector<size_t> estimation_samples;
  std::vector<double> split_weights(num_rows, 0.0);
  std::vector<double> estimation_weights(num_rows, 0.0);
  std::vector<size_t> active_samples;
  std::set<size_t> active_psus;
  std::unordered_map<size_t, bool> psu_in_split_set;

  for (size_t sample = 0; sample < num_rows; ++sample) {
    const double multiplier = data.get_resampling_multiplier(sample,
                                                              tree_index);
    if (multiplier <= 0.0) {
      continue;
    }
    active_samples.push_back(sample);
    active_psus.insert(data.get_psu_id(sample));
  }

  if (active_psus.empty()) {
    throw std::invalid_argument(
        "A multiplier column must activate at least one PSU.");
  }

  // Algorithm 1 draws each active PSU indicator exactly once. An empty split
  // set is a valid stump, but an empty estimation set has no conditional
  // distribution and must fail explicitly rather than silently changing the
  // Bernoulli law through rejection sampling.
  bool has_estimation_psu = false;
  for (size_t psu : active_psus) {
    const bool in_split_set = sampler.sample_bernoulli(q);
    psu_in_split_set[psu] = in_split_set;
    has_estimation_psu = has_estimation_psu || !in_split_set;
  }
  if (!has_estimation_psu) {
    throw std::runtime_error(
        "Algorithm 1 assigned every active PSU to the split set. Use a less "
        "extreme q, provide more active PSUs, or rerun with another tree seed.");
  }

  for (size_t sample : active_samples) {
    const double multiplier = data.get_resampling_multiplier(sample,
                                                              tree_index);
    const bool in_split_set = psu_in_split_set.at(data.get_psu_id(sample));
    const double pi = data.get_inclusion_probability(sample);
    if (in_split_set) {
      split_samples.push_back(sample);
      split_weights[sample] = multiplier / (q * pi);
    } else {
      estimation_samples.push_back(sample);
      estimation_weights[sample] = multiplier / ((1.0 - q) * pi);
    }
  }

  // OOB membership is not used by SDRF. Structural independence comes from
  // the PSU-level split/estimation assignment, and prediction reads only the
  // estimation samples stored in leaves.
  return train_internal(data,
                        sampler,
                        split_samples,
                        estimation_samples,
                        split_weights,
                        estimation_weights,
                        std::vector<size_t>(),
                        options);
}

std::unique_ptr<Tree> TreeTrainer::train_internal(
    const Data& data,
    RandomSampler& sampler,
    const std::vector<size_t>& split_samples,
    const std::vector<size_t>& estimation_samples,
    const std::vector<double>& split_weights,
    const std::vector<double>& estimation_weights,
    const std::vector<size_t>& drawn_samples,
    const TreeOptions& options) const {
  std::vector<std::vector<size_t>> child_nodes(2);
  std::vector<std::vector<size_t>> nodes;
  std::vector<size_t> split_vars;
  std::vector<double> split_values;
  std::vector<size_t> node_depths;

  create_empty_node(child_nodes, nodes, split_vars, split_values);
  nodes[0] = split_samples;
  node_depths.push_back(0);

  std::unique_ptr<SplittingRule> splitting_rule =
      splitting_rule_factory->create(data, options);

  // Fourier-MMD consumes the immutable RFF map and needs no n by d response
  // workspace. This removes one large allocation per concurrent tree.
  std::vector<std::vector<double>> responses_by_sample;
  if (splitting_rule->requires_relabeling()) {
    responses_by_sample.resize(data.get_num_rows());
  }

  size_t num_open_nodes = 1;
  size_t node = 0;
  while (num_open_nodes > 0) {
    const bool is_leaf = split_node(node,
                                    data,
                                    splitting_rule,
                                    sampler,
                                    child_nodes,
                                    nodes,
                                    split_vars,
                                    split_values,
                                    node_depths,
                                    responses_by_sample,
                                    split_weights,
                                    options);
    if (is_leaf) {
      --num_open_nodes;
    } else {
      nodes[node].clear();
      ++num_open_nodes;
    }
    ++node;
  }

  std::unique_ptr<Tree> tree(new Tree(0,
                                      child_nodes,
                                      nodes,
                                      split_vars,
                                      split_values,
                                      drawn_samples,
                                      PredictionValues()));

  // Survey trees must always discard split-set samples, even in the rare case
  // that every active PSU was assigned to the split set and estimation_samples
  // is empty. Legacy trees retain their old behavior.
  if (options.is_survey_mode() || !estimation_samples.empty()) {
    repopulate_leaf_nodes(tree,
                          data,
                          estimation_samples,
                          estimation_weights,
                          options.get_honesty_prune_leaves());
  }

  if (prediction_strategy != nullptr && !options.is_survey_mode()) {
    PredictionValues prediction_values =
        prediction_strategy->precompute_prediction_values(
            tree->get_leaf_samples(), data);
    tree->set_prediction_values(prediction_values);
  }
  return tree;
}

void TreeTrainer::repopulate_leaf_nodes(
    const std::unique_ptr<Tree>& tree,
    const Data& data,
    const std::vector<size_t>& leaf_samples,
    const std::vector<double>& sample_weights,
    bool honesty_prune_leaves) const {
  const size_t num_nodes = tree->get_leaf_samples().size();
  std::vector<std::vector<size_t>> new_leaf_nodes(num_nodes);
  std::vector<std::vector<double>> new_leaf_weights;
  if (!sample_weights.empty()) {
    if (sample_weights.size() != data.get_num_rows()) {
      throw std::invalid_argument(
          "Estimation weights must have one entry per training row.");
    }
    new_leaf_weights.resize(num_nodes);
  }

  const std::vector<size_t> leaf_node_by_sample =
      tree->find_leaf_nodes(data, leaf_samples);
  for (size_t sample : leaf_samples) {
    const size_t leaf_node = leaf_node_by_sample[sample];
    new_leaf_nodes[leaf_node].push_back(sample);
    if (!sample_weights.empty()) {
      new_leaf_weights[leaf_node].push_back(sample_weights[sample]);
    }
  }

  tree->set_leaf_samples(new_leaf_nodes);
  tree->set_leaf_sample_weights(new_leaf_weights);
  if (honesty_prune_leaves) {
    tree->honesty_prune_leaves();
  }
}

void TreeTrainer::create_split_variable_subset(
    std::vector<size_t>& result,
    RandomSampler& sampler,
    const Data& data,
    uint mtry) const {
  const size_t num_independent_variables = data.get_num_cols() -
      data.get_disallowed_split_variables().size();
  const size_t mtry_sample = sampler.sample_poisson(mtry);
  const size_t split_mtry = std::max<size_t>(
      std::min<size_t>(mtry_sample, num_independent_variables), 1uL);

  sampler.draw(result,
               data.get_num_cols(),
               data.get_disallowed_split_variables(),
               split_mtry);
}

bool TreeTrainer::split_node(
    size_t node,
    const Data& data,
    const std::unique_ptr<SplittingRule>& splitting_rule,
    RandomSampler& sampler,
    std::vector<std::vector<size_t>>& child_nodes,
    std::vector<std::vector<size_t>>& samples,
    std::vector<size_t>& split_vars,
    std::vector<double>& split_values,
    std::vector<size_t>& node_depths,
    std::vector<std::vector<double>>& responses_by_sample,
    const std::vector<double>& split_weights,
    const TreeOptions& options) const {
  std::vector<size_t> possible_split_vars;
  create_split_variable_subset(possible_split_vars,
                               sampler,
                               data,
                               options.get_mtry());

  const bool stop = split_node_internal(node,
                                        data,
                                        splitting_rule,
                                        possible_split_vars,
                                        samples,
                                        split_vars,
                                        split_values,
                                        responses_by_sample,
                                        split_weights,
                                        node_depths[node],
                                        options);
  if (stop) {
    return true;
  }

  const size_t split_var = split_vars[node];
  const double split_value = split_values[node];
  const size_t child_depth = node_depths[node] + 1;

  const size_t left_child = samples.size();
  child_nodes[0][node] = left_child;
  create_empty_node(child_nodes, samples, split_vars, split_values);
  node_depths.push_back(child_depth);

  const size_t right_child = samples.size();
  child_nodes[1][node] = right_child;
  create_empty_node(child_nodes, samples, split_vars, split_values);
  node_depths.push_back(child_depth);

  for (size_t sample : samples[node]) {
    if (data.get(sample, split_var) <= split_value) {
      samples[left_child].push_back(sample);
    } else {
      samples[right_child].push_back(sample);
    }
  }
  return false;
}

bool TreeTrainer::split_node_internal(
    size_t node,
    const Data& data,
    const std::unique_ptr<SplittingRule>& splitting_rule,
    const std::vector<size_t>& possible_split_vars,
    const std::vector<std::vector<size_t>>& samples,
    std::vector<size_t>& split_vars,
    std::vector<double>& split_values,
    std::vector<std::vector<double>>& responses_by_sample,
    const std::vector<double>& split_weights,
    size_t depth,
    const TreeOptions& options) const {
  if (samples[node].size() <= options.get_min_node_size() ||
      (options.get_max_depth() > 0 && depth >= options.get_max_depth())) {
    split_values[node] = -1.0;
    return true;
  }

  if (splitting_rule->requires_relabeling() &&
      relabeling_strategy->relabel(samples[node],
                                   data,
                                   responses_by_sample)) {
    split_values[node] = -1.0;
    return true;
  }

  if (splitting_rule->find_best_split(data,
                                      node,
                                      possible_split_vars,
                                      responses_by_sample,
                                      split_weights,
                                      samples,
                                      split_vars,
                                      split_values)) {
    split_values[node] = -1.0;
    return true;
  }
  return false;
}

void TreeTrainer::create_empty_node(
    std::vector<std::vector<size_t>>& child_nodes,
    std::vector<std::vector<size_t>>& samples,
    std::vector<size_t>& split_vars,
    std::vector<double>& split_values) const {
  child_nodes[0].push_back(0);
  child_nodes[1].push_back(0);
  samples.emplace_back();
  split_vars.push_back(0);
  split_values.push_back(0.0);
}

} // namespace drf