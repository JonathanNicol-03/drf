#ifndef DRF_TREE_TRAINER_H_
#define DRF_TREE_TRAINER_H_

#include <memory>
#include <set>
#include <vector>

#include "prediction/OptimizedPredictionStrategy.h"
#include "relabeling/RelabelingStrategy.h"
#include "sampling/RandomSampler.h"
#include "splitting/factory/SplittingRuleFactory.h"
#include "tree/Tree.h"
#include "tree/TreeOptions.h"

namespace drf {

class TreeTrainer {
public:
  TreeTrainer(std::unique_ptr<RelabelingStrategy> relabeling_strategy,
              std::unique_ptr<SplittingRuleFactory> splitting_rule_factory,
              std::unique_ptr<OptimizedPredictionStrategy> prediction_strategy);

  /** Legacy DRF tree construction. */
  std::unique_ptr<Tree> train(const Data& data,
                              RandomSampler& sampler,
                              const std::vector<size_t>& clusters,
                              const TreeOptions& options) const;

  /**
   * SDRF Algorithm 1 tree construction.
   *
   * Every active PSU is assigned wholly to either the split or estimation set.
   * The supplied tree index selects one column of design-respecting resampling
   * multipliers stored on Data.
   */
  std::unique_ptr<Tree> train_survey(const Data& data,
                                     RandomSampler& sampler,
                                     size_t tree_index,
                                     const TreeOptions& options) const;

private:
  std::unique_ptr<Tree> train_internal(
      const Data& data,
      RandomSampler& sampler,
      const std::vector<size_t>& split_samples,
      const std::vector<size_t>& estimation_samples,
      const std::vector<double>& split_weights,
      const std::vector<double>& estimation_weights,
      const std::vector<size_t>& drawn_samples,
      const TreeOptions& options) const;

  void create_empty_node(std::vector<std::vector<size_t>>& child_nodes,
                         std::vector<std::vector<size_t>>& samples,
                         std::vector<size_t>& split_vars,
                         std::vector<double>& split_values) const;

  void repopulate_leaf_nodes(
      const std::unique_ptr<Tree>& tree,
      const Data& data,
      const std::vector<size_t>& leaf_samples,
      const std::vector<double>& sample_weights,
      bool honesty_prune_leaves) const;

  void create_split_variable_subset(std::vector<size_t>& result,
                                    RandomSampler& sampler,
                                    const Data& data,
                                    uint mtry) const;

  bool split_node(size_t node,
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
                  const TreeOptions& options) const;

  bool split_node_internal(
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
      const TreeOptions& options) const;

  std::set<size_t> disallowed_split_variables;
  std::unique_ptr<RelabelingStrategy> relabeling_strategy;
  std::unique_ptr<SplittingRuleFactory> splitting_rule_factory;
  std::unique_ptr<OptimizedPredictionStrategy> prediction_strategy;
};

} // namespace drf

#endif // DRF_TREE_TRAINER_H_