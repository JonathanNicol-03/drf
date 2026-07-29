/*-------------------------------------------------------------------------------
 This file is part of distributional random forest (drf).
 
 drf is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 drf is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with drf. If not, see <http://www.gnu.org/licenses/>.
#-------------------------------------------------------------------------------*/

#include "SampleWeightComputer.h"

#include <cmath>
#include <stdexcept>

#include "tree/Tree.h"

namespace drf {

std::unordered_map<size_t, double> SampleWeightComputer::compute_weights(size_t sample,
                                                                         const Forest& forest,
                                                                         const std::vector<std::vector<size_t>>& leaf_nodes_by_tree,
                                                                         const std::vector<std::vector<bool>>& valid_trees_by_sample, 
                                                                         const size_t tree_index_start, 
                                                                         const size_t tree_index_end) const {
  std::unordered_map<size_t, double> weights_by_sample;

  const size_t n_trees = forest.get_trees().size();
  const size_t index_max = std::min(tree_index_end, n_trees);
  // if(tree_index_start >= n_trees){
  //   throw std::runtime_error("tree_index_start >= n_trees");
  // }
  
  // Create a list of weighted neighbors for this sample.
  for (size_t tree_index = tree_index_start; tree_index < index_max; ++tree_index) {
    
    if (!valid_trees_by_sample[sample][tree_index]) {
      continue;
    }

    const std::vector<size_t>& leaf_nodes = leaf_nodes_by_tree.at(tree_index);
    size_t node = leaf_nodes.at(sample);

    const std::unique_ptr<Tree>& tree = forest.get_trees()[tree_index];
    const std::vector<size_t>& samples = tree->get_leaf_samples()[node];
    if (!samples.empty()) {
      const std::vector<std::vector<double>>& weights_by_leaf =
          tree->get_leaf_sample_weights();
      const std::vector<double> empty_masses;
      const std::vector<double>& sample_masses = weights_by_leaf.empty()
          ? empty_masses
          : weights_by_leaf.at(node);
      add_sample_weights(samples, sample_masses, weights_by_sample);
    }
  }

  normalize_sample_weights(weights_by_sample);
  return weights_by_sample;
}

void SampleWeightComputer::add_sample_weights(const std::vector<size_t>& samples,
                                              const std::vector<double>& sample_masses,
                                              std::unordered_map<size_t, double>& weights_by_sample) const {
  if (sample_masses.empty()) {
    const double sample_weight = 1.0 / samples.size();
    for (size_t sample : samples) {
      weights_by_sample[sample] += sample_weight;
    }
    return;
  }

  if (sample_masses.size() != samples.size()) {
    throw std::logic_error(
        "Serialized leaf sample IDs and SDRF masses are not aligned.");
  }

  double leaf_mass = 0.0;
  for (double mass : sample_masses) {
    if (!std::isfinite(mass) || mass <= 0.0) {
      throw std::logic_error(
          "An SDRF estimation leaf contains a nonpositive mass.");
    }
    leaf_mass += mass;
  }

  // This is the tree-specific Hajek distribution from Algorithm 1. The final
  // normalization below averages these unit-mass distributions across trees.
  for (size_t position = 0; position < samples.size(); ++position) {
    weights_by_sample[samples[position]] += sample_masses[position] / leaf_mass;
  }
}

void SampleWeightComputer::normalize_sample_weights(std::unordered_map<size_t, double>& weights_by_sample) const {
  double total_weight = 0.0;
  for (const auto& entry : weights_by_sample) {
    total_weight += entry.second;
  }

  for (auto& entry : weights_by_sample) {
    entry.second /= total_weight;
  }
}

} // namespace drf
