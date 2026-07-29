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

#include "tree/TreeOptions.h"

#include <cmath>
#include <stdexcept>

namespace drf {

TreeOptions::TreeOptions(uint mtry,
                         uint min_node_size,
                         bool honesty,
                         double honesty_fraction,
                         bool honesty_prune_leaves,
                         double alpha,
                         double imbalance_penalty,
                         size_t num_features,
                         double bandwidth,
                         unsigned int node_scaling,
                         bool survey_mode,
                         double honesty_probability,
                         uint min_obs,
                         double max_weight_ratio,
                         uint max_depth):
  mtry(mtry),
  min_node_size(min_node_size),
  honesty(honesty),
  honesty_fraction(honesty_fraction),
  honesty_prune_leaves(honesty_prune_leaves),
  alpha(alpha),
  imbalance_penalty(imbalance_penalty),
  num_features(num_features),
  bandwidth(bandwidth),
  node_scaling(node_scaling),
  survey_mode(survey_mode),
  honesty_probability(honesty_probability),
  min_obs(min_obs),
  max_weight_ratio(max_weight_ratio),
  max_depth(max_depth) {
  if (survey_mode &&
      (!std::isfinite(honesty_probability) || honesty_probability <= 0.0 ||
       honesty_probability >= 1.0)) {
    throw std::invalid_argument("SDRF q must lie strictly between 0 and 1.");
  }
  if (survey_mode && min_obs == 0) {
    throw std::invalid_argument("SDRF min_obs must be greater than zero.");
  }
  if (survey_mode &&
      (std::isnan(max_weight_ratio) || max_weight_ratio < 1.0)) {
    throw std::invalid_argument(
        "SDRF lambda_max must be at least 1 (or positive infinity). ");
  }
}

uint TreeOptions::get_mtry() const {
  return mtry;
}

uint TreeOptions::get_min_node_size() const {
  return min_node_size;
}

bool TreeOptions::get_honesty() const {
  return honesty;
}

double TreeOptions::get_honesty_fraction() const {
  return honesty_fraction;
}

bool TreeOptions::get_honesty_prune_leaves() const {
  return honesty_prune_leaves;
}

double TreeOptions::get_alpha() const {
  return alpha;
}

double TreeOptions::get_imbalance_penalty() const {
  return imbalance_penalty;
}

size_t TreeOptions::get_num_features() const {
  return num_features;
}

double TreeOptions::get_bandwidth() const {
  return bandwidth;
}

unsigned int TreeOptions::get_node_scaling() const {
  return node_scaling;
}

bool TreeOptions::is_survey_mode() const {
  return survey_mode;
}

double TreeOptions::get_honesty_probability() const {
  return honesty_probability;
}

uint TreeOptions::get_min_obs() const {
  return min_obs;
}

double TreeOptions::get_max_weight_ratio() const {
  return max_weight_ratio;
}

uint TreeOptions::get_max_depth() const {
  return max_depth;
}

} // namespace drf
