#include "splitting/FourierSplittingRule.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "kernel/RandomFourierFeatures.h"

namespace drf {

FourierSplittingRule::FourierSplittingRule(double alpha,
                                           size_t min_obs,
                                           double max_weight_ratio) :
    alpha(alpha),
    min_obs(min_obs),
    max_weight_ratio(max_weight_ratio) {}

bool FourierSplittingRule::requires_relabeling() const {
  return false;
}

bool FourierSplittingRule::find_best_split(
    const Data& data,
    size_t node,
    const std::vector<size_t>& possible_split_vars,
    std::vector<std::vector<double>>& responses_by_sample,
    const std::vector<double>& sample_weights,
    const std::vector<std::vector<size_t>>& samples,
    std::vector<size_t>& split_vars,
    std::vector<double>& split_values) {
  // Relabeling output is part of the common splitting interface. Fourier-MMD
  // uses the window-level response projection instead, so it is intentionally
  // not read or modified here.
  (void) responses_by_sample;

  const size_t size_node = samples[node].size();
    const size_t alpha_child_size = static_cast<size_t>(
      std::ceil(size_node * alpha));
    const size_t min_child_size = std::max<size_t>(
      std::max<size_t>(alpha_child_size, min_obs), 1uL);
  const RandomFourierFeatures& rff = data.get_rff_features();

  size_t best_var = 0;
  double best_value = 0.0;
  double best_score = -1.0;

  for (size_t var : possible_split_vars) {
    find_split(data,
               samples[node],
               var,
               min_child_size,
               rff,
               sample_weights,
               best_value,
               best_var,
               best_score);
  }

  if (best_score <= 0.0) {
    return true;
  }

  split_vars[node] = best_var;
  split_values[node] = best_value;
  return false;
}

void FourierSplittingRule::find_split(
    const Data& data,
    const std::vector<size_t>& node_samples,
    size_t var,
    size_t min_child_size,
    const RandomFourierFeatures& rff,
    const std::vector<double>& sample_weights,
    double& best_value,
    size_t& best_var,
    double& best_score) const {
  const size_t num_samples = node_samples.size();
  if (num_samples < 2 * min_child_size) {
    return;
  }

  // Sort actual training-row IDs. The previous implementation sorted local
  // positions and therefore had to maintain a second index translation for
  // every Fourier lookup.
  std::vector<size_t> ordered_samples(node_samples);
  std::sort(ordered_samples.begin(), ordered_samples.end(),
            [&data, var](size_t left, size_t right) {
              return data.get_index(left, var) < data.get_index(right, var);
            });

  auto weight_for = [&sample_weights](size_t sample) {
    return sample_weights.empty() ? 1.0 : sample_weights.at(sample);
  };

  // Prefix/suffix statistics are independent of the Fourier coordinate and
  // are therefore computed once per candidate predictor. Raw positions enforce
  // min_obs; weighted masses and extrema implement the SDRF Hajek split score
  // and lambda_max constraint.
  std::vector<double> prefix_mass(num_samples);
  std::vector<double> prefix_min(num_samples);
  std::vector<double> prefix_max(num_samples);
  std::vector<double> suffix_min(num_samples);
  std::vector<double> suffix_max(num_samples);

  for (size_t position = 0; position < num_samples; ++position) {
    const double weight = weight_for(ordered_samples[position]);
    if (!std::isfinite(weight) || weight <= 0.0) {
      throw std::invalid_argument(
          "Every sample used for splitting must have positive finite weight.");
    }
    prefix_mass[position] = weight +
        (position == 0 ? 0.0 : prefix_mass[position - 1]);
    prefix_min[position] = position == 0
        ? weight
        : std::min(prefix_min[position - 1], weight);
    prefix_max[position] = position == 0
        ? weight
        : std::max(prefix_max[position - 1], weight);
  }
  for (size_t offset = 0; offset < num_samples; ++offset) {
    const size_t position = num_samples - offset - 1;
    const double weight = weight_for(ordered_samples[position]);
    suffix_min[position] = position + 1 == num_samples
        ? weight
        : std::min(suffix_min[position + 1], weight);
    suffix_max[position] = position + 1 == num_samples
        ? weight
        : std::max(suffix_max[position + 1], weight);
  }

  const double parent_mass = prefix_mass.back();

  std::vector<double> squared_mmd(num_samples - 1, 0.0);
  for (size_t feature = 0; feature < rff.get_num_features(); ++feature) {
    double total_real = 0.0;
    double total_imaginary = 0.0;
    for (size_t sample : ordered_samples) {
      const double weight = weight_for(sample);
      total_real += weight * rff.get_real(sample, feature);
      total_imaginary += weight * rff.get_imaginary(sample, feature);
    }

    double left_real = 0.0;
    double left_imaginary = 0.0;
    for (size_t split = 0; split + 1 < num_samples; ++split) {
      const size_t sample = ordered_samples[split];
        const double weight = weight_for(sample);
        left_real += weight * rff.get_real(sample, feature);
        left_imaginary += weight * rff.get_imaginary(sample, feature);

        const double left_mass = prefix_mass[split];
        const double right_mass = parent_mass - left_mass;
        const double real_difference = left_real / left_mass -
          (total_real - left_real) / right_mass;
        const double imaginary_difference = left_imaginary / left_mass -
          (total_imaginary - left_imaginary) / right_mass;
      squared_mmd[split] += real_difference * real_difference +
          imaginary_difference * imaginary_difference;
    }
  }

  for (size_t split = min_child_size - 1;
       split + min_child_size < num_samples;
       ++split) {
    const size_t left_sample = ordered_samples[split];
    const size_t right_sample = ordered_samples[split + 1];
    if (data.get_index(left_sample, var) == data.get_index(right_sample, var)) {
      continue;
    }

    const double left_ratio = prefix_max[split] / prefix_min[split];
    const double right_ratio = suffix_max[split + 1] / suffix_min[split + 1];
    if (left_ratio > max_weight_ratio || right_ratio > max_weight_ratio) {
      continue;
    }

    const double left_mass = prefix_mass[split];
    const double right_mass = parent_mass - left_mass;
    const double score = (left_mass * right_mass /
        (parent_mass * parent_mass)) * squared_mmd[split];

    if (score > best_score) {
      best_score = score;
      best_value = data.get(left_sample, var);
      best_var = var;
    }
  }
}

} // namespace drf