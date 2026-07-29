#ifndef DRF_FOURIER_SPLITTING_RULE_H_
#define DRF_FOURIER_SPLITTING_RULE_H_

#include <cstddef>
#include <vector>

#include "splitting/SplittingRule.h"

namespace drf {

class RandomFourierFeatures;

/**
 * Finds MMD-maximizing predictor splits using a shared response projection.
 *
 * The RandomFourierFeatures object belongs to Data and is immutable during
 * forest construction. Consequently this rule is lightweight: each tree owns
 * only its temporary prefix sums, while all tree threads reuse one n x D_RFF
 * projection for the current rolling or expanding training window.
 */
class FourierSplittingRule final : public SplittingRule {
public:
  FourierSplittingRule(double alpha,
                       size_t min_obs,
                       double max_weight_ratio);

  bool requires_relabeling() const;

  bool find_best_split(
      const Data& data,
      size_t node,
      const std::vector<size_t>& possible_split_vars,
      std::vector<std::vector<double>>& responses_by_sample,
      const std::vector<double>& sample_weights,
      const std::vector<std::vector<size_t>>& samples,
      std::vector<size_t>& split_vars,
      std::vector<double>& split_values);

private:
  void find_split(const Data& data,
                  const std::vector<size_t>& node_samples,
                  size_t var,
                  size_t min_child_size,
                  const RandomFourierFeatures& rff,
                  const std::vector<double>& sample_weights,
                  double& best_value,
                  size_t& best_var,
                  double& best_score) const;

  double alpha;
  size_t min_obs;
  double max_weight_ratio;

  DISALLOW_COPY_AND_ASSIGN(FourierSplittingRule);
};

} // namespace drf

#endif // DRF_FOURIER_SPLITTING_RULE_H_