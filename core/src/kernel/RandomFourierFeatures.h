#ifndef DRF_RANDOM_FOURIER_FEATURES_H_
#define DRF_RANDOM_FOURIER_FEATURES_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace drf {

class Data;

/**
 * Translation-invariant kernels currently supported by the SDRF projection.
 *
 * GAUSSIAN_RBF represents
 *   k(y, z) = exp(-||y - z||_2^2 / (2 * bandwidth^2)).
 * Its spectral measure is Gaussian with coordinate standard deviation
 * 1 / bandwidth.
 *
 * LAPLACIAN represents the product Laplacian kernel
 *   k(y, z) = exp(-||y - z||_1 / bandwidth).
 * Its spectral measure has independent Cauchy coordinates with scale
 * 1 / bandwidth. Naming the L1 norm here is important: the isotropic L2
 * Laplacian kernel has a different multivariate spectral distribution.
 */
enum class RFFKernel {
  GAUSSIAN_RBF,
  LAPLACIAN
};

RFFKernel parse_rff_kernel(const std::string& kernel_name);

std::string rff_kernel_name(RFFKernel kernel);

struct RFFOptions {
  RFFOptions(size_t num_features,
             double bandwidth,
             RFFKernel kernel,
             uint64_t seed,
             size_t num_threads = 1);

  size_t num_features;
  double bandwidth;
  RFFKernel kernel;
  uint64_t seed;
  size_t num_threads;
};

/**
 * A window-scoped complex random Fourier projection of the response matrix.
 *
 * The projection is computed once before any trees are grown. All trees then
 * read the same immutable values, which both implements the SDRF definition
 * and avoids repeating O(n * response_dimension * num_features) work at every
 * node. Values are stored feature-major as interleaved real/imaginary pairs:
 *
 *   values[((feature * num_rows + sample) * 2) + component]
 *
 * This layout keeps all observations for one Fourier frequency contiguous,
 * matching the prefix-sum loop used by the MMD splitting rule. The explicit
 * pair representation also avoids relying on std::complex memory layout when
 * data are later transferred to or from a GPU implementation.
 */
class RandomFourierFeatures {
public:
  static RandomFourierFeatures compute(const Data& data,
                                       const RFFOptions& options);

  /**
   * Apply a caller-supplied feature-major frequency matrix. The matrix must
   * contain num_features * response_dimension values. This overload is used
   * by the R orchestration layer so CPU and GPU backends consume exactly the
   * same random frequencies.
   */
  static RandomFourierFeatures compute(
      const Data& data,
      size_t num_features,
      const std::vector<double>& frequencies,
      size_t num_threads = 1);

    /**
     * Rehydrate an already projected n by (2 * num_features) matrix after the R
     * layer has loaded it from shared scratch or copied it back from a GPU.
     * Values must use the same feature-major interleaved layout as this class.
     */
    static RandomFourierFeatures from_values(
      size_t num_rows,
      size_t num_features,
      size_t response_dimension,
      const std::vector<double>& values);

  size_t get_num_rows() const;

  size_t get_num_features() const;

  size_t get_response_dimension() const;

  double get_real(size_t sample, size_t feature) const;

  double get_imaginary(size_t sample, size_t feature) const;

  const std::vector<double>& get_values() const;

  const std::vector<double>& get_frequencies() const;

private:
  RandomFourierFeatures(size_t num_rows,
                        size_t num_features,
                        size_t response_dimension,
                        std::vector<double> values,
                        std::vector<double> frequencies);

  size_t value_offset(size_t sample, size_t feature) const;

  size_t num_rows;
  size_t num_features;
  size_t response_dimension;
  std::vector<double> values;
  std::vector<double> frequencies;
};

} // namespace drf

#endif // DRF_RANDOM_FOURIER_FEATURES_H_