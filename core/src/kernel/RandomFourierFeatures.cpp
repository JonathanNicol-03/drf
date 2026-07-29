#include "kernel/RandomFourierFeatures.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>

#include "commons/Data.h"

namespace drf {

namespace {

std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return value;
}

void validate_options(const Data& data, const RFFOptions& options) {
  if (data.get_outcome_index().empty()) {
    throw std::invalid_argument(
        "Cannot compute random Fourier features without a response.");
  }
  if (options.num_features == 0) {
    throw std::invalid_argument("num_rff must be greater than zero.");
  }
  if (!std::isfinite(options.bandwidth) || options.bandwidth <= 0.0) {
    throw std::invalid_argument("RFF bandwidth must be finite and positive.");
  }
}

} // namespace

RFFKernel parse_rff_kernel(const std::string& kernel_name) {
  const std::string normalized = lowercase(kernel_name);
  if (normalized == "gaussian" || normalized == "gaussian_rbf" ||
      normalized == "rbf") {
    return RFFKernel::GAUSSIAN_RBF;
  }
  if (normalized == "laplacian" || normalized == "laplace") {
    return RFFKernel::LAPLACIAN;
  }
  throw std::invalid_argument(
      "Unknown RFF kernel '" + kernel_name +
      "'. Supported kernels are 'gaussian' and 'laplacian'.");
}

std::string rff_kernel_name(RFFKernel kernel) {
  switch (kernel) {
    case RFFKernel::GAUSSIAN_RBF:
      return "gaussian";
    case RFFKernel::LAPLACIAN:
      return "laplacian";
  }
  throw std::invalid_argument("Unknown RFF kernel value.");
}

RFFOptions::RFFOptions(size_t num_features,
                       double bandwidth,
                       RFFKernel kernel,
             uint64_t seed,
             size_t num_threads) :
    num_features(num_features),
    bandwidth(bandwidth),
    kernel(kernel),
  seed(seed),
  num_threads(num_threads) {}

RandomFourierFeatures RandomFourierFeatures::compute(
    const Data& data,
    const RFFOptions& options) {
  validate_options(data, options);

  const std::vector<size_t> outcome_index = data.get_outcome_index();
  const size_t response_dimension = outcome_index.size();
  const double inverse_bandwidth = 1.0 / options.bandwidth;

  // Frequencies are feature-major: all response coordinates for frequency j
  // are adjacent. Keeping them in the result makes a precomputed projection
  // auditable and permits the exact same map to be applied to another Y matrix.
  std::vector<double> frequencies(options.num_features * response_dimension);
  std::mt19937_64 generator(options.seed);
  std::normal_distribution<double> gaussian_frequency(0.0,
                                                       inverse_bandwidth);
  std::cauchy_distribution<double> laplacian_frequency(0.0,
                                                       inverse_bandwidth);

  for (size_t feature = 0; feature < options.num_features; ++feature) {
    for (size_t coordinate = 0; coordinate < response_dimension; ++coordinate) {
      const size_t frequency_offset = feature * response_dimension + coordinate;
      frequencies[frequency_offset] = options.kernel == RFFKernel::GAUSSIAN_RBF
          ? gaussian_frequency(generator)
          : laplacian_frequency(generator);
    }
  }

  return compute(data,
                 options.num_features,
                 frequencies,
                 options.num_threads);
}

RandomFourierFeatures RandomFourierFeatures::compute(
    const Data& data,
    size_t num_features,
    const std::vector<double>& frequencies,
    size_t num_threads) {
  const std::vector<size_t> outcome_index = data.get_outcome_index();
  const size_t response_dimension = outcome_index.size();
  if (response_dimension == 0) {
    throw std::invalid_argument(
        "Cannot compute random Fourier features without a response.");
  }
  if (num_features == 0) {
    throw std::invalid_argument("num_rff must be greater than zero.");
  }
  if (frequencies.size() != num_features * response_dimension) {
    throw std::invalid_argument(
        "RFF frequency matrix dimensions do not match num_rff and Y.");
  }

  const size_t num_rows = data.get_num_rows();
  const double feature_scale = 1.0 /
      std::sqrt(static_cast<double>(num_features));
  std::vector<double> values(num_features * num_rows * 2);

  size_t worker_count = num_threads;
  if (worker_count == 0) {
    worker_count = std::thread::hardware_concurrency();
  }
  worker_count = std::max<size_t>(
      1, std::min<size_t>(worker_count, num_features));
  const size_t features_per_worker =
      (num_features + worker_count - 1) / worker_count;

  auto project_range = [&](size_t feature_begin, size_t feature_end) {
    for (size_t feature = feature_begin; feature < feature_end; ++feature) {
      for (size_t sample = 0; sample < num_rows; ++sample) {
        double phase = 0.0;
        for (size_t coordinate = 0;
             coordinate < response_dimension;
             ++coordinate) {
          phase += frequencies[feature * response_dimension + coordinate] *
              data.get(sample, outcome_index[coordinate]);
        }

        const size_t value_offset = (feature * num_rows + sample) * 2;
        values[value_offset] = feature_scale * std::cos(phase);
        values[value_offset + 1] = feature_scale * std::sin(phase);
      }
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(worker_count > 0 ? worker_count - 1 : 0);
  for (size_t worker = 1; worker < worker_count; ++worker) {
    const size_t feature_begin = worker * features_per_worker;
    const size_t feature_end = std::min(
        num_features, feature_begin + features_per_worker);
    workers.emplace_back(project_range, feature_begin, feature_end);
  }
  project_range(0, std::min(num_features, features_per_worker));
  for (std::thread& worker : workers) {
    worker.join();
  }

  return RandomFourierFeatures(num_rows,
                               num_features,
                               response_dimension,
                               std::move(values),
                               frequencies);
}

RandomFourierFeatures RandomFourierFeatures::from_values(
    size_t num_rows,
    size_t num_features,
    size_t response_dimension,
    const std::vector<double>& values) {
  if (num_rows == 0 || num_features == 0 || response_dimension == 0) {
    throw std::invalid_argument(
        "A precomputed RFF projection must have non-zero dimensions.");
  }
  if (values.size() != num_rows * num_features * 2) {
    throw std::invalid_argument(
        "A precomputed RFF projection must have n by (2 * num_rff) values.");
  }
  for (double value : values) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(
          "A precomputed RFF projection contains a non-finite value.");
    }
  }
  return RandomFourierFeatures(num_rows,
                               num_features,
                               response_dimension,
                               values,
                               std::vector<double>());
}

RandomFourierFeatures::RandomFourierFeatures(
    size_t num_rows,
    size_t num_features,
    size_t response_dimension,
    std::vector<double> values,
    std::vector<double> frequencies) :
    num_rows(num_rows),
    num_features(num_features),
    response_dimension(response_dimension),
    values(std::move(values)),
    frequencies(std::move(frequencies)) {}

size_t RandomFourierFeatures::get_num_rows() const {
  return num_rows;
}

size_t RandomFourierFeatures::get_num_features() const {
  return num_features;
}

size_t RandomFourierFeatures::get_response_dimension() const {
  return response_dimension;
}

double RandomFourierFeatures::get_real(size_t sample, size_t feature) const {
  return values[value_offset(sample, feature)];
}

double RandomFourierFeatures::get_imaginary(size_t sample,
                                            size_t feature) const {
  return values[value_offset(sample, feature) + 1];
}

const std::vector<double>& RandomFourierFeatures::get_values() const {
  return values;
}

const std::vector<double>& RandomFourierFeatures::get_frequencies() const {
  return frequencies;
}

size_t RandomFourierFeatures::value_offset(size_t sample,
                                           size_t feature) const {
  if (sample >= num_rows || feature >= num_features) {
    throw std::out_of_range("Random Fourier feature index is out of range.");
  }
  return (feature * num_rows + sample) * 2;
}

} // namespace drf