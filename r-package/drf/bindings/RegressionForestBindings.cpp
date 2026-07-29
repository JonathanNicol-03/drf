/*-------------------------------------------------------------------------------
  This file is part of ditributional-regression-forest (drf).

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

#include <map>
#include <memory>
#include <Rcpp.h>
#include <sstream>
#include <utility>
#include <vector>

#include "commons/globals.h"
#include "Eigen/Sparse"
#include "forest/ForestPredictors.h"
#include "forest/ForestTrainers.h"
#include "kernel/RandomFourierFeatures.h"
#include "RcppData.h"
#include "RcppUtilities.h"

using namespace drf;

// [[Rcpp::export]]
Rcpp::List gini_train(Rcpp::NumericMatrix train_matrix,
                            Eigen::SparseMatrix<double> sparse_train_matrix,
                            std::vector<size_t> outcome_index, // 
                            size_t sample_weight_index,
                            bool use_sample_weights,
                            unsigned int mtry,
                            unsigned int num_trees,
                            unsigned int min_node_size,
                            double sample_fraction,
                            bool honesty,
                            double honesty_fraction,
                            bool honesty_prune_leaves,
                            size_t ci_group_size,
                            double alpha,
                            double imbalance_penalty,
                            std::vector<size_t> clusters,
                            unsigned int samples_per_cluster,
                            bool compute_oob_predictions,
                            unsigned int num_threads,
                            unsigned int seed,
                            size_t num_features,
                            double bandwidth, 
                            unsigned int node_scaling) {
  //std::cout << "regression_trainer will start" << std::endl;
  
  ForestTrainer trainer = gini_trainer(outcome_index.size());
  
 //std::cout << "convert_data will start" << std::endl;
  std::unique_ptr<Data> data = RcppUtilities::convert_data(train_matrix, sparse_train_matrix);
  //std::cout << "outcome_index will start and size will be printed" << std::endl;
  for (size_t i = 0; i < outcome_index.size(); ++i) {
    outcome_index[i] = outcome_index[i] - 1;
  }
  
 // std::cout << outcome_index.size() << std::endl;
  data->set_outcome_index(outcome_index);
  
  if(use_sample_weights) {
      data->set_weight_index(sample_weight_index - 1);
  }
  data->sort();
  //std::cout << "options will start" << std::endl;
  ForestOptions options(num_trees, ci_group_size, sample_fraction, mtry, min_node_size, honesty,
      honesty_fraction, honesty_prune_leaves, alpha, imbalance_penalty, num_threads, seed, clusters, samples_per_cluster, num_features, bandwidth, node_scaling);
  //std::cout << "trainer.train will start" << std::endl;
  Forest forest = trainer.train(*data, options);

  std::vector<Prediction> predictions;
  if (compute_oob_predictions) {
    ForestPredictor predictor = regression_predictor(num_threads, outcome_index.size());
    predictions = predictor.predict_oob(forest, *data, false);
  }

  return RcppUtilities::create_forest_object(forest, predictions);
}

// [[Rcpp::export]]
Rcpp::List fourier_train(Rcpp::NumericMatrix train_matrix,
                      Eigen::SparseMatrix<double> sparse_train_matrix,
                      std::vector<size_t> outcome_index, // 
                      size_t sample_weight_index,
                      bool use_sample_weights,
                      unsigned int mtry,
                      unsigned int num_trees,
                      unsigned int min_node_size,
                      double sample_fraction,
                      bool honesty,
                      double honesty_fraction,
                      bool honesty_prune_leaves,
                      size_t ci_group_size,
                      double alpha,
                      double imbalance_penalty,
                      std::vector<size_t> clusters,
                      unsigned int samples_per_cluster,
                      bool compute_oob_predictions,
                      unsigned int num_threads,
                      unsigned int seed,
                      size_t num_features,
                      double bandwidth,
                      std::string kernel,
                      unsigned int rff_seed,
                      unsigned int node_scaling) {
  //std::cout << "regression_trainer will start" << std::endl;
  
  ForestTrainer trainer = fourier_trainer(outcome_index.size());
  
  //std::cout << "convert_data will start" << std::endl;
  std::unique_ptr<Data> data = RcppUtilities::convert_data(train_matrix, sparse_train_matrix);
  //std::cout << "outcome_index will start and size will be printed" << std::endl;
  for (size_t i = 0; i < outcome_index.size(); ++i) {
    outcome_index[i] = outcome_index[i] - 1;
  }
  
  // std::cout << outcome_index.size() << std::endl;
  data->set_outcome_index(outcome_index);
  
  if(use_sample_weights) {
    data->set_weight_index(sample_weight_index - 1);
  }

  if (node_scaling != 0) {
    Rcpp::stop(
        "node.scaling is incompatible with a global RFF projection. "
        "Use response.scaling or a window-level response transformation.");
  }

  // The projection is intentionally completed before sorting or launching any
  // tree threads. Data owns the resulting immutable allocation and every
  // FourierSplittingRule reads it by training-row ID.
  RFFOptions rff_options(num_features,
                         bandwidth,
                         parse_rff_kernel(kernel),
                         rff_seed,
                         num_threads);
  RandomFourierFeatures projection = RandomFourierFeatures::compute(
      *data, rff_options);
  data->set_rff_features(std::shared_ptr<const RandomFourierFeatures>(
      new RandomFourierFeatures(std::move(projection))));

  data->sort();
  //std::cout << "options will start" << std::endl;
  ForestOptions options(num_trees, ci_group_size, sample_fraction, mtry, min_node_size, honesty,
                        honesty_fraction, honesty_prune_leaves, alpha, imbalance_penalty, num_threads, seed, clusters, samples_per_cluster, num_features, bandwidth, node_scaling);
  //std::cout << "trainer.train will start" << std::endl;
  Forest forest = trainer.train(*data, options);
  
  std::vector<Prediction> predictions;
  if (compute_oob_predictions) {
    ForestPredictor predictor = regression_predictor(num_threads, outcome_index.size());
    predictions = predictor.predict_oob(forest, *data, false);
  }
  
  return RcppUtilities::create_forest_object(forest, predictions);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix compute_rff_cpp(Rcpp::NumericMatrix response,
                                    Rcpp::NumericMatrix frequencies,
                                    unsigned int num_threads) {
  const size_t num_rows = response.nrow();
  const size_t response_dimension = response.ncol();
  const size_t num_features = frequencies.nrow();

  if (num_rows == 0 || response_dimension == 0) {
    Rcpp::stop("The response matrix must have non-zero dimensions.");
  }
  if (num_features == 0 ||
      static_cast<size_t>(frequencies.ncol()) != response_dimension) {
    Rcpp::stop(
        "The frequency matrix must be num.rff by ncol(response).");
  }

  RcppData data(response, num_rows, response_dimension);
  std::vector<size_t> outcome_index(response_dimension);
  for (size_t coordinate = 0;
       coordinate < response_dimension;
       ++coordinate) {
    outcome_index[coordinate] = coordinate;
  }
  data.set_outcome_index(outcome_index);

  // R matrices are column-major, whereas the core projector keeps all
  // coordinates of one frequency adjacent. Convert explicitly so this layout
  // remains independent of R's storage convention.
  std::vector<double> frequency_values(num_features * response_dimension);
  for (size_t feature = 0; feature < num_features; ++feature) {
    for (size_t coordinate = 0;
         coordinate < response_dimension;
         ++coordinate) {
      frequency_values[feature * response_dimension + coordinate] =
          frequencies(feature, coordinate);
    }
  }

  RandomFourierFeatures projection = RandomFourierFeatures::compute(
      data, num_features, frequency_values, num_threads);
  Rcpp::NumericMatrix result(num_rows, 2 * num_features);
  for (size_t feature = 0; feature < num_features; ++feature) {
    for (size_t sample = 0; sample < num_rows; ++sample) {
      result(sample, 2 * feature) = projection.get_real(sample, feature);
      result(sample, 2 * feature + 1) =
          projection.get_imaginary(sample, feature);
    }
  }
  return result;
}

// [[Rcpp::export]]
Rcpp::List sdrf_train(Rcpp::NumericMatrix train_matrix,
                      Eigen::SparseMatrix<double> sparse_train_matrix,
                      std::vector<size_t> outcome_index,
                      unsigned int mtry,
                      unsigned int num_trees,
                      unsigned int min_obs,
                      double q,
                      double lambda_max,
                      unsigned int max_depth,
                      std::vector<size_t> psu_ids,
                      std::vector<double> inclusion_probabilities,
                      Rcpp::NumericMatrix resampling_multipliers,
                      unsigned int num_threads,
                      unsigned int seed,
                      Rcpp::NumericMatrix rff_features) {
  std::unique_ptr<Data> data = RcppUtilities::convert_data(
      train_matrix, sparse_train_matrix);
  for (size_t index = 0; index < outcome_index.size(); ++index) {
    outcome_index[index] -= 1;
  }
  data->set_outcome_index(outcome_index);

  const size_t num_rows = data->get_num_rows();
  if (static_cast<size_t>(resampling_multipliers.nrow()) != num_rows ||
      static_cast<size_t>(resampling_multipliers.ncol()) != num_trees) {
    Rcpp::stop("resampling.multipliers must have n rows and num.trees columns.");
  }
  std::vector<double> multiplier_values(num_rows * num_trees);
  for (size_t tree = 0; tree < num_trees; ++tree) {
    for (size_t sample = 0; sample < num_rows; ++sample) {
      multiplier_values[tree * num_rows + sample] =
          resampling_multipliers(sample, tree);
    }
  }
  data->set_survey_data(inclusion_probabilities,
                        psu_ids,
                        multiplier_values,
                        num_trees);

  if (static_cast<size_t>(rff_features.nrow()) != num_rows ||
      rff_features.ncol() == 0 || rff_features.ncol() % 2 != 0) {
    Rcpp::stop("rff.features must have n rows and 2 * num.rff columns.");
  }
  const size_t num_features = rff_features.ncol() / 2;
  std::vector<double> projected_values(num_rows * num_features * 2);
  for (size_t feature = 0; feature < num_features; ++feature) {
    for (size_t sample = 0; sample < num_rows; ++sample) {
      const size_t offset = (feature * num_rows + sample) * 2;
      projected_values[offset] = rff_features(sample, 2 * feature);
      projected_values[offset + 1] = rff_features(sample, 2 * feature + 1);
    }
  }
  RandomFourierFeatures projection = RandomFourierFeatures::from_values(
      num_rows,
      num_features,
      outcome_index.size(),
      projected_values);
  data->set_rff_features(std::shared_ptr<const RandomFourierFeatures>(
      new RandomFourierFeatures(std::move(projection))));

  data->sort();
  ForestOptions options(num_trees,
                        1,
                        1.0,
                        mtry,
                        min_obs,
                        true,
                        q,
                        true,
                        0.0,
                        0.0,
                        num_threads,
                        seed,
                        std::vector<size_t>(),
                        0,
                        num_features,
                        1.0,
                        0,
                        true,
                        q,
                        min_obs,
                        lambda_max,
                        max_depth);
  ForestTrainer trainer = fourier_trainer(outcome_index.size());
  Forest forest = trainer.train(*data, options);
  return RcppUtilities::create_forest_object(forest,
                                              std::vector<Prediction>());
}

// [[Rcpp::export]]
Rcpp::List regression_predict(Rcpp::List forest_object,
                              Rcpp::NumericMatrix train_matrix,
                              Eigen::SparseMatrix<double> sparse_train_matrix,
                              std::vector<size_t> outcome_index,
                              Rcpp::NumericMatrix test_matrix,
                              Eigen::SparseMatrix<double> sparse_test_matrix,
                              unsigned int num_threads,
                              unsigned int estimate_variance) {
  std::unique_ptr<Data> train_data = RcppUtilities::convert_data(train_matrix, sparse_train_matrix);
  
  for (size_t i=0; i < outcome_index.size(); ++i) {
    outcome_index[i] = outcome_index[i] - 1;
  }
  train_data->set_outcome_index(outcome_index);

  std::unique_ptr<Data> data = RcppUtilities::convert_data(test_matrix, sparse_test_matrix);
  Forest forest = RcppUtilities::deserialize_forest(forest_object);

  ForestPredictor predictor = regression_predictor(num_threads, outcome_index.size());
  std::vector<Prediction> predictions = predictor.predict(forest, *train_data, *data, estimate_variance);

  return RcppUtilities::create_prediction_object(predictions);
}

// [[Rcpp::export]]
Rcpp::List regression_predict_oob(Rcpp::List forest_object,
                                  Rcpp::NumericMatrix train_matrix,
                                  Eigen::SparseMatrix<double> sparse_train_matrix,
                                  std::vector<size_t> outcome_index,
                                  unsigned int num_threads,
                                  bool estimate_variance) {
  //std::cout << "converting data" << std::endl;
  std::unique_ptr<Data> data = RcppUtilities::convert_data(train_matrix, sparse_train_matrix);
  
  for (size_t i=0; i < outcome_index.size(); ++i) {
    outcome_index[i] = outcome_index[i] - 1;
  }
  data->set_outcome_index(outcome_index);

  Forest forest = RcppUtilities::deserialize_forest(forest_object);
   
  ForestPredictor predictor = regression_predictor(num_threads, outcome_index.size());
  //std::cout << "predicting oob" << std::endl;
  std::vector<Prediction> predictions = predictor.predict_oob(forest, *data, estimate_variance);
  //std::cout << "creating predictions" << std::endl;
  Rcpp::List result = RcppUtilities::create_prediction_object(predictions);
  return result;
}
