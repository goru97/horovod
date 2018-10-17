// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef HOROVOD_BAYESIAN_OPTIMIZATION_H
#define HOROVOD_BAYESIAN_OPTIMIZATION_H

#include <random>

#include <Eigen/Core>

#include "gaussian_process.h"

namespace horovod {
namespace common {

class BayesianOptimization {
public:
  BayesianOptimization(int d, std::vector<std::pair<double, double>> bounds, double alpha);

  void AddSample(const Eigen::VectorXd& x, double y);

  void AddSample(const Eigen::VectorXd& x, const Eigen::VectorXd& y);

  Eigen::VectorXd NextSample();

private:
  // Proposes the next sampling point by optimizing the acquisition function.
  //
  // Args:
  //  acquisition: Acquisition function.
  //  X_sample: Sample locations (n x d).
  //  Y_sample: Sample values (n x 1).
  //  gpr: A GaussianProcessRegressor fitted to samples.
  //
  // Returns: Location of the acquisition function maximum.
  Eigen::VectorXd ProposeLocation(
      const Eigen::MatrixXd& x_sample, const Eigen::MatrixXd& y_sample, int n_restarts=25);

  // Computes the EI at points X based on existing samples X_sample and Y_sample
  // using a Gaussian process surrogate model fitted to the samples.
  //
  // Args:
  //  x: Points at which EI shall be computed (m x d).
  //  x_sample: Sample locations (n x d).
  //  y_sample: Sample values (n x 1).
  //  xi: Exploitation-exploration trade-off parameter.
  //
  // Returns: Expected improvements at points X. '''
  Eigen::VectorXd ExpectedImprovement(const Eigen::MatrixXd& x, const Eigen::MatrixXd& x_sample, double xi=0.1);

  bool CheckBounds(const Eigen::VectorXd& x);

  int d_;
  std::vector<std::pair<double, double>> bounds_;

  std::random_device rd_;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen_ = std::mt19937(rd_()); // Standard mersenne_twister_engine seeded with rd()
  std::vector<std::uniform_real_distribution<>> dists_;

  GaussianProcessRegressor gpr_;
  std::vector<Eigen::VectorXd> x_samples_;
  std::vector<Eigen::VectorXd> y_samples_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_BAYESIAN_OPTIMIZATION_H
