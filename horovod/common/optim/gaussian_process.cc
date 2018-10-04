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

#include "gaussian_process.h"

#include <iostream>
#include <random>

using Eigen::VectorXd;
using Eigen::MatrixXd;


namespace horovod {
namespace common {

// Isotropic squared exponential kernel.
// Computes a covariance matrix from points in X1 and X2.
//
// Args:
//  X1: Array of m points (m x d).
//  X2: Array of n points (n x d).
//
// Returns: Covariance matrix (m x n).
Eigen::MatrixXd GaussianProcessRegressor::Kernel(const MatrixXd& x1, const MatrixXd& x2,
                                                 double l, double sigma_f) {
  auto x1_vec = x1.cwiseProduct(x1).rowwise().sum();
  auto x2_vec = x2.cwiseProduct(x2).rowwise().sum();
  auto x1_x2 = x1_vec.replicate(1, x2_vec.size()).rowwise() + x2_vec.transpose();

  auto& dot = x1 * x2.transpose();
  auto sqdist = x1_x2 - (dot.array() * 2).matrix();

  double sigma_f2 = sigma_f * sigma_f;
  double l2 = l * l;
  auto op = [sigma_f2, l2](double x) {
    return sigma_f2 * std::exp(-0.5 / l2 * x);
  };
  return sqdist.unaryExpr(op);
}

GaussianProcessRegressor::GaussianProcessRegressor(double alpha) : alpha_(alpha) {}

// Evaluate mean and variance at a point.
void GaussianProcessRegressor::Fit(std::shared_ptr<VectorXd> x_train, std::shared_ptr<VectorXd> y_train) {
  x_train_ = x_train;
  y_train_ = y_train;

  auto nll_fn = [&](const VectorXd& x, VectorXd& grad) {

  };
}

} // namespace common
} // namespace horovod