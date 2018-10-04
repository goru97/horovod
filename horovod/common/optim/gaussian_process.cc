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

#include <Eigen/LU>

#include <cmath>
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
void GaussianProcessRegressor::Fit(MatrixXd* x_train, MatrixXd* y_train) {
  x_train_ = x_train;
  y_train_ = y_train;

  auto ln = [](double x) {
    return std::log(x);
  };

  double a2 = alpha_ * alpha_;
  double d3 = 0.5 * x_train_->rows() * std::log(2 * M_PI);
//  auto nll_fn = [&, a2, d3](const VectorXd& x, VectorXd& grad) {
//    auto k = Kernel(*x_train_, *x_train_, 1.0, 1.0) + a2 * MatrixXd::Identity(x_train_->rows(), x_train_->rows());
//
//    // Compute determinant via Cholesky decomposition
//    Eigen::LLT<MatrixXd> llt(k);
//    auto l = llt.matrixL();
//    double d1 = l.toDenseMatrix().diagonal().unaryExpr(ln).sum();
//    double d2 = 0.5 * y_train_->transpose().dot(k.inverse() * (*y_train_));
//    return d1 + d2 + d3;
//  };

  auto step = [&, a2, d3]() {
    MatrixXd k = Kernel(*x_train_, *x_train_, 1.0, 1.0) + (a2 * MatrixXd::Identity(x_train_->rows(), x_train_->rows()));
    MatrixXd k_inv = k.inverse();

    // Compute determinant via Cholesky decomposition
    MatrixXd l = k.llt().matrixL().toDenseMatrix();
    double d1 = l.diagonal().unaryExpr(ln).sum();
    MatrixXd d2 = 0.5 * (y_train_->transpose() * (k_inv * (*y_train_)));
    MatrixXd cov = d2.array() + (d1 + d3);

    return cov;
  };

  MatrixXd step_out = step();
  std::cout << "step: " << step_out << std::endl;
}

} // namespace common
} // namespace horovod