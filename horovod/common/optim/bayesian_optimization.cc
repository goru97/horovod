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

#include "bayesian_optimization.h"

#include <cmath>
#include <iostream>

#include <Eigen/LU>

#include "LBFGS.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace horovod {
namespace common {

const double NORM_PDF_C = std::sqrt(2 * M_PI);

BayesianOptimization::BayesianOptimization(int d, double alpha) : d_(d),
                                                                  dist_(std::uniform_real_distribution<>(-1, 2)),
                                                                  gpr_(GaussianProcessRegressor(alpha)) {}

void BayesianOptimization::AddSample(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
  x_samples_.push_back(x);
  y_samples_.push_back(y);
}

VectorXd BayesianOptimization::NextSample() {
  MatrixXd x_sample(x_samples_.size(), d_);
  for (int i = 0; i < x_samples_.size(); i++) {
    x_sample.row(i) = x_samples_[i];
  }

  MatrixXd y_sample(y_samples_.size(), 1);
  for (int i = 0; i < y_samples_.size(); i++) {
    y_sample.row(i) = y_samples_[i];
  }

  gpr_.Fit(&x_sample, &y_sample);

  return ProposeLocation(x_sample, y_sample);
}

VectorXd BayesianOptimization::ProposeLocation(const MatrixXd& x_sample, const MatrixXd& y_sample, int n_restarts) {
  auto f = [&](const VectorXd& x) {
    // Boundary constraints
    if (x[0] < -1 - 1e-5 || x[0] > 2 + 1e-5) {
      return 1.0;
    }

    // Minimization objective is the negative acquisition function
    return -ExpectedImprovement(x, x_sample)[0];
  };

  auto min_obj = [&](const VectorXd& x, VectorXd& grad) {
    double f0 = f(x);
    GaussianProcessRegressor::ApproxFPrime(x, f, f0, grad);
    return f0;
  };

  LBFGSpp::LBFGSParam<double> param;
  param.epsilon = 1e-5;
  param.max_iterations = 100;

  LBFGSpp::LBFGSSolver<double> solver(param);

//  VectorXd x_next = VectorXd::Zero(1);
//  x_next[0] = -1;
//  double fx_max = -ExpectedImprovement(x_next, x_sample)[0];

  VectorXd x_next;
  double fx_max = 1;
  for (int i = 0; i < n_restarts; i++) {
    VectorXd x = VectorXd::Zero(d_);
    for (int j = 0; j < d_; j++) {
      x[j] = dist_(gen_);
    }

    VectorXd x0 = x;

    double fx;
    solver.minimize(min_obj, x, fx);

    std::cout << "x, fx: " << x0 << " -> " << x << " " << fx << std::endl;

    if (fx < fx_max) {
      fx_max = fx;
      x_next = x;
    }
  }

  std::cout << "x_next = " << x_next.transpose() << std::endl;
  std::cout << "f(x) = " << fx_max << std::endl;

  return x_next;
}

VectorXd BayesianOptimization::ExpectedImprovement(const MatrixXd& x, const MatrixXd& x_sample, double xi) {
  Eigen::VectorXd mu;
  Eigen::VectorXd sigma;
  gpr_.Predict(x, mu, &sigma);

  Eigen::VectorXd mu_sample;
  gpr_.Predict(x_sample, mu_sample);

  // Needed for noise-based model, otherwise use np.max(Y_sample).
  // See also section 2.4 in [...]
  double mu_sample_opt = mu_sample.maxCoeff();

  auto pdf = [](double x) {
    return std::exp(-(x * x) / 2.0) / NORM_PDF_C;
  };

  auto cdf = [](double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
  };

  Eigen::VectorXd imp = mu.array() - mu_sample_opt - xi;

  VectorXd z = imp.array() / sigma.array();

  VectorXd ei = imp.cwiseProduct(z.unaryExpr(cdf)) + sigma.cwiseProduct(z.unaryExpr(pdf));
  ei = (sigma.array() != 0).select(ei, 0.0);
  return ei;
}

} // namespace common
} // namespace horovod
