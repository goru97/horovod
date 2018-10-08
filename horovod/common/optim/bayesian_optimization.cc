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

double erf(double x) {
  double y = 1.0 / ( 1.0 + 0.3275911 * x);
  return 1 - (((((
      + 1.061405429  * y
      - 1.453152027) * y
                 + 1.421413741) * y
                - 0.284496736) * y
               + 0.254829592) * y)
             * exp (-x * x);
}

BayesianOptimization::BayesianOptimization(double alpha) : gpr_(GaussianProcessRegressor(alpha)) {}

void BayesianOptimization::AddSample(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
  x_samples_.push_back(x);
  y_samples_.push_back(y);
}

VectorXd BayesianOptimization::NextSample(const MatrixXd& x) {
  MatrixXd x_sample(x_samples_.size(), x_samples_[0].size());
  for (int i = 0; i < x_samples_.size(); i++) {
    x_sample.row(i) = x_samples_[i];
  }

  MatrixXd y_sample(y_samples_.size(), y_samples_[0].size());
  for (int i = 0; i < y_samples_.size(); i++) {
    y_sample.row(i) = y_samples_[i];
  }

  gpr_.Fit(&x_sample, &y_sample);

  return ExpectedImprovement(x, x_sample);
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
    return 0.5 * (1.0 + erf(x / M_SQRT2));
  };

  Eigen::VectorXd imp = mu.array() - mu_sample_opt - xi;
  VectorXd z = imp.array() / sigma.array();
  VectorXd ei = imp.cwiseProduct(z.unaryExpr(cdf)) + sigma.cwiseProduct(z.unaryExpr(pdf));
  ei = (sigma.array() != 0).select(ei, 0.0);
  return ei;
}

} // namespace common
} // namespace horovod
