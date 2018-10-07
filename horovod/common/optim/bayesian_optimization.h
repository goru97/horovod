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

#include <Eigen/Core>

#include "gaussian_process.h"

namespace horovod {
namespace common {

class BayesianOptimization {
public:
  Eigen::VectorXd ExpectedImprovement(
      const Eigen::MatrixXd& x, const Eigen::MatrixXd& x_sample, const Eigen::MatrixXd& y_sample, double xi=0.1);

private:
  GaussianProcessRegressor gpr_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_BAYESIAN_OPTIMIZATION_H
