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

#ifndef HOROVOD_GAUSSIAN_PROCESS_H
#define HOROVOD_GAUSSIAN_PROCESS_H

#include <Eigen/Cholesky>
#include <vector>

namespace horovod {
namespace common {

class GaussianProcessRegressor {
public:
  GaussianProcessRegressor(double alpha);

  ~GaussianProcessRegressor() {}

  void Fit(std::shared_ptr<Eigen::VectorXd> x_train, std::shared_ptr<Eigen::VectorXd> y_train);

  Eigen::MatrixXd Kernel(const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2, double l=1.0, double sigma_f=1.0);

private:
  double alpha_;

  std::shared_ptr<Eigen::VectorXd> x_train_;
  std::shared_ptr<Eigen::VectorXd> y_train_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_GAUSSIAN_PROCESS_H
