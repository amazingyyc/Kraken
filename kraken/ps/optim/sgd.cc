#include "ps/optim/sgd.h"

#include <iostream>
#include <sstream>
#include <string>

#include "common/error_code.h"
#include "common/log.h"

namespace kraken {

SGD::SGD(const std::unordered_map<std::string, std::string>& conf)
    : Optim(OptimType::kSGD, conf),
      has_weight_decay_(false),
      has_momentum_(false),
      has_dampening_(false),
      nesterov_(false) {
  if (GetConf<float>("weight_decay", &weight_decay_)) {
    has_weight_decay_ = true;
  }

  if (GetConf<float>("momentum", &momentum_)) {
    has_momentum_ = true;
  }

  if (GetConf<float>("dampening", &dampening_)) {
    has_dampening_ = true;
  }

  GetConf<bool>("nesterov", &nesterov_);

  std::ostringstream oss;
  oss << "Create SGD optim";

  if (has_weight_decay_) {
    oss << ", weight_decay:" << weight_decay_;
  } else {
    oss << ", not set weight_decay";
  }

  if (has_momentum_) {
    oss << ", momentum:" << momentum_;
  } else {
    oss << ", not set momentum.";
  }

  if (has_dampening_) {
    oss << ", dampening:" << dampening_;
  } else {
    oss << ", not set dampening";
  }

  if (nesterov_) {
    oss << ", nesterov: true.";
  } else {
    oss << ", nesterov: false.";
  }

  LOG_INFO(oss.str());
}

int32_t SGD::Update(const Tensor& grad, float lr, Tensor* val, Bag* bag) const {
  // Grad maybe Coo tensor.
  Tensor grad_t = grad;
  if (grad_t.IsCoo()) {
    if (grad_t.indices().IsEmpty()) {
      return ErrorCode::kSuccess;
    }

    grad_t = grad_t.ToDense();
  }

  if (grad_t.Size() != val->Size() ||
      grad_t.element_type() != val->element_type()) {
    return ErrorCode::kGradientUnCompatibleError;
  }

  if (has_weight_decay_) {
    grad_t += weight_decay_ * (*val);
  }

  if (has_momentum_) {
    if (bag->state.find(StateType::kMomentumBuffer) == bag->state.end()) {
      bag->state.emplace(StateType::kMomentumBuffer, grad_t.Clone());
    } else {
      Tensor& mb = bag->state[StateType::kMomentumBuffer];
      mb = momentum_ * mb + (1.0 - dampening_) * grad_t;
    }

    if (nesterov_) {
      grad_t += momentum_ * bag->state[StateType::kMomentumBuffer];
    } else {
      grad_t = bag->state[StateType::kMomentumBuffer];
    }
  }

  *val -= (grad_t * lr);

  return ErrorCode::kSuccess;
}

}  // namespace kraken
