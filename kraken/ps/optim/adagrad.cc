#include "ps/optim/adagrad.h"

#include <sstream>

#include "common/error_code.h"
#include "common/log.h"

namespace kraken {

Adagrad::Adagrad(const std::unordered_map<std::string, std::string>& conf)
    : Optim(OptimType::kAdagrad, conf), has_weight_decay_(false), eps_(1e-10) {
  if (GetConf<float>("weight_decay", &weight_decay_)) {
    has_weight_decay_ = true;
  }

  GetConf<float>("eps", &eps_);

  std::ostringstream oss;
  oss << "Create Adagrad optim";

  if (has_weight_decay_) {
    oss << ", weight_decay:" << weight_decay_;
  } else {
    oss << ", not set weight_decay";
  }

  oss << ", eps:" << eps_ << ".";

  LOG_INFO(oss.str());
}

int32_t Adagrad::Update(const Tensor& grad, float lr, Tensor* val,
                        Bag* bag) const {
  if (grad.Size() != val->Size() ||
      grad.element_type() != val->element_type()) {
    return ErrorCode::kGradientUnCompatibleError;
  }

  if (bag->state.find(StateType::kStateSum) == bag->state.end()) {
    bag->state.emplace(StateType::kStateSum, grad.Like().Zero());
  }

  // Use = operator grad_t will share memory with grad.
  Tensor grad_t = grad;
  if (has_weight_decay_) {
    grad_t += weight_decay_ * (*val);
  }

  Tensor& state_sum = bag->state[StateType::kStateSum];
  state_sum += grad_t.Square();

  (*val) -= (lr * (grad_t / (state_sum.Sqrt() + eps_)));

  return ErrorCode::kSuccess;
}
}  // namespace kraken