#include "ps/optim/adagrad.h"

#include <sstream>

#include "common/error_code.h"
#include "common/log.h"
#include "ps/table.h"

namespace kraken {

Adagrad::Adagrad(bool has_weight_decay, float weight_decay, float eps)
    : Optim(OptimType::kAdagrad),
      has_weight_decay_(has_weight_decay),
      weight_decay_(weight_decay),
      eps_(eps) {
}

int32_t Adagrad::Update(const Tensor& grad, float lr, Value* value) const {
  // Grad maybe Coo tensor.
  Tensor grad_t = grad;
  if (grad_t.IsCoo()) {
    if (grad_t.indices().IsEmpty()) {
      return ErrorCode::kSuccess;
    }

    grad_t = grad_t.ToDense();
  }

  if (grad_t.Size() != value->val.Size() ||
      grad_t.element_type() != value->val.element_type()) {
    return ErrorCode::kGradientUnCompatibleError;
  }

  if (value->states.find(StateType::kStateSum) == value->states.end()) {
    value->states.emplace(StateType::kStateSum, grad_t.Like().Zero());
  }

  if (has_weight_decay_) {
    grad_t += weight_decay_ * (value->val);
  }

  Tensor& state_sum = value->states[StateType::kStateSum];
  state_sum += grad_t.Square();

  (value->val) -= (lr * (grad_t / (state_sum.Sqrt() + eps_)));

  return ErrorCode::kSuccess;
}

}  // namespace kraken
