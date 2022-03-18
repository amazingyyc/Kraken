#include "ps/optim/rmsprop.h"

#include <sstream>

#include "common/error_code.h"
#include "common/log.h"
#include "ps/table.h"

namespace kraken {

RMSprop::RMSprop(bool has_weight_decay, float weight_decay, bool has_momentum,
                 float momentum, float alpha, float eps, bool centered)
    : Optim(OptimType::kRMSprop),
      has_weight_decay_(has_weight_decay),
      weight_decay_(weight_decay),
      has_momentum_(has_momentum),
      momentum_(momentum),
      alpha_(alpha),
      eps_(eps),
      centered_(centered) {
}

int32_t RMSprop::Update(const Tensor& grad, float lr, Value* value) const {
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

  if (has_weight_decay_) {
    grad_t += weight_decay_ * value->val;
  }

  if (value->states.find(StateType::kSquareAverage) == value->states.end()) {
    value->states.emplace(StateType::kSquareAverage, grad_t.Like().Zero());
  }

  Tensor vt = value->states[StateType::kSquareAverage];
  vt = alpha_ * vt + (1.0 - alpha_) * grad_t.Square();

  // Update
  value->states[StateType::kSquareAverage] = vt;

  if (centered_) {
    if (value->states.find(StateType::kGAve) == value->states.end()) {
      value->states.emplace(StateType::kGAve, grad_t.Like().Zero());
    }

    Tensor& gave = value->states[StateType::kGAve];
    gave = gave * alpha_ + (1.0 - alpha_) * grad_t;

    vt = vt - gave.Square();
  }

  if (has_momentum_) {
    if (value->states.find(StateType::kMomentumBuffer) == value->states.end()) {
      value->states.emplace(StateType::kMomentumBuffer, grad_t.Like().Zero());
    }

    Tensor& bt = value->states[StateType::kMomentumBuffer];
    bt = bt * momentum_ + grad_t / (vt.Sqrt() + eps_);

    (value->val) -= (lr * bt);
  } else {
    (value->val) -= (lr * grad_t / (vt.Sqrt() + eps_));
  }

  return ErrorCode::kSuccess;
}

}  // namespace kraken
