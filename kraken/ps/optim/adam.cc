#include "ps/optim/adam.h"

#include <cmath>
#include <sstream>

#include "common/error_code.h"
#include "common/log.h"
#include "ps/table.h"

namespace kraken {

Adam::Adam(bool has_weight_decay, float weight_decay, float beta1, float beta2,
           float eps, bool amsgrad)
    : Optim(OptimType::kAdam),
      has_weight_decay_(has_weight_decay),
      weight_decay_(weight_decay),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      amsgrad_(amsgrad) {
}

int32_t Adam::Update(const Tensor& grad, float lr, Value* value) const {
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

  // First moment and second moment.
  if (value->states.find(StateType::kFirstMoment) == value->states.end()) {
    value->states.emplace(StateType::kFirstMoment, grad_t.Like().Zero());
  }

  if (value->states.find(StateType::kSecondMoment) == value->states.end()) {
    value->states.emplace(StateType::kSecondMoment, grad_t.Like().Zero());
  }

  // Weight decay.
  if (has_weight_decay_) {
    grad_t += weight_decay_ * (value->val);
  }

  Tensor& m = value->states[StateType::kFirstMoment];
  Tensor& v = value->states[StateType::kSecondMoment];

  m = beta1_ * m + (1.0 - beta1_) * grad_t;
  v = beta2_ * v + (1.0 - beta2_) * grad_t.Square();

  // step
  int64_t steps = ++(value->states_i[StateType::kSteps]);

  Tensor mt = m / (1.0 - std::pow(beta1_, float(steps)));
  Tensor vt = v / (1.0 - std::pow(beta2_, float(steps)));

  if (amsgrad_) {
    // Get max SecondMomentMax.
    if (value->states.find(StateType::kSecondMomentMax) ==
        value->states.end()) {
      value->states.emplace(StateType::kSecondMomentMax, grad_t.Like().Zero());
    }

    Tensor& v_max = value->states[StateType::kSecondMomentMax];
    v_max = v_max.Max(vt);

    (value->val) -= (lr * mt / (v_max.Sqrt() + eps_));
  } else {
    (value->val) -= (lr * mt / (vt.Sqrt(true) + eps_));
  }

  return ErrorCode::kSuccess;
}

}  // namespace kraken
