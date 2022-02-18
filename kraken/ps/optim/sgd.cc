#include "ps/optim/sgd.h"

#include <iostream>
#include <sstream>
#include <string>

#include "common/error_code.h"
#include "common/log.h"
#include "ps/table.h"

namespace kraken {

SGD::SGD(bool has_weight_decay, float weight_decay, bool has_momentum,
         float momentum, bool has_dampening, float dampening, bool nesterov)
    : Optim(OptimType::kSGD),
      has_weight_decay_(has_weight_decay),
      weight_decay_(weight_decay),
      has_momentum_(has_momentum),
      momentum_(momentum),
      has_dampening_(has_dampening),
      dampening_(dampening),
      nesterov_(nesterov) {
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
