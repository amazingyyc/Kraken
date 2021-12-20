#include "ps/sgd_optim.h"

namespace kraken {

bool SGDOptim::Update(const Tensor& grad, float lr, Tensor* var) {
  if (grad.size() != var->size() ||
      grad.element_type() != var->element_type()) {
    return false;
  }

  *var -= (grad * lr);

  return true;
}

}  // namespace kraken
