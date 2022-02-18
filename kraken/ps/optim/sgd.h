#pragma once

#include <string>

#include "ps/optim/optim.h"
#include "t/tensor.h"

namespace kraken {

// ref: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
class SGD : public Optim {
private:
  bool has_weight_decay_;
  float weight_decay_;

  bool has_momentum_;
  float momentum_;

  bool has_dampening_;
  float dampening_;

  bool nesterov_;

public:
  SGD(bool has_weight_decay, float weight_decay, bool has_momentum,
      float momentum, bool has_dampening, float dampening, bool nesterov);

  int32_t Update(const Tensor& grad, float lr, Tensor* val,
                 Bag* bag) const override;
};

}  // namespace kraken
