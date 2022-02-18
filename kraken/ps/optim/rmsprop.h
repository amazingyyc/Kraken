#pragma once

#include <string>

#include "ps/optim/optim.h"
#include "t/tensor.h"

namespace kraken {

// ref:
// https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop
class RMSprop : public Optim {
private:
  bool has_weight_decay_;
  float weight_decay_;

  bool has_momentum_;
  float momentum_;

  float alpha_;

  float eps_;

  bool centered_;

public:
  RMSprop(bool has_weight_decay, float weight_decay, bool has_momentum,
          float momentum, float alpha, float eps, bool centered);

  int32_t Update(const Tensor& grad, float lr, Tensor* val,
                 Bag* bag) const override;
};

}  // namespace kraken
