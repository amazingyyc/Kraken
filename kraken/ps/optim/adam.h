#pragma once

#include <string>
#include <unordered_map>

#include "ps/optim/optim.h"
#include "t/tensor.h"

namespace kraken {

// ref:
// https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
class Adam : public Optim {
private:
  bool has_weight_decay_;
  float weight_decay_;

  float beta1_;
  float beta2_;

  float eps_;

  bool amsgrad_;

public:
  Adam(bool has_weight_decay, float weight_decay, float beta1, float beta2,
       float eps, bool amsgrad);

  int32_t Update(const Tensor& grad, float lr, Tensor* val,
                 Bag* bag) const override;
};

}  // namespace kraken
