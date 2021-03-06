#pragma once

#include <string>
#include <unordered_map>

#include "ps/optim/optim.h"
#include "t/tensor.h"

namespace kraken {

// ref:
// https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad
class Adagrad : public Optim {
private:
  bool has_weight_decay_;
  float weight_decay_;

  float eps_;

public:
  Adagrad(bool has_weight_decay, float weight_decay, float eps);

  int32_t Update(const Tensor& grad, float lr, Value* value) const override;
};

}  // namespace kraken
