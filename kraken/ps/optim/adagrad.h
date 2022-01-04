#pragma once

#include <string>
#include <unordered_map>

#include "common/tensor.h"
#include "ps/optim/optim.h"

namespace kraken {

// ref: https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad
class Adagrad : public Optim {
private:
  bool has_weight_decay_;
  float weight_decay_;

  float eps_;

public:
  Adagrad(const std::unordered_map<std::string, std::string>& conf);

  int32_t Update(const Tensor& grad, float lr, Tensor* val,
                 Bag* bag) const override;
};

}  // namespace kraken