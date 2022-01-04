#pragma once

#include <string>
#include <unordered_map>

#include "common/tensor.h"
#include "ps/optim/optim.h"

namespace kraken {

// ref: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
class Adam : public Optim {
private:
  bool has_weight_decay_;
  float weight_decay_;

  float beta1_;
  float beta2_;

  float eps_;

  bool amsgrad_;

public:
  Adam(const std::unordered_map<std::string, std::string>& conf);

  int32_t Update(const Tensor& grad, float lr, Tensor* val,
                 Bag* bag) const override;
};

}  // namespace kraken
