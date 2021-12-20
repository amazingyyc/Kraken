#pragma once

#include "common/tensor.h"
#include "ps/optim.h"

namespace kraken {

class SGDOptim : public Optim {
public:
  bool Update(const Tensor& grad, float lr, Tensor* var) override;
};

}  // namespace kraken
