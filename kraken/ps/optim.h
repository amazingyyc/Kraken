#pragma once

#include "common/tensor.h"

namespace kraken {

class Optim {
public:
  virtual ~Optim() = default;

  virtual bool Update(const Tensor& grad, float lr, Tensor* var) = 0;
};

}  // namespace kraken
