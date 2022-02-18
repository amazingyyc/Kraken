#pragma once

#include <cinttypes>
#include <string>
#include <unordered_map>

#include "common/utils.h"
#include "ps/initializer/initializer.h"
#include "t/tensor.h"

namespace kraken {

class UniformInitializer : public Initializer {
private:
  float lower_;
  float upper_;

public:
  UniformInitializer(float lower, float upper);

  std::unordered_map<std::string, std::string> conf() const override;

  void Initialize(Tensor* val) const override;
};

}  // namespace kraken
