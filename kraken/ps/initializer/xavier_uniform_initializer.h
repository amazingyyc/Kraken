#pragma once

#include <cinttypes>
#include <string>
#include <unordered_map>

#include "common/utils.h"
#include "ps/initializer/initializer.h"
#include "t/tensor.h"

namespace kraken {

class XavierUniformInitializer : public Initializer {
private:
  float gain_;

public:
  XavierUniformInitializer(float gain);

  std::unordered_map<std::string, std::string> conf() const override;

  void Initialize(Tensor* val) const override;
};

}  // namespace kraken
