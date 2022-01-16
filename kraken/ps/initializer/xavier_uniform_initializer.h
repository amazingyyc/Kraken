#pragma once

#include <cinttypes>
#include <unordered_map>

#include "common/utils.h"
#include "ps/initializer/initializer.h"
#include "t/tensor.h"

namespace kraken {

class XavierUniformInitializer : public Initializer {
private:
  float gain_;

public:
  XavierUniformInitializer(
      const std::unordered_map<std::string, std::string>& conf);

  void Initialize(Tensor* val) const override;
};

}  // namespace kraken
