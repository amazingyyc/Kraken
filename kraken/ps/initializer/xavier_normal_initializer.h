#pragma once

#include <cinttypes>
#include <unordered_map>

#include "common/tensor.h"
#include "common/utils.h"
#include "ps/initializer/initializer.h"

namespace kraken {

class XavierNormalInitializer : public Initializer {
private:
  float gain_;

public:
  XavierNormalInitializer(
      InitializerType type,
      const std::unordered_map<std::string, std::string>& conf);

  void Initialize(Tensor* val) const override;
};

}  // namespace kraken
