#pragma once

#include <cinttypes>
#include <unordered_map>

#include "common/tensor.h"
#include "common/utils.h"
#include "ps/initializer/initializer.h"

namespace kraken {

class ConstantInitializer : public Initializer {
private:
  float value_;

public:
  ConstantInitializer(InitializerType type,
                      const std::unordered_map<std::string, std::string>& conf);

  void Initialize(Tensor* val) const override;
};

}  // namespace kraken
