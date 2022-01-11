#pragma once

#include <cinttypes>
#include <unordered_map>

#include "common/tensor.h"
#include "common/utils.h"
#include "ps/initializer/initializer.h"

namespace kraken {

class UniformInitializer : public Initializer {
private:
  float lower_;
  float upper_;

public:
  UniformInitializer(InitializerType type,
                     const std::unordered_map<std::string, std::string>& conf);

  void Initialize(Tensor* val) const override;
};

}  // namespace kraken
