#pragma once

#include <cinttypes>
#include <unordered_map>

#include "common/utils.h"
#include "ps/initializer/initializer.h"
#include "t/tensor.h"

namespace kraken {

class NormalInitializer : public Initializer {
private:
  float mean_;
  float stddev_;

public:
  NormalInitializer(float mean, float stddev);

  NormalInitializer(const std::unordered_map<std::string, std::string>& conf);

  void Initialize(Tensor* val) const override;
};

}  // namespace kraken
