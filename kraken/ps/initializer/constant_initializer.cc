#include "ps/initializer/constant_initializer.h"

#include "common/log.h"

namespace kraken {

ConstantInitializer::ConstantInitializer(
    InitializerType type,
    const std::unordered_map<std::string, std::string>& conf)
    : Initializer(type, conf), value_(0) {
  // Parse value.
  GetConf<float>("value", &value_);

  LOG_INFO("Constant Initializer, value:" << value_);
}

void ConstantInitializer::Initialize(Tensor* val) const {
  val->Constant(value_);
}

}  // namespace kraken
