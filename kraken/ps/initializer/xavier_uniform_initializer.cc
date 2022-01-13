#include "ps/initializer/xavier_uniform_initializer.h"

#include "common/log.h"

namespace kraken {

XavierUniformInitializer::XavierUniformInitializer(
    const std::unordered_map<std::string, std::string>& conf)
    : Initializer(InitializerType::kXavierUniform, conf), gain_(1.0) {
  GetConf<float>("gain", &gain_);

  LOG_INFO("XavierUniform Initializer, gain:" << gain_);
}

void XavierUniformInitializer::Initialize(Tensor* val) const {
  val->XavierUniform(gain_);
}

}  // namespace kraken