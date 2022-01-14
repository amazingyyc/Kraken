#include "ps/initializer/xavier_normal_initializer.h"

#include "common/log.h"

namespace kraken {

XavierNormalInitializer::XavierNormalInitializer(
    const std::unordered_map<std::string, std::string>& conf)
    : Initializer(InitializerType::kXavierNormal, conf), gain_(1.0) {
  GetConf<float>("gain", &gain_);

  LOG_INFO("XavierNormal Initializer, gain:" << gain_);
}

void XavierNormalInitializer::Initialize(Tensor* val) const {
  val->XavierNormal(gain_);
}

}  // namespace kraken
