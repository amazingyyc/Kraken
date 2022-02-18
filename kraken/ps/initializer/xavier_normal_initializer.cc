#include "ps/initializer/xavier_normal_initializer.h"

#include "common/log.h"

namespace kraken {

XavierNormalInitializer::XavierNormalInitializer(float gain)
    : Initializer(InitializerType::kXavierNormal), gain_(gain) {
}

std::unordered_map<std::string, std::string> XavierNormalInitializer::conf()
    const {
  return {{"gain", std::to_string(gain_)}};
}

void XavierNormalInitializer::Initialize(Tensor* val) const {
  val->XavierNormal(gain_);
}

}  // namespace kraken
