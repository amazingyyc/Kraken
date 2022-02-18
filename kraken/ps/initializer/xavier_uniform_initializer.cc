#include "ps/initializer/xavier_uniform_initializer.h"

#include "common/log.h"

namespace kraken {

XavierUniformInitializer::XavierUniformInitializer(float gain)
    : Initializer(InitializerType::kXavierUniform), gain_(gain) {
}

std::unordered_map<std::string, std::string> XavierUniformInitializer::conf()
    const {
  return {{"gain", std::to_string(gain_)}};
}

void XavierUniformInitializer::Initialize(Tensor* val) const {
  val->XavierUniform(gain_);
}

}  // namespace kraken
