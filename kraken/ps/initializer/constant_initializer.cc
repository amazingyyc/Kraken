#include "ps/initializer/constant_initializer.h"

#include "common/log.h"

namespace kraken {

ConstantInitializer::ConstantInitializer(float value)
    : Initializer(InitializerType::kConstant), value_(value) {
}

std::unordered_map<std::string, std::string> ConstantInitializer::conf() const {
  return {{"value", std::to_string(value_)}};
}

void ConstantInitializer::Initialize(Tensor* val) const {
  val->Constant(value_);
}

}  // namespace kraken
