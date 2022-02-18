#include "ps/initializer/uniform_initializer.h"

#include "common/log.h"

namespace kraken {

UniformInitializer::UniformInitializer(float lower, float upper)
    : Initializer(InitializerType::kUniform), lower_(lower), upper_(upper) {
}

std::unordered_map<std::string, std::string> UniformInitializer::conf() const {
  return {{"lower", std::to_string(lower_)}, {"upper", std::to_string(upper_)}};
}

void UniformInitializer::Initialize(Tensor* val) const {
  val->Uniform(lower_, upper_);
}

}  // namespace kraken
