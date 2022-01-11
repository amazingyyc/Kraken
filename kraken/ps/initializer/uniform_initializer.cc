#include "ps/initializer/uniform_initializer.h"

#include "common/log.h"

namespace kraken {

UniformInitializer::UniformInitializer(
    InitializerType type,
    const std::unordered_map<std::string, std::string>& conf)
    : Initializer(type, conf), lower_(0.0), upper_(1.0) {
  GetConf<float>("lower", &lower_);
  GetConf<float>("upper", &upper_);

  LOG_INFO("Uniform Initializer, lower:" << lower_ << ", upper:" << upper_);
}

void UniformInitializer::Initialize(Tensor* val) const {
  val->Uniform(lower_, upper_);
}

}  // namespace kraken
