#include "ps/initializer/normal_initializer.h"

#include "common/log.h"

namespace kraken {

NormalInitializer::NormalInitializer(float mean, float stddev)
    : Initializer(InitializerType::kNormal), mean_(mean), stddev_(stddev) {
}

std::unordered_map<std::string, std::string> NormalInitializer::conf() const {
  return {{"mean", std::to_string(mean_)}, {"stddev", std::to_string(stddev_)}};
}

void NormalInitializer::Initialize(Tensor* val) const {
  val->Normal(mean_, stddev_);
}

}  // namespace kraken
