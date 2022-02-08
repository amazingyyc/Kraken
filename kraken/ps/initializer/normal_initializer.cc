#include "ps/initializer/normal_initializer.h"

#include "common/log.h"

namespace kraken {

NormalInitializer::NormalInitializer(
    const std::unordered_map<std::string, std::string>& conf)
    : Initializer(InitializerType::kNormal, conf), mean_(0.0), stddev_(1.0) {
  GetConf<float>("mean", &mean_);
  GetConf<float>("stddev", &stddev_);

  LOG_INFO("Normal Initializer, mean:" << mean_ << ", stddev:" << stddev_);
}

void NormalInitializer::Initialize(Tensor* val) const {
  val->Normal(mean_, stddev_);
}

}  // namespace kraken
