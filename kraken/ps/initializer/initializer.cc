#include "ps/initializer/initializer.h"

#include "ps/initializer/constant_initializer.h"
#include "ps/initializer/normal_initializer.h"
#include "ps/initializer/uniform_initializer.h"
#include "ps/initializer/xavier_normal_initializer.h"
#include "ps/initializer/xavier_uniform_initializer.h"

namespace kraken {

Initializer::Initializer(InitializerType type) : type_(type) {
}

InitializerType Initializer::type() const {
  return type_;
}

std::unordered_map<std::string, std::string> Initializer::conf() const {
  return {};
}

std::unique_ptr<Initializer> Initializer::Create(
    InitializerType init_type,
    const std::unordered_map<std::string, std::string>& init_conf) {
  std::unique_ptr<Initializer> initializer;

  if (init_type == InitializerType::kConstant) {
    float value = 0;
    utils::ParseConf<float>(init_conf, "value", &value);

    initializer.reset(new ConstantInitializer(value));
  } else if (init_type == InitializerType::kNormal) {
    float mean = 0.0;
    float stddev = 1.0;

    utils::ParseConf<float>(init_conf, "mean", &mean);
    utils::ParseConf<float>(init_conf, "stddev", &stddev);

    initializer.reset(new NormalInitializer(mean, stddev));
  } else if (init_type == InitializerType::kUniform) {
    float lower = 0.0;
    float upper = 1.0;

    utils::ParseConf<float>(init_conf, "lower", &lower);
    utils::ParseConf<float>(init_conf, "upper", &upper);

    initializer.reset(new UniformInitializer(lower, upper));
  } else if (init_type == InitializerType::kXavierNormal) {
    float gain = 1.0;
    utils::ParseConf<float>(init_conf, "gain", &gain);

    initializer.reset(new XavierNormalInitializer(gain));
  } else if (init_type == InitializerType::kXavierUniform) {
    float gain = 1.0;
    utils::ParseConf<float>(init_conf, "gain", &gain);

    initializer.reset(new XavierUniformInitializer(gain));
  }

  return initializer;
}

}  // namespace kraken
