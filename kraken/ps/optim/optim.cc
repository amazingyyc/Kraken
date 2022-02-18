#include "ps/optim/optim.h"

#include "ps/optim/adagrad.h"
#include "ps/optim/adam.h"
#include "ps/optim/rmsprop.h"
#include "ps/optim/sgd.h"

namespace kraken {

Optim::Optim(OptimType optim_type) : optim_type_(optim_type) {
}

std::unique_ptr<Optim> Optim::Create(
    OptimType optim_type,
    const std::unordered_map<std::string, std::string>& optim_conf) {
  std::unique_ptr<Optim> optim;

  if (optim_type == OptimType::kAdagrad) {
    bool has_weight_decay = false;
    float weight_decay = 0.0;
    float eps = 1e-10;

    if (utils::ParseConf<float>(optim_conf, "weight_decay", &weight_decay)) {
      has_weight_decay = true;
    }

    utils::ParseConf<float>(optim_conf, "eps", &eps);

    optim.reset(new Adagrad(has_weight_decay, weight_decay, eps));
  } else if (optim_type == OptimType::kAdam) {
    bool has_weight_decay = false;
    float weight_decay = 0.0;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float eps = 1e-08;
    bool amsgrad = false;

    if (utils::ParseConf<float>(optim_conf, "weight_decay", &weight_decay)) {
      has_weight_decay = true;
    }

    utils::ParseConf<float>(optim_conf, "beta1", &beta1);
    utils::ParseConf<float>(optim_conf, "beta2", &beta2);
    utils::ParseConf<float>(optim_conf, "eps", &eps);
    utils::ParseConf<bool>(optim_conf, "amsgrad", &amsgrad);

    optim.reset(
        new Adam(has_weight_decay, weight_decay, beta1, beta2, eps, amsgrad));
  } else if (optim_type == OptimType::kRMSprop) {
    bool has_weight_decay = false;
    float weight_decay = 0.0;
    bool has_momentum = false;
    float momentum = 0.0;
    float alpha = 0.99;
    float eps = 1e-8;
    bool centered = false;

    if (utils::ParseConf<float>(optim_conf, "weight_decay", &weight_decay)) {
      has_weight_decay = true;
    }

    if (utils::ParseConf<float>(optim_conf, "momentum", &momentum)) {
      has_momentum = true;
    }

    utils::ParseConf<float>(optim_conf, "alpha", &alpha);
    utils::ParseConf<float>(optim_conf, "eps", &eps);
    utils::ParseConf<bool>(optim_conf, "centered", &centered);

    optim.reset(new RMSprop(has_weight_decay, weight_decay, has_momentum,
                            momentum, alpha, eps, centered));
  } else if (optim_type == OptimType::kSGD) {
    bool has_weight_decay = false;
    float weight_decay = 0.0;
    bool has_momentum = false;
    float momentum = 0.0;
    bool has_dampening = false;
    float dampening = 0.0;
    bool nesterov = false;

    if (utils::ParseConf<float>(optim_conf, "weight_decay", &weight_decay)) {
      has_weight_decay = true;
    }

    if (utils::ParseConf<float>(optim_conf, "momentum", &momentum)) {
      has_momentum = true;
    }

    if (utils::ParseConf<float>(optim_conf, "dampening", &dampening)) {
      has_dampening = true;
    }

    utils::ParseConf<bool>(optim_conf, "nesterov", &nesterov);

    optim.reset(new SGD(has_weight_decay, weight_decay, has_momentum, momentum,
                        has_dampening, dampening, nesterov));
  }

  return optim;
}

}  // namespace kraken
