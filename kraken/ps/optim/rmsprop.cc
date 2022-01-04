#include "ps/optim/rmsprop.h"

#include <sstream>

#include "common/error_code.h"
#include "common/log.h"

namespace kraken {

RMSprop::RMSprop(const std::unordered_map<std::string, std::string>& conf)
    : Optim(OptimType::kRMSprop, conf),
      has_weight_decay_(false),
      has_momentum_(false),
      alpha_(0.99),
      eps_(1e-8),
      centered_(false) {
  if (GetConf<float>("weight_decay", &weight_decay_)) {
    has_weight_decay_ = true;
  }

  if (GetConf<float>("momentum", &momentum_)) {
    has_momentum_ = true;
  }

  GetConf<float>("alpha", &alpha_);
  GetConf<float>("eps", &eps_);
  GetConf<bool>("centered", &centered_);

  std::ostringstream oss;
  oss << "Create RMSprop optim";

  oss << ", alpha:" << alpha_ << ".";
  oss << ", eps:" << eps_ << ".";

  if (centered_) {
    oss << ", centered: true.";
  } else {
    oss << ", centered: false.";
  }

  LOG_INFO(oss.str());
}

int32_t RMSprop::Update(const Tensor& grad, float lr, Tensor* val,
                        Bag* bag) const {
  if (grad.Size() != val->Size() ||
      grad.element_type() != val->element_type()) {
    return ErrorCode::kGradientUnCompatibleError;
  }

  // Use = operator grad_t will share memory with grad.
  Tensor grad_t = grad;
  if (has_weight_decay_) {
    grad_t += weight_decay_ * (*val);
  }

  if (bag->state.find(StateType::kSquareAverage) == bag->state.end()) {
    bag->state.emplace(StateType::kSquareAverage, grad.Like().Zero());
  }

  Tensor vt = bag->state[StateType::kSquareAverage];
  vt = alpha_ * vt + (1.0 - alpha_) * grad_t.Square();

  // Update
  bag->state[StateType::kSquareAverage] = vt;

  if (centered_) {
    if (bag->state.find(StateType::kGAve) == bag->state.end()) {
      bag->state.emplace(StateType::kGAve, grad.Like().Zero());
    }

    Tensor& gave = bag->state[StateType::kGAve];
    gave = gave * alpha_ + (1.0 - alpha_) * grad_t;

    vt = vt - gave.Square();
  }

  if (has_momentum_) {
    if (bag->state.find(StateType::kMomentumBuffer) == bag->state.end()) {
      bag->state.emplace(StateType::kMomentumBuffer, grad.Like().Zero());
    }

    Tensor& bt = bag->state[StateType::kMomentumBuffer];
    bt = bt * momentum_ + grad_t / (vt.Sqrt() + eps_);

    (*val) -= (lr * bt);
  } else {
    (*val) -= (lr * grad_t / (vt.Sqrt() + eps_));
  }

  return ErrorCode::kSuccess;
}

}  // namespace kraken