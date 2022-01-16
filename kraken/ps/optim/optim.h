#pragma once

#include <cinttypes>
#include <unordered_map>

#include "common/utils.h"
#include "parallel_hashmap/parallel_hashmap/phmap.h"
#include "t/tensor.h"

namespace kraken {

enum class OptimType : uint8_t {
  kAdagrad = 0,
  kAdam = 1,
  kRMSprop = 2,
  kSGD = 3,
};

enum class StateType : uint32_t {
  kSteps = 0,
  kMomentumBuffer = 1,
  kStateSum = 2,
  kFirstMoment = 3,
  kSecondMoment = 4,
  kSecondMomentMax = 5,
  kSquareAverage = 6,
  kGAve = 7,
};

/**
 * \brief This is a simple struct that store some useful resource.
 * Like in SGD optim we may need store some temporary tensor.
 */
struct Bag {
  // tensor state.
  phmap::flat_hash_map<StateType, Tensor> state;

  // integer state.
  phmap::flat_hash_map<StateType, int64_t> state_i;
};

class Optim {
protected:
  OptimType optim_type_;

  // Optim conf.
  std::unordered_map<std::string, std::string> conf_;

public:
  Optim(OptimType optim_type,
        const std::unordered_map<std::string, std::string>& conf)
      : optim_type_(optim_type), conf_(conf) {
  }

  virtual ~Optim() = default;

public:
  template <typename T>
  bool GetConf(const std::string& k, T* v) {
    return false;
  }

  virtual int32_t Update(const Tensor& grad, float lr, Tensor* val,
                         Bag* package) const = 0;
};

template <>
inline bool Optim::GetConf<float>(const std::string& k, float* v) {
  auto it = conf_.find(k);
  if (it == conf_.end()) {
    return false;
  }

  *v = std::stof(it->second);

  return true;
}

template <>
inline bool Optim::GetConf<bool>(const std::string& k, bool* v) {
  auto it = conf_.find(k);
  if (it == conf_.end()) {
    return false;
  }

  std::string lv = utils::ToLower(it->second);

  if (lv == "true" || lv == "1") {
    *v = true;
  } else {
    *v = false;
  }

  return true;
}

}  // namespace kraken
