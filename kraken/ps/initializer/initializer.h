#pragma once

#include <cinttypes>
#include <unordered_map>

#include "common/tensor.h"
#include "common/utils.h"

namespace kraken {

enum class InitializerType : uint8_t {
  kConstant = 0,
  kUniform = 1,
  kNormal = 2,
  kXavierUniform = 3,
  kXavierNormal = 4,
};

class Initializer {
protected:
  InitializerType type_;

  std::unordered_map<std::string, std::string> conf_;

protected:
  Initializer(InitializerType type) : type_(type) {
  }

  Initializer(InitializerType type,
              const std::unordered_map<std::string, std::string>& conf)
      : type_(type), conf_(conf) {
  }

  template <typename T>
  bool GetConf(const std::string& k, T* v) {
    return false;
  }

public:
  virtual ~Initializer() = default;

  InitializerType type() const {
    return type_;
  }

  virtual void Initialize(Tensor* val) const = 0;
};

template <>
inline bool Initializer::GetConf<float>(const std::string& k, float* v) {
  auto it = conf_.find(k);
  if (it == conf_.end()) {
    return false;
  }

  *v = std::stof(it->second);

  return true;
}

template <>
inline bool Initializer::GetConf<bool>(const std::string& k, bool* v) {
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
