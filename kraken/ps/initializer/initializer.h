#pragma once

#include <cinttypes>
#include <unordered_map>

#include "common/tensor.h"
#include "common/utils.h"

namespace kraken {

enum InitializerType : uint8_t {
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

public:
  Initializer(InitializerType type,
              const std::unordered_map<std::string, std::string>& conf)
      : type_(type), conf_(conf) {
  }

  virtual ~Initializer() = default;

protected:
  template <typename T>
  bool GetConf(const std::string& k, T* v) {
    return false;
  }

public:
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
