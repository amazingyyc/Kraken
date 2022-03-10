#pragma once

#include "common/deserialize.h"
#include "common/info.h"
#include "common/serialize.h"

namespace kraken {

struct InitModelRequest {
  std::string name;
  OptimType optim_type;
  std::unordered_map<std::string, std::string> optim_conf;
};

template <>
inline bool Serialize::operator<<(const InitModelRequest& v) {
  return (*this) << v.name && (*this) << v.optim_type &&
         (*this) << v.optim_conf;
}

template <>
inline bool Deserialize::operator>>(InitModelRequest& v) {
  return (*this) >> v.name && (*this) >> v.optim_type &&
         (*this) >> v.optim_conf;
}

struct InitModelResponse {};

template <>
inline bool Serialize::operator<<(const InitModelResponse& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(InitModelResponse& v) {
  return true;
}

}  // namespace kraken
