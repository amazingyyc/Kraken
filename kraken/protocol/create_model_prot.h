#pragma once

#include "common/deserialize.h"
#include "common/info.h"
#include "common/serialize.h"

namespace kraken {

struct CreateModelRequest {
  std::string name;
  OptimType optim_type;
  std::unordered_map<std::string, std::string> optim_conf;
};

template <>
inline bool Serialize::operator<<(const CreateModelRequest& v) {
  return (*this) << v.name && (*this) << v.optim_type &&
         (*this) << v.optim_conf;
}

template <>
inline bool Deserialize::operator>>(CreateModelRequest& v) {
  return (*this) >> v.name && (*this) >> v.optim_type &&
         (*this) >> v.optim_conf;
}

struct CreateModelResponse {};

template <>
inline bool Serialize::operator<<(const CreateModelResponse& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(CreateModelResponse& v) {
  return true;
}

}  // namespace kraken
