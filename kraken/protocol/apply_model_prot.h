#pragma once

#include <cinttypes>
#include <string>
#include <unordered_map>

#include "common/deserialize.h"
#include "common/serialize.h"
#include "ps/optim/optim.h"

namespace kraken {

struct ApplyModelRequest {
  std::string model_name;
  OptimType optim_type;
  std::unordered_map<std::string, std::string> optim_conf;
};

template <>
inline bool Serialize::operator<<(const ApplyModelRequest& v) {
  return (*this) << v.model_name && (*this) << v.optim_type &&
         (*this) << v.optim_conf;
}

template <>
inline bool Deserialize::operator>>(ApplyModelRequest& v) {
  return (*this) >> v.model_name && (*this) >> v.optim_type &&
         (*this) >> v.optim_conf;
}

struct ApplyModelResponse {
  uint64_t model_id;
};

template <>
inline bool Serialize::operator<<(const ApplyModelResponse& v) {
  return (*this) << v.model_id;
}

template <>
inline bool Deserialize::operator>>(ApplyModelResponse& v) {
  return (*this) >> v.model_id;
}

}  // namespace kraken
