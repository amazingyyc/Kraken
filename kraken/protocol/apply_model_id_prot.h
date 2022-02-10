#pragma once

#include <cinttypes>
#include <string>

#include "common/deserialize.h"
#include "common/serialize.h"

namespace kraken {

struct ApplyModelIdRequest {
  std::string model_name;
};

template <>
inline bool Serialize::operator<<(const ApplyModelIdRequest& v) {
  return (*this) << v.model_name;
}

template <>
inline bool Deserialize::operator>>(ApplyModelIdRequest& v) {
  return (*this) >> v.model_name;
}

struct ApplyModelIdResponse {
  uint64_t model_id;
};

template <>
inline bool Serialize::operator<<(const ApplyModelIdResponse& v) {
  return (*this) << v.model_id;
}

template <>
inline bool Deserialize::operator>>(ApplyModelIdResponse& v) {
  return (*this) >> v.model_id;
}

}  // namespace kraken
