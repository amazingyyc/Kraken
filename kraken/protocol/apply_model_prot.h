#pragma once

#include <cinttypes>
#include <string>

#include "common/tensor.h"
#include "rpc/deserialize.h"
#include "rpc/serialize.h"

namespace kraken {

/**
 * \brief Apply a model in PS. Apply donot means really create a model in server,
 * it just giving a new id, the response server guard the id is unique.
 */
struct ApplyModelRequest {
  std::string name;
};

template <>
inline bool Serialize::operator<<(const ApplyModelRequest& v) {
  return (*this) << v.name;
}

template <>
inline bool Deserialize::operator>>(ApplyModelRequest& v) {
  return (*this) >> v.name;
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
