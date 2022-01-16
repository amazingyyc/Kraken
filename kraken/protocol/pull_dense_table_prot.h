#pragma once

#include <cinttypes>

#include "t/tensor.h"
#include "rpc/deserialize.h"
#include "rpc/serialize.h"

namespace kraken {

struct PullDenseTableRequest {
  uint64_t model_id;
  uint64_t table_id;
};

template <>
inline bool Serialize::operator<<(const PullDenseTableRequest& v) {
  return (*this) << v.model_id && (*this) << v.table_id;
}

template <>
inline bool Deserialize::operator>>(PullDenseTableRequest& v) {
  return (*this) >> v.model_id && (*this) >> v.table_id;
}

struct PullDenseTableResponse {
  Tensor val;
};

template <>
inline bool Serialize::operator<<(const PullDenseTableResponse& v) {
  return (*this) << v.val;
}

template <>
inline bool Deserialize::operator>>(PullDenseTableResponse& v) {
  return (*this) >> v.val;
}

}  // namespace kraken
