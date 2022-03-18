#pragma once

#include <cinttypes>

#include "common/deserialize.h"
#include "common/serialize.h"
#include "t/tensor.h"

namespace kraken {

struct PullDenseTableRequest {
  uint64_t router_version;
  uint64_t table_id;
};

template <>
inline bool Serialize::operator<<(const PullDenseTableRequest& v) {
  return (*this) << v.router_version && (*this) << v.table_id;
}

template <>
inline bool Deserialize::operator>>(PullDenseTableRequest& v) {
  return (*this) >> v.router_version && (*this) >> v.table_id;
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
