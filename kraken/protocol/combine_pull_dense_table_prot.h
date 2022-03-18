#pragma once

#include <cinttypes>
#include <vector>

#include "common/deserialize.h"
#include "common/serialize.h"
#include "t/tensor.h"

namespace kraken {

struct CombinePullDenseTableRequest {
  uint64_t router_version;

  std::vector<uint64_t> table_ids;
};

template <>
inline bool Serialize::operator<<(const CombinePullDenseTableRequest& v) {
  return (*this) << v.router_version && (*this) << v.table_ids;
}

template <>
inline bool Deserialize::operator>>(CombinePullDenseTableRequest& v) {
  return (*this) >> v.router_version && (*this) >> v.table_ids;
}

struct CombinePullDenseTableResponse {
  std::vector<Tensor> vals;
};

template <>
inline bool Serialize::operator<<(const CombinePullDenseTableResponse& v) {
  return (*this) << v.vals;
}

template <>
inline bool Deserialize::operator>>(CombinePullDenseTableResponse& v) {
  return (*this) >> v.vals;
}

}  // namespace kraken
