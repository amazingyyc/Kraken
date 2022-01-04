#pragma once

#include <cinttypes>
#include <vector>

#include "common/tensor.h"
#include "rpc/deserialize.h"
#include "rpc/serialize.h"

namespace kraken {

struct PullSparseTableRequest {
  uint64_t model_id;
  uint64_t table_id;

  std::vector<int64_t> indices;
};

template <>
inline bool Serialize::operator<<(const PullSparseTableRequest& v) {
  return (*this) << v.model_id && (*this) << v.table_id && (*this) << v.indices;
}

template <>
inline bool Deserialize::operator>>(PullSparseTableRequest& v) {
  return (*this) >> v.model_id && (*this) >> v.table_id && (*this) >> v.indices;
}

struct PullSparseTableResponse {
  std::vector<Tensor> vals;
};

template <>
inline bool Serialize::operator<<(const PullSparseTableResponse& v) {
  return (*this) << v.vals;
}

template <>
inline bool Deserialize::operator>>(PullSparseTableResponse& v) {
  return (*this) >> v.vals;
}

}  // namespace kraken