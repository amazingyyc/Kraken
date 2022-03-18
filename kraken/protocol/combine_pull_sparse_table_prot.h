#pragma once

#include <cinttypes>
#include <unordered_map>
#include <vector>

#include "common/deserialize.h"
#include "common/serialize.h"
#include "t/tensor.h"

namespace kraken {

struct CombinePullSparseTableRequest {
  uint64_t router_version;

  // <TableId, SparseId> map.
  std::unordered_map<uint64_t, std::vector<uint64_t>> table_sparse_ids;
};

template <>
inline bool Serialize::operator<<(const CombinePullSparseTableRequest& v) {
  return (*this) << v.router_version && (*this) << v.table_sparse_ids;
}

template <>
inline bool Deserialize::operator>>(CombinePullSparseTableRequest& v) {
  return (*this) >> v.router_version && (*this) >> v.table_sparse_ids;
}

struct CombinePullSparseTableResponse {
  // <TableId, Vals> map.
  std::unordered_map<uint64_t, std::vector<Tensor>> table_vals;
};

template <>
inline bool Serialize::operator<<(const CombinePullSparseTableResponse& v) {
  return (*this) << v.table_vals;
}

template <>
inline bool Deserialize::operator>>(CombinePullSparseTableResponse& v) {
  return (*this) >> v.table_vals;
}

}  // namespace kraken
