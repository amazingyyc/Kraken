#pragma once

#include <cinttypes>
#include <vector>

#include "common/deserialize.h"
#include "common/serialize.h"
#include "t/tensor.h"

namespace kraken {

struct PullSparseTableRequest {
  uint64_t router_version;

  uint64_t table_id;
  std::vector<uint64_t> sparse_ids;
};

template <>
inline bool Serialize::operator<<(const PullSparseTableRequest& v) {
  return (*this) << v.router_version && (*this) << v.table_id &&
         (*this) << v.sparse_ids;
}

template <>
inline bool Deserialize::operator>>(PullSparseTableRequest& v) {
  return (*this) >> v.router_version && (*this) >> v.table_id &&
         (*this) >> v.sparse_ids;
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
