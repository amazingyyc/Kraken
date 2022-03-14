#pragma once

#include "common/deserialize.h"
#include "common/info.h"
#include "common/serialize.h"

namespace kraken {

struct TryFetchSparseValuesRequest {
  uint64_t table_id;
  std::vector<uint64_t> sparse_ids;
};

template <>
inline bool Serialize::operator<<(const TryFetchSparseValuesRequest& v) {
  return (*this) << v.table_id && (*this) << v.sparse_ids;
}

template <>
inline bool Deserialize::operator>>(TryFetchSparseValuesRequest& v) {
  return (*this) >> v.table_id && (*this) >> v.sparse_ids;
}

struct TryFetchSparseValuesResponse {
  std::vector<uint64_t> exist_sparse_ids;
  std::vector<Value> values;
};

template <>
inline bool Serialize::operator<<(const TryFetchSparseValuesResponse& v) {
  return (*this) << v.exist_sparse_ids && (*this) << v.values;
}

template <>
inline bool Deserialize::operator>>(TryFetchSparseValuesResponse& v) {
  return (*this) >> v.exist_sparse_ids && (*this) >> v.values;
}

}  // namespace kraken
