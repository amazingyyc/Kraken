#pragma once

#include "common/deserialize.h"
#include "common/info.h"
#include "common/serialize.h"

namespace kraken {

struct TransferSparseValuesRequest {
  uint64_t table_id;

  std::vector<uint64_t> sparse_ids;
  std::vector<Value> values;
};

template <>
inline bool Serialize::operator<<(const TransferSparseValuesRequest& v) {
  return (*this) << v.table_id && (*this) << v.sparse_ids &&
         (*this) << v.values;
}

template <>
inline bool Deserialize::operator>>(TransferSparseValuesRequest& v) {
  return (*this) >> v.table_id && (*this) >> v.sparse_ids &&
         (*this) >> v.values;
}

struct TransferSparseValuesResponse {};

template <>
inline bool Serialize::operator<<(const TransferSparseValuesResponse& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(TransferSparseValuesResponse& v) {
  return true;
}

}  // namespace kraken
