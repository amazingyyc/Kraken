#pragma once

#include <cinttypes>
#include <unordered_map>

#include "common/deserialize.h"
#include "common/serialize.h"
#include "t/tensor.h"

namespace kraken {

struct CombinePushSparseTableItem {
  std::vector<uint64_t> sparse_ids;
  std::vector<Tensor> grads;
};

template <>
inline bool Serialize::operator<<(const CombinePushSparseTableItem& v) {
  return (*this) << v.sparse_ids && (*this) << v.grads;
}

template <>
inline bool Deserialize::operator>>(CombinePushSparseTableItem& v) {
  return (*this) >> v.sparse_ids && (*this) >> v.grads;
}

struct CombinePushSparseTableRequest {
  uint64_t router_version;

  std::unordered_map<uint64_t, CombinePushSparseTableItem> table_items;

  float lr;
};

template <>
inline bool Serialize::operator<<(
    const std::unordered_map<uint64_t, CombinePushSparseTableItem>& v) {
  uint64_t size = v.size();
  if (((*this) << size) == false) {
    return false;
  }

  for (const auto& [key, val] : v) {
    if (((*this) << key) == false || ((*this) << val) == false) {
      return false;
    }
  }

  return true;
}

template <>
inline bool Deserialize::operator>>(
    std::unordered_map<uint64_t, CombinePushSparseTableItem>& v) {
  v.clear();

  uint64_t size;
  if (((*this) >> size) == false) {
    return false;
  }

  v.reserve(size);
  for (uint64_t i = 0; i < size; ++i) {
    uint64_t key;
    CombinePushSparseTableItem value;

    if (((*this) >> key) == false || ((*this) >> value) == false) {
      return false;
    }

    v.emplace(key, std::move(value));
  }

  return true;
}

template <>
inline bool Serialize::operator<<(const CombinePushSparseTableRequest& v) {
  return (*this) << v.router_version && (*this) << v.table_items &&
         (*this) << v.lr;
}

template <>
inline bool Deserialize::operator>>(CombinePushSparseTableRequest& v) {
  return (*this) >> v.router_version && (*this) >> v.table_items &&
         (*this) >> v.lr;
}

struct CombinePushSparseTableResponse {
  /*empty*/
};

template <>
inline bool Serialize::operator<<(const CombinePushSparseTableResponse& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(CombinePushSparseTableResponse& v) {
  return true;
}

}  // namespace kraken
