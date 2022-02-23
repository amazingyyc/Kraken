#pragma once

#include <cinttypes>
#include <unordered_map>
#include <vector>

#include "common/deserialize.h"
#include "common/serialize.h"
#include "t/tensor.h"

namespace kraken {

struct CombinePullSparseTableRequest {
  uint64_t model_id;

  // <TableId, Indices> map.
  std::unordered_map<uint64_t, std::vector<uint64_t>> indices;
};

template <>
inline bool Serialize::operator<<(
    const std::unordered_map<uint64_t, std::vector<uint64_t>>& v) {
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
inline bool Serialize::operator<<(const CombinePullSparseTableRequest& v) {
  return (*this) << v.model_id && (*this) << v.indices;
}

template <>
inline bool Deserialize::operator>>(
    std::unordered_map<uint64_t, std::vector<uint64_t>>& v) {
  v.clear();

  uint64_t size;
  if (((*this) >> size) == false) {
    return false;
  }

  v.reserve(size);
  for (uint64_t i = 0; i < size; ++i) {
    uint64_t key;
    std::vector<uint64_t> value;

    if (((*this) >> key) == false || ((*this) >> value) == false) {
      return false;
    }

    v.emplace(key, std::move(value));
  }

  return true;
}

template <>
inline bool Deserialize::operator>>(CombinePullSparseTableRequest& v) {
  return (*this) >> v.model_id && (*this) >> v.indices;
}

struct CombinePullSparseTableResponse {
  // <TableId, Val> map.
  std::unordered_map<uint64_t, std::vector<Tensor>> vals;
};

template <>
inline bool Serialize::operator<<(
    const std::unordered_map<uint64_t, std::vector<Tensor>>& v) {
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
inline bool Serialize::operator<<(const CombinePullSparseTableResponse& v) {
  return (*this) << v.vals;
}

template <>
inline bool Deserialize::operator>>(
    std::unordered_map<uint64_t, std::vector<Tensor>>& v) {
  v.clear();

  uint64_t size;
  if (((*this) >> size) == false) {
    return false;
  }

  v.reserve(size);
  for (uint64_t i = 0; i < size; ++i) {
    uint64_t key;
    std::vector<Tensor> value;

    if (((*this) >> key) == false || ((*this) >> value) == false) {
      return false;
    }

    v.emplace(key, std::move(value));
  }

  return true;
}

template <>
inline bool Deserialize::operator>>(CombinePullSparseTableResponse& v) {
  return (*this) >> v.vals;
}

}  // namespace kraken
