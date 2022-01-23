#pragma once

#include <cinttypes>
#include <vector>

#include "protocol/pull_sparse_table_prot.h"
#include "rpc/deserialize.h"
#include "rpc/serialize.h"
#include "t/tensor.h"

namespace kraken {

struct CombinePullSparseTableRequest {
  std::vector<PullSparseTableRequest> indices;
};

template <>
inline bool Serialize::operator<<(
    const std::vector<PullSparseTableRequest>& v) {
  uint64_t size = v.size();
  if (((*this) << size) == false) {
    return false;
  }

  for (auto& i : v) {
    if (((*this) << i) == false) {
      return false;
    }
  }

  return true;
}

template <>
inline bool Serialize::operator<<(const CombinePullSparseTableRequest& v) {
  return (*this) << v.indices;
}

template <>
inline bool Deserialize::operator>>(std::vector<PullSparseTableRequest>& v) {
  uint64_t size;
  if (((*this) >> size) == false) {
    return false;
  }

  v.resize(size);
  for (uint64_t i = 0; i < size; ++i) {
    if (((*this) >> v[i]) == false) {
      return false;
    }
  }

  return true;
}

template <>
inline bool Deserialize::operator>>(CombinePullSparseTableRequest& v) {
  return (*this) >> v.indices;
}

struct CombinePullSparseTableResponse {
  std::vector<PullSparseTableResponse> vals;
};

template <>
inline bool Serialize::operator<<(
    const std::vector<PullSparseTableResponse>& v) {
  uint64_t size = v.size();
  if (((*this) << size) == false) {
    return false;
  }

  for (auto& i : v) {
    if (((*this) << i) == false) {
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
inline bool Deserialize::operator>>(std::vector<PullSparseTableResponse>& v) {
  uint64_t size;
  if (((*this) >> size) == false) {
    return false;
  }

  v.resize(size);
  for (uint64_t i = 0; i < size; ++i) {
    if (((*this) >> v[i]) == false) {
      return false;
    }
  }

  return true;
}

template <>
inline bool Deserialize::operator>>(CombinePullSparseTableResponse& v) {
  return (*this) >> v.vals;
}

}  // namespace kraken
