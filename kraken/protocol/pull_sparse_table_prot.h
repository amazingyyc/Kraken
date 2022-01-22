#pragma once

#include <cinttypes>
#include <vector>

#include "rpc/deserialize.h"
#include "rpc/serialize.h"
#include "t/tensor.h"

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
