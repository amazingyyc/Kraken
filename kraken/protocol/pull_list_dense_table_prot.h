#pragma once

#include <cinttypes>
#include <vector>

#include "t/tensor.h"
#include "rpc/deserialize.h"
#include "rpc/serialize.h"

namespace kraken {

struct PullListDenseTableRequest {
  uint64_t model_id;

  std::vector<uint64_t> table_ids;
};

template <>
inline bool Serialize::operator<<(const PullListDenseTableRequest& v) {
  return (*this) << v.model_id && (*this) << v.table_ids;
}

template <>
inline bool Deserialize::operator>>(PullListDenseTableRequest& v) {
  return (*this) >> v.model_id && (*this) >> v.table_ids;
}

struct PullListDenseTableResponse {
  std::vector<Tensor> vals;
};

template <>
inline bool Serialize::operator<<(const PullListDenseTableResponse& v) {
  return (*this) << v.vals;
}

template <>
inline bool Deserialize::operator>>(PullListDenseTableResponse& v) {
  return (*this) >> v.vals;
}

}  // namespace kraken
