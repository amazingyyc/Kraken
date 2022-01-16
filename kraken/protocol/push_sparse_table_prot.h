#pragma once

#include <cinttypes>

#include "t/tensor.h"
#include "rpc/deserialize.h"
#include "rpc/serialize.h"

namespace kraken {

struct PushSparseTableRequest {
  uint64_t model_id;
  uint64_t table_id;

  std::vector<int64_t> indices;
  std::vector<Tensor> grads;

  float lr;
};

template <>
inline bool Serialize::operator<<(const PushSparseTableRequest& v) {
  return (*this) << v.model_id && (*this) << v.table_id &&
         (*this) << v.indices && (*this) << v.grads && (*this) << v.lr;
}

template <>
inline bool Deserialize::operator>>(PushSparseTableRequest& v) {
  return (*this) >> v.model_id && (*this) >> v.table_id &&
         (*this) >> v.indices && (*this) >> v.grads && (*this) >> v.lr;
}

struct PushSparseTableResponse {
  /*empty*/
};

template <>
inline bool Serialize::operator<<(const PushSparseTableResponse& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(PushSparseTableResponse& v) {
  return true;
}

}  // namespace kraken
