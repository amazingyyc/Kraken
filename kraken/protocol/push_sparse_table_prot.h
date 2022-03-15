#pragma once

#include <cinttypes>

#include "common/deserialize.h"
#include "common/serialize.h"
#include "t/tensor.h"

namespace kraken {

struct PushSparseTableRequest {
  uint64_t router_version;
  uint64_t table_id;

  std::vector<uint64_t> sparse_ids;
  std::vector<Tensor> grads;

  float lr;
};

template <>
inline bool Serialize::operator<<(const PushSparseTableRequest& v) {
  return (*this) << v.router_version && (*this) << v.table_id &&
         (*this) << v.sparse_ids && (*this) << v.grads && (*this) << v.lr;
}

template <>
inline bool Deserialize::operator>>(PushSparseTableRequest& v) {
  return (*this) >> v.router_version && (*this) >> v.table_id &&
         (*this) >> v.sparse_ids && (*this) >> v.grads && (*this) >> v.lr;
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
