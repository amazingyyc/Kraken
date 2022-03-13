#pragma once

#include <cinttypes>

#include "common/deserialize.h"
#include "common/serialize.h"
#include "t/tensor.h"

namespace kraken {

struct PushDenseTableRequest {
  uint64_t router_version;

  uint64_t table_id;

  Tensor grad;
  float lr;
};

template <>
inline bool Serialize::operator<<(const PushDenseTableRequest& v) {
  return (*this) << v.router_version && (*this) << v.table_id &&
         (*this) << v.grad && (*this) << v.lr;
}

template <>
inline bool Deserialize::operator>>(PushDenseTableRequest& v) {
  return (*this) >> v.router_version && (*this) >> v.table_id &&
         (*this) >> v.grad && (*this) >> v.lr;
}

struct PushDenseTableResponse {
  /*empty*/
};

template <>
inline bool Serialize::operator<<(const PushDenseTableResponse& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(PushDenseTableResponse& v) {
  return true;
}

}  // namespace kraken
