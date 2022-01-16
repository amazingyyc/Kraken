#pragma once

#include <cinttypes>

#include "t/tensor.h"
#include "rpc/deserialize.h"
#include "rpc/serialize.h"

namespace kraken {

struct PushDenseTableRequest {
  uint64_t model_id;
  uint64_t table_id;

  Tensor grad;
  float lr;
};

template <>
inline bool Serialize::operator<<(const PushDenseTableRequest& v) {
  return (*this) << v.model_id && (*this) << v.table_id && (*this) << v.grad &&
         (*this) << v.lr;
}

template <>
inline bool Deserialize::operator>>(PushDenseTableRequest& v) {
  return (*this) >> v.model_id && (*this) >> v.table_id && (*this) >> v.grad &&
         (*this) >> v.lr;
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
