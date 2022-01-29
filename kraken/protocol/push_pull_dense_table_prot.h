#pragma once

#include <cinttypes>

#include "common/deserialize.h"
#include "common/serialize.h"
#include "t/tensor.h"

namespace kraken {

struct PushPullDenseTableRequest {
  uint64_t model_id;
  uint64_t table_id;

  Tensor grad;
  float lr;
};

template <>
inline bool Serialize::operator<<(const PushPullDenseTableRequest& v) {
  return (*this) << v.model_id && (*this) << v.table_id && (*this) << v.grad &&
         (*this) << v.lr;
}

template <>
inline bool Deserialize::operator>>(PushPullDenseTableRequest& v) {
  return (*this) >> v.model_id && (*this) >> v.table_id && (*this) >> v.grad &&
         (*this) >> v.lr;
}

struct PushPullDenseTableResponse {
  Tensor val;
};

template <>
inline bool Serialize::operator<<(const PushPullDenseTableResponse& v) {
  return (*this) << v.val;
}

template <>
inline bool Deserialize::operator>>(PushPullDenseTableResponse& v) {
  return (*this) >> v.val;
}

}  // namespace kraken
