#pragma once

#include <cinttypes>
#include <string>

#include "common/deserialize.h"
#include "common/serialize.h"
#include "t/tensor.h"

namespace kraken {

struct RegisterDenseTableRequest {
  std::string name;

  Tensor val;
};

template <>
inline bool Serialize::operator<<(const RegisterDenseTableRequest& v) {
  return (*this) << v.name && (*this) << v.val;
}

template <>
inline bool Deserialize::operator>>(RegisterDenseTableRequest& v) {
  return (*this) >> v.name && (*this) >> v.val;
}

struct RegisterDenseTableResponse {
  uint64_t table_id;
};

template <>
inline bool Serialize::operator<<(const RegisterDenseTableResponse& v) {
  return (*this) << v.table_id;
}

template <>
inline bool Deserialize::operator>>(RegisterDenseTableResponse& v) {
  return (*this) >> v.table_id;
}

}  // namespace kraken
