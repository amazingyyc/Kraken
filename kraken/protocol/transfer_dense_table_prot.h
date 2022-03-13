#pragma once

#include "common/deserialize.h"
#include "common/info.h"
#include "common/serialize.h"

namespace kraken {

struct TransferDenseTableRequest {
  uint64_t id;
  std::string name;
  Value value;
};

template <>
inline bool Serialize::operator<<(const TransferDenseTableRequest& v) {
  return (*this) << v.id && (*this) << v.name && (*this) << v.value;
}

template <>
inline bool Deserialize::operator>>(TransferDenseTableRequest& v) {
  return (*this) >> v.id && (*this) >> v.name && (*this) >> v.value;
}

struct TransferDenseTableResponse {};

template <>
inline bool Serialize::operator<<(const TransferDenseTableResponse& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(TransferDenseTableResponse& v) {
  return true;
}

}  // namespace kraken
