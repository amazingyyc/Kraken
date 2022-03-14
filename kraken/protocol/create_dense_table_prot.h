#pragma once

#include "common/deserialize.h"
#include "common/serialize.h"

namespace kraken {

struct CreateDenseTableRequest {
  uint64_t table_id;
  std::string name;
  Tensor val;
};

template <>
inline bool Serialize::operator<<(const CreateDenseTableRequest& v) {
  return (*this) << v.table_id && (*this) << v.name && (*this) << v.val;
}

template <>
inline bool Deserialize::operator>>(CreateDenseTableRequest& v) {
  return (*this) >> v.table_id && (*this) >> v.name && (*this) >> v.val;
}

struct CreateDenseTableResponse {};

template <>
inline bool Serialize::operator<<(const CreateDenseTableResponse& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(CreateDenseTableResponse& v) {
  return true;
}

}  // namespace kraken
