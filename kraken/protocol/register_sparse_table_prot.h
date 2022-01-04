#pragma once

#include <cinttypes>
#include <string>

#include "common/element_type.h"
#include "common/shape.h"
#include "rpc/deserialize.h"
#include "rpc/serialize.h"

namespace kraken {

struct RegisterSparseTableRequest {
  uint64_t model_id;

  uint64_t id;
  std::string name;

  int64_t dimension;
  ElementType etype;
};

template <>
inline bool Serialize::operator<<(const RegisterSparseTableRequest& v) {
  return (*this) << v.model_id && (*this) << v.id && (*this) << v.name &&
         (*this) << v.dimension && (*this) << v.etype;
}

template <>
inline bool Deserialize::operator>>(RegisterSparseTableRequest& v) {
  return (*this) >> v.model_id && (*this) >> v.id && (*this) >> v.name &&
         (*this) >> v.dimension && (*this) >> v.etype;
}

struct RegisterSparseTableResponse {};

template <>
inline bool Serialize::operator<<(const RegisterSparseTableResponse& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(RegisterSparseTableResponse& v) {
  return true;
}

}  // namespace kraken
