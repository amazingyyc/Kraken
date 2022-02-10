#pragma once

#include <cinttypes>
#include <string>

#include "common/deserialize.h"
#include "common/serialize.h"
#include "ps/initializer/initializer.h"
#include "t/element_type.h"

namespace kraken {

struct RegisterSparseTableInfoRequest {
  uint64_t model_id;

  uint64_t id;
  std::string name;

  int64_t dimension;
  ElementType element_type;

  InitializerType init_type;
  std::unordered_map<std::string, std::string> init_conf;
};

template <>
inline bool Serialize::operator<<(const RegisterSparseTableInfoRequest& v) {
  return (*this) << v.model_id && (*this) << v.id && (*this) << v.name &&
         (*this) << v.dimension && (*this) << v.element_type &&
         (*this) << v.init_type && (*this) << v.init_conf;
}

template <>
inline bool Deserialize::operator>>(RegisterSparseTableInfoRequest& v) {
  return (*this) >> v.model_id && (*this) >> v.id && (*this) >> v.name &&
         (*this) >> v.dimension && (*this) >> v.element_type &&
         (*this) >> v.init_type && (*this) >> v.init_conf;
}

struct RegisterSparseTableInfoResponse {};

template <>
inline bool Serialize::operator<<(const RegisterSparseTableInfoResponse& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(RegisterSparseTableInfoResponse& v) {
  return true;
}

}  // namespace kraken
