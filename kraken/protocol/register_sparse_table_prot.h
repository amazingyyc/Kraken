#pragma once

#include <cinttypes>
#include <string>

#include "common/deserialize.h"
#include "common/serialize.h"
#include "ps/initializer/initializer.h"
#include "t/element_type.h"

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

struct RegisterSparseTableV2Request {
  uint64_t model_id;

  uint64_t id;
  std::string name;

  int64_t dimension;
  ElementType etype;

  InitializerType init_type;
  std::unordered_map<std::string, std::string> init_conf;
};

template <>
inline bool Serialize::operator<<(const RegisterSparseTableV2Request& v) {
  return (*this) << v.model_id && (*this) << v.id && (*this) << v.name &&
         (*this) << v.dimension && (*this) << v.etype &&
         (*this) << v.init_type && (*this) << v.init_conf;
}

template <>
inline bool Deserialize::operator>>(RegisterSparseTableV2Request& v) {
  return (*this) >> v.model_id && (*this) >> v.id && (*this) >> v.name &&
         (*this) >> v.dimension && (*this) >> v.etype &&
         (*this) >> v.init_type && (*this) >> v.init_conf;
}

struct RegisterSparseTableV2Response {};

template <>
inline bool Serialize::operator<<(const RegisterSparseTableV2Response& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(RegisterSparseTableV2Response& v) {
  return true;
}

}  // namespace kraken
