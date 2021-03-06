#pragma once

#include "common/deserialize.h"
#include "common/serialize.h"

namespace kraken {

struct CreateSparseTableRequest {
  uint64_t table_id;
  std::string name;

  int64_t dimension;
  ElementType element_type;

  InitializerType init_type;
  std::unordered_map<std::string, std::string> init_conf;
};

template <>
inline bool Serialize::operator<<(const CreateSparseTableRequest& v) {
  return (*this) << v.table_id && (*this) << v.name && (*this) << v.dimension &&
         (*this) << v.element_type && (*this) << v.init_type &&
         (*this) << v.init_conf;
}

template <>
inline bool Deserialize::operator>>(CreateSparseTableRequest& v) {
  return (*this) >> v.table_id && (*this) >> v.name && (*this) >> v.dimension &&
         (*this) >> v.element_type && (*this) >> v.init_type &&
         (*this) >> v.init_conf;
}

struct CreateSparseTableResponse {};

template <>
inline bool Serialize::operator<<(const CreateSparseTableResponse& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(CreateSparseTableResponse& v) {
  return true;
}

}  // namespace kraken
