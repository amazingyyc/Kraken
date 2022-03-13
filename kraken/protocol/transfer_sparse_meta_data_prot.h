#pragma once

#include "common/deserialize.h"
#include "common/info.h"
#include "common/serialize.h"

namespace kraken {

struct TransferSparseMetaDataRequest {
  uint64_t id;
  std::string name;

  int64_t dimension;
  ElementType element_type;

  InitializerType init_type;
  std::unordered_map<std::string, std::string> init_conf;
};

template <>
inline bool Serialize::operator<<(const TransferSparseMetaDataRequest& v) {
  return (*this) << v.id && (*this) << v.name && (*this) << v.dimension &&
         (*this) << v.element_type && (*this) << v.init_type &&
         (*this) << v.init_conf;
}

template <>
inline bool Deserialize::operator>>(TransferSparseMetaDataRequest& v) {
  return (*this) >> v.id && (*this) >> v.name && (*this) >> v.dimension &&
         (*this) >> v.element_type && (*this) >> v.init_type &&
         (*this) >> v.init_conf;
}

struct TransferSparseMetaDataResponse {};

template <>
inline bool Serialize::operator<<(const TransferSparseMetaDataResponse& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(TransferSparseMetaDataResponse& v) {
  return true;
}

}  // namespace kraken
