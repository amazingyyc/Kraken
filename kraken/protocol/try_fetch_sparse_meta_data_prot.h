#pragma once

#include "common/deserialize.h"
#include "common/info.h"
#include "common/serialize.h"

namespace kraken {

struct TryFetchSparseMetaDataRequest {
  uint64_t table_id;
};

template <>
inline bool Serialize::operator<<(const TryFetchSparseMetaDataRequest& v) {
  return (*this) << v.table_id;
}

template <>
inline bool Deserialize::operator>>(TryFetchSparseMetaDataRequest& v) {
  return (*this) >> v.table_id;
}

struct TryFetchSparseMetaDataResponse {
  std::string name;

  int64_t dimension;
  ElementType element_type;

  InitializerType init_type;
  std::unordered_map<std::string, std::string> init_conf;
};

template <>
inline bool Serialize::operator<<(const TryFetchSparseMetaDataResponse& v) {
  return (*this) << v.name && (*this) << v.dimension &&
         (*this) << v.element_type && (*this) << v.init_type &&
         (*this) << v.init_conf;
}

template <>
inline bool Deserialize::operator>>(TryFetchSparseMetaDataResponse& v) {
  return (*this) >> v.name && (*this) >> v.dimension &&
         (*this) >> v.element_type && (*this) >> v.init_type &&
         (*this) >> v.init_conf;
}

}  // namespace kraken
