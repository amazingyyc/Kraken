#pragma once

#include "common/deserialize.h"
#include "common/info.h"
#include "common/serialize.h"

namespace kraken {

struct TryFetchDenseTableRequest {
  uint64_t table_id;
};

template <>
inline bool Serialize::operator<<(const TryFetchDenseTableRequest& v) {
  return (*this) << v.table_id;
}

template <>
inline bool Deserialize::operator>>(TryFetchDenseTableRequest& v) {
  return (*this) >> v.table_id;
}

struct TryFetchDenseTableResponse {
  std::string name;
  Value value;
};

template <>
inline bool Serialize::operator<<(const TryFetchDenseTableResponse& v) {
  return (*this) << v.name && (*this) << v.value;
}

template <>
inline bool Deserialize::operator>>(TryFetchDenseTableResponse& v) {
  return (*this) >> v.name && (*this) >> v.value;
}

}  // namespace kraken
