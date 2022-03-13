#pragma once

#include <vector>

#include "common/deserialize.h"
#include "common/info.h"
#include "common/serialize.h"

namespace kraken {

struct TryCombineFetchDenseTableRequest {
  std::vector<uint64_t> ids;
};

template <>
inline bool Serialize::operator<<(const TryCombineFetchDenseTableRequest& v) {
  return (*this) << v.ids;
}

template <>
inline bool Deserialize::operator>>(TryCombineFetchDenseTableRequest& v) {
  return (*this) >> v.ids;
}

struct TryCombineFetchDenseTableResponse {
  std::vector<std::string> names;
  std::vector<Value> values;
};

template <>
inline bool Serialize::operator<<(const TryCombineFetchDenseTableResponse& v) {
  return (*this) << v.names && (*this) << v.values;
}

template <>
inline bool Deserialize::operator>>(TryCombineFetchDenseTableResponse& v) {
  return (*this) >> v.names && (*this) >> v.values;
}

}  // namespace kraken
