#pragma once

#include "common/deserialize.h"
#include "common/info.h"
#include "common/serialize.h"

namespace kraken {

struct FetchModelMetaDataRequest {};

template <>
inline bool Serialize::operator<<(const FetchModelMetaDataRequest& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(FetchModelMetaDataRequest& v) {
  return true;
}

struct FetchModelMetaDataResponse {
  bool model_init;
  ModelMetaData model_mdata;
};

template <>
inline bool Serialize::operator<<(const FetchModelMetaDataResponse& v) {
  return (*this) << v.model_init && (*this) << v.model_mdata;
}

template <>
inline bool Deserialize::operator>>(FetchModelMetaDataResponse& v) {
  return (*this) >> v.model_init && (*this) >> v.model_mdata;
}

}  // namespace kraken