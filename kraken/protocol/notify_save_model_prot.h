#pragma once

#include "common/deserialize.h"
#include "common/info.h"
#include "common/serialize.h"

namespace kraken {

struct NotifySaveModelRequest {
  ModelMetaData model_mdata;
};

template <>
inline bool Serialize::operator<<(const NotifySaveModelRequest& v) {
  return (*this) << v.model_mdata;
}

template <>
inline bool Deserialize::operator>>(NotifySaveModelRequest& v) {
  return (*this) >> v.model_mdata;
}

struct NotifySaveModelResponse {};

template <>
inline bool Serialize::operator<<(const NotifySaveModelResponse& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(NotifySaveModelResponse& v) {
  return true;
}

}  // namespace kraken
