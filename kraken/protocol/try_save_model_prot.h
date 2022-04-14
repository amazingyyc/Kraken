#pragma once

#include "common/deserialize.h"
#include "common/serialize.h"

namespace kraken {

struct TrySaveModelRequest {};

template <>
inline bool Serialize::operator<<(const TrySaveModelRequest& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(TrySaveModelRequest& v) {
  return true;
}

struct TrySaveModelResponse {
  bool success;
};

template <>
inline bool Serialize::operator<<(const TrySaveModelResponse& v) {
  return (*this) << v.success;
}

template <>
inline bool Deserialize::operator>>(TrySaveModelResponse& v) {
  return (*this) >> v.success;
}

}  // namespace kraken
