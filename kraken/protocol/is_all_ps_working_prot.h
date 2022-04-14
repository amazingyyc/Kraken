#pragma once

#include "common/deserialize.h"
#include "common/serialize.h"

namespace kraken {

struct IsAllPsWorkingRequest {};

template <>
inline bool Serialize::operator<<(const IsAllPsWorkingRequest& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(IsAllPsWorkingRequest& v) {
  return true;
}

struct IsAllPsWorkingResponse {
  bool yes;
};

template <>
inline bool Serialize::operator<<(const IsAllPsWorkingResponse& v) {
  return (*this) << v.yes;
}

template <>
inline bool Deserialize::operator>>(IsAllPsWorkingResponse& v) {
  return (*this) >> v.yes;
}

}  // namespace kraken
