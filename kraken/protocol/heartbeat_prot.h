#pragma once

#include "common/deserialize.h"
#include "common/serialize.h"

namespace kraken {

struct HeartbeatRequest {};

template <>
inline bool Serialize::operator<<(const HeartbeatRequest& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(HeartbeatRequest& v) {
  return true;
}

struct HeartbeatResponse {
  uint32_t status;
};

template <>
inline bool Serialize::operator<<(const HeartbeatResponse& v) {
  return (*this) << v.status;
}

template <>
inline bool Deserialize::operator>>(HeartbeatResponse& v) {
  return (*this) >> v.status;
}

}  // namespace kraken
