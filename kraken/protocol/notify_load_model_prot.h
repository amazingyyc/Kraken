#pragma once

#include "common/deserialize.h"
#include "common/serialize.h"

namespace kraken {

struct NotifyLoadModelRequest {
  std::string load_dir;
};

template <>
inline bool Serialize::operator<<(const NotifyLoadModelRequest& v) {
  return (*this) << v.load_dir;
}

template <>
inline bool Deserialize::operator>>(NotifyLoadModelRequest& v) {
  return (*this) >> v.load_dir;
}

struct NotifyLoadModelResponse {};

template <>
inline bool Serialize::operator<<(const NotifyLoadModelResponse& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(NotifyLoadModelResponse& v) {
  return true;
}

}  // namespace kraken
