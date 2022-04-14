#pragma once

#include "common/deserialize.h"
#include "common/serialize.h"

namespace kraken {

struct TryLoadModelRequest {
  std::string load_dir;
};

template <>
inline bool Serialize::operator<<(const TryLoadModelRequest& v) {
  return (*this) << v.load_dir;
}

template <>
inline bool Deserialize::operator>>(TryLoadModelRequest& v) {
  return (*this) >> v.load_dir;
}

struct TryLoadModelResponse {
  bool success;
};

template <>
inline bool Serialize::operator<<(const TryLoadModelResponse& v) {
  return (*this) << v.success;
}

template <>
inline bool Deserialize::operator>>(TryLoadModelResponse& v) {
  return (*this) >> v.success;
}

}  // namespace kraken
