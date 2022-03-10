#pragma once

#include "common/deserialize.h"
#include "common/router.h"
#include "common/serialize.h"
namespace kraken {

struct FetchRouterRequest {};

template <>
inline bool Serialize::operator<<(const FetchRouterRequest& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(FetchRouterRequest& v) {
  return true;
}

struct FetchRouterResponse {
  Router router;
};

template <>
inline bool Serialize::operator<<(const FetchRouterResponse& v) {
  return (*this) << v.router;
}

template <>
inline bool Deserialize::operator>>(FetchRouterResponse& v) {
  return (*this) >> v.router;
}

}  // namespace kraken
