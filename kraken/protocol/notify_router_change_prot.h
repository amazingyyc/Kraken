#pragma once

#include "common/deserialize.h"
#include "common/router.h"
#include "common/serialize.h"

namespace kraken {

struct NotifyRouterChangeRequest {
  Router old_router;
  Router new_router;
};

template <>
inline bool Serialize::operator<<(const NotifyRouterChangeRequest& v) {
  return (*this) << v.old_router && (*this) << v.new_router;
}

template <>
inline bool Deserialize::operator>>(NotifyRouterChangeRequest& v) {
  return (*this) >> v.old_router && (*this) >> v.new_router;
}

struct NotifyRouterChangeResponse {};

template <>
inline bool Serialize::operator<<(const NotifyRouterChangeResponse& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(NotifyRouterChangeResponse& v) {
  return true;
}

}  // namespace kraken
