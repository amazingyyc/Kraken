#pragma once

#include "common/deserialize.h"
#include "common/router.h"
#include "common/serialize.h"

namespace kraken {

struct NotifyNodeJoinRequest {
  uint64_t joined_id;
  Router old_router;
  Router new_router;
};

template <>
inline bool Serialize::operator<<(const NotifyNodeJoinRequest& v) {
  return (*this) << v.joined_id && (*this) << v.old_router &&
         (*this) << v.new_router;
}

template <>
inline bool Deserialize::operator>>(NotifyNodeJoinRequest& v) {
  return (*this) >> v.joined_id && (*this) >> v.old_router &&
         (*this) >> v.new_router;
}

struct NotifyNodeJoinResponse {};

template <>
inline bool Serialize::operator<<(const NotifyNodeJoinResponse& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(NotifyNodeJoinResponse& v) {
  return true;
}

}  // namespace kraken
