#pragma once

#include "common/deserialize.h"
#include "common/router.h"
#include "common/serialize.h"

namespace kraken {

struct TryJoinRequest {
  std::string addr;
};

template <>
inline bool Serialize::operator<<(const TryJoinRequest& v) {
  return (*this) << v.addr;
}

template <>
inline bool Deserialize::operator>>(TryJoinRequest& v) {
  return (*this) >> v.addr;
}

struct TryJoinResponse {
  bool allow;
  uint64_t node_id;

  Router old_router;
  Router new_router;
};

template <>
inline bool Serialize::operator<<(const TryJoinResponse& v) {
  return (*this) << v.allow && (*this) << v.node_id &&
         (*this) << v.old_router && (*this) << v.new_router;
}

template <>
inline bool Deserialize::operator>>(TryJoinResponse& v) {
  return (*this) >> v.allow && (*this) >> v.node_id &&
         (*this) >> v.old_router && (*this) >> v.new_router;
}

}  // namespace kraken
