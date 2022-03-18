#pragma once

#include "common/deserialize.h"
#include "common/serialize.h"

namespace kraken {

struct NotifyFinishTransferRequest {
  uint64_t from_node_id;
};

template <>
inline bool Serialize::operator<<(const NotifyFinishTransferRequest& v) {
  return (*this) << v.from_node_id;
}

template <>
inline bool Deserialize::operator>>(NotifyFinishTransferRequest& v) {
  return (*this) >> v.from_node_id;
}

struct NotifyFinishTransferResponse {};

template <>
inline bool Serialize::operator<<(const NotifyFinishTransferResponse& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(NotifyFinishTransferResponse& v) {
  return true;
}

}  // namespace kraken
