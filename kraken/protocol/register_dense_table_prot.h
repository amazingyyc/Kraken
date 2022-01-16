#pragma once

#include <cinttypes>
#include <string>

#include "rpc/deserialize.h"
#include "rpc/serialize.h"
#include "t/tensor.h"

namespace kraken {

/**
 * \brief Not like RegisterDenseTableRequest V2 will
 * push a tensor value to server that initialized by worker.
 */
struct RegisterDenseTableRequest {
  uint64_t model_id;

  uint64_t id;
  std::string name;

  Tensor val;
};

template <>
inline bool Serialize::operator<<(const RegisterDenseTableRequest& v) {
  return (*this) << v.model_id && (*this) << v.id && (*this) << v.name &&
         (*this) << v.val;
}

template <>
inline bool Deserialize::operator>>(RegisterDenseTableRequest& v) {
  return (*this) >> v.model_id && (*this) >> v.id && (*this) >> v.name &&
         (*this) >> v.val;
}

struct RegisterDenseTableResponse {};

template <>
inline bool Serialize::operator<<(const RegisterDenseTableResponse& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(RegisterDenseTableResponse& v) {
  return true;
}

}  // namespace kraken
