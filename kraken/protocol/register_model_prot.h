#pragma once

#include <string>
#include <unordered_map>

#include "ps/optim/optim.h"
#include "rpc/deserialize.h"
#include "rpc/serialize.h"

namespace kraken {

/**
 * \brief The worker register a model
 * means the worker has get the model's id.
 *
 */
struct RegisterModelRequest {
  uint64_t id;
  std::string name;
  OptimType optim_type;
  std::unordered_map<std::string, std::string> optim_conf;
};

template <>
inline bool Serialize::operator<<(const RegisterModelRequest& v) {
  return (*this) << v.id && (*this) << v.name && (*this) << v.optim_type &&
         (*this) << v.optim_conf;
}

template <>
inline bool Deserialize::operator>>(RegisterModelRequest& v) {
  return (*this) >> v.id && (*this) >> v.name && (*this) >> v.optim_type &&
         (*this) >> v.optim_conf;
}

struct RegisterModelResponse {};

template <>
inline bool Serialize::operator<<(const RegisterModelResponse& v) {
  return true;
}

template <>
inline bool Deserialize::operator>>(RegisterModelResponse& v) {
  return true;
}

}  // namespace kraken
