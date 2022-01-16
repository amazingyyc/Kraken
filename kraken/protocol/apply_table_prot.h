#pragma once

#include <cinttypes>
#include <string>

#include "ps/table.h"
#include "rpc/deserialize.h"
#include "rpc/serialize.h"

namespace kraken {

/**
 * \brief This will not create a table, just return a unique table id.
 */
struct ApplyTableRequest {
  uint64_t model_id;
  std::string table_name;
  TableType table_type;
};

template <>
inline bool Serialize::operator<<(const ApplyTableRequest& v) {
  return (*this) << v.model_id && (*this) << v.table_name &&
         (*this) << v.table_type;
}

template <>
inline bool Deserialize::operator>>(ApplyTableRequest& v) {
  return (*this) >> v.model_id && (*this) >> v.table_name &&
         (*this) >> v.table_type;
}

struct ApplyTableResponse {
  uint64_t table_id;
};

template <>
inline bool Serialize::operator<<(const ApplyTableResponse& v) {
  return (*this) << v.table_id;
}

template <>
inline bool Deserialize::operator>>(ApplyTableResponse& v) {
  return (*this) >> v.table_id;
}

}  // namespace kraken
