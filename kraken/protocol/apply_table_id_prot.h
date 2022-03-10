// #pragma once

// #include <cinttypes>
// #include <string>

// #include "common/deserialize.h"
// #include "common/serialize.h"

// namespace kraken {

// struct ApplyTableIdRequest {
//   std::string model_name;
//   std::string table_name;
// };

// template <>
// inline bool Serialize::operator<<(const ApplyTableIdRequest& v) {
//   return (*this) << v.model_name && (*this) << v.table_name;
// }

// template <>
// inline bool Deserialize::operator>>(ApplyTableIdRequest& v) {
//   return (*this) >> v.model_name && (*this) >> v.table_name;
// }

// struct ApplyTableIdResponse {
//   uint64_t table_id;
// };

// template <>
// inline bool Serialize::operator<<(const ApplyTableIdResponse& v) {
//   return (*this) << v.table_id;
// }

// template <>
// inline bool Deserialize::operator>>(ApplyTableIdResponse& v) {
//   return (*this) >> v.table_id;
// }

// }  // namespace kraken
