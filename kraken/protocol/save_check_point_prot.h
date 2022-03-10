// #pragma once

// #include <cinttypes>
// #include <string>

// #include "common/deserialize.h"
// #include "common/serialize.h"
// #include "ps/initializer/initializer.h"
// #include "t/element_type.h"

// namespace kraken {

// struct SaveCheckPointRequest {
//   uint64_t model_id;
// };

// template <>
// inline bool Serialize::operator<<(const SaveCheckPointRequest& v) {
//   return (*this) << v.model_id;
// }

// template <>
// inline bool Deserialize::operator>>(SaveCheckPointRequest& v) {
//   return (*this) >> v.model_id;
// }

// struct SaveCheckPointResponse {};

// template <>
// inline bool Serialize::operator<<(const SaveCheckPointResponse& v) {
//   return true;
// }

// template <>
// inline bool Deserialize::operator>>(SaveCheckPointResponse& v) {
//   return true;
// }

// }  // namespace kraken
