// #pragma once

// #include <cinttypes>
// #include <string>

// #include "common/deserialize.h"
// #include "common/serialize.h"
// #include "t/element_type.h"
// #include "t/shape.h"

// namespace kraken {

// struct ApplyDenseTableRequest {
//   uint64_t model_id;

//   std::string name;
//   Shape shape;
//   ElementType element_type;
// };

// template <>
// inline bool Serialize::operator<<(const ApplyDenseTableRequest& v) {
//   return (*this) << v.model_id && (*this) << v.name && (*this) << v.shape &&
//          (*this) << v.element_type;
// }

// template <>
// inline bool Deserialize::operator>>(ApplyDenseTableRequest& v) {
//   return (*this) >> v.model_id && (*this) >> v.name && (*this) >> v.shape &&
//          (*this) >> v.element_type;
// }

// struct ApplyDenseTableResponse {
//   uint64_t table_id;
// };

// template <>
// inline bool Serialize::operator<<(const ApplyDenseTableResponse& v) {
//   return (*this) << v.table_id;
// }

// template <>
// inline bool Deserialize::operator>>(ApplyDenseTableResponse& v) {
//   return (*this) >> v.table_id;
// }

// }  // namespace kraken
