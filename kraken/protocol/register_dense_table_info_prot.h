// #pragma once

// #include <cinttypes>
// #include <string>

// #include "common/deserialize.h"
// #include "common/serialize.h"
// #include "t/element_type.h"
// #include "t/shape.h"

// namespace kraken {

// struct RegisterDenseTableInfoRequest {
//   uint64_t model_id;

//   uint64_t id;
//   std::string name;

//   Shape shape;
//   ElementType element_type;
// };

// template <>
// inline bool Serialize::operator<<(const RegisterDenseTableInfoRequest& v) {
//   return (*this) << v.model_id && (*this) << v.id && (*this) << v.name &&
//          (*this) << v.shape && (*this) << v.element_type;
// }

// template <>
// inline bool Deserialize::operator>>(RegisterDenseTableInfoRequest& v) {
//   return (*this) >> v.model_id && (*this) >> v.id && (*this) >> v.name &&
//          (*this) >> v.shape && (*this) >> v.element_type;
// }

// struct RegisterDenseTableInfoResponse {};

// template <>
// inline bool Serialize::operator<<(const RegisterDenseTableInfoResponse& v) {
//   return true;
// }

// template <>
// inline bool Deserialize::operator>>(RegisterDenseTableInfoResponse& v) {
//   return true;
// }

// }  // namespace kraken
