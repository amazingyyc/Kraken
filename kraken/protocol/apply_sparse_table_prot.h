// #pragma once

// #include <cinttypes>
// #include <string>

// #include "common/deserialize.h"
// #include "common/serialize.h"
// #include "ps/initializer/initializer.h"
// #include "t/element_type.h"

// namespace kraken {

// struct ApplySparseTableRequest {
//   uint64_t model_id;

//   std::string name;

//   int64_t dimension;
//   ElementType element_type;

//   InitializerType init_type;
//   std::unordered_map<std::string, std::string> init_conf;
// };

// template <>
// inline bool Serialize::operator<<(const ApplySparseTableRequest& v) {
//   return (*this) << v.model_id && (*this) << v.name && (*this) << v.dimension &&
//          (*this) << v.element_type && (*this) << v.init_type &&
//          (*this) << v.init_conf;
// }

// template <>
// inline bool Deserialize::operator>>(ApplySparseTableRequest& v) {
//   return (*this) >> v.model_id && (*this) >> v.name && (*this) >> v.dimension &&
//          (*this) >> v.element_type && (*this) >> v.init_type &&
//          (*this) >> v.init_conf;
// }

// struct ApplySparseTableResponse {
//   uint64_t table_id;
// };

// template <>
// inline bool Serialize::operator<<(const ApplySparseTableResponse& v) {
//   return (*this) << v.table_id;
// }

// template <>
// inline bool Deserialize::operator>>(ApplySparseTableResponse& v) {
//   return (*this) >> v.table_id;
// }

// }  // namespace kraken
