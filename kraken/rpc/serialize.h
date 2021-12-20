#pragma once

#include <cstring>
#include <string>
#include <vector>

#include "common/mutable_buffer.h"
#include "rpc/protocol.h"

namespace kraken {

class Serialize {
private:
  // We donot remove or clear the buffer, just append data.
  MutableBuffer* buf_;

public:
  Serialize(MutableBuffer* buf) : buf_(buf) {
  }

  ~Serialize() = default;

  bool Append(const void* ptr, size_t size) {
    buf_->Append((const char*)ptr, size);
    return true;
  }

  template <typename T>
  bool operator<<(const T& v) {
    return false;
  }
};

#define BASIC_TYPE_SERIALIZE(T) \
  template <> \
  inline bool Serialize::operator<<(const T& v) { \
    static_assert(std::is_pod<T>::value, #T " must be a POD type."); \
    return Append(&v, sizeof(v)); \
  }

BASIC_TYPE_SERIALIZE(bool);
BASIC_TYPE_SERIALIZE(uint8_t);
BASIC_TYPE_SERIALIZE(int8_t);
BASIC_TYPE_SERIALIZE(uint16_t);
BASIC_TYPE_SERIALIZE(int16_t);
BASIC_TYPE_SERIALIZE(uint32_t);
BASIC_TYPE_SERIALIZE(int32_t);
BASIC_TYPE_SERIALIZE(uint64_t);
BASIC_TYPE_SERIALIZE(int64_t);
BASIC_TYPE_SERIALIZE(float);
BASIC_TYPE_SERIALIZE(double);

#undef BASIC_TYPE_SERIALIZE

template <>
inline bool Serialize::operator<<(const std::string& v) {
  uint64_t size = v.size();

  if (((*this) << size) == false) {
    return false;
  }

  return Append(v.data(), v.size());
}

template <>
inline bool Serialize::operator<<(const RequestHeader& v) {
  return (*this) << v.timestamp && (*this) << v.type;
}

template <>
inline bool Serialize::operator<<(const ReplyHeader& v) {
  return (*this) << v.timestamp && (*this) << v.error_code;
}

}  // namespace kraken
