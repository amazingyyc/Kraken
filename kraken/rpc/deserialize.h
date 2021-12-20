#pragma once

#include <cstring>
#include <string>
#include <vector>

#include "rpc/protocol.h"

namespace kraken {

class Deserialize {
private:
  const char* ptr_;
  size_t len_;
  size_t offset_;

public:
  Deserialize(const char* ptr, size_t len) : ptr_(ptr), len_(len), offset_(0) {
  }

  ~Deserialize() = default;

public:
  bool read(void* target, size_t size) {
    if (offset_ + size > len_) {
      return false;
    }

    memcpy(target, ptr_ + offset_, size);
    offset_ += size;

    return true;
  }

  template <typename T>
  bool operator>>(T& v) {
    return false;
  }
};

#define BASIC_TYPE_DESERIALIZE(T) \
  template <> \
  inline bool Deserialize::operator>>(T& v) { \
    static_assert(std::is_pod<T>::value, #T " must be a POD type."); \
    return read(&v, sizeof(v)); \
  }

BASIC_TYPE_DESERIALIZE(bool);
BASIC_TYPE_DESERIALIZE(uint8_t);
BASIC_TYPE_DESERIALIZE(int8_t);
BASIC_TYPE_DESERIALIZE(uint16_t);
BASIC_TYPE_DESERIALIZE(int16_t);
BASIC_TYPE_DESERIALIZE(uint32_t);
BASIC_TYPE_DESERIALIZE(int32_t);
BASIC_TYPE_DESERIALIZE(uint64_t);
BASIC_TYPE_DESERIALIZE(int64_t);
BASIC_TYPE_DESERIALIZE(float);
BASIC_TYPE_DESERIALIZE(double);

#undef BASIC_TYPE_DESERIALIZE

template <>
inline bool Deserialize::operator>>(std::string& v) {
  uint64_t size;

  if (((*this) >> size) == false) {
    return false;
  }

  v.resize(size);

  return read((void*)v.data(), size);
}

template <>
inline bool Deserialize::operator>>(RequestHeader& v) {
  return (*this) >> v.timestamp && (*this) >> v.type;
}

template <>
inline bool Deserialize::operator>>(ReplyHeader& v) {
  return (*this) >> v.timestamp && (*this) >> v.error_code;
}

}  // namespace kraken
