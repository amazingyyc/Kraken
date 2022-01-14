#pragma once

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/element_type.h"
#include "common/shape.h"
#include "common/tensor.h"
#include "common/zmq_buffer.h"
#include "ps/initializer/initializer.h"
#include "ps/optim/optim.h"
#include "ps/table.h"
#include "rpc/protocol.h"

namespace kraken {

class IBuffer {
public:
  virtual void Write(const char* ptr, size_t size) = 0;

  virtual void TransferForZMQ(ZMQBuffer* z_buf) = 0;
};

class Serialize {
private:
  // We donot remove or clear the buffer, just append data.
  IBuffer* buf_;

public:
  Serialize(IBuffer* buf) : buf_(buf) {
  }

  ~Serialize() = default;

  bool Write(const void* ptr, size_t size) {
    buf_->Write((const char*)ptr, size);
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
    return Write(&v, sizeof(v)); \
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

#define VEC_BASIC_TYPE_SERIALIZE(T) \
  template <> \
  inline bool Serialize::operator<<(const std::vector<T>& v) { \
    static_assert(std::is_pod<T>::value, #T " must be a POD type."); \
    uint64_t size = v.size(); \
    if (((*this) << size) == false) { \
      return false; \
    } \
    return Write(&(v[0]), size * sizeof(T)); \
  }

VEC_BASIC_TYPE_SERIALIZE(uint8_t);
VEC_BASIC_TYPE_SERIALIZE(int8_t);
VEC_BASIC_TYPE_SERIALIZE(uint16_t);
VEC_BASIC_TYPE_SERIALIZE(int16_t);
VEC_BASIC_TYPE_SERIALIZE(uint32_t);
VEC_BASIC_TYPE_SERIALIZE(int32_t);
VEC_BASIC_TYPE_SERIALIZE(uint64_t);
VEC_BASIC_TYPE_SERIALIZE(int64_t);
VEC_BASIC_TYPE_SERIALIZE(float);
VEC_BASIC_TYPE_SERIALIZE(double);

#undef VEC_BASIC_TYPE_SERIALIZE

template <>
inline bool Serialize::operator<<(const std::string& v) {
  uint64_t size = v.size();
  if (((*this) << size) == false) {
    return false;
  }

  return Write(v.data(), v.size());
}

template <>
inline bool Serialize::operator<<(const CompressType& v) {
  return (*this) << (uint8_t)v;
}

template <>
inline bool Serialize::operator<<(const RequestHeader& v) {
  static_assert(std::is_pod<RequestHeader>::value,
                "RequestHeader must be a POD type.");
  return Write(&v, sizeof(v));
}

template <>
inline bool Serialize::operator<<(const ReplyHeader& v) {
  static_assert(std::is_pod<ReplyHeader>::value,
                "ReplyHeader must be a POD type.");
  return Write(&v, sizeof(v));
}

template <>
inline bool Serialize::operator<<(const InitializerType& v) {
  return (*this) << (uint8_t)v;
}

template <>
inline bool Serialize::operator<<(const OptimType& v) {
  return (*this) << (uint8_t)v;
}

template <>
inline bool Serialize::operator<<(
    const std::unordered_map<std::string, std::string>& v) {
  uint64_t size = v.size();
  if (((*this) << size) == false) {
    return false;
  }

  for (const auto& item : v) {
    if (((*this) << item.first) == false || ((*this) << item.second) == false) {
      return false;
    }
  }

  return true;
}

template <>
inline bool Serialize::operator<<(const TableType& v) {
  return (*this) << ((uint8_t)v);
}

template <>
inline bool Serialize::operator<<(const DType& v) {
  return (*this) << ((uint8_t)v);
}

template <>
inline bool Serialize::operator<<(const ElementType& v) {
  return (*this) << v.dtype;
}

template <>
inline bool Serialize::operator<<(const Shape& v) {
  return (*this) << v.dims();
}

template <>
inline bool Serialize::operator<<(const Tensor& v) {
  if (((*this) << v.shape()) == false ||
      ((*this) << v.element_type()) == false) {
    return false;
  }

  return Write(v.Ptr(), v.NumBytes());
}

template <>
inline bool Serialize::operator<<(const std::vector<Tensor>& v) {
  uint64_t size = v.size();
  if (((*this) << size) == false) {
    return false;
  }

  for (auto& i : v) {
    if (((*this) << i) == false) {
      return false;
    }
  }

  return true;
}

}  // namespace kraken
