#pragma once

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/element_type.h"
#include "common/mutable_buffer.h"
#include "common/shape.h"
#include "common/tensor.h"
#include "common/tensor_storage.h"
#include "ps/initializer/initializer.h"
#include "ps/optim/optim.h"
#include "ps/table.h"
#include "rpc/protocol.h"

namespace kraken {

class Deserialize {
private:
  const char* ptr_;
  size_t length_;
  size_t offset_;

public:
  Deserialize(const char* ptr, size_t length)
      : ptr_(ptr), length_(length), offset_(0) {
  }

  ~Deserialize() {
    ptr_ = nullptr;
    length_ = 0;
    offset_ = 0;
  }

public:
  size_t offset() const {
    return offset_;
  }

  bool Read(void* target, size_t size) {
    if (ptr_ == nullptr || offset_ + size > length_) {
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
    return Read(&v, sizeof(v)); \
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

#define VEC_BASIC_TYPE_DESERIALIZE(T) \
  template <> \
  inline bool Deserialize::operator>>(std::vector<T>& v) { \
    static_assert(std::is_pod<T>::value, #T " must be a POD type."); \
    uint64_t size; \
    if (((*this) >> size) == false) { \
      return false; \
    } \
    v.resize(size); \
    return Read(&(v[0]), size * sizeof(T)); \
  }

VEC_BASIC_TYPE_DESERIALIZE(uint8_t);
VEC_BASIC_TYPE_DESERIALIZE(int8_t);
VEC_BASIC_TYPE_DESERIALIZE(uint16_t);
VEC_BASIC_TYPE_DESERIALIZE(int16_t);
VEC_BASIC_TYPE_DESERIALIZE(uint32_t);
VEC_BASIC_TYPE_DESERIALIZE(int32_t);
VEC_BASIC_TYPE_DESERIALIZE(uint64_t);
VEC_BASIC_TYPE_DESERIALIZE(int64_t);
VEC_BASIC_TYPE_DESERIALIZE(float);
VEC_BASIC_TYPE_DESERIALIZE(double);

#undef VEC_BASIC_TYPE_DESERIALIZE

template <>
inline bool Deserialize::operator>>(std::string& v) {
  uint64_t size;
  if (((*this) >> size) == false) {
    return false;
  }

  v.resize(size);

  return Read((void*)v.data(), size);
}

template <>
inline bool Deserialize::operator>>(CompressType& v) {
  uint8_t type;
  if (!((*this) >> type)) {
    return false;
  }

  v = (CompressType)type;

  return true;
}

template <>
inline bool Deserialize::operator>>(RequestHeader& v) {
  static_assert(std::is_pod<RequestHeader>::value,
                "RequestHeader must be a POD type.");
  return Read(&v, sizeof(v));
}

template <>
inline bool Deserialize::operator>>(ReplyHeader& v) {
  static_assert(std::is_pod<ReplyHeader>::value,
                "ReplyHeader must be a POD type.");
  return Read(&v, sizeof(v));
}

template <>
inline bool Deserialize::operator>>(InitializerType& v) {
  uint8_t type;
  if (!((*this) >> type)) {
    return false;
  }

  v = (InitializerType)type;

  return true;
}

template <>
inline bool Deserialize::operator>>(OptimType& v) {
  uint8_t type;
  if (!((*this) >> type)) {
    return false;
  }

  v = (OptimType)type;

  return true;
}

template <>
inline bool Deserialize::operator>>(
    std::unordered_map<std::string, std::string>& v) {
  v.clear();

  uint64_t size;
  if (((*this) >> size) == false) {
    return false;
  }

  std::string key;
  std::string value;

  for (uint64_t i = 0; i < size; ++i) {
    if (((*this) >> key) == false || ((*this) >> value) == false) {
      return false;
    }

    v.emplace(std::move(key), std::move(value));
  }

  return true;
}

template <>
inline bool Deserialize::operator>>(TableType& v) {
  uint8_t uv;

  if (((*this) >> uv) == false) {
    return false;
  }

  v = (TableType)uv;

  return true;
}

template <>
inline bool Deserialize::operator>>(DType& v) {
  uint8_t uv;

  if (((*this) >> uv) == false) {
    return false;
  }

  v = (DType)uv;

  return true;
}

template <>
inline bool Deserialize::operator>>(ElementType& v) {
  return (*this) >> v.dtype;
}

template <>
inline bool Deserialize::operator>>(Shape& v) {
  std::vector<int64_t> dims;
  if (((*this) >> dims) == false) {
    return false;
  }

  v = Shape(dims);

  return true;
}

template <>
inline bool Deserialize::operator>>(Tensor& v) {
  Shape shape;
  ElementType etype;

  if (((*this) >> shape) == false || ((*this) >> etype) == false) {
    return false;
  }

  size_t size = shape.Size() * etype.ByteWidth();
  auto storage = TensorStorage::Create(size);

  if (Read(storage->ptr(), size) == false) {
    return false;
  }

  v = Tensor::Create(storage, 0, shape, etype);

  return true;
}

template <>
inline bool Deserialize::operator>>(std::vector<Tensor>& v) {
  uint64_t size;
  if (((*this) >> size) == false) {
    return false;
  }

  v.resize(size);
  for (uint64_t i = 0; i < size; ++i) {
    if (((*this) >> v[i]) == false) {
      return false;
    }
  }

  return true;
}

}  // namespace kraken
