#pragma once

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/ireader.h"
#include "ps/initializer/initializer.h"
#include "ps/optim/optim.h"
#include "ps/table.h"
#include "rpc/protocol.h"
#include "t/coo_tensor_impl.h"
#include "t/element_type.h"
#include "t/layout.h"
#include "t/shape.h"
#include "t/storage.h"
#include "t/tensor.h"
#include "t/tensor_impl.h"

namespace kraken {

class Deserialize {
private:
  IReader* reader_;

public:
  Deserialize(IReader* reader) : reader_(reader) {
  }

public:
  bool Read(void* target, size_t size) {
    return reader_->Read(target, size);
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
inline bool Deserialize::operator>>(StateType& v) {
  uint32_t type;
  if (!((*this) >> type)) {
    return false;
  }

  v = (StateType)type;

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
inline bool Deserialize::operator>>(Layout& v) {
  uint8_t uv;

  if (((*this) >> uv) == false) {
    return false;
  }

  v = (Layout)uv;

  return true;
}

template <>
inline bool Deserialize::operator>>(Tensor& v) {
  // Firstly read the layout.
  Layout layout;
  if ((*this) >> layout == false) {
    return false;
  }

  if (layout == Layout::kStride) {
    Shape shape;
    ElementType etype;

    if ((*this) >> shape == false) {
      return false;
    }

    if ((*this) >> etype == false) {
      return false;
    }

    // Read storage.
    size_t nbytes = shape.Size() * etype.ByteWidth();
    auto storage = Storage::Create(nbytes);

    if (Read(storage->ptr(), nbytes) == false) {
      return false;
    }

    auto impl = TensorImpl::Dense(shape, storage, 0, etype);
    v = Tensor(impl);

    return true;
  } else if (layout == Layout::kCoo) {
    Tensor indices;
    Tensor values;
    Shape shape;

    if ((*this) >> indices == false) {
      return false;
    }

    if ((*this) >> values == false) {
      return false;
    }

    if ((*this) >> shape == false) {
      return false;
    }

    auto impl = std::make_shared<CooTensorImpl>(indices, values, shape);
    v = Tensor(impl);

    return true;
  } else {
    return false;
  }
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

template <>
inline bool Deserialize::operator>>(
    std::unordered_map<std::string, std::string>& v) {
  v.clear();

  uint64_t size;
  if (((*this) >> size) == false) {
    return false;
  }

  v.reserve(size);

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
inline bool Deserialize::operator>>(std::unordered_map<StateType, Tensor>& v) {
  v.clear();

  uint64_t size;
  if (((*this) >> size) == false) {
    return false;
  }

  v.reserve(size);

  for (uint64_t i = 0; i < size; ++i) {
    StateType key;
    Tensor value;

    if (((*this) >> key) == false || ((*this) >> value) == false) {
      return false;
    }

    v.emplace(key, value);
  }

  return true;
}

template <>
inline bool Deserialize::operator>>(std::unordered_map<StateType, int64_t>& v) {
  v.clear();

  uint64_t size;
  if (((*this) >> size) == false) {
    return false;
  }

  v.reserve(size);

  for (uint64_t i = 0; i < size; ++i) {
    StateType key;
    int64_t value;

    if (((*this) >> key) == false || ((*this) >> value) == false) {
      return false;
    }

    v.emplace(key, value);
  }

  return true;
}

template <>
inline bool Deserialize::operator>>(
    std::unordered_map<std::string, uint64_t>& v) {
  v.clear();

  uint64_t size;
  if (((*this) >> size) == false) {
    return false;
  }

  v.reserve(size);

  for (uint64_t i = 0; i < size; ++i) {
    std::string key;
    uint64_t value;

    if (((*this) >> key) == false || ((*this) >> value) == false) {
      return false;
    }

    v.emplace(std::move(key), value);
  }

  return true;
}

template <>
inline bool Deserialize::operator>>(Bag& v) {
  return (*this) >> v.state && (*this) >> v.state_i;
}

template <>
inline bool Deserialize::operator>>(Table::Value& v) {
  return (*this) >> v.val && (*this) >> v.bag;
}

}  // namespace kraken
