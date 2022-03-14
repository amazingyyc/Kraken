#pragma once

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/info.h"
#include "common/iwriter.h"
#include "common/router.h"
#include "rpc/protocol.h"
#include "t/coo_tensor_impl.h"
#include "t/element_type.h"
#include "t/layout.h"
#include "t/shape.h"
#include "t/tensor.h"
#include "t/tensor_impl.h"

namespace kraken {

class Serialize {
private:
  // We donot remove or clear the buffer, just append data.
  IWriter* buf_;

public:
  Serialize(IWriter* buf) : buf_(buf) {
  }

  ~Serialize() = default;

  bool Write(const void* ptr, size_t size) {
    return buf_->Write((const char*)ptr, size);
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
inline bool Serialize::operator<<(const StateType& v) {
  return (*this) << (uint32_t)v;
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
inline bool Serialize::operator<<(const Layout& v) {
  return (*this) << ((uint8_t)v);
}

template <>
inline bool Serialize::operator<<(const Tensor& v) {
  if (v.layout() != Layout::kStride && v.layout() != Layout::kCoo) {
    return false;
  }

  if ((*this) << v.layout() == false) {
    return false;
  }

  if (v.layout() == Layout::kStride) {
    if ((*this) << v.shape() == false) {
      return false;
    }

    if ((*this) << v.element_type() == false) {
      return false;
    }

    // Storage.
    return Write(v.Ptr(), v.NumBytes());
  } else if (v.layout() == Layout::kCoo) {
    auto coo_impl = std::dynamic_pointer_cast<CooTensorImpl>(v.impl());

    if ((*this) << coo_impl->indices() == false) {
      return false;
    }

    if ((*this) << coo_impl->values() == false) {
      return false;
    }

    if ((*this) << v.shape() == false) {
      return false;
    }

    return true;
  } else {
    return false;
  }
}

template <>
inline bool Serialize::operator<<(const std::vector<std::string>& v) {
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

template <>
inline bool Serialize::operator<<(
    const std::unordered_map<std::string, std::string>& v) {
  uint64_t size = v.size();
  if (((*this) << size) == false) {
    return false;
  }

  for (const auto& [key, val] : v) {
    if (((*this) << key) == false || ((*this) << val) == false) {
      return false;
    }
  }

  return true;
}

template <>
inline bool Serialize::operator<<(
    const std::unordered_map<StateType, Tensor>& v) {
  uint64_t size = v.size();
  if (((*this) << size) == false) {
    return false;
  }

  for (const auto& [key, val] : v) {
    if (((*this) << key) == false || ((*this) << val) == false) {
      return false;
    }
  }

  return true;
}

template <>
inline bool Serialize::operator<<(
    const std::unordered_map<StateType, int64_t>& v) {
  uint64_t size = v.size();
  if (((*this) << size) == false) {
    return false;
  }

  for (const auto& [key, val] : v) {
    if (((*this) << key) == false || ((*this) << val) == false) {
      return false;
    }
  }

  return true;
}

template <>
inline bool Serialize::operator<<(
    const std::unordered_map<std::string, uint64_t>& v) {
  uint64_t size = v.size();
  if (((*this) << size) == false) {
    return false;
  }

  for (const auto& [key, val] : v) {
    if (((*this) << key) == false || ((*this) << val) == false) {
      return false;
    }
  }

  return true;
}

template <>
inline bool Serialize::operator<<(const Value& v) {
  return ((*this) << v.val) && ((*this) << v.states) && ((*this) << v.states_i);
}

template <>
inline bool Serialize::operator<<(const std::vector<Value>& v) {
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

template <>
inline bool Serialize::operator<<(const TableMetaData& v) {
  return ((*this) << v.id) && ((*this) << v.name) &&
         ((*this) << v.table_type) && ((*this) << v.element_type) &&
         ((*this) << v.shape) && ((*this) << v.dimension) &&
         ((*this) << v.init_type) && ((*this) << v.init_conf);
}

template <>
inline bool Serialize::operator<<(
    const std::unordered_map<uint64_t, TableMetaData>& v) {
  uint64_t size = v.size();
  if (((*this) << size) == false) {
    return false;
  }

  for (const auto& [key, val] : v) {
    if (((*this) << key) == false || ((*this) << val) == false) {
      return false;
    }
  }

  return true;
}

template <>
inline bool Serialize::operator<<(const ModelMetaData& v) {
  return ((*this) << v.name) && ((*this) << v.optim_type) &&
         ((*this) << v.optim_conf) && ((*this) << v.table_mdatas);
}

template <>
inline bool Serialize::operator<<(const Router::Node& v) {
  return ((*this) << v.id) && ((*this) << v.name) && ((*this) << v.vnode_list);
}

template <>
inline bool Serialize::operator<<(const Router::VirtualNode& v) {
  return ((*this) << v.hash_v) && ((*this) << v.node_id) && ((*this) << v.name);
}

template <>
inline bool Serialize::operator<<(const std::map<uint64_t, Router::Node>& v) {
  uint64_t size = v.size();
  if (((*this) << size) == false) {
    return false;
  }

  for (const auto& [key, val] : v) {
    if (((*this) << key) == false || ((*this) << val) == false) {
      return false;
    }
  }

  return true;
}

template <>
inline bool Serialize::operator<<(
    const std::map<uint64_t, Router::VirtualNode>& v) {
  uint64_t size = v.size();
  if (((*this) << size) == false) {
    return false;
  }

  for (const auto& [key, val] : v) {
    if (((*this) << key) == false || ((*this) << val) == false) {
      return false;
    }
  }

  return true;
}

template <>
inline bool Serialize::operator<<(const Router& v) {
  return (*this) << v.version() && (*this) << v.nodes() &&
         (*this) << v.vnodes();
}

}  // namespace kraken
