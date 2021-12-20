#pragma once

#include <memory>

#include "common/element_type.h"
#include "common/shape.h"
#include "common/tensor_storage.h"

namespace kraken {

class Tensor {
private:
  // memory
  std::shared_ptr<TensorStorage> storage_;

  // byte offset of tensor
  size_t offset_;

  // the tensor shape
  Shape shape_;

  // element type
  ElementType element_type_;

public:
  Tensor() = default;

  Tensor(std::shared_ptr<TensorStorage> storage, size_t offset,
         const Shape& shape, ElementType element_type);

  ~Tensor() = default;

public:
  const Shape& shape() const;

  ElementType element_type() const;

  int64_t size() const;

  int64_t num_bytes() const;

  int64_t dim(int64_t) const;

  void* ptr();
  void* ptr() const;

  template <typename T>
  T* data() {
    return (T*)ptr();
  }

  template <typename T>
  T* data() const {
    return (T*)ptr();
  }

public:
  static Tensor create(const std::vector<int64_t>& dims,
                       ElementType element_type);

  static Tensor create(const Shape& shape, ElementType element_type);

public:
  Tensor operator+(const Tensor& other) const;
  Tensor operator-(const Tensor& other) const;
  Tensor operator*(const Tensor& other) const;
  Tensor operator/(const Tensor& other) const;

  Tensor operator+=(const Tensor& other);
  Tensor operator-=(const Tensor& other);
  Tensor operator*=(const Tensor& other);
  Tensor operator/=(const Tensor& other);

  Tensor operator+(float v) const;
  Tensor operator-(float v) const;
  Tensor operator*(float v) const;
  Tensor operator/(float v) const;

  Tensor operator+=(float v);
  Tensor operator-=(float v);
  Tensor operator*=(float v);
  Tensor operator/=(float v);

  // Clone this tensor.
  Tensor clone() const;
};

// operator override
Tensor operator+(float v, const Tensor& t);
Tensor operator-(float v, const Tensor& t);
Tensor operator*(float v, const Tensor& t);
Tensor operator/(float v, const Tensor& t);

}  // namespace kraken
