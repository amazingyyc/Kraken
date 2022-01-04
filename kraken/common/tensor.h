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

  ~Tensor() = default;

private:
  Tensor(std::shared_ptr<TensorStorage> storage, size_t offset,
         const Shape& shape, ElementType etype);

public:
  size_t offset() const;

  const Shape& shape() const;

  ElementType element_type() const;

  int64_t Size() const;

  int64_t NumBytes() const;

  int64_t Dim(int64_t) const;

  void* Ptr();
  void* Ptr() const;

  template <typename T>
  T* Data() {
    return (T*)Ptr();
  }

  template <typename T>
  T* Data() const {
    return (T*)Ptr();
  }

  std::string Str() const;

public:
  static Tensor Create(const std::vector<int64_t>& dims, ElementType etype);

  static Tensor Create(const Shape& shape, ElementType etype);

  static Tensor Create(std::shared_ptr<TensorStorage> storage, size_t offset,
                       const Shape& shape, ElementType etype);

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

  Tensor Reshape(const Shape& nshape) const;
  Tensor Reshape(const std::vector<int64_t>& dims) const;

  // Zero current
  Tensor Zero();

  // normalize initialize
  Tensor Norm(float lower, float upper);

  // Fetch one vector from a tesnor. the tensor must be a matrix.
  // Shape the same storage.
  Tensor Vector(int64_t idx) const;

  // Same shape/element type.
  Tensor Like() const;

  // Clone this tensor.
  Tensor Clone() const;

  // Concat to matrix.
  Tensor ConcatVec(const std::vector<Tensor>& vecs) const;

  // x = x  ^ 2
  Tensor Square(bool in_place=false);

  // x = sqrt(x)
  Tensor Sqrt(bool in_place=false);

  // ret = max(this, other)
  Tensor Max(const Tensor &other);
};

// operator override
Tensor operator+(float v, const Tensor& t);
Tensor operator-(float v, const Tensor& t);
Tensor operator*(float v, const Tensor& t);
Tensor operator/(float v, const Tensor& t);

}  // namespace kraken
