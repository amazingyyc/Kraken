#pragma once

#include <cinttypes>
#include <iostream>
#include <memory>

#include "t/device.h"
#include "t/element_type.h"
#include "t/layout.h"
#include "t/shape.h"
#include "t/storage.h"

namespace kraken {
class Tensor;

class TensorImpl : public std::enable_shared_from_this<TensorImpl> {
protected:
  // Which kind tensor.
  Layout layout_;

  // For dense and sparse will means different.
  Shape shape_;

  // memory
  std::shared_ptr<Storage> storage_;

  // byte offset of tensor
  size_t offset_;

  // element type
  ElementType element_type_;

protected:
  // For sub-class.
  TensorImpl(Layout layout, const Shape& shape);

  // For dense tensor.
  TensorImpl(const Shape& shape, std::shared_ptr<Storage> storage,
             size_t offset, ElementType etype);

  TensorImpl(const TensorImpl&) = delete;
  TensorImpl& operator=(const TensorImpl&) = delete;
  TensorImpl(TensorImpl&&) = delete;
  TensorImpl& operator=(TensorImpl&&) = delete;

public:
  virtual ~TensorImpl() = default;

public:
  // Create a Dense TensorImpl.
  static std::shared_ptr<TensorImpl> Dense(const Shape& shape,
                                           ElementType etype);

  static std::shared_ptr<TensorImpl> Dense(const Shape& shape,
                                           std::shared_ptr<Storage> storage,
                                           size_t offset, ElementType etype);

  static std::shared_ptr<TensorImpl> Empty(ElementType etype);

  static std::shared_ptr<TensorImpl> Coo(std::shared_ptr<TensorImpl> indices,
                                         std::shared_ptr<TensorImpl> values,
                                         const Shape& shape);

  static std::shared_ptr<TensorImpl> EmptyCoo(ElementType value_etype,
                                              const Shape& shape);

public:
  Layout layout() const;

  bool IsCoo() const;

  bool IsDense() const;

  const Shape& shape() const;

  virtual const Tensor& indices() const;

  virtual const Tensor& values() const;

  virtual ElementType element_type() const;

  virtual kraken::Device* Device() const;

  virtual int64_t Size() const;

  virtual int64_t NumBytes() const;

  virtual void* Ptr() const;

  virtual bool IsEmpty() const;

  template <typename T>
  T* Data() const {
    return (T*)Ptr();
  }

public:
  virtual std::shared_ptr<TensorImpl> Add(const TensorImpl& other) const;
  virtual std::shared_ptr<TensorImpl> Sub(const TensorImpl& other) const;
  virtual std::shared_ptr<TensorImpl> Mul(const TensorImpl& other) const;
  virtual std::shared_ptr<TensorImpl> Div(const TensorImpl& other) const;

  virtual std::shared_ptr<TensorImpl> AddAssign(const TensorImpl& other);
  virtual std::shared_ptr<TensorImpl> SubAssign(const TensorImpl& other);
  virtual std::shared_ptr<TensorImpl> MulAssign(const TensorImpl& other);
  virtual std::shared_ptr<TensorImpl> DivAssign(const TensorImpl& other);

  virtual std::shared_ptr<TensorImpl> Add(float v) const;
  virtual std::shared_ptr<TensorImpl> Sub(float v) const;
  virtual std::shared_ptr<TensorImpl> Mul(float v) const;
  virtual std::shared_ptr<TensorImpl> Div(float v) const;

  virtual std::shared_ptr<TensorImpl> AddAssign(float v);
  virtual std::shared_ptr<TensorImpl> SubAssign(float v);
  virtual std::shared_ptr<TensorImpl> MulAssign(float v);
  virtual std::shared_ptr<TensorImpl> DivAssign(float v);

  virtual std::shared_ptr<TensorImpl> BeSub(float v) const;
  virtual std::shared_ptr<TensorImpl> BeDiv(float v) const;

  virtual std::shared_ptr<TensorImpl> Reshape(const Shape& nshape) const;
  virtual std::shared_ptr<TensorImpl> Reshape(
      const std::vector<int64_t>& dims) const;

  virtual std::shared_ptr<TensorImpl> Zero();

  // Same shape/element type.
  virtual std::shared_ptr<TensorImpl> Like() const;

  // Clone this tensor.
  virtual std::shared_ptr<TensorImpl> Clone() const;

  virtual std::shared_ptr<TensorImpl> Constant(float v);

  // x = x ^ 2
  virtual std::shared_ptr<TensorImpl> Square(bool in_place);

  // x = sqrt(x)
  virtual std::shared_ptr<TensorImpl> Sqrt(bool in_place);

  // ret = max(this, other)
  virtual std::shared_ptr<TensorImpl> Max(const TensorImpl& other) const;

  // Fetch one vector from a tesnor. the tensor must be a matrix.
  // Shape the same storage.
  virtual std::shared_ptr<TensorImpl> Vector(int64_t idx) const;

  // Concat to matrix.
  virtual std::shared_ptr<TensorImpl> ConcatVector(
      const std::vector<std::shared_ptr<TensorImpl>>& vecs) const;

  virtual std::shared_ptr<TensorImpl> Normal(float mean, float stddev);

  virtual std::shared_ptr<TensorImpl> XavierNormal(float gain);

  virtual std::shared_ptr<TensorImpl> Uniform(float lower, float upper);

  virtual std::shared_ptr<TensorImpl> XavierUniform(float gain);

  virtual std::shared_ptr<TensorImpl> Abs(bool in_place = false);

  virtual std::shared_ptr<TensorImpl> TopK(int64_t k) const;

  virtual std::shared_ptr<TensorImpl> Take(const TensorImpl& indices) const;

  virtual std::shared_ptr<TensorImpl> FlatNonZero(float th) const;

  virtual std::shared_ptr<TensorImpl> NonZero(float th) const;

  virtual std::shared_ptr<TensorImpl> Transpose(int64_t d0 = 0,
                                                int64_t d1 = 1) const;

  virtual std::shared_ptr<TensorImpl> ToDense() const;

  // Convert to Coo Tensor the indices always be int64.
  // The indices shape will be: [1, nnz]
  // The values shape will be: [nnz]
  // The shape of Coo will be: [this->Size()]
  virtual std::shared_ptr<TensorImpl> ToCoo(float th) const;

  virtual std::shared_ptr<TensorImpl> LtKeep(float th) const;
};

}  // namespace kraken
