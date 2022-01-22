#pragma once

#include <cinttypes>
#include <iostream>

#include "t/device.h"
#include "t/layout.h"
#include "t/tensor.h"
#include "t/tensor_impl.h"

namespace kraken {

class CooTensorImpl : public TensorImpl {
private:
  // number of sparse dimensions
  int64_t sparse_dim_ = 0;
  int64_t dense_dim_ = 0;

  // uint32/int32/uint64/int64
  // Must be dense tensor.
  Tensor indices_;
  Tensor values_;

public:
  CooTensorImpl(const Tensor& indices, const Tensor& values,
                const Shape& shape);

  CooTensorImpl(const CooTensorImpl&) = delete;
  CooTensorImpl& operator=(const CooTensorImpl&) = delete;
  CooTensorImpl(CooTensorImpl&&) = delete;
  CooTensorImpl& operator=(CooTensorImpl&&) = delete;

  ~CooTensorImpl() = default;

public:
  int64_t sparse_dim() const;

  int64_t dense_dim() const;

  int64_t nnz() const;

  const Tensor& indices() const override;

  const Tensor& values() const override;

  ElementType element_type() const override;

  kraken::Device* Device() const override;

  int64_t Size() const override;

  int64_t NumBytes() const override;

  void* Ptr() const override;

  bool IsEmpty() const override;

public:
  std::shared_ptr<TensorImpl> Add(const TensorImpl& other) const override;
  std::shared_ptr<TensorImpl> Sub(const TensorImpl& other) const override;
  std::shared_ptr<TensorImpl> Mul(const TensorImpl& other) const override;
  std::shared_ptr<TensorImpl> Div(const TensorImpl& other) const override;

  std::shared_ptr<TensorImpl> AddAssign(const TensorImpl& other) override;
  std::shared_ptr<TensorImpl> SubAssign(const TensorImpl& other) override;
  std::shared_ptr<TensorImpl> MulAssign(const TensorImpl& other) override;
  std::shared_ptr<TensorImpl> DivAssign(const TensorImpl& other) override;

  std::shared_ptr<TensorImpl> Add(float v) const override;
  std::shared_ptr<TensorImpl> Sub(float v) const override;
  std::shared_ptr<TensorImpl> Mul(float v) const override;
  std::shared_ptr<TensorImpl> Div(float v) const override;

  std::shared_ptr<TensorImpl> AddAssign(float v) override;
  std::shared_ptr<TensorImpl> SubAssign(float v) override;
  std::shared_ptr<TensorImpl> MulAssign(float v) override;
  std::shared_ptr<TensorImpl> DivAssign(float v) override;

  std::shared_ptr<TensorImpl> BeSub(float v) const override;
  std::shared_ptr<TensorImpl> BeDiv(float v) const override;

  std::shared_ptr<TensorImpl> Reshape(const Shape& nshape) const override;
  std::shared_ptr<TensorImpl> Reshape(
      const std::vector<int64_t>& dims) const override;

  std::shared_ptr<TensorImpl> Zero() override;

  std::shared_ptr<TensorImpl> Like() const override;

  std::shared_ptr<TensorImpl> Clone() const override;

  std::shared_ptr<TensorImpl> Constant(float v) override;

  std::shared_ptr<TensorImpl> Square(bool in_place) override;

  std::shared_ptr<TensorImpl> Sqrt(bool in_place) override;

  std::shared_ptr<TensorImpl> Max(const TensorImpl& other) const override;

  std::shared_ptr<TensorImpl> Vector(int64_t idx) const override;

  std::shared_ptr<TensorImpl> ConcatVector(
      const std::vector<std::shared_ptr<TensorImpl>>& vecs) const override;

  std::shared_ptr<TensorImpl> Normal(float mean, float stddev) override;

  std::shared_ptr<TensorImpl> XavierNormal(float gain) override;

  std::shared_ptr<TensorImpl> Uniform(float lower, float upper) override;

  std::shared_ptr<TensorImpl> XavierUniform(float gain) override;

  std::shared_ptr<TensorImpl> Abs(bool in_place = false) override;

  std::shared_ptr<TensorImpl> TopK(int64_t k) const override;

  std::shared_ptr<TensorImpl> Take(const TensorImpl& indices) const override;

  std::shared_ptr<TensorImpl> FlatNonZero(float th) const override;

  std::shared_ptr<TensorImpl> NonZero(float th) const override;

  std::shared_ptr<TensorImpl> Transpose(int64_t d0 = 0,
                                        int64_t d1 = 1) const override;

  std::shared_ptr<TensorImpl> ToDense() const override;

  std::shared_ptr<TensorImpl> ToCoo(float th) const override;

  std::shared_ptr<TensorImpl> LtKeep(float th) const override;

  std::shared_ptr<TensorImpl> Cast(ElementType to_type) override;
};

}  // namespace kraken
