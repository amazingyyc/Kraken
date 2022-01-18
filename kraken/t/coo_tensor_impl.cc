#include "t/coo_tensor_impl.h"

#include "common/exception.h"
#include "t/math.h"
#include "t/tensor.h"

namespace kraken {

CooTensorImpl::CooTensorImpl(const Tensor& indices, const Tensor& values,
                             const Shape& shape)
    : TensorImpl(Layout::kCoo, shape), indices_(indices), values_(values) {
  // ref: https://pytorch.org/docs/stable/sparse.html
  ARGUMENT_CHECK(indices_.IsDense() && values_.IsDense(),
                 "Coo need indices and values is dense.");
  ARGUMENT_CHECK(
      indices_.element_type().Is<uint32_t>() ||
          indices_.element_type().Is<int32_t>() ||
          indices_.element_type().Is<uint64_t>() ||
          indices_.element_type().Is<int64_t>(),
      "Coo Tensor need indices element type is:uint32/int32/uint64/int64.");
  ARGUMENT_CHECK(indices_.shape().NDims() == 2,
                 "Coo need indices's NDim is 2.");
  ARGUMENT_CHECK(
      indices_.shape()[-1] == values_.shape()[0],
      "Coo need indices's last dimension same with values's first dimension.");
  ARGUMENT_CHECK(
      shape_.NDims() + 1 == indices_.shape()[0] + values_.shape().NDims(),
      "Coo shape error!");

  // indices_ shape is: [M, nse]
  // values_ shape is: (nse,) + shape[M : M + K]
  // sparse_dim_ >= 1
  // dense_dim_ >= 0
  sparse_dim_ = indices_.shape()[0];

  for (int64_t i = sparse_dim_, j = 1; i < shape_.NDims(); ++i, ++j) {
    ARGUMENT_CHECK(shape_[i] == values_.shape()[j], "Coo shape error!");
  }

  dense_dim_ = values_.shape().NDims() - 1;
}

int64_t CooTensorImpl::sparse_dim() const {
  return sparse_dim_;
}

int64_t CooTensorImpl::dense_dim() const {
  return dense_dim_;
}

int64_t CooTensorImpl::nnz() const {
  return values_.shape()[0];
}

const Tensor& CooTensorImpl::indices() const {
  return indices_;
}

const Tensor& CooTensorImpl::values() const {
  return values_;
}

ElementType CooTensorImpl::element_type() const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

kraken::Device* CooTensorImpl::Device() const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

int64_t CooTensorImpl::Size() const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

int64_t CooTensorImpl::NumBytes() const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

void* CooTensorImpl::Ptr() const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

bool CooTensorImpl::IsEmpty() const {
  return indices_.IsEmpty();
}

std::shared_ptr<TensorImpl> CooTensorImpl::Add(const TensorImpl& other) const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::Sub(const TensorImpl& other) const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::Mul(const TensorImpl& other) const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::Div(const TensorImpl& other) const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::AddAssign(const TensorImpl& other) {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}
std::shared_ptr<TensorImpl> CooTensorImpl::SubAssign(const TensorImpl& other) {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}
std::shared_ptr<TensorImpl> CooTensorImpl::MulAssign(const TensorImpl& other) {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}
std::shared_ptr<TensorImpl> CooTensorImpl::DivAssign(const TensorImpl& other) {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::Add(float v) const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}
std::shared_ptr<TensorImpl> CooTensorImpl::Sub(float v) const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}
std::shared_ptr<TensorImpl> CooTensorImpl::Mul(float v) const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}
std::shared_ptr<TensorImpl> CooTensorImpl::Div(float v) const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::AddAssign(float v) {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}
std::shared_ptr<TensorImpl> CooTensorImpl::SubAssign(float v) {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}
std::shared_ptr<TensorImpl> CooTensorImpl::MulAssign(float v) {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}
std::shared_ptr<TensorImpl> CooTensorImpl::DivAssign(float v) {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::BeSub(float v) const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}
std::shared_ptr<TensorImpl> CooTensorImpl::BeDiv(float v) const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::Reshape(const Shape& nshape) const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}
std::shared_ptr<TensorImpl> CooTensorImpl::Reshape(
    const std::vector<int64_t>& dims) const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::Zero() {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::Like() const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::Clone() const {
  Tensor indices = indices_.Clone();
  Tensor values = values_.Clone();

  return std::make_shared<CooTensorImpl>(indices, values, shape_);
}

std::shared_ptr<TensorImpl> CooTensorImpl::Constant(float v) {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::Square(bool in_place) {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::Sqrt(bool in_place) {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::Max(const TensorImpl& other) const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::Vector(int64_t idx) const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::ConcatVector(
    const std::vector<std::shared_ptr<TensorImpl>>& vecs) const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::Normal(float mean, float stddev) {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::XavierNormal(float gain) {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::Uniform(float lower, float upper) {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::XavierUniform(float gain) {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::Abs(bool in_place) {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::TopK(int64_t k) const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::Take(
    const TensorImpl& indices) const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::FlatNonZero(float th) const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::NonZero(float th) const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::Transpose(int64_t d0,
                                                     int64_t d1) const {
  RUNTIME_ERROR("CooTensorImpl unsupport.");
}

std::shared_ptr<TensorImpl> CooTensorImpl::ToDense() const {
  auto dense = TensorImpl::Dense(shape_, values_.element_type());

  math::CooToDense(*indices_.impl(), *values_.impl(), *dense);

  return dense;
}

}  // namespace kraken
