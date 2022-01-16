#include "t/tensor.h"

namespace kraken {

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {
}

std::shared_ptr<TensorImpl> Tensor::impl() const {
  return impl_;
}

Layout Tensor::layout() const {
  return impl_->layout();
}

const Shape& Tensor::shape() const {
  return impl_->shape();
}

ElementType Tensor::element_type() const {
  return impl_->element_type();
}

bool Tensor::IsCoo() const {
  return impl_->IsCoo();
}

bool Tensor::IsDense() const {
  return impl_->IsDense();
}

int64_t Tensor::Size() const {
  return impl_->Size();
}

int64_t Tensor::NumBytes() const {
  return impl_->NumBytes();
}

void* Tensor::Ptr() const {
  return impl_->Ptr();
}

Tensor Tensor::Dense(const std::vector<int64_t>& dims, ElementType etype) {
  Shape shape(dims);

  return Dense(shape, etype);
}

Tensor Tensor::Dense(const Shape& shape, ElementType etype) {
  auto impl = TensorImpl::Dense(shape, etype);
  return Tensor(impl);
}

Tensor Tensor::Dense(const Shape& shape, std::shared_ptr<Storage> storage,
                     size_t offset, ElementType etype) {
  auto impl = std::make_shared<TensorImpl>(shape, storage, offset, etype);
  return Tensor(impl);
}

Tensor Tensor::operator+(const Tensor& other) const {
  return Tensor(impl_->Add(*other.impl_));
}

Tensor Tensor::operator-(const Tensor& other) const {
  return Tensor(impl_->Sub(*other.impl_));
}

Tensor Tensor::operator*(const Tensor& other) const {
  return Tensor(impl_->Mul(*other.impl_));
}

Tensor Tensor::operator/(const Tensor& other) const {
  return Tensor(impl_->Div(*other.impl_));
}

Tensor Tensor::operator+=(const Tensor& other) {
  return Tensor(impl_->AddAssign(*other.impl_));
}

Tensor Tensor::operator-=(const Tensor& other) {
  return Tensor(impl_->SubAssign(*other.impl_));
}

Tensor Tensor::operator*=(const Tensor& other) {
  return Tensor(impl_->MulAssign(*other.impl_));
}

Tensor Tensor::operator/=(const Tensor& other) {
  return Tensor(impl_->DivAssign(*other.impl_));
}

Tensor Tensor::operator+(float v) const {
  return Tensor(impl_->Add(v));
}

Tensor Tensor::operator-(float v) const {
  return Tensor(impl_->Sub(v));
}

Tensor Tensor::operator*(float v) const {
  return Tensor(impl_->Mul(v));
}

Tensor Tensor::operator/(float v) const {
  return Tensor(impl_->Div(v));
}

Tensor Tensor::operator+=(float v) {
  return Tensor(impl_->AddAssign(v));
}

Tensor Tensor::operator-=(float v) {
  return Tensor(impl_->SubAssign(v));
}

Tensor Tensor::operator*=(float v) {
  return Tensor(impl_->MulAssign(v));
}

Tensor Tensor::operator/=(float v) {
  return Tensor(impl_->DivAssign(v));
}

Tensor operator+(float v, const Tensor& t) {
  return Tensor(t.impl()->Add(v));
}

Tensor operator-(float v, const Tensor& t) {
  return Tensor(t.impl()->BeSub(v));
}

Tensor operator*(float v, const Tensor& t) {
  return Tensor(t.impl()->Mul(v));
}

Tensor operator/(float v, const Tensor& t) {
  return Tensor(t.impl()->BeDiv(v));
}

Tensor Tensor::Reshape(const Shape& nshape) const {
  return Tensor(impl_->Reshape(nshape));
}

Tensor Tensor::Reshape(const std::vector<int64_t>& dims) const {
  return Tensor(impl_->Reshape(dims));
}

Tensor Tensor::Zero() {
  return Tensor(impl_->Zero());
}

Tensor Tensor::Like() const {
  return Tensor(impl_->Like());
}

Tensor Tensor::Clone() const {
  return Tensor(impl_->Clone());
}

Tensor Tensor::Constant(float v) {
  return Tensor(impl_->Constant(v));
}

Tensor Tensor::Square(bool in_place) {
  return Tensor(impl_->Square(in_place));
}

Tensor Tensor::Sqrt(bool in_place) {
  return Tensor(impl_->Sqrt(in_place));
}

Tensor Tensor::Max(const Tensor& other) const {
  return Tensor(impl_->Max(*other.impl_));
}

Tensor Tensor::Vector(int64_t idx) const {
  return Tensor(impl_->Vector(idx));
}

Tensor Tensor::ConcatVector(const std::vector<Tensor>& vecs) const {
  std::vector<std::shared_ptr<TensorImpl>> v_impls;
  v_impls.reserve(vecs.size());

  for (auto& i : vecs) {
    v_impls.emplace_back(i.impl_);
  }

  return Tensor(impl_->ConcatVector(v_impls));
}

Tensor Tensor::Normal(float mean, float stddev) {
  return Tensor(impl_->Normal(mean, stddev));
}

Tensor Tensor::XavierNormal(float gain) {
  return Tensor(impl_->XavierNormal(gain));
}

Tensor Tensor::Uniform(float lower, float upper) {
  return Tensor(impl_->Uniform(lower, upper));
}

Tensor Tensor::XavierUniform(float gain) {
  return Tensor(impl_->XavierUniform(gain));
}

}  // namespace kraken
