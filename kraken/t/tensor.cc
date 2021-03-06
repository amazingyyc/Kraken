#include "t/tensor.h"

#include <sstream>

#include "common/exception.h"
#include "t/coo_tensor_impl.h"

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

const Tensor& Tensor::indices() const {
  return impl_->indices();
}

const Tensor& Tensor::values() const {
  return impl_->values();
}

ElementType Tensor::element_type() const {
  return impl_->element_type();
}

kraken::Device* Tensor::Device() const {
  return impl_->Device();
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

bool Tensor::IsEmpty() const {
  return impl_->IsEmpty();
}

std::string Tensor::Str() const {
  std::stringstream ss;
  ss << "[";

  int64_t size = Size();
  if (element_type().Is<float>()) {
    for (int64_t i = 0; i < size; ++i) {
      ss << Data<float>()[i] << ", ";
    }
  } else if (element_type().Is<double>()) {
    for (int64_t i = 0; i < size; ++i) {
      ss << Data<double>()[i] << ", ";
    }
  } else if (element_type().Is<int64_t>()) {
    for (int64_t i = 0; i < size; ++i) {
      ss << Data<int64_t>()[i] << ", ";
    }
  } else if (element_type().Is<uint64_t>()) {
    for (int64_t i = 0; i < size; ++i) {
      ss << Data<uint64_t>()[i] << ", ";
    }
  } else {
    RUNTIME_ERROR("Type:" << element_type().Name() << " not support str().");
  }

  ss << "]";
  return ss.str();
}

Tensor Tensor::Dense(const std::vector<int64_t>& dims,
                     ElementType element_type) {
  Shape shape(dims);

  return Dense(shape, element_type);
}

Tensor Tensor::Dense(const Shape& shape, ElementType element_type) {
  auto impl = TensorImpl::Dense(shape, element_type);
  return Tensor(impl);
}

Tensor Tensor::Dense(const Shape& shape, std::shared_ptr<Storage> storage,
                     size_t offset, ElementType element_type) {
  auto impl = TensorImpl::Dense(shape, storage, offset, element_type);
  return Tensor(impl);
}

Tensor Tensor::Empty(const Shape& shape, ElementType element_type) {
  return Tensor(TensorImpl::Empty(shape, element_type));
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

float Tensor::operator[](int64_t index) const {
  ARGUMENT_CHECK(element_type().Is<float>(),
                 "operator[] only support float type.");

  int64_t size = Size();
  while (index < 0) {
    index += size;
  }

  ARGUMENT_CHECK(index < size, "Index outof range.");

  float* ptr = Data<float>();

  return ptr[index];
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

Tensor Tensor::Abs(bool in_place) {
  return Tensor(impl_->Abs(in_place));
}

Tensor Tensor::TopK(int64_t k) const {
  return Tensor(impl_->TopK(k));
}

Tensor Tensor::Take(const Tensor& indices) const {
  return Tensor(impl_->Take(*indices.impl()));
}

Tensor Tensor::FlatNonZero(float th) const {
  return Tensor(impl_->FlatNonZero(th));
}

Tensor Tensor::NonZero(float th) const {
  return Tensor(impl_->NonZero(th));
}

Tensor Tensor::Transpose(int64_t d0, int64_t d1) const {
  return Tensor(impl_->Transpose(d0, d1));
}

Tensor Tensor::ToDense() const {
  return Tensor(impl_->ToDense());
}

Tensor Tensor::ToCoo(float th) const {
  return Tensor(impl_->ToCoo(th));
}

Tensor Tensor::LtKeep(float th) const {
  return Tensor(impl_->LtKeep(th));
}

Tensor Tensor::Cast(ElementType to_type) const {
  return Tensor(impl_->Cast(to_type));
}

}  // namespace kraken
