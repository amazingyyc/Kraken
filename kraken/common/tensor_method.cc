#include "common/exception.h"
#include "common/math.h"
#include "common/tensor.h"

namespace kraken {

Tensor Tensor::create(const std::vector<int64_t>& dims,
                      ElementType element_type) {
  Shape shape(dims);

  auto storage =
      TensorStorage::create(shape.size() * element_type.byte_width());

  return Tensor(storage, 0, shape, element_type);
}

Tensor Tensor::create(const Shape& shape, ElementType element_type) {
  auto storage =
      TensorStorage::create(shape.size() * element_type.byte_width());

  return Tensor(storage, 0, shape, element_type);
}

Tensor Tensor::operator+(const Tensor& other) const {
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "add need two tensor has same element type.")

  Tensor ret = Tensor::create(shape_, element_type_);

  math::add(*this, other, ret);

  return ret;
}

Tensor Tensor::operator-(const Tensor& other) const {
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "sub need two tensor has same element type.")

  Tensor ret = Tensor::create(shape_, element_type_);

  math::sub(*this, other, ret);

  return ret;
}

Tensor Tensor::operator*(const Tensor& other) const {
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "mul need two tensor has same element type.")

  Tensor ret = Tensor::create(shape_, element_type_);

  math::mul(*this, other, ret);

  return ret;
}

Tensor Tensor::operator/(const Tensor& other) const {
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "div need two tensor has same element type.")

  Tensor ret = Tensor::create(shape_, element_type_);

  math::div(*this, other, ret);

  return ret;
}

Tensor Tensor::operator+=(const Tensor& other) {
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "add need two tensor has same element type.")

  math::add(*this, other, *this);

  return *this;
}

Tensor Tensor::operator-=(const Tensor& other) {
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "sub need two tensor has same element type.")

  math::sub(*this, other, *this);

  return *this;
}

Tensor Tensor::operator*=(const Tensor& other) {
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "mul need two tensor has same element type.")

  math::mul(*this, other, *this);

  return *this;
}

Tensor Tensor::operator/=(const Tensor& other) {
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "div need two tensor has same element type.")

  math::div(*this, other, *this);

  return *this;
}

Tensor Tensor::operator+(float v) const {
  Tensor ret = Tensor::create(shape_, element_type_);

  math::add(v, *this, ret);

  return ret;
}

Tensor Tensor::operator-(float v) const {
  Tensor ret = Tensor::create(shape_, element_type_);

  math::sub(*this, v, ret);

  return ret;
}

Tensor Tensor::operator*(float v) const {
  Tensor ret = Tensor::create(shape_, element_type_);

  math::mul(v, *this, ret);

  return ret;
}

Tensor Tensor::operator/(float v) const {
  Tensor ret = Tensor::create(shape_, element_type_);

  math::div(*this, v, ret);

  return ret;
}

Tensor Tensor::operator+=(float v) {
  math::add(v, *this, *this);

  return *this;
}

Tensor Tensor::operator-=(float v) {
  math::sub(*this, v, *this);

  return *this;
}

Tensor Tensor::operator*=(float v) {
  math::mul(v, *this, *this);

  return *this;
}

Tensor Tensor::operator/=(float v) {
  math::div(*this, v, *this);

  return *this;
}

Tensor Tensor::clone() const {
  Tensor ret = Tensor::create(shape_, element_type_);

  storage_->device()->memcpy(ret.ptr(), ptr(), (size_t)num_bytes());

  return ret;
}

Tensor operator+(float v, const Tensor& t) {
  Tensor ret = Tensor::create(t.shape(), t.element_type());

  math::add(v, t, ret);

  return ret;
}

Tensor operator-(float v, const Tensor& t) {
  Tensor ret = Tensor::create(t.shape(), t.element_type());

  math::sub(v, t, ret);

  return ret;
}

Tensor operator*(float v, const Tensor& t) {
  Tensor ret = Tensor::create(t.shape(), t.element_type());

  math::mul(v, t, ret);

  return ret;
}

Tensor operator/(float v, const Tensor& t) {
  Tensor ret = Tensor::create(t.shape(), t.element_type());

  math::div(v, t, ret);

  return ret;
}

}  // namespace kraken
