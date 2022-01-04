#include "common/exception.h"
#include "common/math.h"
#include "common/tensor.h"

namespace kraken {

Tensor Tensor::Create(const std::vector<int64_t>& dims, ElementType etype) {
  Shape shape(dims);

  return Tensor::Create(shape, etype);
}

Tensor Tensor::Create(const Shape& shape, ElementType etype) {
  auto storage = TensorStorage::Create(shape.Size() * etype.ByteWidth());

  return Tensor(storage, 0, shape, etype);
}

Tensor Tensor::Create(std::shared_ptr<TensorStorage> storage, size_t offset,
                      const Shape& shape, ElementType etype) {
  return Tensor(storage, offset, shape, etype);
}

Tensor Tensor::operator+(const Tensor& other) const {
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "add need two tensor has same element type.")

  Tensor ret = Tensor::Create(shape_, element_type_);

  math::Add(*this, other, ret);

  return ret;
}

Tensor Tensor::operator-(const Tensor& other) const {
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "sub need two tensor has same element type.")

  Tensor ret = Tensor::Create(shape_, element_type_);

  math::Sub(*this, other, ret);

  return ret;
}

Tensor Tensor::operator*(const Tensor& other) const {
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "mul need two tensor has same element type.")

  Tensor ret = Tensor::Create(shape_, element_type_);

  math::Mul(*this, other, ret);

  return ret;
}

Tensor Tensor::operator/(const Tensor& other) const {
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "div need two tensor has same element type.")

  Tensor ret = Tensor::Create(shape_, element_type_);

  math::Div(*this, other, ret);

  return ret;
}

Tensor Tensor::operator+=(const Tensor& other) {
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "add need two tensor has same element type.")

  math::Add(*this, other, *this);

  return *this;
}

Tensor Tensor::operator-=(const Tensor& other) {
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "sub need two tensor has same element type.")

  math::Sub(*this, other, *this);

  return *this;
}

Tensor Tensor::operator*=(const Tensor& other) {
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "mul need two tensor has same element type.")

  math::Mul(*this, other, *this);

  return *this;
}

Tensor Tensor::operator/=(const Tensor& other) {
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "div need two tensor has same element type.")

  math::Div(*this, other, *this);

  return *this;
}

Tensor Tensor::operator+(float v) const {
  Tensor ret = Tensor::Create(shape_, element_type_);

  math::Add(v, *this, ret);

  return ret;
}

Tensor Tensor::operator-(float v) const {
  Tensor ret = Tensor::Create(shape_, element_type_);

  math::Sub(*this, v, ret);

  return ret;
}

Tensor Tensor::operator*(float v) const {
  Tensor ret = Tensor::Create(shape_, element_type_);

  math::Mul(v, *this, ret);

  return ret;
}

Tensor Tensor::operator/(float v) const {
  Tensor ret = Tensor::Create(shape_, element_type_);

  math::Div(*this, v, ret);

  return ret;
}

Tensor Tensor::operator+=(float v) {
  math::Add(v, *this, *this);

  return *this;
}

Tensor Tensor::operator-=(float v) {
  math::Sub(*this, v, *this);

  return *this;
}

Tensor Tensor::operator*=(float v) {
  math::Mul(v, *this, *this);

  return *this;
}

Tensor Tensor::operator/=(float v) {
  math::Div(*this, v, *this);

  return *this;
}

Tensor Tensor::Reshape(const Shape& nshape) const {
  ARGUMENT_CHECK(shape_.Size() == nshape.Size(),
                 "Tensor reshape need shape's size same!");

  // Share storage.
  return Tensor(storage_, offset_, nshape, element_type_);
}

Tensor Tensor::Reshape(const std::vector<int64_t>& dims) const {
  Shape shape(dims);

  return Reshape(shape);
}

Tensor Tensor::Zero() {
  storage_->device()->zero(Ptr(), NumBytes());

  return *this;
}

Tensor Tensor::Norm(float lower, float upper) {
  math::Norm(*this, lower, upper);

  return *this;
}

Tensor Tensor::Vector(int64_t idx) const {
  ARGUMENT_CHECK(shape_.NDims() == 2, "Tensor vector need tensor is a matrix.");

  int64_t row = shape_[0];
  int64_t col = shape_[1];

  while (idx < 0) {
    idx += row;
  }

  ARGUMENT_CHECK(idx < row, "Tensor vector out of range!");

  size_t noffset = offset_ + (idx * col) * element_type_.ByteWidth();
  Shape nshape({col});

  return Tensor(storage_, noffset, nshape, element_type_);
}

Tensor Tensor::Like() const {
  return Tensor::Create(shape_, element_type_);
}

Tensor Tensor::Clone() const {
  Tensor ret = Tensor::Create(shape_, element_type_);

  storage_->device()->memcpy(ret.Ptr(), Ptr(), (size_t)NumBytes());

  return ret;
}

Tensor Tensor::ConcatVec(const std::vector<Tensor>& vecs) const {
  ARGUMENT_CHECK(!vecs.empty(), "concat_vec input vecs is empty.");

  int64_t row = (int64_t)vecs.size();
  int64_t col = vecs[0].Size();

  for (const auto& v : vecs) {
    ARGUMENT_CHECK(v.Size() == col && v.shape().NDims() == 1,
                   "concat_vec need inputs is vector.");
  }

  Shape shape({row, col});
  Tensor output = Tensor::Create(shape, vecs[0].element_type());

  math::ConcatVec(vecs, output);

  return output;
}

Tensor Tensor::Square(bool in_place) {
  if (in_place) {
    (*this) *= (*this);
    return (*this);
  }

  return (*this) * (*this);
}

Tensor Tensor::Sqrt(bool in_place) {
  if (in_place) {
    math::Sqrt(*this, *this);
    return *this;
  } else {
    Tensor y = Tensor::Create(shape_, element_type_);
    math::Sqrt(*this, y);
    return y;
  }
}

Tensor Tensor::Max(const Tensor& other) {
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "Max need two tensor has same element type.")

  Tensor ret = Like();

  math::Max(*this, other, ret);

  return ret;
}

Tensor operator+(float v, const Tensor& t) {
  Tensor ret = Tensor::Create(t.shape(), t.element_type());

  math::Add(v, t, ret);

  return ret;
}

Tensor operator-(float v, const Tensor& t) {
  Tensor ret = Tensor::Create(t.shape(), t.element_type());

  math::Sub(v, t, ret);

  return ret;
}

Tensor operator*(float v, const Tensor& t) {
  Tensor ret = Tensor::Create(t.shape(), t.element_type());

  math::Mul(v, t, ret);

  return ret;
}

Tensor operator/(float v, const Tensor& t) {
  Tensor ret = Tensor::Create(t.shape(), t.element_type());

  math::Div(v, t, ret);

  return ret;
}

}  // namespace kraken
