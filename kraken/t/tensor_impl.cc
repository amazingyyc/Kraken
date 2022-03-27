#include "t/tensor_impl.h"

#include <algorithm>

#include "common/exception.h"
#include "t/coo_tensor_impl.h"
#include "t/math.h"

namespace kraken {

TensorImpl::TensorImpl(Layout layout, const Shape& shape)
    : layout_(layout),
      shape_(shape),
      storage_(nullptr),
      offset_(0),
      element_type_(ElementType::From<UnKnown>()) {
}

TensorImpl::TensorImpl(const Shape& shape, std::shared_ptr<Storage> storage,
                       size_t offset, ElementType etype)
    : layout_(Layout::kStride),
      shape_(shape),
      storage_(storage),
      offset_(offset),
      element_type_(etype) {
}

std::shared_ptr<TensorImpl> TensorImpl::Dense(const Shape& shape,
                                              ElementType etype) {
  auto storage = Storage::Create(shape.Size() * etype.ByteWidth());

  return std::shared_ptr<TensorImpl>(new TensorImpl(shape, storage, 0, etype));
}

std::shared_ptr<TensorImpl> TensorImpl::Dense(const Shape& shape,
                                              std::shared_ptr<Storage> storage,
                                              size_t offset,
                                              ElementType etype) {
  return std::shared_ptr<TensorImpl>(
      new TensorImpl(shape, storage, offset, etype));
}

std::shared_ptr<TensorImpl> TensorImpl::Empty(ElementType etype) {
  // Empty still include a storage, avoid unexpected error.
  auto storage = Storage::Create(0);

  return std::shared_ptr<TensorImpl>(
      new TensorImpl(Shape(), storage, 0, etype));
}

std::shared_ptr<TensorImpl> TensorImpl::Coo(std::shared_ptr<TensorImpl> indices,
                                            std::shared_ptr<TensorImpl> values,
                                            const Shape& shape) {
  ARGUMENT_CHECK(indices->IsDense() && values->IsDense(),
                 "Coo need indices/values is Dense.")

  Tensor t_indices(indices);
  Tensor t_values(values);

  return std::shared_ptr<CooTensorImpl>(
      new CooTensorImpl(t_indices, t_values, shape));
}

std::shared_ptr<TensorImpl> TensorImpl::EmptyCoo(ElementType value_etype,
                                                 const Shape& shape) {
  return std::shared_ptr<CooTensorImpl>(
      new CooTensorImpl(Tensor::Empty(ElementType::From<int64_t>()),
                        Tensor::Empty(value_etype), shape));
}

Layout TensorImpl::layout() const {
  return layout_;
}

bool TensorImpl::IsCoo() const {
  return layout_ == Layout::kCoo;
}

bool TensorImpl::IsDense() const {
  return layout_ == Layout::kStride;
}

const Shape& TensorImpl::shape() const {
  return shape_;
}

const Tensor& TensorImpl::indices() const {
  RUNTIME_ERROR("Not Coo TensorImpl.");
}

const Tensor& TensorImpl::values() const {
  RUNTIME_ERROR("Not Coo TensorImpl.");
}

ElementType TensorImpl::element_type() const {
  return element_type_;
}

kraken::Device* TensorImpl::Device() const {
  return storage_->device();
}

int64_t TensorImpl::Size() const {
  return shape_.Size();
}

int64_t TensorImpl::NumBytes() const {
  return element_type_.ByteWidth() * Size();
}

void* TensorImpl::Ptr() const {
  return ((uint8_t*)storage_->ptr()) + offset_;
}

bool TensorImpl::IsEmpty() const {
  return Size() == 0;
}

std::shared_ptr<TensorImpl> TensorImpl::Add(const TensorImpl& other) const {
  ARGUMENT_CHECK(IsDense() && other.IsDense(), "Add need Dense TensorImpl.");
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "Add need two TensorImpl has same element type.")
  ARGUMENT_CHECK(Size() == other.Size(),
                 "Add need two TensorImpl has same size.")

  auto out = TensorImpl::Dense(shape_, element_type_);

  math::Add(*this, other, *out);

  return out;
}

std::shared_ptr<TensorImpl> TensorImpl::Sub(const TensorImpl& other) const {
  ARGUMENT_CHECK(IsDense() && other.IsDense(), "Sub need Dense TensorImpl.");
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "Sub need two TensorImpl has same element type.")
  ARGUMENT_CHECK(Size() == other.Size(),
                 "Sub need two TensorImpl has same size.")

  auto out = TensorImpl::Dense(shape_, element_type_);

  math::Sub(*this, other, *out);

  return out;
}

std::shared_ptr<TensorImpl> TensorImpl::Mul(const TensorImpl& other) const {
  ARGUMENT_CHECK(IsDense() && other.IsDense(), "Mul need Dense TensorImpl.");
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "Mul need two TensorImpl has same element type.")
  ARGUMENT_CHECK(Size() == other.Size(),
                 "Mul need two TensorImpl has same size.")

  auto out = TensorImpl::Dense(shape_, element_type_);

  math::Mul(*this, other, *out);

  return out;
}

std::shared_ptr<TensorImpl> TensorImpl::Div(const TensorImpl& other) const {
  ARGUMENT_CHECK(IsDense() && other.IsDense(), "Div need Dense TensorImpl.");
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "Div need two TensorImpl has same element type.")
  ARGUMENT_CHECK(Size() == other.Size(),
                 "Div need two TensorImpl has same size.")

  auto out = TensorImpl::Dense(shape_, element_type_);

  math::Div(*this, other, *out);

  return out;
}

std::shared_ptr<TensorImpl> TensorImpl::AddAssign(const TensorImpl& other) {
  ARGUMENT_CHECK(IsDense() && other.IsDense(), "Div need Dense TensorImpl.");
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "Div need two TensorImpl has same element type.")
  ARGUMENT_CHECK(Size() == other.Size(),
                 "Div need two TensorImpl has same size.")

  math::Add(*this, other, *this);

  return shared_from_this();
}

std::shared_ptr<TensorImpl> TensorImpl::SubAssign(const TensorImpl& other) {
  ARGUMENT_CHECK(IsDense() && other.IsDense(), "Div need Dense TensorImpl.");
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "Div need two TensorImpl has same element type.")
  ARGUMENT_CHECK(Size() == other.Size(),
                 "Div need two TensorImpl has same size.")

  math::Sub(*this, other, *this);

  return shared_from_this();
}

std::shared_ptr<TensorImpl> TensorImpl::MulAssign(const TensorImpl& other) {
  ARGUMENT_CHECK(IsDense() && other.IsDense(), "Div need Dense TensorImpl.");
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "Div need two TensorImpl has same element type.")
  ARGUMENT_CHECK(Size() == other.Size(),
                 "Div need two TensorImpl has same size.")

  math::Mul(*this, other, *this);

  return shared_from_this();
}

std::shared_ptr<TensorImpl> TensorImpl::DivAssign(const TensorImpl& other) {
  ARGUMENT_CHECK(IsDense() && other.IsDense(), "Div need Dense TensorImpl.");
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "Div need two TensorImpl has same element type.")
  ARGUMENT_CHECK(Size() == other.Size(),
                 "Div need two TensorImpl has same size.")

  math::Div(*this, other, *this);

  return shared_from_this();
}

std::shared_ptr<TensorImpl> TensorImpl::Add(float v) const {
  ARGUMENT_CHECK(IsDense(), "Add need Dense TensorImpl.");

  auto out = TensorImpl::Dense(shape_, element_type_);
  math::Add(v, *this, *out);

  return out;
}

std::shared_ptr<TensorImpl> TensorImpl::Sub(float v) const {
  ARGUMENT_CHECK(IsDense(), "Sub need Dense TensorImpl.");

  auto out = TensorImpl::Dense(shape_, element_type_);
  math::Sub(*this, v, *out);

  return out;
}

std::shared_ptr<TensorImpl> TensorImpl::Mul(float v) const {
  ARGUMENT_CHECK(IsDense(), "Mul need Dense TensorImpl.");

  auto out = TensorImpl::Dense(shape_, element_type_);
  math::Mul(v, *this, *out);

  return out;
}

std::shared_ptr<TensorImpl> TensorImpl::Div(float v) const {
  ARGUMENT_CHECK(IsDense(), "Div need Dense TensorImpl.");

  auto out = TensorImpl::Dense(shape_, element_type_);
  math::Div(*this, v, *out);

  return out;
}

std::shared_ptr<TensorImpl> TensorImpl::AddAssign(float v) {
  ARGUMENT_CHECK(IsDense(), "AddAssign need Dense TensorImpl.");

  math::Add(v, *this, *this);

  return shared_from_this();
}

std::shared_ptr<TensorImpl> TensorImpl::SubAssign(float v) {
  ARGUMENT_CHECK(IsDense(), "SubAssign need Dense TensorImpl.");

  math::Sub(*this, v, *this);

  return shared_from_this();
}

std::shared_ptr<TensorImpl> TensorImpl::MulAssign(float v) {
  ARGUMENT_CHECK(IsDense(), "MulAssign need Dense TensorImpl.");

  math::Mul(v, *this, *this);

  return shared_from_this();
}

std::shared_ptr<TensorImpl> TensorImpl::DivAssign(float v) {
  ARGUMENT_CHECK(IsDense(), "DivAssign need Dense TensorImpl.");

  math::Div(*this, v, *this);

  return shared_from_this();
}

std::shared_ptr<TensorImpl> TensorImpl::BeSub(float v) const {
  ARGUMENT_CHECK(IsDense(), "BeSub need Dense TensorImpl.");

  auto out = TensorImpl::Dense(shape_, element_type_);
  math::Sub(v, *this, *out);

  return out;
}

std::shared_ptr<TensorImpl> TensorImpl::BeDiv(float v) const {
  ARGUMENT_CHECK(IsDense(), "BeDiv need Dense TensorImpl.");

  auto out = TensorImpl::Dense(shape_, element_type_);
  math::Div(v, *this, *out);

  return out;
}

std::shared_ptr<TensorImpl> TensorImpl::Reshape(const Shape& nshape) const {
  ARGUMENT_CHECK(IsDense(), "Reshape need Dense TensorImpl.");
  ARGUMENT_CHECK(shape_.Size() == nshape.Size(),
                 "Rshape need shape's size same.");

  return TensorImpl::Dense(nshape, storage_, offset_, element_type_);
}

std::shared_ptr<TensorImpl> TensorImpl::Reshape(
    const std::vector<int64_t>& dims) const {
  Shape shape(dims);

  return Reshape(shape);
}

std::shared_ptr<TensorImpl> TensorImpl::Zero() {
  ARGUMENT_CHECK(IsDense(), "Zero need Dense TensorImpl.");
  storage_->device()->Zero(Ptr(), NumBytes());

  return shared_from_this();
}

std::shared_ptr<TensorImpl> TensorImpl::Like() const {
  ARGUMENT_CHECK(IsDense(), "Like need Dense TensorImpl.");

  return TensorImpl::Dense(shape_, element_type_);
}

std::shared_ptr<TensorImpl> TensorImpl::Clone() const {
  ARGUMENT_CHECK(IsDense(), "Clone need Dense TensorImpl.");

  auto out = TensorImpl::Dense(shape_, element_type_);

  storage_->device()->Memcpy(out->Ptr(), Ptr(), (size_t)NumBytes());

  return out;
}

std::shared_ptr<TensorImpl> TensorImpl::Constant(float v) {
  ARGUMENT_CHECK(IsDense(), "Constant need Dense TensorImpl.");

  math::Constant(*this, v);

  return shared_from_this();
}

std::shared_ptr<TensorImpl> TensorImpl::Square(bool in_place) {
  ARGUMENT_CHECK(IsDense(), "Square need Dense TensorImpl.");

  if (in_place) {
    return MulAssign(*this);
  } else {
    return Mul(*this);
  }
}

std::shared_ptr<TensorImpl> TensorImpl::Sqrt(bool in_place) {
  ARGUMENT_CHECK(IsDense(), "Sqrt need Dense TensorImpl.");

  if (in_place) {
    math::Sqrt(*this, *this);

    return shared_from_this();
  } else {
    auto y = TensorImpl::Dense(shape_, element_type_);
    math::Sqrt(*this, *y);

    return y;
  }
}

std::shared_ptr<TensorImpl> TensorImpl::Max(const TensorImpl& other) const {
  ARGUMENT_CHECK(IsDense() && other.IsDense(), "Max need Dense TensorImpl.");
  ARGUMENT_CHECK(element_type_ == other.element_type_,
                 "Max need two TensorImpl has same element type.");

  auto out = Like();
  math::Max(*this, other, *out);

  return out;
}

std::shared_ptr<TensorImpl> TensorImpl::Vector(int64_t idx) const {
  ARGUMENT_CHECK(IsDense(), "Vector need TensorImpl is dense.")
  ARGUMENT_CHECK(shape_.NDims() == 2, "Tensor vector need tensor is a matrix.");

  int64_t row = shape_[0];
  int64_t col = shape_[1];

  while (idx < 0) {
    idx += row;
  }

  ARGUMENT_CHECK(idx < row, "TensorImpl vector out of range!");

  size_t noffset = offset_ + (idx * col) * element_type_.ByteWidth();
  Shape nshape({col});

  return TensorImpl::Dense(nshape, storage_, noffset, element_type_);
}

std::shared_ptr<TensorImpl> TensorImpl::ConcatVector(
    const std::vector<std::shared_ptr<TensorImpl>>& vecs) const {
  ARGUMENT_CHECK(IsDense(), "ConcatVector need Dense TensorImpl.");
  ARGUMENT_CHECK(!vecs.empty(), "ConcatVector input vecs is empty.");

  int64_t row = (int64_t)vecs.size();
  int64_t col = vecs[0]->Size();

  for (const auto& v : vecs) {
    ARGUMENT_CHECK(v->IsDense(), "ConcatVector need Dense TensorImpl.");
    ARGUMENT_CHECK(v->Size() == col && v->shape().NDims() == 1,
                   "concat_vec need inputs is vector.");
  }

  Shape shape({row, col});
  auto out = TensorImpl::Dense(shape, vecs[0]->element_type());

  math::ConcatVector(vecs, *out);

  return out;
}

std::shared_ptr<TensorImpl> TensorImpl::Normal(float mean, float stddev) {
  ARGUMENT_CHECK(IsDense(), "Normal need Dense TensorImpl.");

  math::Normal(*this, mean, stddev);

  return shared_from_this();
}

std::shared_ptr<TensorImpl> TensorImpl::XavierNormal(float gain) {
  ARGUMENT_CHECK(IsDense(), "XavierNormal need Dense TensorImpl.");

  math::XavierNormal(*this, gain);

  return shared_from_this();
}

std::shared_ptr<TensorImpl> TensorImpl::Uniform(float lower, float upper) {
  ARGUMENT_CHECK(IsDense(), "Uniform need Dense TensorImpl.");

  math::Uniform(*this, lower, upper);

  return shared_from_this();
}

std::shared_ptr<TensorImpl> TensorImpl::XavierUniform(float gain) {
  ARGUMENT_CHECK(IsDense(), "XavierUniform need Dense TensorImpl.");

  math::XavierUniform(*this, gain);

  return shared_from_this();
}

std::shared_ptr<TensorImpl> TensorImpl::Abs(bool in_place) {
  if (in_place) {
    math::Abs(*this, *this);

    return shared_from_this();
  } else {
    auto out = Like();
    math::Abs(*this, *out);

    return out;
  }
}

std::shared_ptr<TensorImpl> TensorImpl::TopK(int64_t k) const {
  ARGUMENT_CHECK(k > 0 && k <= Size(), "k outof range.");

  auto out = TensorImpl::Dense(Shape({k}), element_type_);
  math::TopK(*this, *out);

  return out;
}

std::shared_ptr<TensorImpl> TensorImpl::Take(const TensorImpl& indices) const {
  ARGUMENT_CHECK(indices.IsDense(), "Take need indices is Dense.");
  ARGUMENT_CHECK(!indices.IsEmpty(), "Take need indices not empty.");

  auto out = TensorImpl::Dense(indices.shape(), element_type_);
  math::Take(*this, indices, *out);

  return out;
}

std::shared_ptr<TensorImpl> TensorImpl::FlatNonZero(float th) const {
  return math::FlatNonZero(*this, th);
}

std::shared_ptr<TensorImpl> TensorImpl::NonZero(float th) const {
  return math::NonZero(*this, th);
}

std::shared_ptr<TensorImpl> TensorImpl::Transpose(int64_t d0,
                                                  int64_t d1) const {
  while (d0 < 0) {
    d0 += shape_.NDims();
  }

  while (d1 < 0) {
    d1 += shape_.NDims();
  }

  ARGUMENT_CHECK(d0 < shape_.NDims() && d1 < shape_.NDims(),
                 "Transpose d0/d1 outof range.");

  std::vector<int64_t> dims = shape_.dims();
  std::swap(dims[d0], dims[d1]);

  auto out = TensorImpl::Dense(Shape(dims), element_type_);
  math::Transpose(*this, *out, d0, d1);

  return out;
}

std::shared_ptr<TensorImpl> TensorImpl::ToDense() const {
  RUNTIME_ERROR("Dense TensorImpl unsupport ToDense.");
}

std::shared_ptr<TensorImpl> TensorImpl::ToCoo(float th) const {
  ARGUMENT_CHECK(IsDense(), "ToCoo need Dense TensorImpl.");
  ARGUMENT_CHECK(th >= 0, "ToCoo need th >= 0.");

  // Current tensor element count.
  int64_t size = Size();
  Shape shape({size});

  if (IsEmpty()) {
    return TensorImpl::EmptyCoo(element_type_, shape);
  }

  // shape: [nnz]
  auto indices = math::FlatNonZero(*this, th);
  if (indices->IsEmpty()) {
    return TensorImpl::EmptyCoo(element_type_, shape);
  }

  // to [1, nnz]
  int64_t nnz = indices->Size();
  indices = indices->Reshape({1, nnz});

  // shape: [nnz]
  auto values = Take(*indices);
  values = values->Reshape({nnz});

  return TensorImpl::Coo(indices, values, shape);
}

std::shared_ptr<TensorImpl> TensorImpl::LtKeep(float th) const {
  ARGUMENT_CHECK(th >= 0, "LtKeep need th >= 0.");

  auto out = Like();
  math::LtKeep(*this, th, *out);

  return out;
}

std::shared_ptr<TensorImpl> TensorImpl::Cast(ElementType to_type) {
  if (element_type_ == to_type) {
    return shared_from_this();
  }

  auto to = TensorImpl::Dense(shape_, to_type);
  math::Cast(*this, *to);

  return to;
}

}  // namespace kraken
