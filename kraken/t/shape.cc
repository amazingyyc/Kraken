#include "t/shape.h"

#include <sstream>

#include "common/exception.h"

namespace kraken {

Shape::Shape(const Shape& other)
    : dims_(other.dims_), strides_(other.strides_) {
}

Shape::Shape(const std::vector<int64_t>& dims) : dims_(dims) {
  for (auto d : dims_) {
    ARGUMENT_CHECK(d >= 0, "dimension need >= 0");
  }

  UpdateStrides();
}

Shape::Shape(std::vector<int64_t>&& dims) : dims_(std::move(dims)) {
  for (auto d : dims_) {
    ARGUMENT_CHECK(d >= 0, "dimension need >= 0");
  }

  UpdateStrides();
}

void Shape::UpdateStrides() {
  int64_t ndims = (int64_t)dims_.size();

  strides_.resize(ndims);

  if (ndims > 0) {
    if (dims_[ndims - 1] == 0) {
      strides_[ndims - 1] = 0;
    } else {
      strides_[ndims - 1] = 1;
    }

    for (int64_t i = ndims - 2; i >= 0; --i) {
      strides_[i] = strides_[i + 1] * dims_[i + 1];
    }
  }
}

const Shape& Shape::operator=(const Shape& other) {
  dims_ = other.dims_;
  strides_ = other.strides_;

  return *this;
}

const Shape& Shape::operator=(Shape&& other) {
  dims_ = std::move(other.dims_);
  strides_ = std::move(other.strides_);

  return *this;
}

bool Shape::operator==(const Shape& other) const {
  if (dims_.size() != other.dims_.size()) {
    return false;
  }

  for (size_t i = 0; i < dims_.size(); ++i) {
    if (dims_[i] != other.dims_[i]) {
      return false;
    }
  }

  return true;
}

bool Shape::operator!=(const Shape& other) const {
  return !((*this) == other);
}

int64_t Shape::operator[](int64_t axis) const {
  return Dim(axis);
}

const std::vector<int64_t>& Shape::dims() const {
  return dims_;
}

const std::vector<int64_t>& Shape::strides() const {
  return strides_;
}

int64_t Shape::NDims() const {
  return (int64_t)dims_.size();
}

int64_t Shape::Size() const {
  if (dims_.empty()) {
    return 0;
  }

  int64_t size = 1;
  for (auto d : dims_) {
    size *= d;
  }

  return size;
}

int64_t Shape::Dim(int64_t axis) const {
  while (axis < 0) {
    axis += NDims();
  }

  ARGUMENT_CHECK(0 <= axis && axis < NDims(),
                 "the axis is out of rang: [0, " << NDims() << "]");

  return dims_[axis];
}

int64_t Shape::Stride(int64_t axis) const {
  while (axis < 0) {
    axis += NDims();
  }

  ARGUMENT_CHECK(0 <= axis && axis < NDims(),
                 "the axis is out of rang: [0, " << NDims() << "]");

  return strides_[axis];
}

std::string Shape::Str() const {
  std::stringstream ss;

  ss << "[";
  for (int64_t i = 0; i < NDims() - 1; ++i) {
    ss << Dim(i) << ",";
  }

  if (NDims() > 0) {
    ss << Dim(-1);
  }
  ss << "]";

  return ss.str();
}

}  // namespace kraken
