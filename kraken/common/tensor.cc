#include "common/tensor.h"

#include <sstream>

#include "common/exception.h"

namespace kraken {

Tensor::Tensor(std::shared_ptr<TensorStorage> storage, size_t offset,
               const Shape& shape, ElementType element_type)
    : storage_(storage),
      offset_(offset),
      shape_(shape),
      element_type_(element_type) {
}

size_t Tensor::offset() const {
  return offset_;
}

const Shape& Tensor::shape() const {
  return shape_;
}

ElementType Tensor::element_type() const {
  return element_type_;
}

int64_t Tensor::Size() const {
  return shape_.Size();
}

int64_t Tensor::NumBytes() const {
  return element_type_.ByteWidth() * Size();
}

int64_t Tensor::Dim(int64_t axis) const {
  return shape_[axis];
}

void* Tensor::Ptr() {
  return ((uint8_t*)storage_->ptr()) + offset_;
}

void* Tensor::Ptr() const {
  return ((uint8_t*)storage_->ptr()) + offset_;
}

std::string Tensor::Str() const {
  std::stringstream ss;
  ss << "[";

  int64_t size = shape_.Size();
  if (element_type_.Is<float>()) {
    for (int64_t i = 0; i < size; ++i) {
      ss << Data<float>()[i] << ", ";
    }
  } else if (element_type_.Is<float>()) {
    for (int64_t i = 0; i < size; ++i) {
      ss << Data<double>()[i] << ", ";
    }
  } else {
    RUNTIME_ERROR("Type:" << element_type_.Name() << " not support str().");
  }

  ss << "]";
  return ss.str();
}

}  // namespace kraken
