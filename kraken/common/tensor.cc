#include "common/tensor.h"

namespace kraken {

Tensor::Tensor(std::shared_ptr<TensorStorage> storage, size_t offset,
               const Shape& shape, ElementType element_type)
    : storage_(storage),
      offset_(offset),
      shape_(shape),
      element_type_(element_type) {
}

const Shape& Tensor::shape() const {
  return shape_;
}

ElementType Tensor::element_type() const {
  return element_type_;
}

int64_t Tensor::size() const {
  return shape_.size();
}

int64_t Tensor::num_bytes() const {
  return element_type_.byte_width() * size();
}

int64_t Tensor::dim(int64_t axis) const {
  return shape_[axis];
}

void* Tensor::ptr() {
  return ((uint8_t*)storage_->ptr()) + offset_;
}

void* Tensor::ptr() const {
  return ((uint8_t*)storage_->ptr()) + offset_;
}

}  // namespace kraken
