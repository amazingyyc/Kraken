#include "common/tensor_storage.h"

namespace kraken {

TensorStorage::TensorStorage(Device* device, void* ptr, size_t size, bool own)
    : device_(device), ptr_(ptr), size_(size), own_(own) {
}

TensorStorage::~TensorStorage() {
  if (own_) {
    device_->free(ptr_);
  }

  device_ = nullptr;

  ptr_ = nullptr;
  size_ = 0;
  own_ = false;
}

Device* TensorStorage::device() {
  return device_;
}

void* TensorStorage::ptr() {
  return ptr_;
}

size_t TensorStorage::size() {
  return size_;
}

std::shared_ptr<TensorStorage> TensorStorage::create(size_t size) {
  Device* device = Device::default_device();

  void* ptr = device->malloc(size);

  std::shared_ptr<TensorStorage> storage(
      new TensorStorage(device, ptr, size, true));

  return storage;
}

std::shared_ptr<TensorStorage> TensorStorage::create_from(void* ptr,
                                                          size_t size) {
  std::shared_ptr<TensorStorage> storage(
      new TensorStorage(nullptr, ptr, size, false));

  return storage;
}

}  // namespace kraken
