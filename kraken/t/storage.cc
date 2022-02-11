#include "t/storage.h"

namespace kraken {

Storage::Storage(Device* device, void* ptr, size_t size, bool own)
    : device_(device), ptr_(ptr), size_(size), own_(own) {
}

Storage::~Storage() {
  if (own_) {
    device_->Free(ptr_);
  }

  device_ = nullptr;

  ptr_ = nullptr;
  size_ = 0;
  own_ = false;
}

Device* Storage::device() {
  return device_;
}

void* Storage::ptr() {
  return ptr_;
}

size_t Storage::size() {
  return size_;
}

std::shared_ptr<Storage> Storage::Create(size_t size) {
  Device* device = Device::Shared();

  void* ptr = device->Malloc(size);

  return std::shared_ptr<Storage>(new Storage(device, ptr, size, true));
}

std::shared_ptr<Storage> Storage::From(void* ptr, size_t size) {
  Device* device = Device::Shared();

  return std::shared_ptr<Storage>(new Storage(device, ptr, size, false));
}

}  // namespace kraken
