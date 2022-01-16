#include "t/device.h"

#include "common/exception.h"

namespace kraken {

void* CPUAllocator::Malloc(size_t size) {
  return std::malloc(size);
}

void CPUAllocator::Free(void* ptr) {
  std::free(ptr);
}

void CPUAllocator::Zero(void* ptr, size_t size) {
  std::memset(ptr, 0, size);
}

void CPUAllocator::Memcpy(void* dst, const void* src, size_t n) {
  std::memcpy(dst, src, n);
}

Device::Device(int16_t id, DeviceType type) : id_(id), type_(type) {
  if (DeviceType::kCPU == type_) {
    allocator_.reset(new CPUAllocator());
  } else {
    RUNTIME_ERROR("the device type:" << (uint8_t)type << " is not support");
  }
}

int16_t Device::id() {
  return id_;
}

DeviceType Device::type() {
  return type_;
}

void* Device::Malloc(size_t size) {
  return allocator_->Malloc(size);
}

void Device::Free(void* ptr) {
  allocator_->Free(ptr);
}

void Device::Zero(void* ptr, size_t size) {
  allocator_->Zero(ptr, size);
}

void Device::Memcpy(void* dst, const void* src, size_t n) {
  allocator_->Memcpy(dst, src, n);
}

Device* Device::Shared() {
  static Device device(0, DeviceType::kCPU);

  return &device;
}

}  // namespace kraken
