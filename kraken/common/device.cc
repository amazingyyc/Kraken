#include "common/device.h"

#include "common/exception.h"

namespace kraken {

void* CPUAllocator::malloc(size_t size) {
  return std::malloc(size);
}

void CPUAllocator::free(void* ptr) {
  std::free(ptr);
}

void CPUAllocator::zero(void* ptr, size_t size) {
  std::memset(ptr, 0, size);
}

void CPUAllocator::memcpy(void* dst, const void* src, size_t n) {
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

void* Device::malloc(size_t size) {
  return allocator_->malloc(size);
}

void Device::free(void* ptr) {
  allocator_->free(ptr);
}

void Device::zero(void* ptr, size_t size) {
  allocator_->zero(ptr, size);
}

void Device::memcpy(void* dst, const void* src, size_t n) {
  allocator_->memcpy(dst, src, n);
}

Device* Device::Shared() {
  static Device device(0, DeviceType::kCPU);

  return &device;
}

}  // namespace kraken
