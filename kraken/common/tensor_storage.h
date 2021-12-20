#pragma once

#include <memory>

#include "common/device.h"

namespace kraken {

class TensorStorage {
private:
  // the device that hold the memory
  Device* device_;

  // memory pointer
  void* ptr_;

  // the memory size
  size_t size_;

  // whether malloc by device_
  bool own_;

private:
  explicit TensorStorage(Device* device, void* ptr, size_t size, bool own);

public:
  ~TensorStorage();

  Device* device();

  void* ptr();

  size_t size();

public:
  static std::shared_ptr<TensorStorage> create(size_t size);
  static std::shared_ptr<TensorStorage> create_from(void* ptr, size_t size);
};

}  // namespace kraken
