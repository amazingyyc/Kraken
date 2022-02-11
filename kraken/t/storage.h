#pragma once

#include <memory>

#include "t/device.h"

namespace kraken {

class Storage {
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
  Storage(Device* device, void* ptr, size_t size, bool own);

  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  Storage(Storage&&) = delete;
  Storage& operator=(Storage&&) = delete;

public:
  ~Storage();

  Device* device();

  void* ptr();

  size_t size();

public:
  static std::shared_ptr<Storage> Create(size_t size);
  static std::shared_ptr<Storage> From(void* ptr, size_t size);
};

}  // namespace kraken
