#pragma once

#include <cstdint>
#include <cstring>
#include <memory>

namespace kraken {

/**
 * \brief For now only CPU type.
 */
enum class DeviceType : uint8_t {
  kCPU = 0,
};

class IAllocator {
public:
  virtual ~IAllocator() = default;

  virtual void* Malloc(size_t) = 0;

  virtual void Free(void*) = 0;

  virtual void Zero(void*, size_t) = 0;

  virtual void Memcpy(void*, const void*, size_t) = 0;
};

class CPUAllocator : public IAllocator {
public:
  void* Malloc(size_t) override;

  void Free(void*) override;

  void Zero(void*, size_t) override;

  void Memcpy(void*, const void*, size_t) override;
};

class Device {
private:
  int16_t id_;

  DeviceType type_;

  std::unique_ptr<IAllocator> allocator_;

private:
  Device(int16_t id, DeviceType type);

public:
  ~Device() = default;

  int16_t id();

  DeviceType type();

  void* Malloc(size_t);

  void Free(void*);

  void Zero(void*, size_t);

  void Memcpy(void*, const void*, size_t);

public:
  static Device* Shared();
};

}  // namespace kraken
