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

  virtual void* malloc(size_t) = 0;

  virtual void free(void*) = 0;

  virtual void zero(void*, size_t) = 0;

  virtual void memcpy(void*, const void*, size_t) = 0;
};

class CPUAllocator : public IAllocator {
public:
  void* malloc(size_t) override;

  void free(void*) override;

  void zero(void*, size_t) override;

  void memcpy(void*, const void*, size_t) override;
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

  void* malloc(size_t);

  void free(void*);

  void zero(void*, size_t);

  void memcpy(void*, const void*, size_t);

public:
  static Device* Shared();
};

}  // namespace kraken
