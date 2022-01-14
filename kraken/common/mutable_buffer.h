#pragma once

#include <cstdlib>

#include "rpc/serialize.h"

namespace kraken {

/**
 * \brief A mutable buffer not thread-safe.
 *
 * This buffer use to store binary data and can be increase length automaticlly.
 */
class MutableBuffer : public IBuffer {
private:
  char* ptr_;

  size_t capacity_;
  size_t offset_;

public:
  MutableBuffer();

  explicit MutableBuffer(size_t expect);
  explicit MutableBuffer(MutableBuffer&&);

  const MutableBuffer& operator=(MutableBuffer&&);

  MutableBuffer(const MutableBuffer&) = delete;
  MutableBuffer& operator=(const MutableBuffer&) = delete;

  ~MutableBuffer();

private:
  size_t Growth(size_t new_size) const;

public:
  char* ptr() const;

  size_t capacity() const;

  size_t offset() const;

  void Write(const char* bytes, size_t size) override;

  void TransferForZMQ(ZMQBuffer* z_buf) override;

public:
  static void* Malloc(size_t);
  static void Free(void*);

  // A special free func for ZMQ.
  static void ZMQFree(void*, void*);
};

}  // namespace kraken
