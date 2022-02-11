#pragma once

#include <cstdlib>

#include "common/iwriter.h"
#include "common/zmq_buffer.h"

namespace kraken {

/**
 * \brief A mutable buffer not thread-safe.
 *
 * This buffer use to store binary data and can be increase length automaticlly.
 */
class MemBuffer : public IWriter {
private:
  char* ptr_;

  size_t capacity_;
  size_t offset_;

public:
  MemBuffer();

  explicit MemBuffer(size_t expect);
  explicit MemBuffer(MemBuffer&&);

  const MemBuffer& operator=(MemBuffer&&);

  MemBuffer(const MemBuffer&) = delete;
  MemBuffer& operator=(const MemBuffer&) = delete;

  ~MemBuffer();

private:
  size_t Growth(size_t new_size) const;

public:
  char* ptr() const;

  size_t capacity() const;

  size_t offset() const;

  bool Write(const char* bytes, size_t size) override;

  void TransferForZMQ(ZMQBuffer* z_buf);

public:
  static void* Malloc(size_t);
  static void Free(void*);

  // A special free func for ZMQ.
  static void ZMQFree(void*, void*);
};

}  // namespace kraken
