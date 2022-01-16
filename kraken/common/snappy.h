#pragma once

#include "rpc/ibuffer.h"
#include "snappy-sinksource.h"

namespace kraken {

class SnappySource : public snappy::Source {
private:
  const char* ptr_;
  size_t offset_;
  size_t length_;

public:
  SnappySource(const char* ptr, size_t length);

  ~SnappySource() override;

public:
  size_t Available() const override;

  const char* Peek(size_t* len) override;

  void Skip(size_t n) override;
};

class SnappySink : public snappy::Sink, public IBuffer {
private:
  char* ptr_;
  size_t capacity_;
  size_t offset_;

public:
  SnappySink();

  ~SnappySink() override;

private:
  void Growth(size_t new_size);

public:
  char* ptr() const;

  size_t capacity() const;

  size_t offset() const;

  // Fo IBuffer.
  void Write(const char* bytes, size_t n) override;

  void TransferForZMQ(ZMQBuffer* z_buf) override;

  // For snappy sink.
  void Append(const char* bytes, size_t n) override;

  char* GetAppendBuffer(size_t length, char* scratch) override;

  void AppendAndTakeOwnership(char* bytes, size_t n,
                              void (*deleter)(void*, const char*, size_t),
                              void* deleter_arg) override;

  char* GetAppendBufferVariable(size_t min_size, size_t desired_size_hint,
                                char* scratch, size_t scratch_size,
                                size_t* allocated_size) override;

public:
  static void* Malloc(size_t);
  static void Free(void*);
  static void ZMQFree(void*, void*);
};

}  // namespace kraken
