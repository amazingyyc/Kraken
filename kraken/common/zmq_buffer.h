#pragma once

#include <cstdlib>

namespace kraken {

/**
 * \brief This is a special buffer for ZMQ.
 */
class ZMQBuffer {
private:
  // memory pointer.
  void* ptr_;

  // The capacity of this memory.
  size_t capacity_;

  // The usesful memory.
  size_t offset_;

  // A free funciton pointer used for zmq.
  void (*zmq_free_)(void*, void*);

public:
  ZMQBuffer();

  ZMQBuffer(void* ptr, size_t capacity, size_t offset,
            void (*zmq_free)(void*, void*));

  explicit ZMQBuffer(ZMQBuffer&& other);

  const ZMQBuffer& operator=(ZMQBuffer&& other);

  ZMQBuffer(const ZMQBuffer&) = delete;
  ZMQBuffer& operator=(const ZMQBuffer&) = delete;

  ~ZMQBuffer();

public:
  void* ptr() const;

  size_t capacity() const;

  size_t offset() const;

  void Reset(void* ptr, size_t capacity, size_t offset,
             void (*zmq_free)(void*, void*));

  void Transfer(void** ptr, size_t* capacity, size_t* offset,
                void (**zmq_free)(void*, void*));
};

}  // namespace kraken
