#include "common/zmq_buffer.h"

#include <cstring>

namespace kraken {

ZMQBuffer::ZMQBuffer()
    : ptr_(nullptr), capacity_(0), offset_(0), zmq_free_(nullptr) {
}

ZMQBuffer::ZMQBuffer(void* ptr, size_t capacity, size_t offset,
                     void (*zmq_free)(void*, void*))
    : ptr_(ptr), capacity_(capacity), offset_(offset), zmq_free_(zmq_free) {
}

ZMQBuffer::ZMQBuffer(ZMQBuffer&& other)
    : ptr_(other.ptr_),
      capacity_(other.capacity_),
      offset_(other.offset_),
      zmq_free_(other.zmq_free_) {
  other.ptr_ = nullptr;
  other.capacity_ = 0;
  other.offset_ = 0;
  other.zmq_free_ = nullptr;
}

const ZMQBuffer& ZMQBuffer::operator=(ZMQBuffer&& other) {
  if (zmq_free_ != nullptr && ptr_ != nullptr) {
    (*zmq_free_)(ptr_, nullptr);
  }

  ptr_ = other.ptr_;
  capacity_ = other.capacity_;
  offset_ = other.offset_;
  zmq_free_ = other.zmq_free_;

  other.ptr_ = nullptr;
  other.capacity_ = 0;
  other.offset_ = 0;
  other.zmq_free_ = nullptr;

  return *this;
}

ZMQBuffer::~ZMQBuffer() {
  if (zmq_free_ != nullptr && ptr_ != nullptr) {
    (*zmq_free_)(ptr_, nullptr);
  }

  ptr_ = nullptr;
  capacity_ = 0;
  offset_ = 0;

  zmq_free_ = nullptr;
}

void* ZMQBuffer::ptr() const {
  return ptr_;
}

size_t ZMQBuffer::capacity() const {
  return capacity_;
}

size_t ZMQBuffer::offset() const {
  return offset_;
}

void ZMQBuffer::Reset(void* ptr, size_t capacity, size_t offset,
                      void (*zmq_free)(void*, void*)) {
  if (zmq_free_ != nullptr && ptr_ != nullptr) {
    (*zmq_free_)(ptr_, nullptr);
  }

  ptr_ = ptr;
  capacity_ = capacity;
  offset_ = offset;
  zmq_free_ = zmq_free;
}

void ZMQBuffer::Transfer(void** ptr, size_t* capacity, size_t* offset,
                         void (**zmq_free)(void*, void*)) {
  *ptr = ptr_;
  *capacity = capacity_;
  *offset = offset_;
  *zmq_free = zmq_free_;

  ptr_ = nullptr;
  capacity_ = 0;
  offset_ = 0;
  zmq_free_ = nullptr;
}

}  // namespace kraken
