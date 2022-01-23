#include "common/snappy.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace kraken {

SnappySource::SnappySource(const char* ptr, size_t length)
    : ::snappy::Source(), ptr_(ptr), offset_(0), length_(length) {
}

SnappySource::~SnappySource() {
  ptr_ = nullptr;
  offset_ = 0;
  length_ = 0;
}

size_t SnappySource::Available() const {
  return (length_ - offset_);
}

const char* SnappySource::Peek(size_t* len) {
  *len = (length_ - offset_);
  return ptr_ + offset_;
}

void SnappySource::Skip(size_t n) {
  offset_ += n;
}

SnappySink::SnappySink() : ptr_(nullptr), capacity_(0), offset_(0) {
}

SnappySink::~SnappySink() {
  if (ptr_ != nullptr) {
    SnappySink::Free(ptr_);
  }

  ptr_ = nullptr;
  capacity_ = 0;
  offset_ = 0;
}

void SnappySink::Growth(size_t new_size) {
  size_t new_capacity = capacity_ + capacity_ / 2;
  if (new_capacity < new_size) {
    new_capacity = new_size;
  }

  if (new_capacity <= capacity_) {
    return;
  }

  char* new_ptr = (char*)SnappySink::Malloc(new_capacity);

  if (ptr_ != nullptr) {
    if (offset_ > 0) {
      std::memcpy(new_ptr, ptr_, offset_);
    }

    SnappySink::Free(ptr_);
  }

  ptr_ = new_ptr;
  capacity_ = new_capacity;
}

char* SnappySink::ptr() const {
  return ptr_;
}

size_t SnappySink::capacity() const {
  return capacity_;
}

size_t SnappySink::offset() const {
  return offset_;
}

void SnappySink::Write(const char* bytes, size_t n) {
  Append(bytes, n);
}

void SnappySink::Append(const char* bytes, size_t n) {
  if (ptr_ == nullptr || offset_ + n > capacity_) {
    Growth(offset_ + n);
  }

  if (ptr_ + offset_ != bytes) {
    std::memcpy(ptr_ + offset_, bytes, n);
  }

  offset_ += n;
}

char* SnappySink::GetAppendBuffer(size_t length, char* scratch) {
  if (ptr_ != nullptr && offset_ + length < capacity_) {
    return ptr_ + offset_;
  }

  return scratch;
}

void SnappySink::AppendAndTakeOwnership(char* bytes, size_t n,
                                        void (*deleter)(void*, const char*,
                                                        size_t),
                                        void* deleter_arg) {
  if (ptr_ == nullptr || (ptr_ + offset_) != bytes) {
    if (ptr_ == nullptr || offset_ + n > capacity_) {
      Growth(offset_ + n);
    }

    std::memcpy(ptr_ + offset_, bytes, n);

    (*deleter)(deleter_arg, bytes, n);
  }

  offset_ += n;
}

char* SnappySink::GetAppendBufferVariable(size_t min_size,
                                          size_t desired_size_hint,
                                          char* scratch, size_t scratch_size,
                                          size_t* allocated_size) {
  if (ptr_ == nullptr || offset_ + min_size > capacity_) {
    // No enough capacity for min size.
    size_t need_size = std::max<size_t>(desired_size_hint, min_size);

    Growth(offset_ + need_size);
  }

  *allocated_size = capacity_ - offset_;

  return ptr_ + offset_;
}

void SnappySink::TransferForZMQ(void** ptr, size_t* capacity, size_t* offset,
                                void (**zmq_free)(void*, void*)) {
  *ptr = ptr_;
  *capacity = capacity_;
  *offset = offset_;
  *zmq_free = SnappySink::ZMQFree;

  ptr_ = nullptr;
  capacity_ = 0;
  offset_ = 0;
}

void SnappySink::TransferForZMQ(ZMQBuffer* z_buf) {
  z_buf->Reset(ptr_, capacity_, offset_, SnappySink::ZMQFree);

  ptr_ = nullptr;
  capacity_ = 0;
  offset_ = 0;
}

void* SnappySink::Malloc(size_t size) {
  return std::malloc(size);
}

void SnappySink::Free(void* ptr) {
  std::free(ptr);
}

void SnappySink::ZMQFree(void* data, void* /*not use*/) {
  SnappySink::Free(data);
}

}  // namespace kraken
