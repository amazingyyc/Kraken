#include "common/mutable_buffer.h"

#include <cstring>

namespace kraken {

MutableBuffer::MutableBuffer() : ptr_(nullptr), capacity_(0), offset_(0) {
}

MutableBuffer::MutableBuffer(size_t expect) : ptr_(nullptr), offset_(0) {
  ptr_ = (char*)MutableBuffer::Malloc(expect);
  capacity_ = expect;
}

MutableBuffer::MutableBuffer(MutableBuffer&& other)
    : ptr_(other.ptr_), capacity_(other.capacity_), offset_(other.offset_) {
  other.ptr_ = nullptr;
  other.capacity_ = 0;
  other.offset_ = 0;
}

const MutableBuffer& MutableBuffer::operator=(MutableBuffer&& other) {
  if (ptr_ != nullptr) {
    MutableBuffer::Free(ptr_);

    ptr_ = nullptr;
    capacity_ = 0;
    offset_ = 0;
  }

  ptr_ = other.ptr_;
  capacity_ = other.capacity_;
  offset_ = other.offset_;

  other.ptr_ = nullptr;
  other.capacity_ = 0;
  other.offset_ = 0;

  return *this;
}

MutableBuffer::~MutableBuffer() {
  if (ptr_ != nullptr) {
    MutableBuffer::Free(ptr_);
  }

  ptr_ = nullptr;
  capacity_ = 0;
  offset_ = 0;
}

size_t MutableBuffer::Growth(size_t new_size) const {
  size_t new_capacity = capacity_ + capacity_ / 2;

  if (new_capacity >= new_size) {
    return new_capacity;
  }

  return new_size;
}

char* MutableBuffer::ptr() const {
  return ptr_;
}

size_t MutableBuffer::capacity() const {
  return capacity_;
}

size_t MutableBuffer::offset() const {
  return offset_;
}

void MutableBuffer::Attach(const char* bytes, size_t size) {
  if (ptr_ == nullptr || offset_ + size > capacity_) {
    // increase the buffer.
    size_t new_capacity = Growth(offset_ + size);

    char* new_ptr = (char*)MutableBuffer::Malloc(new_capacity);

    // copy old data
    if (offset_ > 0) {
      std::memcpy(new_ptr, ptr_, offset_);
    }

    if (ptr_ != nullptr) {
      MutableBuffer::Free(ptr_);
    }

    ptr_ = new_ptr;
    capacity_ = new_capacity;
  }

  std::memcpy(ptr_ + offset_, bytes, size);

  offset_ += size;
}

void MutableBuffer::TransferForZMQ(ZMQBuffer* z_buf) {
  z_buf->Reset(ptr_, capacity_, offset_, MutableBuffer::ZMQFree);

  ptr_ = nullptr;
  capacity_ = 0;
  offset_ = 0;
}

void* MutableBuffer::Malloc(size_t size) {
  return std::malloc(size);
}

void MutableBuffer::Free(void* ptr) {
  std::free(ptr);
}

void MutableBuffer::ZMQFree(void* data, void* /*no use*/) {
  MutableBuffer::Free(data);
}

}  // namespace kraken
