#include "common/mem_buffer.h"

#include <cstring>

namespace kraken {

MemBuffer::MemBuffer() : ptr_(nullptr), capacity_(0), offset_(0) {
}

MemBuffer::MemBuffer(size_t expect) : ptr_(nullptr), offset_(0) {
  ptr_ = (char*)MemBuffer::Malloc(expect);
  capacity_ = expect;
}

MemBuffer::MemBuffer(MemBuffer&& other)
    : ptr_(other.ptr_), capacity_(other.capacity_), offset_(other.offset_) {
  other.ptr_ = nullptr;
  other.capacity_ = 0;
  other.offset_ = 0;
}

const MemBuffer& MemBuffer::operator=(MemBuffer&& other) {
  if (ptr_ != nullptr) {
    MemBuffer::Free(ptr_);

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

MemBuffer::~MemBuffer() {
  if (ptr_ != nullptr) {
    MemBuffer::Free(ptr_);
  }

  ptr_ = nullptr;
  capacity_ = 0;
  offset_ = 0;
}

size_t MemBuffer::Growth(size_t new_size) const {
  size_t new_capacity = capacity_ + capacity_ / 2;

  if (new_capacity >= new_size) {
    return new_capacity;
  }

  return new_size;
}

char* MemBuffer::ptr() const {
  return ptr_;
}

size_t MemBuffer::capacity() const {
  return capacity_;
}

size_t MemBuffer::offset() const {
  return offset_;
}

bool MemBuffer::Write(const char* bytes, size_t size) {
  if (ptr_ == nullptr || offset_ + size > capacity_) {
    // increase the buffer.
    size_t new_capacity = Growth(offset_ + size);

    char* new_ptr = (char*)MemBuffer::Malloc(new_capacity);

    // copy old data
    if (offset_ > 0) {
      std::memcpy(new_ptr, ptr_, offset_);
    }

    if (ptr_ != nullptr) {
      MemBuffer::Free(ptr_);
    }

    ptr_ = new_ptr;
    capacity_ = new_capacity;
  }

  std::memcpy(ptr_ + offset_, bytes, size);
  offset_ += size;

  return true;
}

void MemBuffer::TransferForZMQ(ZMQBuffer* z_buf) {
  z_buf->Reset(ptr_, capacity_, offset_, MemBuffer::ZMQFree);

  ptr_ = nullptr;
  capacity_ = 0;
  offset_ = 0;
}

void* MemBuffer::Malloc(size_t size) {
  return std::malloc(size);
}

void MemBuffer::Free(void* ptr) {
  std::free(ptr);
}

void MemBuffer::ZMQFree(void* data, void* /*no use*/) {
  MemBuffer::Free(data);
}

}  // namespace kraken
