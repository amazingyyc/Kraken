#include "common/mutable_buffer.h"

#include <cstring>

namespace kraken {

MutableBuffer::MutableBuffer() : ptr_(nullptr), length_(0), offset_(0) {
}

MutableBuffer::MutableBuffer(size_t expect) : ptr_(nullptr), offset_(0) {
  ptr_ = (char*)MutableBuffer::Malloc(expect);
  length_ = expect;
}

MutableBuffer::MutableBuffer(MutableBuffer&& other)
    : ptr_(other.ptr_), length_(other.length_), offset_(other.offset_) {
  other.ptr_ = nullptr;
  other.length_ = 0;
  other.offset_ = 0;
}

const MutableBuffer& MutableBuffer::operator=(MutableBuffer&& other) {
  if (ptr_ != nullptr) {
    MutableBuffer::Free(ptr_);

    ptr_ = nullptr;
    length_ = 0;
    offset_ = 0;
  }

  ptr_ = other.ptr_;
  length_ = other.length_;
  offset_ = other.offset_;

  other.ptr_ = nullptr;
  other.length_ = 0;
  other.offset_ = 0;

  return *this;
}

MutableBuffer::~MutableBuffer() {
  if (ptr_ != nullptr) {
    MutableBuffer::Free(ptr_);
  }

  ptr_ = nullptr;
  length_ = 0;
  offset_ = 0;
}

size_t MutableBuffer::Growth(size_t new_size) const {
  size_t new_length = length_ + length_ / 2;

  if (new_length >= new_size) {
    return new_length;
  }

  return new_size;
}

size_t MutableBuffer::Length() const {
  return length_;
}

size_t MutableBuffer::Offset() const {
  return offset_;
}

void MutableBuffer::Append(const char* data, size_t size) {
  if (ptr_ == nullptr || offset_ + size > length_) {
    // increase the buffer.
    size_t new_length = Growth(offset_ + size);

    char* new_ptr = (char*)MutableBuffer::Malloc(new_length);

    // copy old data
    if (offset_ > 0) {
      memcpy(new_ptr, ptr_, offset_);
    }

    if (ptr_ != nullptr) {
      MutableBuffer::Free(ptr_);
    }

    ptr_ = new_ptr;
    length_ = new_length;
  }

  memcpy(ptr_ + offset_, data, size);

  offset_ += size;
}

void MutableBuffer::Transfer(char** ptr) {
  *ptr = ptr_;

  ptr_ = nullptr;
  length_ = 0;
  offset_ = 0;
}

void MutableBuffer::Transfer(char** ptr, size_t* length, size_t* offset) {
  *ptr = ptr_;
  *length = length_;
  *offset = offset_;

  ptr_ = nullptr;
  length_ = 0;
  offset_ = 0;
}

void* MutableBuffer::Malloc(size_t size) {
  return malloc(size);
}

void MutableBuffer::Free(void* ptr) {
  free(ptr);
}

void MutableBuffer::ZMQFree(void* data, void* /*no use*/) {
  MutableBuffer::Free(data);
}

}  // namespace kraken
