#include "common/snappy.h"

#include <cstdlib>

namespace kraken {

SnappySource::SnappySource(char* ptr, size_t capacity, void (*deleter)(void*))
    : ::snappy::Source(),
      ptr_(ptr),
      offset_(0),
      capacity_(capacity),
      deleter_(deleter) {
}

SnappySource::~SnappySource() {
  if (deleter_ != nullptr) {
    (*deleter_)(ptr_);
  }

  ptr_ = nullptr;
  offset_ = 0;
  capacity_ = 0;
  deleter_ = nullptr;
}

size_t SnappySource::Available() const {
  return (capacity_ - offset_);
}

const char* SnappySource::Peek(size_t* len) {
  *len = (capacity_ - offset_);
  return ptr_ + offset_;
}

void SnappySource::Skip(size_t n) {
  offset_ += n;
}

// SnappySink::SnappySink() : ptr_(nullptr), offset_(0), capacity_(0) {
// }

// SnappySink::~SnappySink() {
//   if (ptr_ != nullptr) {
//     std::free(ptr_);
//   }

//   ptr_ = nullptr;
//   offset_ = 0;
//   capacity_ = 0;
// }

// size_t SnappySink::Growth(size_t new_size) const {
//   size_t new_capacity = capacity_ + capacity_ / 2;

//   if (new_capacity >= new_size) {
//     return new_capacity;
//   }

//   return new_size;
// }

// void SnappySink::Append(const char* bytes, size_t n) {
//   if (ptr_ == nullptr || offset_ + n > capacity_) {
//     size_t new_capacity = Growth(offset_ + n);

//     char* new_ptr = (char*)std::malloc(new_capacity);

//     // copy old data.
//     if (offset_ > 0) {
//       std::memcpy(new_ptr, ptr_, offset_);
//     }

//     if (ptr_ != nullptr) {
//       std::free(ptr_);
//     }

//     ptr_ = new_ptr;
//     capacity_ = new_capacity;
//   }

//   if (ptr_ + offset_ != bytes) {
//     std::memcpy(ptr_ + offset_, bytes, n);
//   }

//   offset_ += n;
// }

// char* SnappySink::GetAppendBuffer(size_t length, char* scratch) {
//   if (offset_ + length < capacity_) {
//     return (char*)(ptr_ + offset_);
//   }

//   return scratch;
// }

// void SnappySink::AppendAndTakeOwnership(char* bytes, size_t n,
//                                         void (*deleter)(void*, const char*,
//                                                         size_t),
//                                         void* deleter_arg) {
// }

// char* SnappySink::GetAppendBufferVariable(size_t min_size,
//                                           size_t desired_size_hint,
//                                           char* scratch, size_t scratch_size,
//                                           size_t* allocated_size) {
//   return scratch;
// }

}  // namespace kraken
