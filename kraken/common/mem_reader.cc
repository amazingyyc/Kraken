#include "common/mem_reader.h"

#include <cstring>

namespace kraken {

MemReader::MemReader(const char* ptr, size_t length, void (*free)(void*))
    : ptr_(ptr), length_(length), free_(free), offset_(0) {
}

bool MemReader::Read(void* target, size_t size) {
  if (ptr_ == nullptr || offset_ + size > length_) {
    return false;
  }

  memcpy(target, ptr_ + offset_, size);
  offset_ += size;

  return true;
}

}  // namespace kraken
