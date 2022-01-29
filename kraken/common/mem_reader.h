#pragma once

#include <cstdlib>

#include "common/ireader.h"

namespace kraken {

class MemReader : public IReader {
private:
  const char* ptr_;
  size_t length_;
  size_t offset_;

  void (*free_)(void*);

public:
  MemReader(const char* ptr, size_t length, void (*free_)(void*) = nullptr);

  bool Read(void* target, size_t size) override;
};

}  // namespace kraken
