#pragma once

#include <cstddef>

namespace kraken {

class IWriter {
public:
  virtual bool Write(const char* ptr, size_t size) = 0;
};

}  // namespace kraken
