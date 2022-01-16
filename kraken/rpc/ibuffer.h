#pragma once

#include <cinttypes>

#include "common/zmq_buffer.h"

namespace kraken {

class IBuffer {
public:
  virtual void Write(const char* ptr, size_t size) = 0;

  virtual void TransferForZMQ(ZMQBuffer* z_buf) = 0;
};

}  // namespace kraken
