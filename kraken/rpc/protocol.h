#pragma once

#include <cinttypes>

namespace kraken {

struct RequestHeader {
  // every a message has a timestamp is defined by client.
  // the server will send back to client what he received.
  uint64_t timestamp;

  // which func will be call in server.
  uint32_t type;
};

struct ReplyHeader {
  // same as client.
  uint64_t timestamp;

  // 0 means success or will be error code.
  int32_t error_code;
};

}  // namespace kraken
