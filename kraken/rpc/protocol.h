#pragma once

#include <cinttypes>

namespace kraken {

enum class CompressType : uint8_t {
  kNo = 0,
  kSnappy = 1,
};

#pragma pack(1)
struct RequestHeader {
  // every a message has a timestamp is defined by client.
  // the server will send back to client what he received.
  uint64_t timestamp;

  // which func will be call in server.
  uint32_t type;

  // compress type.
  CompressType compress_type;
};

static_assert(sizeof(RequestHeader) == 13);
#pragma pack()

#pragma pack(1)
struct ReplyHeader {
  // same as client.
  uint64_t timestamp;

  // 0 means success or will be error code.
  int32_t error_code;

  // compress type.
  CompressType compress_type;
};

static_assert(sizeof(RequestHeader) == 13);
#pragma pack()

}  // namespace kraken
