#pragma once

#include <cinttypes>

namespace kraken {

enum ErrorCode : int32_t {
  kSuccess = 0,
  kRequestHeaderError = 1,
  kUnRegisterFuncError = 2,
  kSerializeRequestError = 3,
  kSerializeReplyError = 4,
  kDeserializeRequestError = 5,
  kDeserializeReplyError = 6,
};

#pragma pack(1)
struct RequestHeader {
  // every a message has a timestamp is defined by client.
  // the server will send back to client what he received.
  uint64_t timestamp;

  // which func will be call in server.
  uint32_t type;
};
static_assert(12 == sizeof(RequestHeader));
#pragma pack()

#pragma pack(1)
struct ReplyHeader {
  // same as client.
  uint64_t timestamp;

  // 0 means success or will be error code.
  int32_t error_code;
};
static_assert(12 == sizeof(ReplyHeader));
#pragma pack()

}  // namespace kraken
