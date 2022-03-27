#pragma once

#include <snappy.h>
#include <zmq.h>

#include <atomic>
#include <cinttypes>
#include <functional>
#include <queue>
#include <thread>
#include <unordered_map>

#include "common/compress.h"
#include "common/error_code.h"
#include "common/thread_barrier.h"
#include "common/zmq_buffer.h"
#include "rpc/protocol.h"

namespace kraken {

// Like Station but it's sync.
class SyncStation {
private:
  using FUNC = std::function<int32_t(const RequestHeader&, const char*, size_t,
                                     ZMQBuffer*)>;

  uint32_t port_;

  // zmp content
  void* zmq_context_;
  void* zmq_scoket_;

  // register funcs.
  std::unordered_map<uint32_t, FUNC> funcs_;

public:
  SyncStation(uint32_t port);

  ~SyncStation() = default;

private:
  void Run();

  void HandleError(uint64_t timestamp, int32_t error_code, zmq_msg_t& identity,
                   void* socket);

  void HandleMsg(zmq_msg_t& identity, zmq_msg_t& msg, void* socket);

public:
  // Cannot call this function after start.
  template <typename RequestType, typename ReplyType>
  void RegisterFunc(
      uint32_t type,
      std::function<int32_t(const RequestType&, ReplyType*)>&& callback) {
    auto func = [this, callback{std::move(callback)}](
                    const RequestHeader& req_header, const char* body,
                    size_t body_len, ZMQBuffer* z_buf) -> int32_t {
      RequestType req;
      ReplyType reply;

      if (req_header.compress_type == CompressType::kNo) {
        if (Compress::NoUnCompressDeser<RequestType>(body, body_len, &req) ==
            false) {
          return ErrorCode::kDeserializeRequestError;
        }
      } else if (req_header.compress_type == CompressType::kSnappy) {
        if (Compress::SnappyUnCompressDeser<RequestType>(body, body_len,
                                                         &req) == false) {
          return ErrorCode::kDeserializeRequestError;
        }
      } else {
        return ErrorCode::kUnSupportCompressTypeError;
      }

      int32_t error_code = callback(req, &reply);
      if (error_code != ErrorCode::kSuccess) {
        return error_code;
      }

      ReplyHeader reply_header;
      reply_header.timestamp = req_header.timestamp;
      reply_header.error_code = ErrorCode::kSuccess;
      reply_header.compress_type = req_header.compress_type;

      if (reply_header.compress_type == CompressType::kNo) {
        if (Compress::NoCompressSeria<ReplyType>(reply_header, reply, z_buf) ==
            false) {
          return ErrorCode::kSerializeReplyError;
        }
      } else if (reply_header.compress_type == CompressType::kSnappy) {
        if (Compress::SnappyCompressSeria<ReplyType>(reply_header, reply,
                                                     z_buf) == false) {
          return ErrorCode::kSerializeReplyError;
        }
      } else {
        return ErrorCode::kUnSupportCompressTypeError;
      }

      return ErrorCode::kSuccess;
    };

    funcs_.emplace(type, std::move(func));
  }

  void Start();
};

}  // namespace kraken
