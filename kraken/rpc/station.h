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
#include "common/deserialize.h"
#include "common/error_code.h"
#include "common/exception.h"
#include "common/mem_buffer.h"
#include "common/mem_reader.h"
#include "common/serialize.h"
#include "common/snappy.h"
#include "common/thread_barrier.h"
#include "common/zmq_buffer.h"
#include "rpc/protocol.h"

namespace kraken {

class Station {
private:
  using FUNC = std::function<int32_t(const RequestHeader&, const char*, size_t,
                                     ZMQBuffer*)>;

  uint32_t port_;

  uint32_t thread_nums_;
  std::vector<std::thread> workers_;

  // A seperate thread to listen connect.
  std::thread listen_t_;

  std::atomic_bool started_;
  std::atomic_bool stop_;

  // register funcs.
  std::unordered_map<uint32_t, FUNC> funcs_;

public:
  Station(uint32_t port, uint32_t thread_nums);

  ~Station();

private:
  void ZMQReceiveAndSend(void* from, void* to);

  void HandleError(uint64_t timestamp, int32_t error_code, zmq_msg_t& identity,
                   void* socket);

  void HandleMsg(zmq_msg_t& identity, zmq_msg_t& msg, void* socket);

  void Run(void* zmp_context);

public:
  template <typename RequestType, typename ReplyType>
  void RegisterFunc(
      uint32_t type,
      std::function<int32_t(const RequestType&, ReplyType*)>&& callback) {
    // check whether the server has been started.
    ARGUMENT_CHECK(!started_.load(),
                   "The server has been started, must call register_func "
                   "before start.");

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

  void Wait();

  void Stop();
};

}  // namespace kraken
