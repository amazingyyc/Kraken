#pragma once

#include <zmq.h>

#include <atomic>
#include <cinttypes>
#include <functional>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/deserialize.h"
#include "common/error_code.h"
#include "common/exception.h"
#include "common/log.h"
#include "common/mem_buffer.h"
#include "common/mem_reader.h"
#include "common/serialize.h"
#include "common/snappy.h"
#include "common/zmq_buffer.h"
#include "rpc/protocol.h"
#include "snappy.h"

namespace kraken {

// A RPC server will bind a port and RPC functions to responds client's request.
class Server {
private:
  using FUNC = std::function<int32_t(const RequestHeader&, const char*, size_t,
                                     ZMQBuffer*)>;

  uint32_t port_;

  uint32_t thread_nums_;
  std::vector<std::thread> workers_;

  // whether the server has been started.
  std::atomic_bool started_;

  // stop the server.
  std::atomic_bool stop_;

  // zmp content
  void* zmq_context_;

  // zmq clients.
  void* zmq_clients_;

  // zmq worker.
  void* zmq_workers_;

  // register funcs.
  std::unordered_map<uint32_t, FUNC> funcs_;

public:
  Server(uint32_t port, uint32_t thread_nums);

  ~Server();

private:
  void HandleError(uint64_t timestamp, int32_t error_code, zmq_msg_t& identity,
                   void* socket);

  void HandleMsg(zmq_msg_t& identity, zmq_msg_t& msg, void* socket);

  void WorkerRun(void* zmp_context);

  template <typename RequestType>
  int32_t NoUnCompress(const char* body, size_t body_len, RequestType* req) {
    MemReader reader(body, body_len);
    Deserialize deserialize(&reader);
    if ((deserialize >> (*req)) == false) {
      return ErrorCode::kDeserializeRequestError;
    }

    return ErrorCode::kSuccess;
  }

  template <typename RequestType>
  int32_t SnappyUnCompress(const char* body, size_t body_len,
                           RequestType* req) {
    SnappySource source(body, body_len);
    SnappySink sink;

    if (snappy::Uncompress(&source, &sink) == false) {
      return ErrorCode::kSnappyUncompressError;
    }

    MemReader reader(sink.ptr(), sink.offset());
    Deserialize deserialize(&reader);
    if ((deserialize >> (*req)) == false) {
      return ErrorCode::kDeserializeRequestError;
    }

    return ErrorCode::kSuccess;
  }

  template <typename ReplyType>
  int32_t NoCompress(const ReplyHeader& reply_header, const ReplyType& reply,
                     ZMQBuffer* z_buf) {
    MemBuffer buffer;
    Serialize serialize(&buffer);

    ARGUMENT_CHECK(serialize << reply_header, "Serialize reply header error!");
    if ((serialize << reply) == false) {
      return ErrorCode::kSerializeReplyError;
    }

    // Transfer to ZMQBuffer.
    buffer.TransferForZMQ(z_buf);

    return ErrorCode::kSuccess;
  }

  template <typename ReplyType>
  int32_t SnappyCompress(const ReplyHeader& reply_header,
                         const ReplyType& reply, ZMQBuffer* z_buf) {
    SnappySink sink;

    // step1 serialize header.
    {
      Serialize serialize(&sink);
      ARGUMENT_CHECK(serialize << reply_header,
                     "Serialize reply header error!");
    }

    // step2 serialize body to tmp buffer.
    MemBuffer body_buf;
    {
      Serialize serialize(&body_buf);
      if ((serialize << reply) == false) {
        return ErrorCode::kSerializeReplyError;
      }
    }

    // Step3 compress body data to sink.
    {
      SnappySource source(body_buf.ptr(), body_buf.offset());
      if (snappy::Compress(&source, &sink) == false) {
        return ErrorCode::kSnappyCompressError;
      }
    }

    sink.TransferForZMQ(z_buf);

    return ErrorCode::kSuccess;
  }

public:
  /**
   * \brief Reguster a RPC function to this server. Not thread-safe.
   *
   * Every RPC func has a Request and Response, the server accept the request
   * and return the response.
   *
   * \tparam RequestType The RPC request type.
   * \tparam ReplyType The RPC response type.
   * \param type Which function will be called, every RPC function has q unique id.
   * \param callback The real RPC function that use to handle the request.
   */
  template <typename RequestType, typename ReplyType>
  void RegisterFunc(
      uint32_t type,
      std::function<int32_t(const RequestType&, ReplyType*)>&& callback) {
    // check whether the server has been started.
    ARGUMENT_CHECK(!started_.load(),
                   "The server has been started, must call register_func "
                   "before call start.");

    auto func = [this, callback{std::move(callback)}](
                    const RequestHeader& req_header, const char* body,
                    size_t body_len, ZMQBuffer* z_buf) -> int32_t {
      RequestType req;
      ReplyType reply;

      auto ecode = ErrorCode::kSuccess;

      if (req_header.compress_type == CompressType::kNo) {
        ecode = this->NoUnCompress<RequestType>(body, body_len, &req);
      } else if (req_header.compress_type == CompressType::kSnappy) {
        ecode = this->SnappyUnCompress<RequestType>(body, body_len, &req);
      } else {
        return ErrorCode::kUnSupportCompressTypeError;
      }

      if (ecode != ErrorCode::kSuccess) {
        return ecode;
      }

      ecode = callback(req, &reply);

      if (ecode != ErrorCode::kSuccess) {
        return ecode;
      }

      ReplyHeader reply_header;
      reply_header.timestamp = req_header.timestamp;
      reply_header.error_code = ErrorCode::kSuccess;
      reply_header.compress_type = req_header.compress_type;

      if (reply_header.compress_type == CompressType::kNo) {
        return this->NoCompress<ReplyType>(reply_header, reply, z_buf);
      } else if (reply_header.compress_type == CompressType::kSnappy) {
        return this->SnappyCompress<ReplyType>(reply_header, reply, z_buf);
      } else {
        return ErrorCode::kUnSupportCompressTypeError;
      }
    };

    funcs_.emplace(type, std::move(func));
  }

  void Start();

  void Stop();
};

}  // namespace kraken
