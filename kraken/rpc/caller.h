#pragma once

#include <zmq.h>

#include <atomic>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>

#include "common/error_code.h"
#include "common/exception.h"
#include "common/mutable_buffer.h"
#include "common/snappy.h"
#include "common/thread_barrier.h"
#include "common/zmq_buffer.h"
#include "rpc/deserialize.h"
#include "rpc/protocol.h"
#include "rpc/serialize.h"
#include "snappy.h"

namespace kraken {

// Every caller will create a thread to send request and accept response.
class Caller {
private:
  using CALLBACK_FUNC =
      std::function<void(const ReplyHeader&, const char*, size_t)>;

  struct Message {
    uint64_t timestamp;
    uint32_t type;

    // Becareful z_buf include the RequestHeader.
    ZMQBuffer z_buf;
  };

  // use thread to handle io
  std::thread woker_;

  // zmp content
  void* zmq_context_;

  // use a queue to store message that needed to be send.
  std::mutex msg_que_mu_;
  std::queue<Message> msg_que_;
  zmq_fd_t efd_;

  std::atomic_uint64_t timestamp_;

  std::mutex callback_funcs_mu_;
  std::unordered_map<uint64_t /*timestamp*/, CALLBACK_FUNC> callback_funcs_;

  // server address.
  std::string addr_;

  // compress algorithm
  CompressType compress_type_;

  std::atomic_bool started_;
  std::atomic_bool stop_;

public:
  Caller(const std::string& addr,
         CompressType compress_type = CompressType::kNo);

  ~Caller();

private:
  void HandleMsg(zmq_msg_t& reply);

  void Run(void* zmp_context, const std::string& addr, zmq_fd_t qfd);

  void SendMessage(Message&& msg);

  template <typename ReplyType>
  int32_t NoUnCompress(const char* body, size_t body_len, ReplyType* reply) {
    Deserialize deserialize(body, body_len);
    if ((deserialize >> (*reply)) == false) {
      return ErrorCode::kDeserializeReplyError;
    }

    return ErrorCode::kSuccess;
  }

  template <typename ReplyType>
  int32_t SnappyUnCompress(const char* body, size_t body_len,
                           ReplyType* reply) {
    SnappySource source(body, body_len);
    SnappySink sink;

    if (snappy::Uncompress(&source, &sink) == false) {
      return ErrorCode::kSnappyUncompressError;
    }

    Deserialize deserialize(sink.ptr(), sink.offset());
    if ((deserialize >> (*reply)) == false) {
      return ErrorCode::kDeserializeReplyError;
    }

    return ErrorCode::kSuccess;
  }

public:
  void Start();

  void Stop();

  template <typename ReqType, typename ReplyType>
  int32_t Call(uint32_t type, const ReqType& req, ReplyType* reply) {
    ThreadBarrier barrier(1);
    int32_t ecode;

    auto callback = [&ecode, &barrier, reply](int32_t rcode,
                                              ReplyType& rreply) {
      ecode = rcode;
      *reply = std::move(rreply);

      barrier.Release();
    };

    CallAsync<ReqType, ReplyType>(type, req, std::move(callback));

    barrier.Wait();

    return ecode;
  }

  template <typename ReqType, typename ReplyType>
  void CallAsync(uint32_t type, const ReqType& req,
                 std::function<void(int32_t, ReplyType&)>&& callback) {
    uint64_t timestamp = timestamp_.fetch_add(1);

    ReplyType dummy_reply;

    RequestHeader req_header;
    req_header.timestamp = timestamp;
    req_header.type = type;
    req_header.compress_type = CompressType::kNo;

    // Serialize the header and request.
    MutableBuffer buffer;
    Serialize serialize(&buffer);

    ARGUMENT_CHECK(serialize << req_header, "Serialize request header error!");
    if ((serialize << req) == false) {
      callback(ErrorCode::kSerializeRequestError, dummy_reply);
      return;
    }

    // Wrapper a message and put in queue.
    Message msg;
    msg.timestamp = timestamp;
    msg.type = type;
    buffer.TransferForZMQ(&msg.z_buf);

    auto func = [this, callback{std::move(callback)}](const ReplyHeader& header,
                                                      const char* body,
                                                      size_t body_len) {
      ReplyType reply;

      if (header.error_code != ErrorCode::kSuccess) {
        callback(header.error_code, reply);
        return;
      }

      auto ecode = ErrorCode::kSuccess;
      if (header.compress_type == CompressType::kNo) {
        ecode = this->NoUnCompress<ReplyType>(body, body_len, &reply);
      } else if (header.compress_type == CompressType::kSnappy) {
        ecode = this->SnappyUnCompress<ReplyType>(body, body_len, &reply);
      } else {
        ecode = ErrorCode::kUnSupportCompressTypeError;
      }

      callback(ecode, reply);
    };

    // add to map
    {
      std::unique_lock<std::mutex> lock(callback_funcs_mu_);
      callback_funcs_.emplace(timestamp, func);
    }

    SendMessage(std::move(msg));
  }
};

}  // namespace kraken
