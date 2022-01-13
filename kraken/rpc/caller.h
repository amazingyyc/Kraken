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

  // use thread to handle io
  std::thread woker_;

  // zmp content
  void* zmq_context_;

  // use a queue to store message that needed to be send.
  std::mutex buf_que_mu_;
  std::queue<ZMQBuffer> buf_que_;
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

  void SendBuffer(ZMQBuffer&& buffer);

  template <typename ReqType>
  int32_t NoCompress(const RequestHeader& req_header, const ReqType& req,
                     ZMQBuffer* z_buf) {
    MutableBuffer buffer;
    Serialize serialize(&buffer);

    ARGUMENT_CHECK(serialize << req_header, "Serialize request header error!");

    if ((serialize << req) == false) {
      return ErrorCode::kSerializeRequestError;
    }

    // Transfer to ZMQBuffer.
    buffer.TransferForZMQ(z_buf);

    return ErrorCode::kSuccess;
  }

  template <typename ReqType>
  int32_t SnappyCompress(const RequestHeader& req_header, const ReqType& req,
                         ZMQBuffer* z_buf) {
    SnappySink sink;

    // step1 serialize header.
    {
      Serialize serialize(&sink);
      ARGUMENT_CHECK(serialize << req_header,
                     "Serialize request header error!");
    }

    // step2 serialize body to tmp buffer.
    MutableBuffer body_buf;
    {
      Serialize serialize(&body_buf);
      if ((serialize << req) == false) {
        return ErrorCode::kSerializeRequestError;
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
    req_header.compress_type = compress_type_;

    auto ecode = ErrorCode::kSuccess;

    ZMQBuffer z_buf;
    if (compress_type_ == CompressType::kNo) {
      ecode = NoCompress<ReqType>(req_header, req, &z_buf);
    } else if (compress_type_ == CompressType::kSnappy) {
      ecode = SnappyCompress<ReqType>(req_header, req, &z_buf);
    } else {
      ecode = ErrorCode::kUnSupportCompressTypeError;
    }

    if (ecode != ErrorCode::kSuccess) {
      callback(ecode, dummy_reply);
      return;
    }

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

    // send to server.
    SendBuffer(std::move(z_buf));
  }
};

}  // namespace kraken
