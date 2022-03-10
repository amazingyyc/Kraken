#pragma once

#include <zmq.h>

#include <atomic>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>

#include "common/compress.h"
#include "common/mem_buffer.h"
#include "common/thread_barrier.h"
#include "common/zmq_buffer.h"
#include "rpc/connecter.h"
#include "rpc/protocol.h"

namespace kraken {

class IndepConnecter : public Connecter {
private:
  struct Task {
    uint64_t timestamp;

    uint32_t rpc_type;

    // Becareful z_buf include the RequestHeader.
    ZMQBuffer z_buf;

    ZMQ_CALLBACK z_callback;

    // Timeout milliseconds.
    int64_t timeout_ms;
  };

  // target address.
  std::string addr_;

  // compress algorithm
  CompressType compress_type_;

  std::atomic_bool started_;
  std::atomic_bool stop_;

  std::atomic_uint64_t timestamp_;

  std::mutex task_que_mu_;
  std::queue<Task> task_que_;
  zmq_fd_t efd_;

  std::unordered_map<uint64_t /*timestamp*/, ZMQ_CALLBACK> z_callbacks_;

  std::priority_queue<TimerEvent, std::vector<TimerEvent>, TimerEventGrater>
      timers_;

  std::thread worker_;

  // zmp content
  void* zmq_context_;

public:
  IndepConnecter(const std::string& addr, CompressType compress_type);

  ~IndepConnecter();

private:
  void HandleReply(zmq_msg_t& reply);

  void Run();

  void EnqueTask(Task&& task);

public:
  void Start();

  void Stop();

  template <typename ReqType, typename ReplyType>
  int32_t Call(uint32_t rpc_type, const ReqType& req, ReplyType* reply,
               int64_t timeout_ms = 5000) {
    ThreadBarrier barrier(1);
    int32_t error_code;

    auto callback = [&barrier, &error_code, reply](int32_t rcode,
                                                   ReplyType& rreply) {
      error_code = rcode;
      *reply = std::move(rreply);

      barrier.Release();
    };

    CallAsync<ReqType, ReplyType>(rpc_type, req, std::move(callback),
                                  timeout_ms);

    barrier.Wait();

    return error_code;
  }

  template <typename ReqType, typename ReplyType>
  void CallAsync(uint32_t rpc_type, const ReqType& req,
                 std::function<void(int32_t, ReplyType&)>&& callback,
                 int64_t timeout_ms = 5000 /*default 5s*/) {
    uint64_t timestamp = timestamp_.fetch_add(1);

    ReplyType dummy_reply;

    RequestHeader req_header;
    req_header.timestamp = timestamp;
    req_header.type = rpc_type;
    req_header.compress_type = CompressType::kNo;

    // Serialize the header and request.
    MemBuffer buffer;
    Serialize serialize(&buffer);

    ARGUMENT_CHECK(serialize << req_header, "Serialize request header error!");
    if ((serialize << req) == false) {
      callback(ErrorCode::kSerializeRequestError, dummy_reply);
      return;
    }

    auto z_callback = [this, callback{std::move(callback)}](
                          const ReplyHeader& header, const char* body,
                          size_t body_len) {
      ReplyType reply;

      if (header.error_code != ErrorCode::kSuccess) {
        callback(header.error_code, reply);
        return;
      }

      auto error_code = ErrorCode::kSuccess;
      if (header.compress_type == CompressType::kNo) {
        if (Compress::NoUnCompressDeser<ReplyType>(body, body_len, &reply) ==
            false) {
          error_code = ErrorCode::kDeserializeReplyError;
        }
      } else if (header.compress_type == CompressType::kSnappy) {
        if (Compress::SnappyUnCompressDeser<ReplyType>(body, body_len,
                                                       &reply) == false) {
          error_code = ErrorCode::kDeserializeReplyError;
        }
      } else {
        error_code = ErrorCode::kUnSupportCompressTypeError;
      }

      callback(error_code, reply);
    };

    Task task;
    task.timestamp = timestamp;
    task.rpc_type = rpc_type;
    task.z_callback = std::move(z_callback);
    task.timeout_ms = timeout_ms;

    buffer.TransferForZMQ(&task.z_buf);

    EnqueTask(std::move(task));
  }
};

}  // namespace kraken
