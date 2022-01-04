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
#include "common/thread_barrier.h"
#include "rpc/deserialize.h"
#include "rpc/protocol.h"
#include "rpc/serialize.h"

namespace kraken {

// Every caller will create a thread to send request and accept response.
class Caller {
private:
  using CALLBACK_FUNC =
      std::function<void(const ReplyHeader&, Deserialize& deserializer)>;

  // use thread to handle io
  std::thread woker_;

  // zmp content
  void* zmq_context_;

  // use a queue to store message that needed to be send.
  std::mutex buf_que_mu_;
  std::queue<MutableBuffer> buf_que_;
  zmq_fd_t efd_;

  std::atomic_uint64_t timestamp_;

  std::mutex callback_funcs_mu_;
  std::unordered_map<uint64_t /*timestamp*/, CALLBACK_FUNC> callback_funcs_;

  // server address.
  std::string addr_;

  std::atomic_bool started_;
  std::atomic_bool stop_;

public:
  Caller(const std::string& addr);

  ~Caller();

private:
  void HandleMsg(zmq_msg_t& reply);

  void Run(void* zmp_context, const std::string& addr, zmq_fd_t qfd);

  void SendBuffer(MutableBuffer&& buffer);

public:
  void Start();

  void Stop();

  template <typename ReqType, typename RspType>
  int32_t Call(uint32_t type, const ReqType& req, RspType* rsp) {
    ThreadBarrier barrier(1);

    int32_t ecode;

    auto callback = [&ecode, &barrier, rsp](int32_t rcode, RspType& rrsp) {
      ecode = rcode;
      *rsp = std::move(rrsp);

      barrier.Release();
    };

    CallAsync<ReqType, RspType>(type, req, std::move(callback));

    barrier.Wait();

    return ecode;
  }

  template <typename ReqType, typename RspType>
  void CallAsync(uint32_t type, const ReqType& req,
                 std::function<void(int32_t, RspType&)>&& callback) {
    uint64_t timestamp = timestamp_.fetch_add(1);

    RspType fake_rsp;

    RequestHeader req_header;
    req_header.timestamp = timestamp;
    req_header.type = type;

    MutableBuffer buffer;
    Serialize serialize(&buffer);

    ARGUMENT_CHECK(serialize << req_header, "serialize request header error!");
    if ((serialize << req) == false) {
      callback(ErrorCode::kSerializeRequestError, fake_rsp);
      return;
    }

    // create callback
    auto func = [callback{std::move(callback)}](
                    const ReplyHeader& header,
                    Deserialize& deserializer) mutable {
      RspType rsp;

      if (header.error_code != ErrorCode::kSuccess) {
        callback(header.error_code, rsp);
        return;
      }

      // deserialize.
      if ((deserializer >> rsp) == false) {
        callback(ErrorCode::kDeserializeReplyError, rsp);
        return;
      }

      callback(ErrorCode::kSuccess, rsp);
    };

    // add to map
    {
      std::unique_lock<std::mutex> lock(callback_funcs_mu_);
      callback_funcs_.emplace(timestamp, func);
    }

    // send to server.
    SendBuffer(std::move(buffer));
  }
};

}  // namespace kraken
