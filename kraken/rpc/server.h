#pragma once

#include <zmq.h>

#include <atomic>
#include <cinttypes>
#include <functional>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/error_code.h"
#include "common/exception.h"
#include "common/log.h"
#include "rpc/deserialize.h"
#include "rpc/protocol.h"
#include "rpc/serialize.h"

namespace kraken {

// A RPC server will bind a port and RPC functions to responds client's request.
class Server {
private:
  using FUNC =
      std::function<int32_t(const RequestHeader&, Deserialize&, Serialize*)>;

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

public:
  /**
   * \brief Reguster a RPC function to this server.
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

    auto func = [callback{std::move(callback)}](
                    const RequestHeader& req_header, Deserialize& deserializer,
                    Serialize* serializer) mutable -> int32_t {
      // deserializer has been fetch the RequestHeader
      // serializer is empty, this func should responsible to insert reply
      // header.
      RequestType req;
      ReplyType reply;

      if ((deserializer >> req) == false) {
        return ErrorCode::kDeserializeRequestError;
      }

      int32_t error_code = callback(req, &reply);
      if (error_code != ErrorCode::kSuccess) {
        return error_code;
      }

      ReplyHeader reply_header;
      reply_header.timestamp = req_header.timestamp;
      reply_header.error_code = ErrorCode::kSuccess;

      // serialize repy header just crash.
      ARGUMENT_CHECK((*serializer) << reply_header,
                     "serialize reply header error!");

      if (((*serializer) << reply) == false) {
        return ErrorCode::kSerializeReplyError;
      }

      return ErrorCode::kSuccess;
    };

    funcs_.emplace(type, std::move(func));
  }

  void Start();

  void Stop();
};

}  // namespace kraken
