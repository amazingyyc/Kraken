#include "rpc/server.h"

#include <cassert>
#include <cstring>

#include "common/log.h"

namespace kraken {

Server::Server(uint32_t port, uint32_t thread_nums)
    : port_(port), thread_nums_(thread_nums), started_(false), stop_(false) {
}

Server::~Server() {
}

void Server::HandleError(uint64_t timestamp, int32_t error_code,
                         zmq_msg_t& identity, void* socket) {
  ReplyHeader reply_header;
  reply_header.timestamp = timestamp;
  reply_header.error_code = error_code;
  reply_header.compress_type = CompressType::kNo;

  MutableBuffer buf;
  Serialize serializer(&buf);

  ARGUMENT_CHECK(serializer << reply_header, "serialize reply header error!");

  // init replyid
  zmq_msg_t replyid;
  ZMQ_CALL(zmq_msg_init(&replyid));
  ZMQ_CALL(zmq_msg_copy(&replyid, &identity));

  ZMQBuffer z_buf;
  buf.TransferForZMQ(&z_buf);

  void* ptr = nullptr;
  size_t capacity = 0;
  size_t offset = 0;
  void (*zmq_free)(void*, void*) = nullptr;

  z_buf.Transfer(&ptr, &capacity, &offset, &zmq_free);

  // init reply
  zmq_msg_t reply;
  ZMQ_CALL(zmq_msg_init_data(&reply, ptr, offset, zmq_free, nullptr));

  ZMQ_CALL(zmq_msg_send(&replyid, socket, ZMQ_SNDMORE));
  ZMQ_CALL(zmq_msg_send(&reply, socket, 0));

  ZMQ_CALL(zmq_msg_close(&replyid));
  ZMQ_CALL(zmq_msg_close(&reply));
}

void Server::HandleMsg(zmq_msg_t& identity, zmq_msg_t& msg, void* socket) {
  size_t req_size = zmq_msg_size(&msg);
  const char* req_data = (const char*)zmq_msg_data(&msg);

  Deserialize header_d(req_data, req_size);

  RequestHeader req_header;
  ARGUMENT_CHECK(header_d >> req_header, "Deserialize request header error!");

  auto it = funcs_.find(req_header.type);
  if (it == funcs_.end()) {
    HandleError(req_header.timestamp, ErrorCode::kUnRegisterFuncError, identity,
                socket);
    return;
  }

  ZMQBuffer z_buf;
  int32_t ecode = it->second(req_header, req_data + sizeof(req_header),
                             req_size - sizeof(req_header), &z_buf);

  if (ecode != ErrorCode::kSuccess) {
    HandleError(req_header.timestamp, ecode, identity, socket);
    return;
  }

  void* ptr = nullptr;
  size_t capacity = 0;
  size_t offset = 0;
  void (*zmq_free)(void*, void*) = nullptr;

  z_buf.Transfer(&ptr, &capacity, &offset, &zmq_free);

  // send reply.
  zmq_msg_t replyid;
  ZMQ_CALL(zmq_msg_init(&replyid));
  ZMQ_CALL(zmq_msg_copy(&replyid, &identity));

  zmq_msg_t reply;
  ZMQ_CALL(zmq_msg_init_data(&reply, ptr, offset, zmq_free, nullptr));

  ZMQ_CALL(zmq_msg_send(&replyid, socket, ZMQ_SNDMORE));
  ZMQ_CALL(zmq_msg_send(&reply, socket, 0));

  ZMQ_CALL(zmq_msg_close(&replyid));
  ZMQ_CALL(zmq_msg_close(&reply));
}

void Server::WorkerRun(void* zmp_context) {
  void* receiver = zmq_socket(zmp_context, ZMQ_DEALER);
  ARGUMENT_CHECK(receiver != nullptr, "zmq_socket return nullptr, error:"
                                          << zmq_strerror(zmq_errno()));

  ZMQ_CALL(zmq_connect(receiver, "inproc://workers"));

  while (!stop_.load()) {
    // For Router and DEALER model the worker will recieve 2 message one is
    // identity another is real msg.
    zmq_msg_t identity;
    zmq_msg_t msg;

    ZMQ_CALL(zmq_msg_init(&identity));
    ZMQ_CALL(zmq_msg_init(&msg));

    ZMQ_CALL(zmq_msg_recv(&identity, receiver, 0));
    ZMQ_CALL(zmq_msg_recv(&msg, receiver, 0));

    HandleMsg(identity, msg, receiver);

    ZMQ_CALL(zmq_msg_close(&identity));
    ZMQ_CALL(zmq_msg_close(&msg));
  }

  ZMQ_CALL(zmq_close(receiver));
}

void Server::Start() {
  zmq_context_ = zmq_init(1);
  ARGUMENT_CHECK(zmq_context_ != nullptr, "zmq_init return nullptr, error:"
                                              << zmq_strerror(zmq_errno()));

  zmq_clients_ = zmq_socket(zmq_context_, ZMQ_ROUTER);
  ARGUMENT_CHECK(zmq_clients_ != nullptr, "zmq_socket return nullptr, error:"
                                              << zmq_strerror(zmq_errno()));

  // bind port for client
  std::string addr = "tcp://*:" + std::to_string(port_);
  ZMQ_CALL(zmq_bind(zmq_clients_, addr.c_str()));

  // create socket for worker. inner process socket.
  zmq_workers_ = zmq_socket(zmq_context_, ZMQ_DEALER);
  ARGUMENT_CHECK(zmq_workers_ != nullptr, "zmq_socket return nullptr, error:"
                                              << zmq_strerror(zmq_errno()));

  ZMQ_CALL(zmq_bind(zmq_workers_, "inproc://workers"));

  // create a thread pool.
  for (uint32_t i = 0; i < thread_nums_; ++i) {
    std::thread t(&Server::WorkerRun, this, zmq_context_);
    workers_.emplace_back(std::move(t));
  }

  // set flag.
  started_.store(true);

  LOG_INFO("Server start at port:" << port_);

  // start
  ZMQ_CALL(zmq_device(ZMQ_QUEUE, zmq_clients_, zmq_workers_));
}

void Server::Stop() {
  stop_.store(true);

  for (auto& t : workers_) {
    if (t.joinable()) {
      t.join();
    }
  }

  // close socket and destroy context.
  ZMQ_CALL(zmq_close(zmq_workers_));
  zmq_workers_ = nullptr;

  ZMQ_CALL(zmq_close(zmq_clients_));
  zmq_clients_ = nullptr;

  ZMQ_CALL(zmq_term(zmq_context_));
  zmq_context_ = nullptr;
}

}  // namespace kraken
