#include "rpc/station.h"

#include <errno.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include <cstring>

#include "common/exception.h"
#include "common/log.h"

namespace kraken {

Station::Station(uint32_t port, uint32_t thread_nums)
    : port_(port), thread_nums_(thread_nums), started_(false), stop_(false) {
}

Station::~Station() {
}

void Station::ZMQReceiveAndSend(void* from, void* to) {
  int64_t more;
  size_t more_len = sizeof(more);
  do {
    zmq_msg_t part;
    ZMQ_CALL(zmq_msg_init(&part));
    ZMQ_CALL(zmq_msg_recv(&part, from, 0));

    // Check whether send more.
    ZMQ_CALL(zmq_getsockopt(from, ZMQ_RCVMORE, &more, &more_len));

    // http://api.zeromq.org/4-1:zmq-msg-send
    ZMQ_CALL(zmq_msg_send(&part, to, more ? ZMQ_SNDMORE : 0));

  } while (more);
}

void Station::HandleError(uint64_t timestamp, int32_t error_code,
                          zmq_msg_t& identity, void* socket) {
  ReplyHeader reply_header;
  reply_header.timestamp = timestamp;
  reply_header.error_code = error_code;
  reply_header.compress_type = CompressType::kNo;

  MemBuffer buf;
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

  // http://api.zeromq.org/4-1:zmq-msg-send
  // ZMQ_CALL(zmq_msg_close(&replyid));
  // ZMQ_CALL(zmq_msg_close(&reply));
}

void Station::HandleMsg(zmq_msg_t& identity, zmq_msg_t& msg, void* socket) {
  size_t req_size = zmq_msg_size(&msg);
  const char* req_data = (const char*)zmq_msg_data(&msg);

  MemReader reader(req_data, req_size);
  Deserialize header_d(&reader);

  RequestHeader req_header;
  ARGUMENT_CHECK(header_d >> req_header, "Deserialize request header error!");

  // Only read not need mutex.
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

  // http://api.zeromq.org/4-1:zmq-msg-send
  // ZMQ_CALL(zmq_msg_close(&replyid));
  // ZMQ_CALL(zmq_msg_close(&reply));
}

void Station::Run(void* zmp_context) {
  void* receiver = zmq_socket(zmp_context, ZMQ_DEALER);
  ARGUMENT_CHECK(receiver != nullptr, "zmq_socket return nullptr, error:"
                                          << zmq_strerror(zmq_errno()));

  ZMQ_CALL(zmq_connect(receiver, "inproc://workers"));

  while (stop_.load() == false) {
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

void Station::Start() {
  auto callback = [this]() {
    void* zmq_context = zmq_ctx_new();
    ARGUMENT_CHECK(zmq_context != nullptr, "zmq_ctx_new return nullptr, error:"
                                               << zmq_strerror(zmq_errno()));

    void* frontend = zmq_socket(zmq_context, ZMQ_ROUTER);
    ARGUMENT_CHECK(frontend != nullptr, "zmq_socket return nullptr, error:"
                                            << zmq_strerror(zmq_errno()));

    // Bind to tcp.
    std::string addr = "tcp://*:" + std::to_string(port_);
    ZMQ_CALL(zmq_bind(frontend, addr.c_str()));

    // create socket for worker. inner process socket.
    void* backend = zmq_socket(zmq_context, ZMQ_DEALER);
    ARGUMENT_CHECK(backend != nullptr, "zmq_socket return nullptr, error:"
                                           << zmq_strerror(zmq_errno()));

    ZMQ_CALL(zmq_bind(backend, "inproc://workers"));

    // Start thread pool.
    for (uint32_t i = 0; i < thread_nums_; ++i) {
      std::thread t(&Station::Run, this, zmq_context);
      workers_.emplace_back(std::move(t));
    }

    zmq_pollitem_t items[] = {{frontend, 0, ZMQ_POLLIN, 0},
                              {backend, 0, ZMQ_POLLIN, 0}};

    started_.store(true);
    LOG_INFO("Station start at port:[" << port_ << "]");

    while (stop_.load() == false) {
      zmq_poll(items, 2, -1);

      if (items[0].revents & ZMQ_POLLIN) {
        // frontend to backend.
        // http://api.zeromq.org/4-0:zmq-msg-recv
        // http://thisthread.blogspot.com/2012/02/zeromq-31-multithreading-reviewed.html
        ZMQReceiveAndSend(frontend, backend);
      }

      if (items[1].revents & ZMQ_POLLIN) {
        // backend to frontend.
        ZMQReceiveAndSend(backend, frontend);
      }
    }

    // Wait worker thread finish.
    for (auto& t : workers_) {
      if (t.joinable()) {
        t.join();
      }
    }

    // close socket and destroy context.
    ZMQ_CALL(zmq_close(backend));
    backend = nullptr;

    ZMQ_CALL(zmq_close(frontend));
    frontend = nullptr;

    ZMQ_CALL(zmq_term(zmq_context));
    zmq_context = nullptr;
  };

  // use separate thread to listen the connection.
  listen_t_ = std::thread(std::move(callback));

  // Wait start finish.
  while (started_.load() == false) {
  }
}

void Station::Wait() {
  listen_t_.join();
}

void Station::Stop() {
  // (TODO) fix.
}

}  // namespace kraken
