#include "rpc/caller.h"

#include <errno.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include <cstring>

#include "common/exception.h"
#include "common/log.h"

namespace kraken {

Caller::Caller(const std::string& addr)
    : addr_(addr), started_(false), stop_(false) {
}

Caller::~Caller() {
}

void Caller::HandleMsg(zmq_msg_t& reply) {
  size_t reply_size = zmq_msg_size(&reply);
  const char* reply_data = (const char*)zmq_msg_data(&reply);

  Deserialize deserializer(reply_data, reply_size);

  ReplyHeader reply_header;
  ARGUMENT_CHECK(deserializer >> reply_header,
                 "Deserialize ReplyHeader error!");

  // find callback by timestamp.
  CALLBACK_FUNC callback;
  {
    std::unique_lock<std::mutex> lock(callback_funcs_mu_);

    auto it = callback_funcs_.find(reply_header.timestamp);
    if (it == callback_funcs_.end()) {
      return;
    }

    callback = std::move(it->second);
    callback_funcs_.erase(it);
  }

  callback(reply_header, deserializer);
}

void Caller::Run(void* zmp_context, const std::string& addr, zmq_fd_t efd) {
  // create zmq socket.
  void* zsocket = zmq_socket(zmp_context, ZMQ_DEALER);
  ARGUMENT_CHECK(zsocket != nullptr, "zmq_socket return nullptr, error:"
                                         << zmq_strerror(zmq_errno()));

  // connect server.
  ZMQ_CALL(zmq_connect(zsocket, addr.c_str()));

  zmq_pollitem_t items[2];

  // 0 to accept socket response.
  items[0].socket = zsocket;
  items[0].fd = 0;
  items[0].events = ZMQ_POLLIN;
  items[0].revents = 0;

  // 1 to check msg queue send to server.
  items[1].socket = nullptr;
  items[1].fd = efd;
  items[1].events = ZMQ_POLLIN;
  items[1].revents = 0;

  // consume efd.
  uint64_t u;

  while (!stop_.load()) {
    ZMQ_CALL(zmq_poll(items, 2, -1));

    // zmq socket get message.
    if (items[0].revents & ZMQ_POLLIN) {
      zmq_msg_t msg;

      ZMQ_CALL(zmq_msg_init(&msg));
      ZMQ_CALL(zmq_msg_recv(&msg, zsocket, 0));

      HandleMsg(msg);

      ZMQ_CALL(zmq_msg_close(&msg));
    }

    // send message.
    if (items[1].revents & ZMQ_POLLIN) {
      ARGUMENT_CHECK(read(efd, &u, sizeof(uint64_t)) == sizeof(uint64_t),
                     "read eventfd error.");

      for (;;) {
        // Read buffer from queue.
        MutableBuffer buffer;
        {
          std::unique_lock<std::mutex> lock(buf_que_mu_);

          if (buf_que_.empty()) {
            break;
          }

          buffer = std::move(buf_que_.front());
          buf_que_.pop();
        }

        char* ptr;
        size_t length;
        size_t offset;

        buffer.Transfer(&ptr, &length, &offset);

        // send by socket.
        zmq_msg_t msg;

        // zero copy.
        ZMQ_CALL(zmq_msg_init_data(&msg, ptr, offset, MutableBuffer::ZMQFree,
                                   nullptr));
        ZMQ_CALL(zmq_msg_send(&msg, zsocket, 0));
        ZMQ_CALL(zmq_msg_close(&msg));
      }
    }
  }

  ZMQ_CALL(zmq_close(zsocket));
}

void Caller::SendBuffer(MutableBuffer&& buffer) {
  {
    std::unique_lock<std::mutex> lock(buf_que_mu_);
    buf_que_.emplace(std::move(buffer));
  }

  // tell worker to send message.
  uint64_t u = 1;
  ARGUMENT_CHECK(
      write(efd_, &u, sizeof(uint64_t)) == sizeof(uint64_t),
      "write eventfd errno:" << errno << ", msg:" << strerror(errno));
}

void Caller::Start() {
  zmq_context_ = zmq_init(1);
  ARGUMENT_CHECK(zmq_context_ != nullptr, "zmq_init return nullptr, error:"
                                              << zmq_strerror(zmq_errno()));

  // create eventfd.
  efd_ = eventfd(0, EFD_SEMAPHORE);
  ARGUMENT_CHECK(efd_ != -1, "eventfd error:" << efd_);

  // start worker thread.
  woker_ = std::thread(&Caller::Run, this, zmq_context_, addr_, efd_);

  started_.store(true);

  LOG_INFO("Caller start for addr:" << addr_);
}

void Caller::Stop() {
  stop_.store(true);

  if (woker_.joinable()) {
    woker_.join();
  }

  // close eventfd
  close(efd_);

  // close zmq context.
  ZMQ_CALL(zmq_term(zmq_context_));
  zmq_context_ = nullptr;
}

}  // namespace kraken
