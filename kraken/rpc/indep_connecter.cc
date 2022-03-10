#include "rpc/indep_connecter.h"

#include <errno.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include "common/exception.h"
#include "common/log.h"

namespace kraken {

IndepConnecter::IndepConnecter(const std::string& addr,
                               CompressType compress_type)
    : addr_(addr),
      compress_type_(compress_type),
      started_(false),
      stop_(false),
      timestamp_(0) {
}

IndepConnecter::~IndepConnecter() {
}

void IndepConnecter::HandleReply(zmq_msg_t& reply) {
  size_t reply_size = zmq_msg_size(&reply);
  const char* reply_data = (const char*)zmq_msg_data(&reply);

  MemReader reader(reply_data, reply_size);
  Deserialize deserializer(&reader);

  ReplyHeader reply_header;
  ARGUMENT_CHECK(deserializer >> reply_header,
                 "Deserialize ReplyHeader error!");

  // find callback by timestamp.
  auto it = z_callbacks_.find(reply_header.timestamp);
  if (it != z_callbacks_.end()) {
    it->second(reply_header, reply_data + sizeof(reply_header),
               reply_size - sizeof(reply_header));

    z_callbacks_.erase(it);
  }
}

void IndepConnecter::Run() {
  // create zmq socket.
  void* zsocket = zmq_socket(zmq_context_, ZMQ_DEALER);
  ARGUMENT_CHECK(zsocket != nullptr, "zmq_socket return nullptr, error:"
                                         << zmq_strerror(zmq_errno()));

  std::string tcp_addr = "tcp://" + addr_;

  // connect server.
  ZMQ_CALL(zmq_connect(zsocket, tcp_addr.c_str()));

  zmq_pollitem_t items[2];

  // 0 to accept socket response.
  items[0].socket = zsocket;
  items[0].fd = 0;
  items[0].events = ZMQ_POLLIN;
  items[0].revents = 0;

  // 1 to check msg queue send to server.
  items[1].socket = nullptr;
  items[1].fd = efd_;
  items[1].events = ZMQ_POLLIN;
  items[1].revents = 0;

  // consume efd.
  uint64_t u;
  long wait_timeout = -1;

  started_.store(true);

  while (stop_.load() == false) {
    ZMQ_CALL(zmq_poll(items, 2, wait_timeout));

    // zmq socket get message.
    if (items[0].revents & ZMQ_POLLIN) {
      zmq_msg_t msg;

      ZMQ_CALL(zmq_msg_init(&msg));
      ZMQ_CALL(zmq_msg_recv(&msg, zsocket, 0));

      HandleReply(msg);

      ZMQ_CALL(zmq_msg_close(&msg));
    }

    // Send message.
    if (items[1].revents & ZMQ_POLLIN) {
      ARGUMENT_CHECK(read(efd_, &u, sizeof(uint64_t)) == sizeof(uint64_t),
                     "read eventfd error.");

      for (;;) {
        Task task;
        {
          std::unique_lock<std::mutex> lock(task_que_mu_);
          if (task_que_.empty()) {
            break;
          }

          task = std::move(task_que_.front());
          task_que_.pop();
        }

        void* ptr;
        size_t capacity;
        size_t offset;
        void (*zmq_free)(void*, void*);

        if (compress_type_ == CompressType::kNo) {
          // No need to compress the data.
          task.z_buf.Transfer(&ptr, &capacity, &offset, &zmq_free);
        } else if (compress_type_ == CompressType::kSnappy) {
          // Compress the data, becareful the raw data already include the
          // RequestHeader but we only compress the body.
          RequestHeader req_header;
          req_header.timestamp = task.timestamp;
          req_header.type = task.rpc_type;
          req_header.compress_type = CompressType::kSnappy;

          SnappySink sink;
          // step1 serialize header.
          {
            Serialize serialize(&sink);
            ARGUMENT_CHECK(serialize << req_header,
                           "Serialize request header error!");
          }

          // Compress body.
          {
            SnappySource source((char*)task.z_buf.ptr() + sizeof(req_header),
                                task.z_buf.offset() - sizeof(req_header));

            ARGUMENT_CHECK(snappy::Compress(&source, &sink) > 0,
                           "snappy::Compress error.");
          }

          sink.TransferForZMQ(&ptr, &capacity, &offset, &zmq_free);
        }

        // send by socket.
        zmq_msg_t zmq_msg;

        // zero copy.
        ZMQ_CALL(zmq_msg_init_data(&zmq_msg, ptr, offset, zmq_free, nullptr));
        ZMQ_CALL(zmq_msg_send(&zmq_msg, zsocket, 0));
        ZMQ_CALL(zmq_msg_close(&zmq_msg));

        // put callback into map.
        if (task.z_callback) {
          z_callbacks_.emplace(task.timestamp, std::move(task.z_callback));

          if (task.timeout_ms > 0) {
            // Set a timeout event.
            TimerEvent event;
            event.timestamp = task.timestamp;
            event.when = std::chrono::steady_clock::now() +
                         std::chrono::milliseconds(task.timeout_ms);

            timers_.push(event);
          }
        }
      }
    }

    // Handle timeout event.
    wait_timeout = -1;
    while (timers_.empty() == false) {
      auto now = std::chrono::steady_clock::now();
      if (now < timers_.top().when) {
        wait_timeout = std::chrono::duration_cast<std::chrono::milliseconds>(
                           timers_.top().when - now)
                           .count();
        break;
      }

      TimerEvent event = timers_.top();
      timers_.pop();

      auto it = z_callbacks_.find(event.timestamp);
      if (it != z_callbacks_.end()) {
        ReplyHeader timeout_header;
        timeout_header.compress_type = CompressType::kNo;
        timeout_header.error_code = ErrorCode::kTimeoutError;
        timeout_header.timestamp = event.timestamp;

        it->second(timeout_header, nullptr, 0);

        z_callbacks_.erase(it);
      }
    }
  }

  ZMQ_CALL(zmq_close(zsocket));
}

void IndepConnecter::EnqueTask(Task&& task) {
  {
    std::unique_lock<std::mutex> lock(task_que_mu_);
    task_que_.emplace(std::move(task));
  }

  // tell worker to send message.
  uint64_t u = 1;
  ARGUMENT_CHECK(
      write(efd_, &u, sizeof(uint64_t)) == sizeof(uint64_t),
      "write eventfd errno:" << errno << ", msg:" << strerror(errno));
}

void IndepConnecter::Start() {
  zmq_context_ = zmq_init(1);
  ARGUMENT_CHECK(zmq_context_ != nullptr, "zmq_init return nullptr, error:"
                                              << zmq_strerror(zmq_errno()));

  // create eventfd.
  efd_ = eventfd(0, EFD_SEMAPHORE);
  ARGUMENT_CHECK(efd_ != -1, "eventfd error:" << efd_);

  // start worker thread.
  worker_ = std::thread(&IndepConnecter::Run, this);

  while (started_.load() == false) {
  }
}

void IndepConnecter::Stop() {
  stop_.store(true);

  // tell thread to stop.
  uint64_t u = 1;
  ARGUMENT_CHECK(
      write(efd_, &u, sizeof(uint64_t)) == sizeof(uint64_t),
      "write eventfd errno:" << errno << ", msg:" << strerror(errno));

  if (worker_.joinable()) {
    worker_.join();
  }

  // close eventfd
  close(efd_);

  // close zmq context.
  ZMQ_CALL(zmq_term(zmq_context_));
  zmq_context_ = nullptr;
}

}  // namespace kraken
