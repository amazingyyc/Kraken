#include "rpc/combine_connecter.h"

#include <errno.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include "common/deserialize.h"
#include "common/error_code.h"
#include "common/exception.h"
#include "common/log.h"
#include "common/serialize.h"
#include "common/snappy.h"
#include "snappy.h"

namespace kraken {

CombineConnecter::CombineConnecter(CompressType compress_type)
    : compress_type_(compress_type),
      started_(false),
      stop_(false),
      timestamp_(0) {
}

CombineConnecter::~CombineConnecter() {
}

bool CombineConnecter::AddConnect(uint64_t id, const std::string& addr,
                                  zmq_pollitem_t** zmq_polls,
                                  int* zmq_polls_size) {
  if (sender_idx_.find(id) != sender_idx_.end()) {
    return false;
  }

  void* sender = zmq_socket(zmq_context_, ZMQ_DEALER);
  ARGUMENT_CHECK(sender != nullptr, "zmq_socket return nullptr, error:"
                                        << zmq_strerror(zmq_errno()));

  std::string tcp_addr = "tcp://" + addr;

  // connect server.
  if (zmq_connect(sender, tcp_addr.c_str()) != 0) {
    return false;
  }

  sender_idx_[id] = senders_.size();
  senders_.emplace_back(sender);

  // free old zmq_pollitem
  free(*zmq_polls);

  *zmq_polls_size = (1 + senders_.size());
  *zmq_polls =
      (zmq_pollitem_t*)malloc((*zmq_polls_size) * sizeof(zmq_pollitem_t));

  (*zmq_polls)[0].socket = nullptr;
  (*zmq_polls)[0].fd = efd_;
  (*zmq_polls)[0].events = ZMQ_POLLIN;
  (*zmq_polls)[0].revents = 0;

  for (int i = 1; i < *zmq_polls_size; ++i) {
    (*zmq_polls)[i].socket = senders_[i - 1];
    (*zmq_polls)[i].fd = 0;
    (*zmq_polls)[i].events = ZMQ_POLLIN;
    (*zmq_polls)[i].revents = 0;
  }

  return true;
}

bool CombineConnecter::RemoveConnect(uint64_t id, zmq_pollitem_t** zmq_polls,
                                     int* zmq_polls_size) {
  if (sender_idx_.find(id) == sender_idx_.end()) {
    return false;
  }

  size_t remove_idx = sender_idx_[id];
  void* remove_sender = senders_[remove_idx];

  if (zmq_close(remove_sender) != 0) {
    return false;
  }

  sender_idx_.erase(id);
  senders_.erase(senders_.begin() + remove_idx);

  // free old zmq_pollitem
  free(*zmq_polls);

  *zmq_polls_size = (1 + senders_.size());
  *zmq_polls =
      (zmq_pollitem_t*)malloc((*zmq_polls_size) * sizeof(zmq_pollitem_t));

  (*zmq_polls)[0].socket = nullptr;
  (*zmq_polls)[0].fd = efd_;
  (*zmq_polls)[0].events = ZMQ_POLLIN;
  (*zmq_polls)[0].revents = 0;

  for (int i = 1; i < *zmq_polls_size; ++i) {
    (*zmq_polls)[i].socket = senders_[i - 1];
    (*zmq_polls)[i].fd = 0;
    (*zmq_polls)[i].events = ZMQ_POLLIN;
    (*zmq_polls)[i].revents = 0;
  }

  return true;
}

int32_t CombineConnecter::SendMsg(uint64_t id, uint64_t timestamp,
                                  uint32_t rpc_type, ZMQBuffer* z_buf) {
  if (sender_idx_.find(id) == sender_idx_.end()) {
    return ErrorCode::kNotExistError;
  }

  void* sender = senders_[sender_idx_[id]];

  void* ptr;
  size_t capacity;
  size_t offset;
  void (*zmq_free)(void*, void*);

  if (compress_type_ == CompressType::kNo) {
    // No need to compress the data.
    z_buf->Transfer(&ptr, &capacity, &offset, &zmq_free);
  } else if (compress_type_ == CompressType::kSnappy) {
    RequestHeader req_header;
    req_header.timestamp = timestamp;
    req_header.type = rpc_type;
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
      SnappySource source((char*)z_buf->ptr() + sizeof(req_header),
                          z_buf->offset() - sizeof(req_header));

      ARGUMENT_CHECK(snappy::Compress(&source, &sink) > 0,
                     "snappy::Compress error.");
    }

    sink.TransferForZMQ(&ptr, &capacity, &offset, &zmq_free);
  }

  // send by socket.
  zmq_msg_t zmq_msg;

  // zero copy.
  ZMQ_CALL(zmq_msg_init_data(&zmq_msg, ptr, offset, zmq_free, nullptr));
  ZMQ_CALL(zmq_msg_send(&zmq_msg, sender, 0));
  ZMQ_CALL(zmq_msg_close(&zmq_msg));

  return ErrorCode::kSuccess;
}

void CombineConnecter::HandleReply(zmq_msg_t& reply) {
  size_t reply_size = zmq_msg_size(&reply);
  const char* reply_data = (const char*)zmq_msg_data(&reply);

  MemReader reader(reply_data, reply_size);
  Deserialize deserializer(&reader);

  ReplyHeader reply_header;
  ARGUMENT_CHECK(deserializer >> reply_header,
                 "Deserialize ReplyHeader error!");

  auto it = z_callbacks_.find(reply_header.timestamp);
  if (it != z_callbacks_.end()) {
    it->second(reply_header, reply_data + sizeof(reply_header),
               reply_size - sizeof(reply_header));

    z_callbacks_.erase(it);
  }
}

void CombineConnecter::Run() {
  // We use zmq_pollitem_t to monitor the message queue and socket.
  // zmq_poll[0] always be the efd.
  int zmq_polls_size = 1;

  zmq_pollitem_t* zmq_polls =
      (zmq_pollitem_t*)malloc(zmq_polls_size * sizeof(zmq_pollitem_t));

  zmq_polls[0].socket = nullptr;
  zmq_polls[0].fd = efd_;
  zmq_polls[0].events = ZMQ_POLLIN;
  zmq_polls[0].revents = 0;

  // consume efd.
  uint64_t u;
  long wait_timeout = -1;

  started_.store(true);

  while (stop_.load() == false) {
    ZMQ_CALL(zmq_poll(zmq_polls, zmq_polls_size, wait_timeout));

    // We should handle the socket event firstly then handle Task.
    for (int i = 1; i < zmq_polls_size; ++i) {
      if (zmq_polls[i].revents & ZMQ_POLLIN) {
        zmq_msg_t reply;

        ZMQ_CALL(zmq_msg_init(&reply));
        ZMQ_CALL(zmq_msg_recv(&reply, zmq_polls[i].socket, 0));

        HandleReply(reply);

        ZMQ_CALL(zmq_msg_close(&reply));
      }
    }

    // Task
    if (zmq_polls[0].revents & ZMQ_POLLIN) {
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

        if (task.type == 0) {
          // Add connect.
          bool success =
              AddConnect(task.id, task.addr, &zmq_polls, &zmq_polls_size);

          if (task.callback) {
            task.callback(success);
          }
        } else if (task.type == 1) {
          // Remove connect.
          bool success = RemoveConnect(task.id, &zmq_polls, &zmq_polls_size);

          if (task.callback) {
            task.callback(success);
          }
        } else if (task.type == 2) {
          // Send message.
          int32_t error_code =
              SendMsg(task.id, task.timestamp, task.rpc_type, &task.z_buf);

          if (task.z_callback) {
            if (error_code != ErrorCode::kSuccess) {
              ReplyHeader dummy_header;
              dummy_header.timestamp = task.timestamp;
              dummy_header.error_code = error_code;
              dummy_header.compress_type = CompressType::kNo;

              task.z_callback(dummy_header, nullptr, 0);
            } else {
              z_callbacks_.emplace(task.timestamp, std::move(task.z_callback));

              if (task.timeout_ms > 0) {
                TimerEvent event;
                event.timestamp = task.timestamp;
                event.when = std::chrono::steady_clock::now() +
                             std::chrono::milliseconds(task.timeout_ms);

                timers_.push(event);
              }
            }
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

  for (auto sender : senders_) {
    ZMQ_CALL(zmq_close(sender));
  }

  sender_idx_.clear();
  senders_.clear();

  free(zmq_polls);
}

void CombineConnecter::EnqueTask(Task&& task) {
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

void CombineConnecter::Start() {
  zmq_context_ = zmq_init(1);
  ARGUMENT_CHECK(zmq_context_ != nullptr, "zmq_init return nullptr, error:"
                                              << zmq_strerror(zmq_errno()));

  // create eventfd.
  efd_ = eventfd(0, EFD_SEMAPHORE);
  ARGUMENT_CHECK(efd_ != -1, "eventfd error:" << efd_);

  // start worker thread.
  worker_ = std::thread(&CombineConnecter::Run, this);

  while (started_.load() == false) {
  }
}

void CombineConnecter::Stop() {
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

bool CombineConnecter::AddConnect(uint64_t id, const std::string& addr) {
  ThreadBarrier barrier(1);
  bool success = true;

  auto callback = [&barrier, &success](bool s) {
    success = s;
    barrier.Release();
  };

  AddConnectAsync(id, addr, std::move(callback));

  barrier.Wait();

  return success;
}

void CombineConnecter::AddConnectAsync(uint64_t id, const std::string& addr,
                                       CALLBACK&& callback) {
  Task task;
  task.type = 0;
  task.id = id;
  task.addr = addr;
  task.callback = std::move(callback);

  EnqueTask(std::move(task));
}

bool CombineConnecter::RemoveConnect(uint64_t id) {
  ThreadBarrier barrier(1);
  bool success = true;

  auto callback = [&barrier, &success](bool s) {
    success = s;

    barrier.Release();
  };

  RemoveConnectAsync(id, std::move(callback));

  barrier.Wait();

  return success;
}

void CombineConnecter::RemoveConnectAsync(uint64_t id, CALLBACK&& callback) {
  Task task;
  task.type = 1;
  task.id = id;
  task.callback = std::move(callback);

  EnqueTask(std::move(task));
}

}  // namespace kraken
