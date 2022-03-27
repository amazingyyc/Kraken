#include "rpc/sync_station.h"

#include "common/log.h"

namespace kraken {

SyncStation::SyncStation(uint32_t port) : port_(port) {
}

void SyncStation::HandleError(uint64_t timestamp, int32_t error_code,
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

  // ZMQ_CALL(zmq_msg_close(&replyid));
  // ZMQ_CALL(zmq_msg_close(&reply));
}

void SyncStation::HandleMsg(zmq_msg_t& identity, zmq_msg_t& msg, void* socket) {
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

void SyncStation::Run() {
  zmq_context_ = zmq_ctx_new();
  ARGUMENT_CHECK(zmq_context_ != nullptr, "zmq_ctx_new return nullptr, error:"
                                              << zmq_strerror(zmq_errno()));

  zmq_scoket_ = zmq_socket(zmq_context_, ZMQ_ROUTER);
  ARGUMENT_CHECK(zmq_scoket_ != nullptr, "zmq_socket return nullptr, error:"
                                             << zmq_strerror(zmq_errno()));

  std::string addr = "tcp://*:" + std::to_string(port_);
  ZMQ_CALL(zmq_bind(zmq_scoket_, addr.c_str()));

  LOG_INFO("SyncStation start at port:[" << port_ << "]");

  while (true) {
    zmq_msg_t identity;
    zmq_msg_t msg;

    ZMQ_CALL(zmq_msg_init(&identity));
    ZMQ_CALL(zmq_msg_init(&msg));

    ZMQ_CALL(zmq_msg_recv(&identity, zmq_scoket_, 0));
    ZMQ_CALL(zmq_msg_recv(&msg, zmq_scoket_, 0));

    HandleMsg(identity, msg, zmq_scoket_);

    ZMQ_CALL(zmq_msg_close(&identity));
    ZMQ_CALL(zmq_msg_close(&msg));
  }

  // Never be called.
  ZMQ_CALL(zmq_close(zmq_scoket_));
  zmq_scoket_ = nullptr;

  ZMQ_CALL(zmq_term(zmq_context_));
  zmq_context_ = nullptr;
}

void SyncStation::Start() {
  Run();
}

}  // namespace kraken
