#include "ps/ps_server.h"

#include "protocol/rpc_func_type.h"

namespace kraken {

PsServer::PsServer(uint32_t port, uint32_t thread_nums, const std::string& addr,
                   const std::string& s_addr)
    : station_(port, thread_nums, true), ps_(addr, s_addr) {
}

int32_t PsServer::Heartbeat(const HeartbeatRequest& req,
                            HeartbeatResponse* rsp) {
  return ps_.Heartbeat(&(rsp->status));
}

int32_t PsServer::NotifyFinishTransfer(const NotifyFinishTransferRequest& req,
                                       NotifyFinishTransferResponse* rsp) {
  return ps_.NotifyFinishTransfer(req.node_id);
}

int32_t PsServer::NotifyNodeJoin(const NotifyNodeJoinRequest& req,
                                 NotifyNodeJoinResponse* rsp) {
  return ps_.NotifyNodeJoin(req.joined_id, req.old_router, req.new_router);
}

int32_t PsServer::InitModel(const InitModelRequest& req,
                            InitModelResponse* rsp) {
  return ps_.InitModel(req.name, req.optim_type, req.optim_conf);
}

void PsServer::RegisterFuncs() {
  using namespace std::placeholders;

#define REGISTER_FUNC(TYPE, FUNC) \
  station_.RegisterFunc<TYPE##Request, TYPE##Response>( \
      RPCFuncType::k##TYPE##Type, std::bind(&PsServer::FUNC, this, _1, _2));

  REGISTER_FUNC(Heartbeat, Heartbeat);
  REGISTER_FUNC(NotifyFinishTransfer, NotifyFinishTransfer);
  REGISTER_FUNC(NotifyNodeJoin, NotifyNodeJoin);
  REGISTER_FUNC(InitModel, InitModel);
}

void PsServer::Start() {
  RegisterFuncs();

  station_.Start();
  ps_.Start();

  // Station is async we have to wait.
  station_.Wait();
}

void PsServer::Stop() {
  station_.Stop();
}

}  // namespace kraken
