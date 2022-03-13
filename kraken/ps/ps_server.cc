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

int32_t PsServer::CreateDenseTable(const CreateDenseTableRequest& req,
                                   CreateDenseTableResponse* rsp) {
  return ps_.CreateDenseTable(req.id, req.name, req.val);
}

int32_t PsServer::CreateSparseTable(const CreateSparseTableRequest& req,
                                    CreateSparseTableResponse* rsp) {
  return ps_.CreateSparseTable(req.id, req.name, req.dimension,
                               req.element_type, req.init_type, req.init_conf);
}

int32_t PsServer::TransferDenseTable(const TransferDenseTableRequest& req,
                                     TransferDenseTableResponse* rsp) {
  return ps_.TransferDenseTable(req.id, req.name, req.value);
}

int32_t PsServer::TransferSparseMetaData(
    const TransferSparseMetaDataRequest& req,
    TransferSparseMetaDataResponse* rsp) {
  return ps_.TransferSparseMetaData(req.id, req.name, req.dimension,
                                    req.element_type, req.init_type,
                                    req.init_conf);
}

int32_t PsServer::TransferSparseValues(const TransferSparseValuesRequest& req,
                                       TransferSparseValuesResponse* rsp) {
  return ps_.TransferSparseValues(req.id, req.sparse_ids, req.values);
}

int32_t PsServer::TryFetchDenseTable(const TryFetchDenseTableRequest& req,
                                     TryFetchDenseTableResponse* rsp) {
  return ps_.TryFetchDenseTable(req.id, &(rsp->name), &(rsp->value));
}

int32_t PsServer::TryCombineFetchDenseTable(
    const TryCombineFetchDenseTableRequest& req,
    TryCombineFetchDenseTableResponse* rsp) {
  return ps_.TryCombineFetchDenseTable(req.ids, &(rsp->names), &(rsp->values));
}

int32_t PsServer::PullDenseTable(const PullDenseTableRequest& req,
                                 PullDenseTableResponse* rsp) {
  return ps_.PullDenseTable(req.router_version, req.table_id, &(rsp->val));
}

int32_t PsServer::CombinePullDenseTable(const CombinePullDenseTableRequest& req,
                                        CombinePullDenseTableResponse* rsp) {
  return ps_.CombinePullDenseTable(req.router_version, req.table_ids,
                                   &(rsp->vals));
}

int32_t PsServer::PushDenseTable(const PushDenseTableRequest& req,
                                 PushDenseTableResponse* rsp) {
  return ps_.PushDenseTable(req.router_version, req.table_id, req.grad, req.lr);
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
  REGISTER_FUNC(CreateDenseTable, CreateDenseTable);
  REGISTER_FUNC(CreateSparseTable, CreateSparseTable);
  REGISTER_FUNC(TransferDenseTable, TransferDenseTable);
  REGISTER_FUNC(TransferSparseMetaData, TransferSparseMetaData);
  REGISTER_FUNC(TransferSparseValues, TransferSparseValues);
  REGISTER_FUNC(TryFetchDenseTable, TryFetchDenseTable);
  REGISTER_FUNC(TryCombineFetchDenseTable, TryCombineFetchDenseTable);
  REGISTER_FUNC(PullDenseTable, PullDenseTable);
  REGISTER_FUNC(CombinePullDenseTable, CombinePullDenseTable);
  REGISTER_FUNC(PushDenseTable, PushDenseTable);
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
