#include "ps/ps_server.h"

#include "protocol/rpc_func_type.h"

namespace kraken {

PsServer::PsServer(uint32_t port, uint32_t thread_nums)
    : server_(port, thread_nums) {
}

int32_t PsServer::ApplyModel(const ApplyModelRequest& req,
                             ApplyModelResponse* rsp) {
  return ps_.ApplyModel(req.name, &(rsp->model_id));
}

int32_t PsServer::ApplyTable(const ApplyTableRequest& req,
                             ApplyTableResponse* rsp) {
  return ps_.ApplyTable(req.model_id, req.table_name, req.table_type,
                        &(rsp->table_id));
}

int32_t PsServer::RegisterModel(const RegisterModelRequest& req,
                                RegisterModelResponse* rsp) {
  return ps_.RegisterModel(req.id, req.name, req.optim_type, req.optim_conf);
}

int32_t PsServer::RegisterDenseTable(const RegisterDenseTableRequest& req,
                                     RegisterDenseTableResponse* rsp) {
  return ps_.RegisterDenseTable(req.model_id, req.id, req.name, req.val);
}

int32_t PsServer::RegisterSparseTable(const RegisterSparseTableRequest& req,
                                      RegisterSparseTableResponse* rsp) {
  return ps_.RegisterSparseTable(req.model_id, req.id, req.name, req.dimension,
                                 req.etype);
}

int32_t PsServer::PushDenseTable(const PushDenseTableRequest& req,
                                 PushDenseTableResponse* rsp) {
  return ps_.PushDenseTable(req.model_id, req.table_id, req.grad, req.lr);
}

int32_t PsServer::PullDenseTable(const PullDenseTableRequest& req,
                                 PullDenseTableResponse* rsp) {
  return ps_.PullDenseTable(req.model_id, req.table_id, &(rsp->val));
}

int32_t PsServer::PullListDenseTable(const PullListDenseTableRequest& req,
                                     PullListDenseTableResponse* rsp) {
  return ps_.PullListDenseTable(req.model_id, req.table_ids, &(rsp->vals));
}

int32_t PsServer::PushPullDenseTable(const PushPullDenseTableRequest& req,
                                     PushPullDenseTableResponse* rsp) {
  return ps_.PushPullDenseTable(req.model_id, req.table_id, req.grad, req.lr,
                                &(rsp->val));
}

int32_t PsServer::PushSparseTable(const PushSparseTableRequest& req,
                                  PushSparseTableResponse* rsp) {
  return ps_.PushSparseTable(req.model_id, req.table_id, req.indices, req.grads,
                             req.lr);
}

int32_t PsServer::PullSparseTable(const PullSparseTableRequest& req,
                                  PullSparseTableResponse* rsp) {
  return ps_.PullSparseTable(req.model_id, req.table_id, req.indices,
                             &(rsp->vals));
}

void PsServer::RegisterFuncs() {
  using namespace std::placeholders;

#define REGISTER_FUNC(TYPE, FUNC) \
  server_.RegisterFunc<TYPE##Request, TYPE##Response>( \
      RPCFuncType::k##TYPE##Type, std::bind(&PsServer::FUNC, this, _1, _2));

  REGISTER_FUNC(ApplyModel, ApplyModel);
  REGISTER_FUNC(ApplyTable, ApplyTable);
  REGISTER_FUNC(RegisterModel, RegisterModel);
  REGISTER_FUNC(RegisterDenseTable, RegisterDenseTable);
  REGISTER_FUNC(RegisterSparseTable, RegisterSparseTable);
  REGISTER_FUNC(PushDenseTable, PushDenseTable);
  REGISTER_FUNC(PullDenseTable, PullDenseTable);
  REGISTER_FUNC(PullListDenseTable, PullListDenseTable);
  REGISTER_FUNC(PushPullDenseTable, PushPullDenseTable);
  REGISTER_FUNC(PushSparseTable, PushSparseTable);
  REGISTER_FUNC(PullSparseTable, PullSparseTable);
}

void PsServer::Start() {
  RegisterFuncs();

  server_.Start();
}

void PsServer::Stop() {
  server_.Stop();
}

}  // namespace kraken
