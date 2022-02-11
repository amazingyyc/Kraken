#include "ps/ps_server.h"

#include "protocol/rpc_func_type.h"

namespace kraken {

PsServer::PsServer(uint32_t port, uint32_t thread_nums, size_t shard_num,
                   size_t shard_id, const std::string& save_dir,
                   size_t max_save_count)
    : server_(port, thread_nums),
      ps_(shard_num, shard_id, save_dir, max_save_count) {
}

int32_t PsServer::ApplyModel(const ApplyModelRequest& req,
                             ApplyModelResponse* rsp) {
  return ps_.ApplyModel(req.model_name, req.optim_type, req.optim_conf,
                        &(rsp->model_id));
}

int32_t PsServer::ApplyDenseTable(const ApplyDenseTableRequest& req,
                                  ApplyDenseTableResponse* rsp) {
  return ps_.ApplyDenseTable(req.model_id, req.name, req.shape,
                             req.element_type, &(rsp->table_id));
}

int32_t PsServer::ApplySparseTable(const ApplySparseTableRequest& req,
                                   ApplySparseTableResponse* rsp) {
  return ps_.ApplySparseTable(req.model_id, req.name, req.dimension,
                              req.element_type, req.init_type, req.init_conf,
                              &(rsp->table_id));
}

int32_t PsServer::RegisterModel(const RegisterModelRequest& req,
                                RegisterModelResponse* rsp) {
  return ps_.RegisterModel(req.id, req.name, req.optim_type, req.optim_conf);
}

int32_t PsServer::RegisterDenseTableInfo(
    const RegisterDenseTableInfoRequest& req,
    RegisterDenseTableInfoResponse* rsp) {
  return ps_.RegisterDenseTableInfo(req.model_id, req.id, req.name, req.shape,
                                    req.element_type);
}

int32_t PsServer::RegisterDenseTable(const RegisterDenseTableRequest& req,
                                     RegisterDenseTableResponse* rsp) {
  return ps_.RegisterDenseTable(req.model_id, req.id, req.name, req.val);
}

int32_t PsServer::RegisterSparseTable(const RegisterSparseTableRequest& req,
                                      RegisterSparseTableResponse* rsp) {
  return ps_.RegisterSparseTable(req.model_id, req.id, req.name, req.dimension,
                                 req.element_type, req.init_type,
                                 req.init_conf);
}

int32_t PsServer::PullDenseTable(const PullDenseTableRequest& req,
                                 PullDenseTableResponse* rsp) {
  return ps_.PullDenseTable(req.model_id, req.table_id, &(rsp->val));
}

int32_t PsServer::CombinePullDenseTable(const CombinePullDenseTableRequest& req,
                                        CombinePullDenseTableResponse* rsp) {
  return ps_.CombinePullDenseTable(req.model_id, req.table_ids, &(rsp->vals));
}

int32_t PsServer::PushPullDenseTable(const PushPullDenseTableRequest& req,
                                     PushPullDenseTableResponse* rsp) {
  return ps_.PushPullDenseTable(req.model_id, req.table_id, req.grad, req.lr,
                                &(rsp->val));
}

int32_t PsServer::PushDenseTable(const PushDenseTableRequest& req,
                                 PushDenseTableResponse* rsp) {
  return ps_.PushDenseTable(req.model_id, req.table_id, req.grad, req.lr);
}

int32_t PsServer::PullSparseTable(const PullSparseTableRequest& req,
                                  PullSparseTableResponse* rsp) {
  return ps_.PullSparseTable(req.model_id, req.table_id, req.indices,
                             &(rsp->vals));
}

int32_t PsServer::CombinePullSparseTable(
    const CombinePullSparseTableRequest& req,
    CombinePullSparseTableResponse* rsp) {
  rsp->vals.resize(req.indices.size());

  for (size_t i = 0; i < req.indices.size(); ++i) {
    auto ecode =
        ps_.PullSparseTable(req.indices[i].model_id, req.indices[i].table_id,
                            req.indices[i].indices, &(rsp->vals[i].vals));

    if (ecode != ErrorCode::kSuccess) {
      return ecode;
    }
  }

  return ErrorCode::kSuccess;
}

int32_t PsServer::PushSparseTable(const PushSparseTableRequest& req,
                                  PushSparseTableResponse* rsp) {
  return ps_.PushSparseTable(req.model_id, req.table_id, req.indices, req.grads,
                             req.lr);
}

int32_t PsServer::SaveCheckPoint(const SaveCheckPointRequest& req,
                                 SaveCheckPointResponse* rsp) {
  return ps_.SaveCheckPoint(req.model_id);
}

void PsServer::RegisterFuncs() {
  using namespace std::placeholders;

#define REGISTER_FUNC(TYPE, FUNC) \
  server_.RegisterFunc<TYPE##Request, TYPE##Response>( \
      RPCFuncType::k##TYPE##Type, std::bind(&PsServer::FUNC, this, _1, _2));

  REGISTER_FUNC(ApplyModel, ApplyModel);
  REGISTER_FUNC(ApplyDenseTable, ApplyDenseTable);
  REGISTER_FUNC(ApplySparseTable, ApplySparseTable);
  REGISTER_FUNC(RegisterModel, RegisterModel);
  REGISTER_FUNC(RegisterDenseTableInfo, RegisterDenseTableInfo);
  REGISTER_FUNC(RegisterDenseTable, RegisterDenseTable);
  REGISTER_FUNC(RegisterSparseTable, RegisterSparseTable);
  REGISTER_FUNC(PullDenseTable, PullDenseTable);
  REGISTER_FUNC(CombinePullDenseTable, CombinePullDenseTable);
  REGISTER_FUNC(PushPullDenseTable, PushPullDenseTable);
  REGISTER_FUNC(PushDenseTable, PushDenseTable);
  REGISTER_FUNC(PullSparseTable, PullSparseTable);
  REGISTER_FUNC(CombinePullSparseTable, CombinePullSparseTable);
  REGISTER_FUNC(PushSparseTable, PushSparseTable);
  REGISTER_FUNC(SaveCheckPoint, SaveCheckPoint);
}

void PsServer::Load(const std::string& load_dir) {
  ps_.Load(load_dir);
}

void PsServer::Start() {
  RegisterFuncs();

  server_.Start();
}

void PsServer::Stop() {
  server_.Stop();
  ps_.Stop();
}

}  // namespace kraken
