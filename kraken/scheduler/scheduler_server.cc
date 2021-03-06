#include "scheduler/scheduler_server.h"

#include "protocol/rpc_func_type.h"
#include "rpc/protocol.h"

namespace kraken {

SchedulerServer::SchedulerServer(uint32_t port) : station_(port), scheduler_() {
}

int32_t SchedulerServer::TryJoin(const TryJoinRequest& req,
                                 TryJoinResponse* rsp) {
  return scheduler_.TryJoin(req.addr, &(rsp->allow), &(rsp->node_id),
                            &(rsp->old_router), &(rsp->new_router),
                            &(rsp->model_init), &(rsp->model_mdata));
}

int32_t SchedulerServer::FetchRouter(const FetchRouterRequest& req,
                                     FetchRouterResponse* rsp) {
  return scheduler_.FetchRouter(&(rsp->router));
}

int32_t SchedulerServer::InitModel(const InitModelRequest& req,
                                   InitModelResponse* rsp) {
  return scheduler_.InitModel(req.name, req.optim_type, req.optim_conf);
}

int32_t SchedulerServer::RegisterDenseTable(
    const RegisterDenseTableRequest& req, RegisterDenseTableResponse* rsp) {
  return scheduler_.RegisterDenseTable(req.name, req.val, &(rsp->table_id));
}

int32_t SchedulerServer::RegisterSparseTable(
    const RegisterSparseTableRequest& req, RegisterSparseTableResponse* rsp) {
  return scheduler_.RegisterSparseTable(req.name, req.dimension,
                                        req.element_type, req.init_type,
                                        req.init_conf, &(rsp->table_id));
}

int32_t SchedulerServer::TrySaveModel(const TrySaveModelRequest& req,
                                      TrySaveModelResponse* rsp) {
  return scheduler_.TrySaveModel(&(rsp->success));
}

int32_t SchedulerServer::TryLoadModel(const TryLoadModelRequest& req,
                                      TryLoadModelResponse* rsp) {
  return scheduler_.TryLoadModel(req.load_dir, &(rsp->success));
}

int32_t SchedulerServer::IsAllPsWorking(const IsAllPsWorkingRequest& req,
                                        IsAllPsWorkingResponse* rsp) {
  return scheduler_.IsAllPsWorking(&(rsp->yes));
}

void SchedulerServer::RegisterFuncs() {
  using namespace std::placeholders;

#define REGISTER_FUNC(TYPE, FUNC) \
  station_.RegisterFunc<TYPE##Request, TYPE##Response>( \
      RPCFuncType::k##TYPE##Type, \
      std::bind(&SchedulerServer::FUNC, this, _1, _2));

  REGISTER_FUNC(TryJoin, TryJoin);
  REGISTER_FUNC(FetchRouter, FetchRouter);
  REGISTER_FUNC(InitModel, InitModel);
  REGISTER_FUNC(RegisterDenseTable, RegisterDenseTable);
  REGISTER_FUNC(RegisterSparseTable, RegisterSparseTable);
  REGISTER_FUNC(TrySaveModel, TrySaveModel);
  REGISTER_FUNC(TryLoadModel, TryLoadModel);
  REGISTER_FUNC(IsAllPsWorking, IsAllPsWorking);
}

void SchedulerServer::Start() {
  RegisterFuncs();

  scheduler_.Start();
  station_.Start();
}

void SchedulerServer::Stop() {
  scheduler_.Stop();
}

}  // namespace kraken
