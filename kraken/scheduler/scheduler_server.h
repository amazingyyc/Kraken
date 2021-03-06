#pragma once

#include "protocol/fetch_model_meta_data_prot.h"
#include "protocol/fetch_router_prot.h"
#include "protocol/init_model_prot.h"
#include "protocol/is_all_ps_working_prot.h"
#include "protocol/register_dense_table_prot.h"
#include "protocol/register_sparse_table_prot.h"
#include "protocol/try_join_prot.h"
#include "protocol/try_load_model_prot.h"
#include "protocol/try_save_model_prot.h"
#include "rpc/sync_station.h"
#include "scheduler/scheduler.h"

namespace kraken {

class SchedulerServer {
private:
  SyncStation station_;
  Scheduler scheduler_;

public:
  SchedulerServer(uint32_t port);

  ~SchedulerServer() = default;

private:
  int32_t TryJoin(const TryJoinRequest& req, TryJoinResponse* rsp);

  int32_t FetchRouter(const FetchRouterRequest& req, FetchRouterResponse* rsp);

  int32_t InitModel(const InitModelRequest& req, InitModelResponse* rsp);

  int32_t RegisterDenseTable(const RegisterDenseTableRequest& req,
                             RegisterDenseTableResponse* rsp);

  int32_t RegisterSparseTable(const RegisterSparseTableRequest& req,
                              RegisterSparseTableResponse* rsp);

  int32_t TrySaveModel(const TrySaveModelRequest& req,
                       TrySaveModelResponse* rsp);

  int32_t TryLoadModel(const TryLoadModelRequest& req,
                       TryLoadModelResponse* rsp);

  int32_t IsAllPsWorking(const IsAllPsWorkingRequest& req,
                         IsAllPsWorkingResponse* rsp);

  void RegisterFuncs();

public:
  void Start();

  void Stop();
};

}  // namespace kraken
