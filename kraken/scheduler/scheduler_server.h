#pragma once

#include "protocol/fetch_model_meta_data_prot.h"
#include "protocol/fetch_router_prot.h"
#include "protocol/init_model_prot.h"
#include "protocol/register_dense_table_prot.h"
#include "protocol/register_sparse_table_prot.h"
#include "protocol/try_join_prot.h"
#include "rpc/simple_station.h"
#include "scheduler/scheduler.h"

namespace kraken {

class SchedulerServer {
private:
  SimpleStation station_;
  Scheduler scheduler_;

public:
  SchedulerServer(uint32_t port);

  ~SchedulerServer() = default;

private:
  int32_t TryJoin(const TryJoinRequest& req, TryJoinResponse* rsp);

  int32_t FetchModelMetaData(const FetchModelMetaDataRequest& req,
                             FetchModelMetaDataResponse* rsp);

  int32_t FetchRouter(const FetchRouterRequest& req, FetchRouterResponse* rsp);

  int32_t InitModel(const InitModelRequest& req, InitModelResponse* rsp);

  int32_t RegisterDenseTable(const RegisterDenseTableRequest& req,
                             RegisterDenseTableResponse* rsp);

  int32_t RegisterSparseTable(const RegisterSparseTableRequest& req,
                              RegisterSparseTableResponse* rsp);

  void RegisterFuncs();

public:
  void Start();

  void Stop();
};

}  // namespace kraken
