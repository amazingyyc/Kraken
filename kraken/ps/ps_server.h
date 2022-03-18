#pragma once

#include "protocol/combine_pull_dense_table_prot.h"
#include "protocol/combine_pull_sparse_table_prot.h"
#include "protocol/create_dense_table_prot.h"
#include "protocol/create_model_prot.h"
#include "protocol/create_sparse_table_prot.h"
#include "protocol/heartbeat_prot.h"
#include "protocol/notify_finish_transfer_prot.h"
#include "protocol/notify_node_join_prot.h"
#include "protocol/pull_dense_table_prot.h"
#include "protocol/pull_sparse_table_prot.h"
#include "protocol/push_dense_table_prot.h"
#include "protocol/push_sparse_table_prot.h"
#include "protocol/transfer_dense_table_prot.h"
#include "protocol/transfer_sparse_meta_data_prot.h"
#include "protocol/transfer_sparse_values_prot.h"
#include "protocol/try_combine_fetch_dense_table_prot.h"
#include "protocol/try_fetch_dense_table_prot.h"
#include "protocol/try_fetch_sparse_meta_data_prot.h"
#include "protocol/try_fetch_sparse_values_prot.h"
#include "ps/ps.h"
#include "rpc/station.h"

namespace kraken {

class PsServer {
private:
  Station station_;
  Ps ps_;

public:
  PsServer(uint32_t port, uint32_t thread_nums, const std::string& addr,
           const std::string& s_addr);

private:
  int32_t Heartbeat(const HeartbeatRequest& req, HeartbeatResponse* rsp);

  int32_t NotifyFinishTransfer(const NotifyFinishTransferRequest& req,
                               NotifyFinishTransferResponse* rsp);

  int32_t NotifyNodeJoin(const NotifyNodeJoinRequest& req,
                         NotifyNodeJoinResponse* rsp);

  int32_t CreateModel(const CreateModelRequest& req, CreateModelResponse* rsp);

  int32_t CreateDenseTable(const CreateDenseTableRequest& req,
                           CreateDenseTableResponse* rsp);

  int32_t CreateSparseTable(const CreateSparseTableRequest& req,
                            CreateSparseTableResponse* rsp);

  int32_t TransferDenseTable(const TransferDenseTableRequest& req,
                             TransferDenseTableResponse* rsp);

  int32_t TransferSparseMetaData(const TransferSparseMetaDataRequest& req,
                                 TransferSparseMetaDataResponse* rsp);

  int32_t TransferSparseValues(const TransferSparseValuesRequest& req,
                               TransferSparseValuesResponse* rsp);

  int32_t TryFetchDenseTable(const TryFetchDenseTableRequest& req,
                             TryFetchDenseTableResponse* rsp);

  int32_t TryCombineFetchDenseTable(const TryCombineFetchDenseTableRequest& req,
                                    TryCombineFetchDenseTableResponse* rsp);

  int32_t TryFetchSparseMetaData(const TryFetchSparseMetaDataRequest& req,
                                 TryFetchSparseMetaDataResponse* rsp);

  int32_t TryFetchSparseValues(const TryFetchSparseValuesRequest& req,
                               TryFetchSparseValuesResponse* rsp);

  int32_t PullDenseTable(const PullDenseTableRequest& req,
                         PullDenseTableResponse* rsp);

  int32_t CombinePullDenseTable(const CombinePullDenseTableRequest& req,
                                CombinePullDenseTableResponse* rsp);

  int32_t PushDenseTable(const PushDenseTableRequest& req,
                         PushDenseTableResponse* rsp);

  int32_t PullSparseTable(const PullSparseTableRequest& req,
                          PullSparseTableResponse* rsp);

  int32_t CombinePullSparseTable(const CombinePullSparseTableRequest& req,
                                 CombinePullSparseTableResponse* rsp);

  int32_t PushSparseTable(const PushSparseTableRequest& req,
                          PushSparseTableResponse* rsp);

  int32_t CombinePushSparseTable(const CombinePushSparseTableRequest& req,
                                 CombinePushSparseTableResponse* rsp);

  void RegisterFuncs();

public:
  void Start();

  void Stop();
};

}  // namespace kraken
