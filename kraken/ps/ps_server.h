#pragma once

#include "protocol/apply_model_id_prot.h"
#include "protocol/apply_table_id_prot.h"
#include "protocol/combine_pull_dense_table_prot.h"
#include "protocol/combine_pull_sparse_table_prot.h"
#include "protocol/pull_dense_table_prot.h"
#include "protocol/pull_sparse_table_prot.h"
#include "protocol/push_dense_table_prot.h"
#include "protocol/push_pull_dense_table_prot.h"
#include "protocol/push_sparse_table_prot.h"
#include "protocol/register_dense_table_info_prot.h"
#include "protocol/register_dense_table_prot.h"
#include "protocol/register_model_prot.h"
#include "protocol/register_sparse_table_info_prot.h"
#include "protocol/register_sparse_table_prot.h"
#include "ps/ps.h"
#include "rpc/server.h"

namespace kraken {

class PsServer {
private:
  Server server_;
  Ps ps_;

public:
  PsServer(uint32_t port, uint32_t thread_nums, size_t shard_num,
           size_t shard_id);

private:
  int32_t ApplyModelId(const ApplyModelIdRequest&, ApplyModelIdResponse*);

  int32_t ApplyTableId(const ApplyTableIdRequest&, ApplyTableIdResponse*);

  int32_t RegisterModel(const RegisterModelRequest&, RegisterModelResponse*);

  int32_t RegisterDenseTableInfo(const RegisterDenseTableInfoRequest&,
                                 RegisterDenseTableInfoResponse*);

  int32_t RegisterDenseTable(const RegisterDenseTableRequest&,
                             RegisterDenseTableResponse*);

  int32_t RegisterSparseTableInfo(const RegisterSparseTableInfoRequest&,
                                  RegisterSparseTableInfoResponse*);

  int32_t RegisterSparseTable(const RegisterSparseTableRequest&,
                              RegisterSparseTableResponse*);

  int32_t PullDenseTable(const PullDenseTableRequest&, PullDenseTableResponse*);

  int32_t CombinePullDenseTable(const CombinePullDenseTableRequest&,
                                CombinePullDenseTableResponse*);

  int32_t PushPullDenseTable(const PushPullDenseTableRequest&,
                             PushPullDenseTableResponse*);

  int32_t PushDenseTable(const PushDenseTableRequest&, PushDenseTableResponse*);

  int32_t PullSparseTable(const PullSparseTableRequest&,
                          PullSparseTableResponse*);

  int32_t CombinePullSparseTable(const CombinePullSparseTableRequest&,
                                 CombinePullSparseTableResponse*);

  int32_t PushSparseTable(const PushSparseTableRequest&,
                          PushSparseTableResponse*);

  void RegisterFuncs();

public:
  void Start();

  void Stop();
};

}  // namespace kraken
