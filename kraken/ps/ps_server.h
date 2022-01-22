#pragma once

#include "protocol/apply_model_prot.h"
#include "protocol/apply_table_prot.h"
#include "protocol/pull_dense_table_prot.h"
#include "protocol/pull_list_dense_table_prot.h"
#include "protocol/pull_sparse_table_prot.h"
#include "protocol/push_dense_table_prot.h"
#include "protocol/push_pull_dense_table_prot.h"
#include "protocol/push_sparse_table_prot.h"
#include "protocol/register_dense_table_prot.h"
#include "protocol/register_model_prot.h"
#include "protocol/register_sparse_table_prot.h"
#include "ps/ps.h"
#include "rpc/server.h"

namespace kraken {

class PsServer {
private:
  Ps ps_;
  Server server_;

public:
  PsServer(uint32_t port, uint32_t thread_nums);

private:
  int32_t ApplyModel(const ApplyModelRequest&, ApplyModelResponse*);

  int32_t ApplyTable(const ApplyTableRequest&, ApplyTableResponse*);

  int32_t RegisterModel(const RegisterModelRequest&, RegisterModelResponse*);

  int32_t RegisterDenseTable(const RegisterDenseTableRequest&,
                             RegisterDenseTableResponse*);

  int32_t RegisterSparseTable(const RegisterSparseTableRequest&,
                              RegisterSparseTableResponse*);

  int32_t RegisterSparseTableV2(const RegisterSparseTableV2Request&,
                                RegisterSparseTableV2Response*);

  int32_t PushDenseTable(const PushDenseTableRequest&, PushDenseTableResponse*);

  int32_t PullDenseTable(const PullDenseTableRequest&, PullDenseTableResponse*);

  int32_t PullListDenseTable(const PullListDenseTableRequest&,
                             PullListDenseTableResponse*);

  int32_t PushPullDenseTable(const PushPullDenseTableRequest&,
                             PushPullDenseTableResponse*);

  int32_t PushSparseTable(const PushSparseTableRequest&,
                          PushSparseTableResponse*);

  int32_t PullSparseTable(const PullSparseTableRequest&,
                          PullSparseTableResponse*);

  int32_t CombinePullSparseTable(const CombinePullSparseTableRequest&,
                                 CombinePullSparseTableResponse*);

  void RegisterFuncs();

public:
  void Start();

  void Stop();
};

}  // namespace kraken
