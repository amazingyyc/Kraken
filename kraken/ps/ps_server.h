#pragma once

// #include "protocol/apply_dense_table_prot.h"
// #include "protocol/apply_model_prot.h"
// #include "protocol/apply_sparse_table_prot.h"
// #include "protocol/apply_table_id_prot.h"
// #include "protocol/combine_pull_dense_table_prot.h"
// #include "protocol/combine_pull_sparse_table_prot.h"
// #include "protocol/pull_dense_table_prot.h"
// #include "protocol/pull_sparse_table_prot.h"
// #include "protocol/push_dense_table_prot.h"
// #include "protocol/push_pull_dense_table_prot.h"
// #include "protocol/push_sparse_table_prot.h"
// #include "protocol/register_dense_table_info_prot.h"
// #include "protocol/register_dense_table_prot.h"
// #include "protocol/register_model_prot.h"
// #include "protocol/register_sparse_table_info_prot.h"
// #include "protocol/register_sparse_table_prot.h"
// #include "protocol/save_check_point_prot.h"
#include "protocol/heartbeat_prot.h"
#include "protocol/init_model_prot.h"
#include "protocol/notify_finish_transfer_prot.h"
#include "protocol/notify_node_join_prot.h"
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

  int32_t InitModel(const InitModelRequest& req, InitModelResponse* rsp);

  void RegisterFuncs();

public:
  void Start();

  void Stop();
};

}  // namespace kraken
