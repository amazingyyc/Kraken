#include "scheduler/scheduler.h"

#include <iostream>

#include "checkpoint/checkpoint.h"
#include "checkpoint/checkpoint_exec.h"
#include "common/log.h"
#include "protocol/create_dense_table_prot.h"
#include "protocol/create_model_prot.h"
#include "protocol/create_sparse_table_prot.h"
#include "protocol/heartbeat_prot.h"
#include "protocol/notify_load_model_prot.h"
#include "protocol/notify_node_join_prot.h"
#include "protocol/notify_router_change_prot.h"
#include "protocol/notify_save_model_prot.h"
#include "protocol/rpc_func_type.h"

namespace kraken {

Scheduler::Scheduler() : connecter_(CompressType::kNo), model_init_(false) {
}

Scheduler::~Scheduler() {
}

bool Scheduler::LoadModelMetaData(const std::string& load_dir) {
  // In Scheduler we only need load the model metadata.
  std::string model_mdata_binary_path;

  io::CheckpointExec checkpoint_exec;
  if (checkpoint_exec.GetLatestModelMetaDataBinaryPath(
          load_dir, &model_mdata_binary_path) == false) {
    LOG_ERROR("Get model meta data binary path from:[" << load_dir
                                                       << "] error!");
    return false;
  }

  if (io::Checkpoint::LoadModelMetaDataBinary(model_mdata_binary_path,
                                              &model_mdata_) == false) {
    LOG_ERROR("Load ModelMetaData from:[" << model_mdata_binary_path
                                          << "] error!");
    return false;
  }

  return true;
}

void Scheduler::Start() {
  connecter_.Start();
}

void Scheduler::Stop() {
  connecter_.Stop();
}

int32_t Scheduler::TryJoin(const std::string& addr, bool* allow,
                           uint64_t* node_id, Router* old_router,
                           Router* new_router, bool* model_init,
                           ModelMetaData* model_mdata) {
  // Allow a Node join in that need all exist ps Node be kWorker status.
  LOG_INFO("A node:[" << addr << "] try to join.");

  {
    std::vector<uint64_t> ids;
    for (const auto& [k, v] : nodes_) {
      if (v.type == NodeType::kPs) {
        ids.emplace_back(v.id);
      }
    }

    HeartbeatRequest req;
    std::vector<HeartbeatResponse> replies;

    if (connecter_.Call(RPCFuncType::kHeartbeatType, ids, req, &replies) !=
        ErrorCode::kSuccess) {
      *allow = false;
      return ErrorCode::kSuccess;
    }

    for (auto& reply : replies) {
      if (reply.status != NodeStatus::kWork) {
        *allow = false;

        LOG_INFO(
            "One of Node status is't kWork, Reject the join node:" << addr);

        return ErrorCode::kSuccess;
      }
    }
  }

  LOG_INFO("Allow node:" << addr << " join.");

  uint64_t real_id = nodes_.size();
  while (nodes_.find(real_id) != nodes_.end()) {
    real_id++;
  }

  if (connecter_.AddConnect(real_id, addr) == false) {
    return ErrorCode::kConnectNodeError;
  }

  Node node;
  node.type = NodeType::kPs;
  node.id = real_id;
  node.addr = addr;

  nodes_.emplace(real_id, std::move(node));

  *allow = true;
  *node_id = real_id;
  *old_router = router_;

  ARGUMENT_CHECK(router_.Add(real_id, addr), "Router add error!");
  *new_router = router_;

  *model_init = model_init_;
  *model_mdata = model_mdata_;

  LOG_INFO("Old router:" << old_router->Str());
  LOG_INFO("New router:" << new_router->Str());

  // Notify other ps a new node join in.
  {
    // NotifyNodeJoinRequest
    std::vector<uint64_t> ids;
    for (const auto& [k, v] : nodes_) {
      if (v.type == NodeType::kPs && v.id != real_id) {
        ids.emplace_back(v.id);
      }
    }

    NotifyNodeJoinRequest req;
    req.joined_id = real_id;
    req.old_router = *old_router;
    req.new_router = *new_router;

    std::vector<NotifyNodeJoinResponse> replies;

    // We have to make sure the notify success.
    RPC_CALL(
        connecter_.Call(RPCFuncType::kNotifyNodeJoinType, ids, req, &replies));
  }

  return ErrorCode::kSuccess;
}

int32_t Scheduler::FetchRouter(Router* router) {
  *router = router_;

  return ErrorCode::kSuccess;
}

int32_t Scheduler::InitModel(
    const std::string& name, OptimType optim_type,
    const std::unordered_map<std::string, std::string>& optim_conf) {
  if (model_init_) {
    LOG_INFO("Model already initialized!");
    return ErrorCode::kSuccess;
  }

  model_mdata_.name = name;
  model_mdata_.optim_type = optim_type;
  model_mdata_.optim_conf = optim_conf;
  model_init_ = true;

  // Ask Ps to init model.
  {
    std::vector<uint64_t> ids;
    for (const auto& node : router_.nodes()) {
      ids.emplace_back(node.id);
    }

    CreateModelRequest req;
    req.name = name;
    req.optim_type = optim_type;
    req.optim_conf = optim_conf;

    std::vector<CreateModelResponse> replies;

    // We have to make sure the notify success.
    RPC_CALL(
        connecter_.Call(RPCFuncType::kCreateModelType, ids, req, &replies));
  }

  LOG_INFO("Init model:[" << name << "], optim_type:[" << optim_type
                          << "], optim_conf:[" << optim_conf << "]");

  return ErrorCode::kSuccess;
}

int32_t Scheduler::RegisterDenseTable(std::string name, const Tensor& val,
                                      uint64_t* table_id) {
  if (model_init_ == false) {
    return ErrorCode::kModelNotInitializedError;
  }

  for (const auto& [k, v] : model_mdata_.table_mdatas) {
    if (v.name == name) {
      if (v.table_type != TableType::kDense || v.shape != val.shape() ||
          v.element_type != val.element_type()) {
        return ErrorCode::kDenseTableUnCompatibleError;
      }

      *table_id = v.id;

      LOG_INFO("DenseTable:[" << name << "] already registered!");

      return ErrorCode::kSuccess;
    }
  }

  uint64_t real_id = model_mdata_.table_mdatas.size();
  while (model_mdata_.table_mdatas.find(real_id) !=
         model_mdata_.table_mdatas.end()) {
    real_id++;
  }

  // Select a ps node.
  uint64_t node_id = router_.Hit(utils::Hash(real_id));

  CreateDenseTableRequest req;
  req.table_id = real_id;
  req.name = name;
  req.val = val;

  CreateDenseTableResponse reply;

  RPC_CALL(connecter_.Call(RPCFuncType::kCreateDenseTableType, node_id, req,
                           &reply));

  TableMetaData table_mdata;
  table_mdata.id = real_id;
  table_mdata.name = name;
  table_mdata.table_type = TableType::kDense;
  table_mdata.element_type = val.element_type();
  table_mdata.shape = val.shape();

  model_mdata_.table_mdatas.emplace(real_id, std::move(table_mdata));

  *table_id = real_id;

  LOG_INFO("Register DenseTable:["
           << name << "], id:[" << real_id << "], ElementType:["
           << val.element_type().Name() << "], shape:" << val.shape().Str()
           << " in Ps:[" << node_id << "]");

  return ErrorCode::kSuccess;
}

int32_t Scheduler::RegisterSparseTable(
    std::string name, int64_t dimension, ElementType element_type,
    InitializerType init_type,
    const std::unordered_map<std::string, std::string>& init_conf,
    uint64_t* table_id) {
  if (model_init_ == false) {
    return ErrorCode::kModelNotInitializedError;
  }

  for (const auto& [k, v] : model_mdata_.table_mdatas) {
    if (v.name == name) {
      if (v.table_type != TableType::kSparse || v.dimension != dimension ||
          v.element_type != element_type || v.init_type != init_type) {
        return ErrorCode::kSparseTableUnCompatibleError;
      }

      *table_id = v.id;

      LOG_INFO("SparseTable:[" << name << "] already registered!");

      return ErrorCode::kSuccess;
    }
  }

  uint64_t real_id = model_mdata_.table_mdatas.size();
  while (model_mdata_.table_mdatas.find(real_id) !=
         model_mdata_.table_mdatas.end()) {
    real_id++;
  }

  {
    std::vector<uint64_t> ids;
    ids.reserve(router_.nodes().size());
    for (const auto& node : router_.nodes()) {
      ids.emplace_back(node.id);
    }

    CreateSparseTableRequest req;
    req.table_id = real_id;
    req.name = name;
    req.dimension = dimension;
    req.element_type = element_type;
    req.init_type = init_type;
    req.init_conf = init_conf;

    std::vector<CreateSparseTableResponse> replies;

    RPC_CALL(connecter_.Call(RPCFuncType::kCreateSparseTableType, ids, req,
                             &replies));
  }

  TableMetaData table_mdata;
  table_mdata.id = real_id;
  table_mdata.name = name;
  table_mdata.table_type = TableType::kSparse;
  table_mdata.dimension = dimension;
  table_mdata.element_type = element_type;
  table_mdata.init_type = init_type;
  table_mdata.init_conf = init_conf;

  model_mdata_.table_mdatas.emplace(real_id, std::move(table_mdata));

  *table_id = real_id;

  LOG_INFO("Register SparseTable:["
           << name << "], id:[" << real_id << "], dimension:[" << dimension
           << "], ElementType:[" << element_type.Name() << "], init_type:["
           << init_type << "], init_conf:[" << init_conf << "], in all Ps.");

  return ErrorCode::kSuccess;
}

int32_t Scheduler::TrySaveModel(bool* success) {
  if (model_init_ == false) {
    return ErrorCode::kModelNotInitializedError;
  }

  // Check Ps status.
  {
    std::vector<uint64_t> ids;
    for (const auto& [k, v] : nodes_) {
      if (v.type == NodeType::kPs) {
        ids.emplace_back(v.id);
      }
    }

    HeartbeatRequest req;
    std::vector<HeartbeatResponse> replies;

    if (connecter_.Call(RPCFuncType::kHeartbeatType, ids, req, &replies) !=
        ErrorCode::kSuccess) {
      *success = false;
      return ErrorCode::kSuccess;
    }

    for (auto& reply : replies) {
      if (reply.status != NodeStatus::kWork) {
        *success = false;

        LOG_INFO("One of Ps status is not kWork, reject save model.");

        return ErrorCode::kSuccess;
      }
    }
  }

  std::vector<uint64_t> ids;
  ids.reserve(router_.nodes().size());
  for (const auto& node : router_.nodes()) {
    ids.emplace_back(node.id);
  }

  NotifySaveModelRequest req;
  req.model_mdata = model_mdata_;

  std::vector<NotifySaveModelResponse> replies;

  RPC_CALL(
      connecter_.Call(RPCFuncType::kNotifySaveModelType, ids, req, &replies));

  *success = true;

  return ErrorCode::kSuccess;
}

int32_t Scheduler::TryLoadModel(const std::string& load_dir, bool* success) {
  LOG_INFO("Try load model from:[" << load_dir << "].");

  if (model_init_ == true) {
    LOG_WARNING("Scheduler already initialize the model, name:["
                << model_mdata_.name << "].");
    *success = true;
    return ErrorCode::kSuccess;
  }

  // Check Ps status.
  {
    std::vector<uint64_t> ids;
    for (const auto& [k, v] : nodes_) {
      if (v.type == NodeType::kPs) {
        ids.emplace_back(v.id);
      }
    }

    HeartbeatRequest req;
    std::vector<HeartbeatResponse> replies;

    if (connecter_.Call(RPCFuncType::kHeartbeatType, ids, req, &replies) !=
        ErrorCode::kSuccess) {
      *success = false;
      return ErrorCode::kSuccess;
    }

    for (auto& reply : replies) {
      if (reply.status != NodeStatus::kWork) {
        *success = false;

        LOG_INFO("One of Ps status is not kWork, reject load model.");
        return ErrorCode::kSuccess;
      }
    }
  }

  if (LoadModelMetaData(load_dir) == false) {
    LOG_ERROR("Scheduler load ModelMetaData error!");
    return ErrorCode::kLoadModelError;
  }

  std::vector<uint64_t> ids;
  ids.reserve(router_.nodes().size());
  for (const auto& node : router_.nodes()) {
    ids.emplace_back(node.id);
  }

  // Tell ps to load model.
  NotifyLoadModelRequest req;
  req.load_dir = load_dir;
  std::vector<NotifyLoadModelResponse> replies;

  RPC_CALL(
      connecter_.Call(RPCFuncType::kNotifyLoadModelType, ids, req, &replies));

  *success = true;

  // Set flag.
  model_init_ = true;

  return ErrorCode::kSuccess;
}

int32_t Scheduler::IsAllPsWorking(bool* yes) {
  std::vector<uint64_t> ids;
  for (const auto& [k, v] : nodes_) {
    if (v.type == NodeType::kPs) {
      ids.emplace_back(v.id);
    }
  }

  *yes = true;

  HeartbeatRequest req;
  std::vector<HeartbeatResponse> replies;

  if (connecter_.Call(RPCFuncType::kHeartbeatType, ids, req, &replies) !=
      ErrorCode::kSuccess) {
    *yes = false;
  } else {
    for (const auto& reply : replies) {
      if (reply.status != NodeStatus::kWork) {
        *yes = false;
      }
    }
  }

  return ErrorCode::kSuccess;
}

}  // namespace kraken
