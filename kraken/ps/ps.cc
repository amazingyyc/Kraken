#include "ps/ps.h"

#include <thread>

#include "common/error_code.h"
#include "common/exception.h"
#include "common/info.h"
#include "common/log.h"
#include "protocol/fetch_model_meta_data_prot.h"
#include "protocol/rpc_func_type.h"
#include "protocol/try_join_prot.h"
#include "ps/dense_table.h"
#include "ps/sparse_table.h"
#include "ps/transfer.h"
#include "rpc/indep_connecter.h"

namespace kraken {

Ps::Ps(const std::string& addr, const std::string& s_addr)
    : task_que_(1),
      addr_(addr),
      s_addr_(s_addr),
      status_(NodeStatus::kInit),
      node_id_(0),
      model_init_(false) {
}

void Ps::Clean() {
  LOG_INFO("Begin to clean old data.");

  uint64_t node_id;
  Router router;

  // Copy to local.
  {
    std::shared_lock<std::shared_mutex> _(mu_);
    node_id = node_id_;
    router = router_;
  }

  // Clean DenseTable.
  {
    // For DenseTable the table num is not huge. so we will lock it only once.
    std::unique_lock<std::shared_mutex> _(model_mu_);

    uint64_t table_id = 0;

    do {
      auto it = tables_.FindGreaterOrEqual(table_id);

      if (it.Valid() == false) {
        break;
      }

      if (it.value()->type() != TableType::kDense) {
        table_id = it.key() + 1;
        continue;
      }

      if (router.Hit(utils::Hash(it.key())) == node_id) {
        table_id = it.key() + 1;
        continue;
      }

      LOG_INFO("Clean DensTable:[" << it.value()->name() << "], id:["
                                   << it.key() << "]");

      // Becareful we have update the table_id before Remove when call Remove
      // the 'it' will become invalid.
      table_id = it.key() + 1;
      tables_.Remove(it);
    } while (true);
  }

  // Clean SparseTable.
  {
    uint64_t table_id = 0;

    while (true) {
      std::shared_lock<std::shared_mutex> _(model_mu_);

      auto it = tables_.FindGreaterOrEqual(table_id);
      if (it.Valid() == false) {
        break;
      }

      if (it.value()->type() != TableType::kSparse) {
        // Jump to next SparseTable.
        table_id = it.key() + 1;
        continue;
      }

      // Hit a new SparseTable update.
      if (table_id != it.key()) {
        table_id = it.key();
      }

      SparseTable* table = (SparseTable*)it.value().get();
      auto* parallel_vals = table->vals();

      size_t remove_c = 0;

      for (size_t slot = 0; slot < parallel_vals->slot_count(); ++slot) {
        auto skip_list_h = parallel_vals->UniqueSkipListHandler(slot);

        uint64_t sparse_id = 0;

        while (true) {
          auto sit = skip_list_h.skip_list.FindGreaterOrEqual(sparse_id);
          if (sit.Valid() == false) {
            break;
          }

          // Must update sparse_id before call Remove or the 'it' will be
          // invalid.
          sparse_id = sit.key() + 1;

          if (router.Hit(utils::Hash(it.key(), sit.key())) != node_id) {
            skip_list_h.skip_list.Remove(sit);
            remove_c++;
          }
        }
      }

      LOG_INFO("Clean SparseValues from SparseTable:["
               << table_id << "], count:[" << remove_c << "]");

      table_id = it.key() + 1;
    }
  }

  LOG_INFO("Finish clean old data.");
}

// Transfer data to new node.
void Ps::Transfer(uint64_t target_id) {
  LOG_INFO("Begin to transfer data to node:[" << target_id << "]");

  Router router;
  uint64_t node_id;

  // Copy to local.
  {
    std::shared_lock<std::shared_mutex> _(mu_);
    router = router_;
    node_id = node_id_;
  }

  Router::Node target_node;
  ARGUMENT_CHECK(router.node(target_id, &target_node),
                 "Router not include node:" << target_id);

  // Create a connecter.
  kraken::Transfer transfer(target_node.id, target_node.name,
                            CompressType::kSnappy);

  // Transfer DenseTable.
  {
    uint64_t table_id_offset = 0;

    uint64_t table_id = 0;
    std::string name;
    Value val;

    while (true) {
      {
        std::shared_lock<std::shared_mutex> _(model_mu_);

        auto it = tables_.FindGreaterOrEqual(table_id_offset);
        if (it.Valid() == false) {
          break;
        }

        if (it.value()->type() != TableType::kDense) {
          table_id_offset = it.key() + 1;
          continue;
        }

        if (router.Hit(utils::Hash(it.key())) != target_id) {
          table_id_offset = it.key() + 1;
          continue;
        }

        // Send to target Node. If the DenseTable need send to target Node. It
        // means it's not belong to this Node, So it Ok use shared locker.
        DenseTable* table = (DenseTable*)it.value().get();

        // If a dense table need transfer to other node it means it will never
        // modify on this node. So shallow copy is OK.
        auto l = table->shared_handler();
        table_id = table->id();
        name = table->name();
        val = table->val();

        table_id_offset = it.key() + 1;
      }

      RPC_CALL(transfer.TransferDenseTable(node_id, table_id, name, val));

      // At here the mutex for tables_ has been released. So when
      // send data we will not block other thread to modify it.
      LOG_INFO("Transfer DenseTable:[" << name << "], id:[" << table_id
                                       << "] to node:[" << target_id << "]");
    }
  }

  // Transfer SparseTable MetaData.
  {
    uint64_t table_id_offset = 0;

    uint64_t table_id;
    std::string name;
    int64_t dimension;
    ElementType element_type;
    InitializerType init_type;
    std::unordered_map<std::string, std::string> init_conf;

    while (true) {
      {
        std::shared_lock<std::shared_mutex> _(model_mu_);

        auto it = tables_.FindGreaterOrEqual(table_id_offset);
        if (it.Valid() == false) {
          break;
        }

        if (it.value()->type() != TableType::kSparse) {
          table_id_offset = it.key() + 1;
          continue;
        }

        SparseTable* table = (SparseTable*)it.value().get();

        // Copy.
        table_id = table->id();
        name = table->name();
        dimension = table->dimension();
        element_type = table->element_type();
        init_type = table->initializer()->type();
        init_conf = table->initializer()->conf();

        table_id_offset += 1;
      }

      RPC_CALL(transfer.TransferSparseMetaData(node_id, table_id, name,
                                               dimension, element_type,
                                               init_type, init_conf));

      LOG_INFO("Transfer SparseTable:[" << table_id << "] MetaData to node:["
                                        << target_id << "]");
    }
  }

  // Transfer SparseTable.
  {
    uint64_t table_id_offset = 0;
    size_t slot_id_offset = 0;
    uint64_t sparse_id_offset = 0;

    const size_t step = 1024;
    uint64_t table_id = 0;

    std::vector<uint64_t> sparse_ids;
    sparse_ids.reserve(step);

    std::vector<Value> vals;
    vals.reserve(step);

    while (true) {
      sparse_ids.clear();
      vals.clear();

      {
        std::shared_lock<std::shared_mutex> _(model_mu_);

        auto it = tables_.FindGreaterOrEqual(table_id_offset);
        if (it.Valid() == false) {
          break;
        }

        if (it.value()->type() != TableType::kSparse) {
          // Jump to next SparseTable.
          table_id_offset = it.key() + 1;
          slot_id_offset = 0;
          sparse_id_offset = 0;

          continue;
        }

        // Hit a new SparseTable update.
        if (table_id_offset != it.key()) {
          table_id_offset = it.key();
          slot_id_offset = 0;
          sparse_id_offset = 0;
        }

        SparseTable* table = (SparseTable*)it.value().get();
        auto* parallel_vals = table->vals();

        // Jump to next SparseTable.
        if (slot_id_offset >= parallel_vals->slot_count()) {
          table_id_offset = it.key() + 1;
          slot_id_offset = 0;
          sparse_id_offset = 0;

          continue;
        }

        // Store the real table id.
        table_id = it.key();

        auto skip_list_h = parallel_vals->SharedSkipListHandler(slot_id_offset);
        for (size_t i = 0; i < step; ++i) {
          auto sit = skip_list_h.skip_list.FindGreaterOrEqual(sparse_id_offset);

          // Jump to next slot.
          if (sit.Valid() == false) {
            slot_id_offset++;
            sparse_id_offset = 0;
            break;
          }

          if (router.Hit(utils::Hash(table_id, sit.key())) == target_id) {
            sparse_ids.emplace_back(sit.key());
            vals.emplace_back(sit.value());
          }

          sparse_id_offset = sit.key() + 1;
        }
      }

      if (sparse_ids.empty() == false) {
        RPC_CALL(
            transfer.TransferSparseValues(node_id, table_id, sparse_ids, vals));

        LOG_INFO("Transfer SparseValues of SparseTable:["
                 << table_id << "] to node:[" << target_id << "], count:["
                 << sparse_ids.size() << "]");
      }
    }
  }

  LOG_INFO("Finish transfer data to node:[" << target_id << "]");

  // Clean the data that not belong to this Node.
  Clean();

  // Notify the target node that we already finish trasfer.
  RPC_CALL(transfer.NotifyFinishTransfer(node_id));

  std::unique_lock<std::shared_mutex> _(mu_);
  status_ = status_ & (~NodeStatus::kTransfer);

  LOG_INFO("Finish transfer/clean, status become:[" << NodeStatusStr(status_)
                                                    << "]");
}

void Ps::Start() {
  // The Ps start.
  // 1: Connect to scheduler.
  // 2: Get old_router and new_router. Status become: kWork & kProxy.
  // 3: Response/Proxy worker request.
  // 4: Wait Proxy Node finish transfer data. Status become to: kWork.

  // Try to join the Cluster. Maybe join fail (a node is join/leave will
  // let other node join fail).
  // We have to lock the mutex avoid some other thread change the status.
  std::unique_lock<std::shared_mutex> lock(mu_);

  // Connect to scheduler.
  LOG_INFO("Try connect to scheduler:" << s_addr_);
  IndepConnecter s_connecter(s_addr_, CompressType::kNo);
  s_connecter.Start();

  bool allow = false;
  int64_t sleep_s = 10;
  do {
    LOG_INFO("Try to join in cluster.");

    TryJoinRequest req;
    req.addr = addr_;
    TryJoinResponse reply;

    RPC_CALL(s_connecter.Call(RPCFuncType::kTryJoinType, req, &reply));

    allow = reply.allow;

    if (allow) {
      node_id_ = reply.node_id;
      old_router_ = reply.old_router;
      router_ = reply.new_router;
    } else {
      LOG_INFO("Joint fail will sleep:" << sleep_s << "s");

      // Sleep.
      std::this_thread::sleep_for(std::chrono::seconds(sleep_s));
      sleep_s += sleep_s / 2;
    }
  } while (allow == false);

  LOG_INFO("Join cluster success.");
  LOG_INFO("Old router:" << old_router_.Str());
  LOG_INFO("New router:" << router_.Str());

  // Fetch ModelMetaData and init model.
  {
    FetchModelMetaDataRequest req;
    FetchModelMetaDataResponse reply;

    RPC_CALL(
        s_connecter.Call(RPCFuncType::kFetchModelMetaDataType, req, &reply));

    if (reply.model_init) {
      // Init the model.
      std::unique_lock<std::shared_mutex> _(model_mu_);

      optim_ = Optim::Create(reply.model_mdata.optim_type,
                             reply.model_mdata.optim_conf);
      ARGUMENT_CHECK(optim_ != nullptr, "Unsupport Optim type.");

      model_name_ = reply.model_mdata.name;
      model_init_ = true;

      LOG_INFO("Init Model, optim_type:"
               << (uint32_t)reply.model_mdata.optim_type
               << ", optim conf:" << reply.model_mdata.optim_conf);
    } else {
      LOG_INFO("Model not initialized.")
    }
  }

  // For now This node has been join the cluster success.
  // So this node will proxy the new request and wait other Node finish transfer
  // data.
  if (router_.nodes().size() == 1) {
    // means this is the first node in this cluster.
    // So we donot need to create proxy.
    status_ = NodeStatus::kWork;

    LOG_INFO("I'm the first node, status:[" << NodeStatusStr(status_) << "]");
  } else {
    Router::Node cur_node;
    ARGUMENT_CHECK(router_.node(node_id_, &cur_node),
                   "Router not include node:" << node_id_);

    // Try to create proxy.
    std::unordered_set<uint64_t> proxy_ids;
    for (auto hash : cur_node.vnode_list) {
      uint64_t hit_node_id = old_router_.Hit(hash);
      if (hit_node_id != node_id_) {
        proxy_ids.emplace(hit_node_id);
      }
    }

    proxy_.reset(new Proxy(proxy_ids, old_router_, CompressType::kSnappy));

    // Store event wait proxy node finish transfer.
    events_[EventType::kProxyFinishTransfer] = proxy_ids;

    status_ = NodeStatus::kWork | NodeStatus::kProxy;

    LOG_INFO("Will proxy nodes:" << proxy_ids << ", status:["
                                 << NodeStatusStr(status_) << "]");
  }

  s_connecter.Stop();
  LOG_INFO("Disconnect from scheduler.");
}

int32_t Ps::Heartbeat(uint32_t* status) {
  std::shared_lock<std::shared_mutex> _(mu_);

  *status = status_;

  return ErrorCode::kSuccess;
}

int32_t Ps::NotifyFinishTransfer(uint64_t node_id) {
  std::unique_lock<std::shared_mutex> _(mu_);

  LOG_INFO("Got notificaion from node:[" << node_id << "] finish transfer.");

  if (!(status_ & NodeStatus::kProxy)) {
    return ErrorCode::kNodeStatusError;
  }

  if (events_.find(EventType::kProxyFinishTransfer) == events_.end()) {
    return ErrorCode::kUnSupportEventTypeError;
  }

  if (events_[EventType::kProxyFinishTransfer].find(node_id) ==
      events_[EventType::kProxyFinishTransfer].end()) {
    return ErrorCode::kUnSupportEventTypeError;
  }

  events_[EventType::kProxyFinishTransfer].erase(node_id);

  if (events_[EventType::kProxyFinishTransfer].empty()) {
    events_.erase(EventType::kProxyFinishTransfer);

    // Change status.
    proxy_.reset();
    status_ = status_ & (~NodeStatus::kProxy);

    LOG_INFO("All node finish transfer data, stop proxy, status become:["
             << NodeStatusStr(status_) << "]");
  }

  return ErrorCode::kSuccess;
}

int32_t Ps::NotifyNodeJoin(uint64_t joined_id, const Router& old_router,
                           const Router& new_router) {
  std::unique_lock<std::shared_mutex> _(mu_);

  // A node is allowed to join must need all node status is kWork.
  if (status_ != NodeStatus::kWork) {
    LOG_INFO("Status is not kWork, So not allowed join new node.");
    return ErrorCode::kNodeStatusInappropriateError;
  }

  LOG_INFO("Got notification: node:" << joined_id
                                     << " join in, status:" << status_);
  LOG_INFO("Old router:" << old_router.Str());
  LOG_INFO("New router:" << new_router.Str());

  ARGUMENT_CHECK(old_router == router_, "The router is error!");

  old_router_ = old_router;
  router_ = new_router;

  Router::Node node;
  Router::Node target_node;

  ARGUMENT_CHECK(router_.node(node_id_, &node),
                 "Cannot find node:" << node_id_);
  ARGUMENT_CHECK(router_.node(joined_id, &target_node),
                 "Cannot find node:" << joined_id);

  // Check whether we need transfer the data to the new node or not.
  bool need_transfer = false;
  for (auto h : target_node.vnode_list) {
    if (node_id_ == old_router_.Hit(h)) {
      need_transfer = true;
      break;
    }
  }

  if (need_transfer) {
    // Modify status.
    status_ = NodeStatus::kWork | NodeStatus::kTransfer;

    // Async task to transfer data.
    auto task = [this, joined_id] { this->Transfer(joined_id); };

    task_que_.Enque(std::move(task));

    LOG_INFO("Affect me will transfer data to node:["
             << joined_id << "], status:[" << NodeStatusStr(status_) << "]");
  } else {
    LOG_INFO("New node not affect me, status:[" << NodeStatusStr(status_)
                                                << "]");
  }

  return ErrorCode::kSuccess;
}

int32_t Ps::CreateModel(
    std::string name, OptimType optim_type,
    const std::unordered_map<std::string, std::string>& optim_conf) {
  // Check status.
  std::shared_lock<std::shared_mutex> l(mu_);
  if (!(status_ & NodeStatus::kWork)) {
    return ErrorCode::kNodeStatusError;
  }

  std::unique_lock<std::shared_mutex> ll(model_mu_);

  if (model_init_) {
    // (TODO) check model config.
    LOG_INFO("Model:" << model_name_ << " already initied!");
    return ErrorCode::kSuccess;
  }

  optim_ = Optim::Create(optim_type, optim_conf);
  if (optim_ == nullptr) {
    return ErrorCode::kUnSupportOptimTypeError;
  }

  model_name_ = name;
  model_init_ = true;

  LOG_INFO("Create model:" << name << ", optim_type:" << optim_type
                           << ", optim_conf:" << optim_conf);

  return ErrorCode::kSuccess;
}

int32_t Ps::CreateDenseTable(uint64_t table_id, std::string name,
                             const Tensor& val) {
  std::shared_lock<std::shared_mutex> l(mu_);
  if (!(status_ & NodeStatus::kWork)) {
    return ErrorCode::kNodeStatusError;
  }

  std::unique_lock<std::shared_mutex> ll(model_mu_);

  auto it = tables_.Find(table_id);
  if (it.Valid()) {
    // (TODO) check DenseTabel conf.
    LOG_INFO("DenseTable:" << name << " already created!");
    return ErrorCode::kSuccess;
  }

  std::unique_ptr<DenseTable> table(new DenseTable(table_id, name, val));

  tables_.Insert(table_id, std::move(table));

  LOG_INFO("Create DenseTable:" << name << ", id:" << table_id
                                << ", ElementType:" << val.element_type().Name()
                                << ", shape:" << val.shape().Str());

  return ErrorCode::kSuccess;
}

int32_t Ps::CreateSparseTable(
    uint64_t table_id, std::string name, int64_t dimension,
    ElementType element_type, InitializerType init_type,
    const std::unordered_map<std::string, std::string>& init_conf) {
  std::shared_lock<std::shared_mutex> l(mu_);
  if (!(status_ & NodeStatus::kWork)) {
    return ErrorCode::kNodeStatusError;
  }

  std::unique_lock<std::shared_mutex> ll(model_mu_);

  if (dimension <= 0) {
    return ErrorCode::kSparseDimensionError;
  }

  auto it = tables_.Find(table_id);
  if (it.Valid()) {
    // (TODO) check SparseTable conf.
    LOG_INFO("SparseTable:" << name << " already created!");
    return ErrorCode::kSuccess;
  }

  std::unique_ptr<Initializer> initializer =
      Initializer::Create(init_type, init_conf);
  if (initializer == nullptr) {
    return ErrorCode::kUnSupportInitializerTypeError;
  }

  std::unique_ptr<SparseTable> table(new SparseTable(
      table_id, name, dimension, element_type, std::move(initializer)));

  tables_.Insert(table_id, std::move(table));

  LOG_INFO("Create SparseTable:"
           << name << ", id:" << table_id << ", dimension:" << dimension
           << ", ElementType:" << element_type.Name()
           << ", init_type:" << init_type << ", init_conf:" << init_conf);

  return ErrorCode::kSuccess;
}

int32_t Ps::TransferDenseTable(uint64_t from_node_id, uint64_t table_id,
                               const std::string& name, const Value& value) {
  std::shared_lock<std::shared_mutex> l(mu_);
  if (status_ != (NodeStatus::kWork | NodeStatus::kProxy)) {
    return ErrorCode::kNodeStatusError;
  }

  std::unique_lock<std::shared_mutex> ll(model_mu_);

  // If exist just return success.
  auto it = tables_.Find(table_id);
  if (it.Valid()) {
    LOG_INFO("Transfered DenseTable:" << table_id << " from node:"
                                      << from_node_id << " already exist!");
    return ErrorCode::kSuccess;
  }

  std::unique_ptr<DenseTable> table(new DenseTable(table_id, name, value));
  tables_.Insert(table_id, std::move(table));

  LOG_INFO("Get Transfered DenseTable:[" << table_id << "] from node:["
                                         << from_node_id << "]");

  return ErrorCode::kSuccess;
}

int32_t Ps::TransferSparseMetaData(
    uint64_t from_node_id, uint64_t table_id, std::string name,
    int64_t dimension, ElementType element_type, InitializerType init_type,
    const std::unordered_map<std::string, std::string>& init_conf) {
  std::shared_lock<std::shared_mutex> l(mu_);
  if (status_ != (NodeStatus::kWork | NodeStatus::kProxy)) {
    return ErrorCode::kNodeStatusError;
  }

  std::unique_lock<std::shared_mutex> ll(model_mu_);

  // If exist just return success.
  auto it = tables_.Find(table_id);
  if (it.Valid()) {
    return ErrorCode::kSuccess;
  }

  if (dimension <= 0) {
    return ErrorCode::kSparseDimensionError;
  }

  std::unique_ptr<Initializer> initializer =
      Initializer::Create(init_type, init_conf);
  if (initializer == nullptr) {
    return ErrorCode::kUnSupportInitializerTypeError;
  }

  std::unique_ptr<SparseTable> table(new SparseTable(
      table_id, name, dimension, element_type, std::move(initializer)));

  tables_.Insert(table_id, std::move(table));

  LOG_INFO("Get Transfered SparseTableMetaData:[" << table_id << "] from node:["
                                                  << from_node_id << "]");

  return ErrorCode::kSuccess;
}

int32_t Ps::TransferSparseValues(uint64_t from_node_id, uint64_t table_id,
                                 const std::vector<uint64_t>& sparse_ids,
                                 const std::vector<Value>& values) {
  assert(sparse_ids.size() == values.size());

  std::shared_lock<std::shared_mutex> l(mu_);
  if (status_ != (NodeStatus::kWork | NodeStatus::kProxy)) {
    return ErrorCode::kNodeStatusError;
  }

  std::shared_lock<std::shared_mutex> ll(model_mu_);

  auto it = tables_.Find(table_id);
  if (it.Valid() == false || it.value()->type() != TableType::kSparse) {
    return ErrorCode::kTableNotExistError;
  }

  SparseTable* table = (SparseTable*)it.value().get();
  table->vals()->Insert(sparse_ids, values);

  LOG_INFO("Get Transfered SparseValues of SparseTable:["
           << table_id << "], count:[" << sparse_ids.size() << "], from node:["
           << from_node_id << "]");

  return ErrorCode::kSuccess;
}

int32_t Ps::TryFetchDenseTable(uint64_t table_id, std::string* name,
                               Value* value) {
  std::shared_lock<std::shared_mutex> ll(model_mu_);

  auto it = tables_.Find(table_id);
  if (it.Valid() == false || it.value()->type() != TableType::kDense) {
    return ErrorCode::kTableNotExistError;
  }

  DenseTable* table = (DenseTable*)it.value().get();

  auto h = table->shared_handler();

  *name = table->name();
  *value = table->val().Clone();  // must clone

  return ErrorCode::kSuccess;
}

int32_t Ps::TryCombineFetchDenseTable(const std::vector<uint64_t>& table_ids,
                                      std::vector<uint64_t>* exist_table_ids,
                                      std::vector<std::string>* names,
                                      std::vector<Value>* values) {
  std::shared_lock<std::shared_mutex> ll(model_mu_);

  size_t count = table_ids.size();

  exist_table_ids->reserve(count);
  names->reserve(count);
  values->reserve(count);

  for (size_t i = 0; i < count; ++i) {
    auto it = tables_.Find(table_ids[i]);
    if (it.Valid() == false || it.value()->type() != TableType::kDense) {
      continue;
    }

    DenseTable* table = (DenseTable*)it.value().get();

    auto h = table->shared_handler();

    exist_table_ids->emplace_back(table_ids[i]);
    names->emplace_back(table->name());
    values->emplace_back(table->val().Clone());  // must clone
  }

  return ErrorCode::kSuccess;
}

int32_t Ps::TryFetchSparseMetaData(
    uint64_t table_id, std::string* name, int64_t* dimension,
    ElementType* element_type, InitializerType* init_type,
    std::unordered_map<std::string, std::string>* init_conf) {
  std::shared_lock<std::shared_mutex> ll(model_mu_);

  auto it = tables_.Find(table_id);
  if (it.Valid() == false || it.value()->type() != TableType::kSparse) {
    return ErrorCode::kTableNotExistError;
  }

  SparseTable* table = (SparseTable*)it.value().get();

  *name = table->name();
  *dimension = table->dimension();
  *element_type = table->element_type();
  *init_type = table->initializer()->type();
  *init_conf = table->initializer()->conf();

  return ErrorCode::kSuccess;
}

int32_t Ps::TryFetchSparseValues(uint64_t table_id,
                                 const std::vector<uint64_t>& sparse_ids,
                                 std::vector<uint64_t>* exist_sparse_ids,
                                 std::vector<Value>* values) {
  std::shared_lock<std::shared_mutex> ll(model_mu_);

  auto it = tables_.Find(table_id);
  if (it.Valid() == false || it.value()->type() != TableType::kSparse) {
    return ErrorCode::kTableNotExistError;
  }

  SparseTable* table = (SparseTable*)it.value().get();

  exist_sparse_ids->reserve(sparse_ids.size());
  values->reserve(sparse_ids.size());

  auto* parallel_vals = table->vals();

  std::unordered_map<size_t, std::vector<size_t>> slot_idx_map;
  slot_idx_map.reserve(parallel_vals->slot_count());

  for (size_t i = 0; i < sparse_ids.size(); ++i) {
    slot_idx_map[parallel_vals->HitSlot(sparse_ids[i])].emplace_back(
        sparse_ids[i]);
  }

  for (auto& [slot, v] : slot_idx_map) {
    // Lock the slot.
    auto h = parallel_vals->UniqueSkipListHandler(slot);

    for (auto sparse_id : v) {
      auto it = h.skip_list.Find(sparse_id);

      if (it.Valid()) {
        exist_sparse_ids->emplace_back(sparse_id);
        values->emplace_back(it.value().Clone());
      }
    }
  }

  return ErrorCode::kSuccess;
}

}  // namespace kraken
