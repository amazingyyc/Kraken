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
#include "rpc/indep_connecter.h"

namespace kraken {

Ps::Ps(const std::string& addr, const std::string& s_addr,
       const std::string& saved_dir, size_t max_save_count)
    : task_que_(1),
      addr_(addr),
      s_addr_(s_addr),
      checkpoint_exec_(saved_dir, max_save_count),
      status_(NodeStatus::kInit),
      node_id_(0),
      model_init_(false) {
}

void Ps::CleanDenseTables() {
  LOG_INFO("Begin to clean DenseTables.");

  uint64_t node_id;
  Router router;

  // Copy to local.
  {
    std::shared_lock<std::shared_mutex> l(mu_);
    node_id = node_id_;
    router = router_;
  }

  std::unique_lock<std::shared_mutex> ll(model_mu_);

  auto it = tables_.Begin();
  while (it.Valid()) {
    if (it.value()->type() != TableType::kDense ||
        router.Hit(utils::Hash(it.key())) == node_id) {
      it.Next();
    } else {
      LOG_INFO("Clean DensTable:[" << it.value()->name() << "], id:["
                                   << it.key() << "]");

      it = tables_.Remove(it);
    }
  }

  LOG_INFO("Finish clean DenseTables.");
}

void Ps::CleanSparseTables() {
  LOG_INFO("Begin to clean SparseTables.");

  uint64_t node_id;
  Router router;

  // Copy to local.
  {
    std::shared_lock<std::shared_mutex> l(mu_);
    node_id = node_id_;
    router = router_;
  }

  uint64_t table_id = 0;

  while (true) {
    std::shared_lock<std::shared_mutex> ll(model_mu_);

    auto it = tables_.FindGreaterOrEqual(table_id);
    if (it.Valid() == false) {
      break;
    }

    if (it.value()->type() != TableType::kSparse) {
      // Jump to next SparseTable.
      table_id = it.key() + 1;
      continue;
    }

    // Update.
    table_id = it.key();

    SparseTable* table = (SparseTable*)it.value().get();
    auto* parallel_vals = table->mutable_vals();

    size_t remove_c = 0;

    for (size_t slot = 0; slot < parallel_vals->slot_count(); ++slot) {
      auto h = parallel_vals->UniqueSkipListHandler(slot);

      auto sit = h.skip_list.Begin();
      while (sit.Valid()) {
        if (router.Hit(utils::Hash(it.key(), sit.key())) != node_id) {
          sit = h.skip_list.Remove(sit);
          remove_c++;
        } else {
          sit.Next();
        }
      }
    }

    LOG_INFO("Clean SparseValues from SparseTable:[" << table_id << "], count:["
                                                     << remove_c << "]");

    table_id = it.key() + 1;
  }

  LOG_INFO("Finish clean SparseTables.");
}

void Ps::CleanTables() {
  CleanDenseTables();
  CleanSparseTables();
}

void Ps::TransferDenseTableTo(const Transfer& transfer, uint64_t target_id) {
  Router router;
  uint64_t node_id;

  // Copy to local.
  {
    std::shared_lock<std::shared_mutex> l(mu_);
    router = router_;
    node_id = node_id_;
  }

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

      if (it.value()->type() != TableType::kDense ||
          router.Hit(utils::Hash(it.key())) != target_id) {
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

    // At here the mutex for tables_ has been released. So when
    // send data we will not block other thread to modify it.
    RPC_CALL(transfer.TransferDenseTable(node_id, table_id, name, val));

    LOG_INFO("Transfer DenseTable:[" << name << "], id:[" << table_id
                                     << "] to node:[" << target_id << "]");
  }
}

void Ps::TransferSparseMetaDataTo(const Transfer& transfer,
                                  uint64_t target_id) {
  uint64_t node_id;

  // Copy to local.
  {
    std::shared_lock<std::shared_mutex> l(mu_);
    node_id = node_id_;
  }

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

      // Jump to next SparseTable.
      table_id_offset = it.key() + 1;
    }

    RPC_CALL(transfer.TransferSparseMetaData(node_id, table_id, name, dimension,
                                             element_type, init_type,
                                             init_conf));

    LOG_INFO("Transfer SparseTable:[" << table_id << "] MetaData to node:["
                                      << target_id << "]");
  }
}

void Ps::TransferSparseValuesTo(const Transfer& transfer, uint64_t target_id) {
  Router router;
  uint64_t node_id;

  // Copy to local.
  {
    std::shared_lock<std::shared_mutex> l(mu_);
    router = router_;
    node_id = node_id_;
  }

  uint64_t table_id_offset = 0;
  size_t slot_id_offset = 0;
  uint64_t sparse_id_offset = 0;

  const size_t step = 1024;

  uint64_t table_id = 0;

  std::vector<uint64_t> sparse_ids;
  sparse_ids.reserve(step);

  std::vector<Value> sparse_vals;
  sparse_vals.reserve(step);

  while (true) {
    sparse_ids.clear();
    sparse_vals.clear();

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
      auto* parallel_vals = table->mutable_vals();

      // Jump to next SparseTable.
      if (slot_id_offset >= parallel_vals->slot_count()) {
        table_id_offset = it.key() + 1;
        slot_id_offset = 0;
        sparse_id_offset = 0;

        continue;
      }

      // Store the real table id.
      table_id = it.key();

      auto h = parallel_vals->SharedSkipListHandler(slot_id_offset);
      auto sit = h.skip_list.FindGreaterOrEqual(sparse_id_offset);

      while (sparse_ids.size() < step && sit.Valid()) {
        if (router.Hit(utils::Hash(it.key(), sit.key())) == target_id) {
          sparse_ids.emplace_back(sit.key());
          sparse_vals.emplace_back(sit.value());
        }

        sit.Next();
      }

      if (sit.Valid() == false) {
        slot_id_offset++;
        sparse_id_offset = 0;
      } else {
        sparse_id_offset = sit.key();
      }
    }

    if (sparse_ids.empty() == false) {
      RPC_CALL(transfer.TransferSparseValues(node_id, table_id, sparse_ids,
                                             sparse_vals));

      LOG_INFO("Transfer SparseValues of SparseTable:["
               << table_id << "] to node:[" << target_id << "], count:["
               << sparse_ids.size() << "]");
    }
  }
}

// Transfer data to new node.
void Ps::TransferTo(uint64_t target_id) {
  LOG_INFO("Begin to transfer data to node:[" << target_id << "]");

  Router router;
  uint64_t node_id;

  // Copy to local.
  {
    std::shared_lock<std::shared_mutex> l(mu_);
    router = router_;
    node_id = node_id_;
  }

  const Router::Node& target_node = router.node(target_id);

  // Create a connecter.
  kraken::Transfer transfer(target_node.id, target_node.name,
                            CompressType::kSnappy);

  TransferDenseTableTo(transfer, target_id);
  TransferSparseMetaDataTo(transfer, target_id);
  TransferSparseValuesTo(transfer, target_id);

  LOG_INFO("Finish transfer data to node:[" << target_id << "]");

  // Clean the data that not belong to this Node.
  CleanTables();

  // Notify the target node that we already finish trasfer.
  RPC_CALL(transfer.NotifyFinishTransfer(node_id));

  std::unique_lock<std::shared_mutex> l(mu_);
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
  std::unique_lock<std::shared_mutex> l(mu_);
  std::unique_lock<std::shared_mutex> ll(model_mu_);

  // Connect to scheduler.
  LOG_INFO("Try connect to scheduler:[" << s_addr_ << "]");
  IndepConnecter s_connecter(s_addr_, CompressType::kNo);
  s_connecter.Start();

  Router old_router;
  ModelMetaData model_mdata;

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
      old_router = reply.old_router;
      model_mdata = std::move(reply.model_mdata);

      node_id_ = reply.node_id;
      router_ = reply.new_router;
      model_init_ = reply.model_init;
    } else {
      LOG_INFO("Join fail will sleep:" << sleep_s << "s");

      // Sleep.
      std::this_thread::sleep_for(std::chrono::seconds(sleep_s));
      sleep_s += sleep_s / 2;
    }
  } while (allow == false);

  LOG_INFO("Join cluster success.");
  LOG_INFO("Old router:" << old_router.Str());
  LOG_INFO("New router:" << router_.Str());

  if (model_init_) {
    model_name_ = model_mdata.name;

    // Init the model.
    optim_ = Optim::Create(model_mdata.optim_type, model_mdata.optim_conf);
    ARGUMENT_CHECK(optim_ != nullptr, "Unsupport Optim type.");

    LOG_INFO("Init Model, optim_type:" << model_mdata.optim_type
                                       << ", optim conf:"
                                       << model_mdata.optim_conf);
  } else {
    LOG_INFO("Model not initialized.")
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
    const Router::Node& cur_node = router_.node(node_id_);

    // Try to create proxy.
    std::unordered_set<uint64_t> proxy_ids;
    for (auto hash : cur_node.vnode_list) {
      uint64_t hit_node_id = old_router.Hit(hash);

      if (hit_node_id != node_id_) {
        proxy_ids.emplace(hit_node_id);
      }
    }

    proxy_.reset(new Proxy(proxy_ids, old_router, CompressType::kSnappy));

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

int32_t Ps::NotifySaveModel(const ModelMetaData& model_mdata) {
  std::unique_lock<std::shared_mutex> _(mu_);

  if (status_ != NodeStatus::kWork) {
    LOG_INFO("Status is:" << NodeStatusStr(status_)
                          << ", not allowed to save model.");

    return ErrorCode::kNodeStatusError;
  }

  ModelMetaData l_model_mdata = model_mdata;

  // In async queue to save model.
  auto task = [this, l_model_mdata] {
    if (checkpoint_exec_.Save(this, l_model_mdata) == false) {
      LOG_ERROR("Save model error!");
    }

    // Reset status.
    std::shared_lock<std::shared_mutex> _(mu_);
    status_ = status_ & ~NodeStatus::kSave;

    LOG_INFO("Finish save model, status become:[" << NodeStatusStr(status_)
                                                  << "].");
  };

  status_ |= NodeStatus::kSave;
  task_que_.Enque(std::move(task));

  LOG_INFO("Allow to save model, status become:[" << NodeStatusStr(status_)
                                                  << "].");

  return ErrorCode::kSuccess;
}

int32_t Ps::NotifyLoadModel(const std::string& load_dir) {
  std::unique_lock<std::shared_mutex> l(mu_);

  if (status_ != NodeStatus::kWork) {
    LOG_INFO("Status is:" << NodeStatusStr(status_)
                          << ", not allowed to load model.");

    return ErrorCode::kNodeStatusError;
  }

  // In async queue to load model.
  auto task = [this, load_dir] {
    ARGUMENT_CHECK(checkpoint_exec_.Load(this, load_dir), "Load model error!");

    // Reset status.
    std::unique_lock<std::shared_mutex> l(mu_);
    status_ = NodeStatus::kWork;

    LOG_INFO("Finish load model, status become:[" << NodeStatusStr(status_)
                                                  << "].");
  };

  status_ = NodeStatus::kLoad;
  task_que_.Enque(std::move(task));

  LOG_INFO("Allow to load model, status become:[" << NodeStatusStr(status_)
                                                  << "].");

  return ErrorCode::kSuccess;
}

int32_t Ps::NotifyNodeJoin(uint64_t joined_id, const Router& old_router,
                           const Router& new_router) {
  std::unique_lock<std::shared_mutex> lock_ps(mu_);

  // A node is allowed to join must need all node status is kWork.
  if (status_ != NodeStatus::kWork) {
    LOG_INFO("Status is not kWork, So not allowed join new node.");
    return ErrorCode::kNodeStatusError;
  }

  LOG_INFO("Got notification: node:[" << joined_id << "] join in, my status:["
                                      << status_ << "]");
  LOG_INFO("Old router:" << old_router.Str());
  LOG_INFO("New router:" << new_router.Str());

  ARGUMENT_CHECK(old_router == router_, "The router is error!");

  // Update router.
  router_ = new_router;

  ARGUMENT_CHECK(router_.Contains(node_id_), "Can't find node:" << node_id_);
  const Router::Node& target_node = router_.node(joined_id);

  // Check whether we need transfer the data to the new node or not.
  bool need_transfer = false;
  for (auto h : target_node.vnode_list) {
    if (node_id_ == old_router.Hit(h)) {
      need_transfer = true;
      break;
    }
  }

  if (need_transfer) {
    // Async task to transfer data.
    auto task = [this, joined_id] { this->TransferTo(joined_id); };

    // Modify status.
    status_ = NodeStatus::kWork | NodeStatus::kTransfer;

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
  if (status_ != NodeStatus::kWork) {
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

  LOG_INFO("Create DenseTable:["
           << name << "], id:[" << table_id << "], ElementType:["
           << val.element_type().Name() << "], shape:" << val.shape().Str());

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

  LOG_INFO("Create SparseTable:["
           << name << "], id:[" << table_id << "], dimension:[" << dimension
           << "], ElementType:[" << element_type.Name() << "], init_type:["
           << init_type << "], init_conf:[" << init_conf << "]");

  return ErrorCode::kSuccess;
}

int32_t Ps::NotifyFinishTransfer(uint64_t from_node_id) {
  std::unique_lock<std::shared_mutex> l(mu_);

  LOG_INFO("Got notificaion from node:[" << from_node_id
                                         << "] finish transfer.");

  if (!(status_ & NodeStatus::kProxy)) {
    return ErrorCode::kNodeStatusError;
  }

  if (events_.find(EventType::kProxyFinishTransfer) == events_.end()) {
    return ErrorCode::kUnSupportEventError;
  }

  if (events_[EventType::kProxyFinishTransfer].find(from_node_id) ==
      events_[EventType::kProxyFinishTransfer].end()) {
    return ErrorCode::kUnSupportEventError;
  }

  events_[EventType::kProxyFinishTransfer].erase(from_node_id);

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

int32_t Ps::TransferDenseTable(uint64_t from_node_id, uint64_t table_id,
                               const std::string& name, const Value& value) {
  std::shared_lock<std::shared_mutex> l(mu_);
  if (status_ != (NodeStatus::kWork | NodeStatus::kProxy)) {
    return ErrorCode::kNodeStatusError;
  }

  std::unique_lock<std::shared_mutex> ll(model_mu_);

  // If exist just return success.
  if (tables_.Contains(table_id)) {
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
  if (tables_.Contains(table_id)) {
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
  table->mutable_vals()->Insert(sparse_ids, values);

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

  auto* parallel_vals = table->mutable_vals();

  std::unordered_map<size_t, std::vector<size_t>> slot_idx_map;
  slot_idx_map.reserve(parallel_vals->slot_count());

  for (size_t i = 0; i < sparse_ids.size(); ++i) {
    slot_idx_map[parallel_vals->HitSlot(sparse_ids[i])].emplace_back(
        sparse_ids[i]);
  }

  for (auto& [slot, v] : slot_idx_map) {
    // Lock the slot.
    auto h = parallel_vals->SharedSkipListHandler(slot);

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
