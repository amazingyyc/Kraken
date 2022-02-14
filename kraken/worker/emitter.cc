#include "worker/emitter.h"

#include <tuple>

#include "common/log.h"
#include "protocol/apply_dense_table_prot.h"
#include "protocol/apply_model_prot.h"
#include "protocol/apply_sparse_table_prot.h"
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
#include "protocol/rpc_func_type.h"
#include "protocol/save_check_point_prot.h"

namespace kraken {

Emitter::Emitter() : Emitter(EmitterType::kDefault) {
}

Emitter::Emitter(EmitterType type)
    : type_(type), initialized_(false), router_(1) {
}

size_t Emitter::DenseTableRouter(uint64_t model_id, uint64_t table_id) {
  return router_(model_id, table_id);
}

size_t Emitter::SparseTableRouter(uint64_t model_id, uint64_t table_id,
                                  int64_t sparse_id) {
  return router_(model_id, table_id, sparse_id);
}

void Emitter::Initialize(const std::string& addrs, CompressType compress_type) {
  if (initialized_.load()) {
    return;
  }

  // Parse address.
  std::vector<std::string> tokens;
  utils::Split(addrs, ",", &tokens);

  for (uint32_t i = 0; i < tokens.size(); ++i) {
    std::unique_ptr<Client> client(new Client(i, tokens[i], compress_type));
    clients_.emplace_back(std::move(client));
  }

  // Start client.
  for (uint32_t i = 0; i < tokens.size(); ++i) {
    clients_[i]->Start();
  }

  // Update router.
  router_ = ConsistentHasher(clients_.size());

  LOG_INFO("Create connection with:" << addrs);

  // set flag.
  initialized_.store(true);
}

void Emitter::Stop() {
  ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

  for (auto& c : clients_) {
    c->Stop();
  }

  initialized_.store(false);
}

uint64_t Emitter::RegisterModel(
    const std::string& model_name, OptimType optim_type,
    const std::unordered_map<std::string, std::string>& optim_conf) {
  ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

  model_name_ = model_name;

  {
    // Step1: Apply a model id in leader server.
    ApplyModelRequest req;
    req.model_name = model_name;
    req.optim_type = optim_type;
    req.optim_conf = optim_conf;

    ApplyModelResponse rsp;

    RPC_CALL(clients_[0]->Call(RPCFuncType::kApplyModelType, req, &rsp));

    // Store model id.
    model_id_ = rsp.model_id;

    LOG_INFO("Apply model: " << model_name_ << ", id: " << model_id_
                             << ", from server 0.");
  }

  {
    // Step2: Register model in all server.
    RegisterModelRequest req;
    req.id = model_id_;
    req.name = model_name;
    req.optim_type = optim_type;
    req.optim_conf = optim_conf;

    std::vector<RegisterModelResponse> rsps;

    // We do not care about the response, just check the error code.
    RPC_CALL(ParallelCallAll(RPCFuncType::kRegisterModelType, req, &rsps));

    LOG_INFO("Register model: " << model_name_ << ", id: " << model_id_
                                << " in all server.");
  }

  return model_id_;
}

void Emitter::UpdateLR(float lr) {
  lr_.store(lr);
}

uint64_t Emitter::RegisterDenseTable(const std::string& name,
                                     const Tensor& val) {
  ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

  uint64_t table_id;
  {
    // Apply table id.
    ApplyDenseTableRequest req;
    ApplyDenseTableResponse rsp;

    req.model_id = model_id_;
    req.name = name;
    req.shape = val.shape();
    req.element_type = val.element_type();

    RPC_CALL(clients_[0]->Call(RPCFuncType::kApplyDenseTableType, req, &rsp));

    table_id = rsp.table_id;

    LOG_INFO("Apply DenseTable: " << name << ", id: " << table_id
                                  << ", from server 0.");
  }

  {
    // Register DenseTable in all server.
    RegisterDenseTableInfoRequest req;
    req.model_id = model_id_;
    req.id = table_id;
    req.name = name;
    req.shape = val.shape();
    req.element_type = val.element_type();

    std::vector<RegisterDenseTableInfoResponse> rsps;

    RPC_CALL(
        ParallelCallAll(RPCFuncType::kRegisterDenseTableInfoType, req, &rsps));

    LOG_INFO("Register DenseTableInfo:" << name << ", id:" << table_id
                                        << " in all server.");
  }

  {
    // Get server id.
    size_t server_id = DenseTableRouter(model_id_, table_id);

    // Register dense table in special server for training.
    RegisterDenseTableRequest req;
    RegisterDenseTableResponse rsp;

    req.model_id = model_id_;
    req.id = table_id;
    req.name = name;
    req.val = val;

    RPC_CALL(clients_[server_id]->Call(RPCFuncType::kRegisterDenseTableType,
                                       req, &rsp));

    LOG_INFO("Register DenseTable: " << name << ", id: " << table_id
                                     << ", in server:" << server_id);
  }

  return table_id;
}

uint64_t Emitter::RegisterSparseTable(
    const std::string& name, int64_t dimension, ElementType element_type,
    InitializerType init_type,
    const std::unordered_map<std::string, std::string>& init_conf) {
  ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

  uint64_t table_id;
  {
    // Apply table id.
    ApplySparseTableRequest req;
    ApplySparseTableResponse rsp;

    req.model_id = model_id_;
    req.name = name;
    req.dimension = dimension;
    req.element_type = element_type;
    req.init_type = init_type;
    req.init_conf = init_conf;

    RPC_CALL(clients_[0]->Call(RPCFuncType::kApplySparseTableType, req, &rsp));

    table_id = rsp.table_id;

    LOG_INFO("Apply SparseTable: " << name << ", id: " << table_id
                                   << ", from server 0.");
  }

  {
    // Register sparse table in all server.
    RegisterSparseTableRequest req;
    std::vector<RegisterSparseTableResponse> rsps;

    req.model_id = model_id_;
    req.id = table_id;
    req.name = name;
    req.dimension = dimension;
    req.element_type = element_type;
    req.init_type = init_type;
    req.init_conf = init_conf;

    RPC_CALL(
        ParallelCallAll(RPCFuncType::kRegisterSparseTableType, req, &rsps));

    LOG_INFO("Register SparseTable: " << name << ", id: " << table_id
                                      << ", in all server.");
  }

  return table_id;
}

Tensor Emitter::PullDenseTable(uint64_t table_id) {
  ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

  size_t server_id = DenseTableRouter(model_id_, table_id);

  PullDenseTableRequest req;
  PullDenseTableResponse rsp;

  req.model_id = model_id_;
  req.table_id = table_id;

  RPC_CALL(
      clients_[server_id]->Call(RPCFuncType::kPullDenseTableType, req, &rsp));

  return rsp.val;
}

std::vector<Tensor> Emitter::CombinePullDenseTable(
    const std::vector<uint64_t>& table_ids) {
  ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

  std::vector<size_t> table_val_idx;
  table_val_idx.resize(table_ids.size());

  std::vector<CombinePullDenseTableRequest> reqs;
  reqs.resize(clients_.size());

  std::vector<CombinePullDenseTableResponse> rsps;

  std::vector<bool> mask(clients_.size(), false);

  for (size_t i = 0; i < table_ids.size(); ++i) {
    uint64_t table_id = table_ids[i];
    size_t server_id = DenseTableRouter(model_id_, table_id);

    mask[server_id] = true;

    table_val_idx[i] = reqs[server_id].table_ids.size();
    reqs[server_id].table_ids.emplace_back(table_id);
  }

  for (size_t i = 0; i < mask.size(); ++i) {
    if (mask[i]) {
      reqs[i].model_id = model_id_;
    }
  }

  RPC_CALL(
      ParallelCall(RPCFuncType::kCombinePullDenseTableType, mask, reqs, &rsps));

  std::vector<Tensor> vals;
  vals.reserve(table_ids.size());

  for (size_t i = 0; i < table_ids.size(); ++i) {
    uint64_t table_id = table_ids[i];
    size_t server_id = DenseTableRouter(model_id_, table_id);

    vals.emplace_back(rsps[server_id].vals.at(table_val_idx[i]));
  }

  return vals;
}

void Emitter::PushDenseTable(uint64_t table_id, const Tensor& grad) {
  ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

  size_t server_id = DenseTableRouter(model_id_, table_id);

  PushDenseTableRequest req;
  req.model_id = model_id_;
  req.table_id = table_id;
  req.grad = grad;
  req.lr = lr_.load();

  auto callback = [](int32_t ecode, PushDenseTableResponse& /*not care*/) {
    RPC_CALL(ecode);
  };

  clients_[server_id]->CallAsync<PushDenseTableRequest, PushDenseTableResponse>(
      RPCFuncType::kPushDenseTableType, req, std::move(callback));
}

Tensor Emitter::PushPullDenseTable(uint64_t table_id, const Tensor& grad) {
  ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

  size_t server_id = DenseTableRouter(model_id_, table_id);

  PushPullDenseTableRequest req;
  PushPullDenseTableResponse rsp;

  req.model_id = model_id_;
  req.table_id = table_id;
  req.grad = grad;
  req.lr = lr_.load();

  RPC_CALL(clients_[server_id]->Call(RPCFuncType::kPushPullDenseTableType, req,
                                     &rsp));

  return rsp.val;
}

Tensor Emitter::PullSparseTable(uint64_t table_id, const Tensor& indices) {
  ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

  // Maybe share memory with pytorch.
  Tensor indices_i64 = indices.Cast(ElementType::From<int64_t>());

  int64_t row = indices_i64.Size();
  int64_t* idp = indices_i64.Data<int64_t>();

  // The val index in req/rsp
  std::vector<std::unordered_map<int64_t, size_t>> val_indices;
  val_indices.resize(clients_.size());

  std::vector<PullSparseTableRequest> reqs;
  reqs.resize(clients_.size());

  std::vector<PullSparseTableResponse> rsps;

  std::vector<bool> mask(clients_.size(), false);

  for (int64_t i = 0; i < row; ++i) {
    int64_t sparse_id = idp[i];
    size_t server_id = SparseTableRouter(model_id_, table_id, sparse_id);

    mask[server_id] = true;

    if (val_indices[server_id].find(sparse_id) ==
        val_indices[server_id].end()) {
      val_indices[server_id][sparse_id] = reqs[server_id].indices.size();
      reqs[server_id].indices.emplace_back(sparse_id);
    }
  }

  for (size_t i = 0; i < reqs.size(); ++i) {
    if (mask[i]) {
      reqs[i].model_id = model_id_;
      reqs[i].table_id = table_id;
    }
  }

  // Send to server.
  RPC_CALL(ParallelCall(RPCFuncType::kPullSparseTableType, mask, reqs, &rsps));

  std::vector<Tensor> vals;
  vals.reserve(row);

  for (int64_t i = 0; i < row; ++i) {
    int64_t sparse_id = idp[i];
    size_t server_id = SparseTableRouter(model_id_, table_id, sparse_id);

    size_t val_idx = val_indices[server_id][sparse_id];

    vals.emplace_back(rsps.at(server_id).vals.at(val_idx));
  }

  Tensor val = indices_i64.ConcatVector(vals);

  std::vector<int64_t> dims = indices_i64.shape().dims();
  int64_t col = val.Size() / indices_i64.Size();

  dims.emplace_back(col);

  return val.Reshape(dims);
}

std::vector<Tensor> Emitter::CombinePullSparseTable(
    const std::vector<uint64_t>& table_ids,
    const std::vector<Tensor>& indices) {
  ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");
  ARGUMENT_CHECK(
      table_ids.size() == indices.size(),
      "CombinePullSparseTable need table_ids and indices has same size.");

  // Maybe share memory with pytorch.
  std::vector<Tensor> indice_i64s;
  indice_i64s.reserve(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    indice_i64s.emplace_back(indices[i].Cast(ElementType::From<int64_t>()));
  }

  std::vector<CombinePullSparseTableRequest> reqs;
  reqs.resize(clients_.size());
  std::vector<CombinePullSparseTableResponse> rsps;

  std::vector<bool> mask(clients_.size(), false);

  // <model_id, table_id>
  using KeyT = std::tuple<uint64_t, uint64_t>;
  struct KeyHash : public std::unary_function<KeyT, std::size_t> {
    std::size_t operator()(const KeyT& k) const {
      return std::get<0>(k) ^ std::get<1>(k);
    }
  };

  std::vector<std::unordered_map<KeyT, size_t, KeyHash>> req_table_idx_map;
  req_table_idx_map.resize(clients_.size());

  std::vector<std::unordered_map<int64_t, size_t>> req_sparse_val_map;
  req_sparse_val_map.resize(clients_.size());

  for (size_t i = 0; i < table_ids.size(); ++i) {
    uint64_t table_id = table_ids[i];

    int64_t row = indice_i64s[i].Size();
    int64_t* idp = indice_i64s[i].Data<int64_t>();

    for (int64_t j = 0; j < row; ++j) {
      int64_t sparse_id = (int64_t)idp[i];
      size_t server_id = SparseTableRouter(model_id_, table_id, sparse_id);

      KeyT key = std::make_tuple(model_id_, table_id);

      mask[server_id] = true;

      if (req_table_idx_map[server_id].find(key) ==
          req_table_idx_map[server_id].end()) {
        req_table_idx_map[server_id][key] = reqs[server_id].indices.size();

        PullSparseTableRequest c_req;
        c_req.model_id = model_id_;
        c_req.table_id = table_id;

        reqs[server_id].indices.emplace_back(c_req);
      }

      size_t table_idx = req_table_idx_map[server_id][key];

      if (req_sparse_val_map[server_id].find(sparse_id) ==
          req_sparse_val_map[server_id].end()) {
        req_sparse_val_map[server_id][sparse_id] =
            reqs[server_id].indices[table_idx].indices.size();
        reqs[server_id].indices[table_idx].indices.emplace_back(sparse_id);
      }
    }
  }

  // Send to server.
  RPC_CALL(ParallelCall(RPCFuncType::kCombinePullSparseTableType, mask, reqs,
                        &rsps));

  std::vector<Tensor> vals;
  vals.reserve(indice_i64s.size());

  for (size_t i = 0; i < table_ids.size(); ++i) {
    uint64_t table_id = table_ids[i];

    int64_t row = indice_i64s[i].Size();
    int64_t* idp = indice_i64s[i].Data<int64_t>();

    std::vector<Tensor> vecs;
    vecs.reserve(row);

    for (int64_t j = 0; j < row; ++j) {
      int64_t sparse_id = (int64_t)idp[i];
      size_t server_id = SparseTableRouter(model_id_, table_id, sparse_id);

      KeyT key = std::make_tuple(model_id_, table_id);

      size_t table_idx = req_table_idx_map[server_id][key];
      size_t val_idx = req_sparse_val_map[server_id][sparse_id];

      vecs.emplace_back(rsps[server_id].vals.at(table_idx).vals.at(val_idx));
    }

    Tensor val = indice_i64s[i].ConcatVector(vecs);

    std::vector<int64_t> dims = indice_i64s[i].shape().dims();
    int64_t col = val.Size() / indice_i64s[i].Size();

    dims.emplace_back(col);

    vals.emplace_back(val.Reshape(dims));
  }

  return vals;
}

void Emitter::PushSparseTable(uint64_t table_id, const Tensor& indices,
                              const Tensor& grads) {
  ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

  // If the indices shape is [do, d1,...,dk] than grads shape must be [do,
  // d1,...,dk, dimension]
  std::vector<int64_t> dims = indices.shape().dims();
  int64_t dimension = grads.shape()[-1];
  dims.emplace_back(dimension);

  ARGUMENT_CHECK(Shape(dims) == grads.shape(),
                 "PushSparseTable indices and grads shape error.");

  Tensor indices_i64 = indices.Cast(ElementType::From<int64_t>());
  int64_t row = indices_i64.Size();
  int64_t* idp = indices_i64.Data<int64_t>();

  Tensor m_grads = grads.Reshape({row, dimension});

  std::vector<std::unordered_map<int64_t, size_t>> grad_indices;
  grad_indices.resize(clients_.size());

  std::vector<PushSparseTableRequest> reqs;
  reqs.resize(clients_.size());

  std::vector<bool> mask(clients_.size(), false);

  for (int64_t i = 0; i < row; ++i) {
    int64_t sparse_id = idp[i];
    size_t server_id = SparseTableRouter(model_id_, table_id, sparse_id);

    mask[server_id] = true;

    auto it = grad_indices[server_id].find(sparse_id);
    if (it == grad_indices[server_id].end()) {
      grad_indices[server_id][sparse_id] = reqs[server_id].indices.size();

      reqs[server_id].indices.emplace_back(sparse_id);
      reqs[server_id].grads.emplace_back(m_grads.Vector(i).Clone());
    } else {
      reqs[server_id].grads[it->second] += m_grads.Vector(i);
    }
  }

  for (size_t i = 0; i < mask.size(); ++i) {
    if (mask[i]) {
      reqs[i].model_id = model_id_;
      reqs[i].table_id = table_id;
      reqs[i].lr = lr_.load();
    }
  }

  ParallelCallAsync<PushSparseTableRequest, PushSparseTableResponse>(
      RPCFuncType::kPushSparseTableType, mask, reqs);
}

void Emitter::SaveCheckPoint() {
  SaveCheckPointRequest req;
  std::vector<SaveCheckPointResponse> rsps;

  req.model_id = model_id_;

  RPC_CALL(ParallelCallAll(RPCFuncType::kSaveCheckPointType, req, &rsps));
}

}  // namespace kraken
