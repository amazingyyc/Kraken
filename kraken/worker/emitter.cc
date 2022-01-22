#include "worker/emitter.h"

#include <tuple>

#include "common/log.h"
#include "protocol/apply_model_prot.h"
#include "protocol/apply_table_prot.h"
#include "protocol/pull_dense_table_prot.h"
#include "protocol/pull_list_dense_table_prot.h"
#include "protocol/push_dense_table_prot.h"
#include "protocol/push_pull_dense_table_prot.h"
#include "protocol/register_dense_table_prot.h"
#include "protocol/register_model_prot.h"
#include "protocol/register_sparse_table_prot.h"
#include "protocol/rpc_func_type.h"

namespace kraken {

Emitter::Emitter() : Emitter(EmitterType::kDefault) {
}

Emitter::Emitter(EmitterType type) : type_(type), initialized_(false) {
}

size_t Emitter::DenseTableRouter(uint64_t model_id, uint64_t table_id) {
  size_t seed = 0;
  seed ^= model_id + 0x9e3779b9;
  seed ^= table_id + 0x9e3779b9 + (seed << 6) + (seed >> 2);

  return seed % clients_.size();
}

size_t Emitter::SparseTableRouter(uint64_t model_id, uint64_t table_id,
                                  int64_t sparse_id) {
  size_t seed = 0;
  seed ^= model_id + 0x9e3779b9;
  seed ^= table_id + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= sparse_id + 0x9e3779b9 + (seed << 6) + (seed >> 2);

  return seed % clients_.size();
}

void Emitter::Initialize(const std::string& addrs) {
  if (initialized_.load()) {
    return;
  }

  // Parse address.
  std::vector<std::string> tokens;
  utils::Split(addrs, ",", &tokens);

  for (uint32_t i = 0; i < tokens.size(); ++i) {
    std::unique_ptr<Client> client(new Client(i, tokens[i]));
    clients_.emplace_back(std::move(client));
  }

  // Start client.
  for (uint32_t i = 0; i < tokens.size(); ++i) {
    clients_[i]->Start();
  }

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

  // Step1: Apply a model in master server.
  ApplyModelRequest req;
  req.name = model_name_;

  ApplyModelResponse rsp;
  RPC_CALL(clients_[0]->Call(RPCFuncType::kApplyModelType, req, &rsp));

  // Store model id.
  model_id_ = rsp.model_id;

  LOG_INFO("Apply model: " << model_name_ << ", id: " << model_id_
                           << ", from server 0.");

  // Step2: Register model in all server.
  RegisterModelRequest register_req;
  register_req.id = model_id_;
  register_req.name = model_name_;
  register_req.optim_type = optim_type;
  register_req.optim_conf = optim_conf;

  std::vector<RegisterModelResponse> register_rsps;

  // We do not care about the response, just check the error code.
  RPC_CALL(ParallelCallAll(RPCFuncType::kRegisterModelType, register_req,
                           &register_rsps));

  LOG_INFO("Register model: " << model_name_ << ", id: " << model_id_
                              << " in all server.");

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
    ApplyTableRequest req;
    ApplyTableResponse rsp;

    req.model_id = model_id_;
    req.table_name = name;
    req.table_type = TableType::kDense;

    RPC_CALL(clients_[0]->Call(RPCFuncType::kApplyTableType, req, &rsp));

    table_id = rsp.table_id;

    LOG_INFO("Apply DenseTable: " << name << ", id: " << table_id
                                  << ", from server 0.");
  }

  {
    // Get server id.
    size_t server_id = DenseTableRouter(model_id_, table_id);

    // Register dense table.
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

uint64_t Emitter::RegisterSparseTable(const std::string& name,
                                      int64_t dimension, ElementType etype) {
  ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

  // Register a SparseTable step:
  // 1: Apply a table id from master.
  // 2: Register the SparseTable in all server.
  uint64_t table_id;

  {
    // Apply table id.
    ApplyTableRequest req;
    ApplyTableResponse rsp;

    req.model_id = model_id_;
    req.table_name = name;
    req.table_type = TableType::kSparse;

    RPC_CALL(clients_[0]->Call(RPCFuncType::kApplyTableType, req, &rsp));

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
    req.etype = etype;

    RPC_CALL(
        ParallelCallAll(RPCFuncType::kRegisterSparseTableType, req, &rsps));

    LOG_INFO("Register SparseTable: " << name << ", id: " << table_id
                                      << ", in all server.");
  }

  return table_id;
}

uint64_t Emitter::RegisterSparseTableV2(
    const std::string& name, int64_t dimension, ElementType etype,
    InitializerType init_type,
    const std::unordered_map<std::string, std::string>& init_conf) {
  ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

  uint64_t table_id;

  {
    // Apply table id.
    ApplyTableRequest req;
    ApplyTableResponse rsp;

    req.model_id = model_id_;
    req.table_name = name;
    req.table_type = TableType::kSparse;

    RPC_CALL(clients_[0]->Call(RPCFuncType::kApplyTableType, req, &rsp));

    table_id = rsp.table_id;

    LOG_INFO("Apply SparseTable: " << name << ", id: " << table_id
                                   << ", from server 0.");
  }

  {
    // Register sparse table in all server.
    RegisterSparseTableV2Request req;
    std::vector<RegisterSparseTableV2Response> rsps;

    req.model_id = model_id_;
    req.id = table_id;
    req.name = name;
    req.dimension = dimension;
    req.etype = etype;
    req.init_type = init_type;
    req.init_conf = init_conf;

    RPC_CALL(
        ParallelCallAll(RPCFuncType::kRegisterSparseTableV2Type, req, &rsps));

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

std::vector<Tensor> Emitter::PullListDenseTable(
    const std::vector<uint64_t>& table_ids) {
  ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

  std::unordered_map<size_t, PullListDenseTableRequest> server_reqs;
  std::unordered_map<size_t, std::vector<size_t>> origin_indice_map;

  for (size_t i = 0; i < table_ids.size(); ++i) {
    uint64_t table_id = table_ids[i];
    size_t server_id = DenseTableRouter(model_id_, table_id);

    if (server_reqs.find(server_id) == server_reqs.end()) {
      PullListDenseTableRequest req;
      req.model_id = model_id_;

      server_reqs.emplace(server_id, req);
    }

    server_reqs[server_id].table_ids.emplace_back(table_id);
    origin_indice_map[server_id].emplace_back(i);
  }

  std::vector<size_t> server_indices;
  server_indices.reserve(server_reqs.size());

  std::vector<PullListDenseTableRequest> reqs;
  std::vector<PullListDenseTableResponse> rsps;

  reqs.reserve(server_reqs.size());
  for (auto& item : server_reqs) {
    server_indices.emplace_back(item.first);
    reqs.emplace_back(std::move(item.second));
  }

  RPC_CALL(ParallelCall(RPCFuncType::kPullListDenseTableType, server_indices,
                        reqs, &rsps));

  std::vector<Tensor> vals;
  vals.resize(table_ids.size());

  for (size_t i = 0; i < server_indices.size(); ++i) {
    uint64_t server_id = server_indices[i];
    const auto& origin_indice = origin_indice_map[server_id];
    const auto& rsp_vals = rsps[i].vals;

    ARGUMENT_CHECK(origin_indice.size() == rsp_vals.size(), "Internal error.");

    for (size_t j = 0; j < origin_indice.size(); ++j) {
      vals[origin_indice[j]] = rsp_vals[j];
    }
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

  Tensor indices_i64 = indices.Cast(ElementType::From<int64_t>());
  int64_t row = indices_i64.Size();

  // Sparse table store in all server. So we need select which server to
  // send.
  std::unordered_map<size_t, PullSparseTableRequest> server_reqs;

  // <server_id, <sparse id, request indices_i64's index>> map
  std::unordered_map<size_t, std::unordered_map<int64_t, size_t>>
      server_indice_map;

  int64_t* idp = indices_i64.Data<int64_t>();

  for (int64_t i = 0; i < row; ++i) {
    int64_t sparse_id = idp[i];
    size_t server_id = SparseTableRouter(model_id_, table_id, sparse_id);

    if (server_indice_map[server_id].find(sparse_id) ==
        server_indice_map[server_id].end()) {
      server_indice_map[server_id][sparse_id] =
          server_reqs[server_id].indices.size();

      server_reqs[server_id].indices.emplace_back(sparse_id);
    }
  }

  std::vector<size_t> server_indices;
  server_indices.reserve(server_reqs.size());

  std::vector<PullSparseTableRequest> reqs;
  std::vector<PullSparseTableResponse> rsps;

  reqs.reserve(server_reqs.size());
  for (auto& item : server_reqs) {
    item.second.model_id = model_id_;
    item.second.table_id = table_id;

    server_indices.emplace_back(item.first);
    reqs.emplace_back(std::move(item.second));
  }

  // Send to server.
  RPC_CALL(ParallelCall(RPCFuncType::kPullSparseTableType, server_indices, reqs,
                        &rsps));

  // For now we already get result from server. So let's merge it.
  std::unordered_map<size_t, size_t> server_rsp_map;
  for (size_t i = 0; i < server_indices.size(); ++i) {
    server_rsp_map[server_indices[i]] = i;
  }

  std::vector<Tensor> vecs;
  vecs.reserve(row);

  for (int64_t i = 0; i < row; ++i) {
    int64_t sparse_id = (int64_t)idp[i];
    size_t server_id = SparseTableRouter(model_id_, table_id, sparse_id);

    size_t rsp_idx = server_rsp_map[server_id];
    size_t vec_idx = server_indice_map[server_id][sparse_id];

    vecs.emplace_back(rsps.at(rsp_idx).vals.at(vec_idx));
  }

  // concat to matrix than reshape.
  Tensor val = indices_i64.ConcatVector(vecs);

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

  std::vector<Tensor> indice_i64s;
  indice_i64s.reserve(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    indice_i64s.emplace_back(indices[i].Cast(ElementType::From<int64_t>()));
  }

  // <server_id, model_id, table_id>
  using KeyT = std::tuple<uint64_t, uint64_t>;
  struct KeyHash : public std::unary_function<KeyT, std::size_t> {
    std::size_t operator()(const KeyT& k) const {
      return std::get<0>(k) ^ std::get<1>(k);
    }
  };

  // key: <server_id, <model_id, table_id>>
  // value: [sparse_id]
  std::unordered_map<size_t,
                     std::unordered_map<KeyT, std::vector<int64_t>, KeyHash>>
      req_indices;

  // key: <server_id, <model_id, table_id>>
  // value: <sparse_id, index>
  std::unordered_map<
      size_t,
      std::unordered_map<KeyT, std::unordered_map<int64_t, size_t>, KeyHash>>
      req_indices_map;

  for (size_t i = 0; i < table_ids.size(); ++i) {
    uint64_t table_id = table_ids[i];

    int64_t row = indice_i64s[i].Size();
    int64_t* idp = indice_i64s[i].Data<int64_t>();

    auto key = std::make_tuple(model_id_, table_id);

    for (int64_t j = 0; j < row; ++j) {
      int64_t sparse_id = (int64_t)idp[i];
      size_t server_id = SparseTableRouter(model_id_, table_id, sparse_id);

      if (req_indices_map[server_id][key].find(sparse_id) ==
          req_indices_map[server_id][key].end()) {
        req_indices_map[server_id][key][sparse_id] =
            req_indices[server_id][key].size();

        req_indices[server_id][key].emplace_back(sparse_id);
      }
    }
  }

  std::vector<size_t> server_indices;
  server_indices.reserve(req_indices.size());

  std::vector<CombinePullSparseTableRequest> reqs;
  reqs.reserve(req_indices.size());

  std::vector<CombinePullSparseTableResponse> rsps;

  // <server_id, <[model_id, table_id], index in rsp>>
  std::unordered_map<size_t, std::unordered_map<KeyT, size_t, KeyHash>>
      server_key_val_map;

  for (auto& item : req_indices) {
    CombinePullSparseTableRequest c_req;
    c_req.indices.reserve(item.second.size());

    for (auto& row : item.second) {
      PullSparseTableRequest req;
      req.model_id = std::get<0>(row.first);
      req.table_id = std::get<1>(row.first);
      req.indices = std::move(row.second);

      server_key_val_map[item.first][row.first] = c_req.indices.size();
      c_req.indices.emplace_back(std::move(req));
    }

    server_indices.emplace_back(item.first);
    reqs.emplace_back(std::move(c_req));
  }

  // Send to server.
  RPC_CALL(ParallelCall(RPCFuncType::kCombinePullSparseTableType,
                        server_indices, reqs, &rsps));

  std::unordered_map<size_t, size_t> server_rsp_map;
  for (size_t i = 0; i < server_indices.size(); ++i) {
    server_rsp_map[server_indices[i]] = i;
  }

  std::vector<Tensor> vals;
  vals.reserve(indice_i64s.size());

  // Get result.
  for (size_t i = 0; i < table_ids.size(); ++i) {
    uint64_t table_id = table_ids[i];

    int64_t row = indice_i64s[i].Size();
    int64_t* idp = indice_i64s[i].Data<int64_t>();

    auto key = std::make_tuple(model_id_, table_id);

    std::vector<Tensor> vecs;
    vecs.reserve(row);

    for (int64_t j = 0; j < row; ++j) {
      int64_t sparse_id = (int64_t)idp[i];
      size_t server_id = SparseTableRouter(model_id_, table_id, sparse_id);

      size_t rsp_idx = server_rsp_map[server_id];
      size_t val_idx = server_key_val_map[server_id][key];
      size_t vec_idx = req_indices_map[server_id][key][sparse_id];

      vecs.emplace_back(rsps.at(rsp_idx).vals.at(val_idx).vals.at(vec_idx));
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

  Tensor indices_i64 = indices.Cast(ElementType::From<int64_t>());

  // If the indices shape is [do, d1,...,dk] than grads shape must be [do,
  // d1,...,dk, dimension]
  std::vector<int64_t> dims = indices_i64.shape().dims();
  int64_t dimension = grads.shape()[-1];
  dims.emplace_back(dimension);

  ARGUMENT_CHECK(Shape(dims) == grads.shape(),
                 "PushSparseTable indices and grads shape error.");

  int64_t row = indices_i64.Size();
  int64_t col = dimension;
  Tensor mgrads = grads.Reshape({row, col});

  // <server_id, <sparse_id, indice>>
  std::unordered_map<size_t, std::unordered_map<int64_t, size_t>>
      server_req_indice_map;

  std::unordered_map<size_t, PushSparseTableRequest> server_reqs;

  int64_t* idp = indices_i64.Data<int64_t>();

  for (int64_t i = 0; i < row; ++i) {
    int64_t sparse_id = idp[i];
    size_t server_id = SparseTableRouter(model_id_, table_id, sparse_id);

    auto it = server_req_indice_map[server_id].find(sparse_id);
    if (it == server_req_indice_map[server_id].end()) {
      // Get a new SparseId.
      server_req_indice_map[server_id][sparse_id] =
          server_reqs[server_id].indices.size();

      server_reqs[server_id].indices.emplace_back(sparse_id);

      // Here must clone. The grads is share memory with torch tensor.
      server_reqs[server_id].grads.emplace_back(grads.Vector(i).Clone());
    } else {
      // Already exist accumulate the gradient.
      size_t r_idx = it->second;
      server_reqs[server_id].grads[r_idx] += grads.Vector(i);
    }
  }

  std::vector<size_t> server_indices;
  server_indices.reserve(server_reqs.size());

  std::vector<PushSparseTableRequest> reqs;
  std::vector<PushSparseTableResponse> rsps;

  reqs.reserve(server_reqs.size());

  for (auto& item : server_reqs) {
    item.second.model_id = model_id_;
    item.second.table_id = table_id;
    item.second.lr = lr_.load();

    server_indices.emplace_back(item.first);
    reqs.emplace_back(std::move(item.second));
  }

  ParallelCallAsync<PushSparseTableRequest, PushSparseTableResponse>(
      RPCFuncType::kPushSparseTableType, server_indices, reqs);
}

}  // namespace kraken
