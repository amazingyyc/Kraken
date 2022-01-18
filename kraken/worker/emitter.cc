#include "worker/emitter.h"

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

Emitter::Emitter(EmitterType type) : type_(type) {
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

  if (indices.element_type().Is<int32_t>()) {
    return PullSparseTableImpl<int32_t>(table_id, indices);
  } else if (indices.element_type().Is<uint32_t>()) {
    return PullSparseTableImpl<uint32_t>(table_id, indices);
  } else if (indices.element_type().Is<int64_t>()) {
    return PullSparseTableImpl<int64_t>(table_id, indices);
  } else if (indices.element_type().Is<uint64_t>()) {
    return PullSparseTableImpl<uint64_t>(table_id, indices);
  } else {
    RUNTIME_ERROR("Unsupported element type:" << indices.element_type().Name());
  }
}

void Emitter::PushSparseTable(uint64_t table_id, const Tensor& indices,
                              const Tensor& grads) {
  ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

  if (indices.element_type().Is<int32_t>()) {
    return PushSparseTableImpl<int32_t>(table_id, indices, grads);
  } else if (indices.element_type().Is<uint32_t>()) {
    return PushSparseTableImpl<uint32_t>(table_id, indices, grads);
  } else if (indices.element_type().Is<int64_t>()) {
    return PushSparseTableImpl<int64_t>(table_id, indices, grads);
  } else if (indices.element_type().Is<uint64_t>()) {
    return PushSparseTableImpl<uint64_t>(table_id, indices, grads);
  } else {
    RUNTIME_ERROR("Unsupported element type:" << indices.element_type().Name());
  }
}

}  // namespace kraken
