#include "worker/emitter.h"

#include <assert.h>

#include "common/exception.h"
#include "common/log.h"
#include "protocol/combine_pull_dense_table_prot.h"
#include "protocol/combine_pull_sparse_table_prot.h"
#include "protocol/combine_push_sparse_table_prot.h"
#include "protocol/fetch_router_prot.h"
#include "protocol/init_model_prot.h"
#include "protocol/is_all_ps_working_prot.h"
#include "protocol/pull_dense_table_prot.h"
#include "protocol/pull_sparse_table_prot.h"
#include "protocol/push_dense_table_prot.h"
#include "protocol/push_sparse_table_prot.h"
#include "protocol/register_dense_table_prot.h"
#include "protocol/register_sparse_table_prot.h"
#include "protocol/rpc_func_type.h"
#include "protocol/try_load_model_prot.h"
#include "protocol/try_save_model_prot.h"

namespace kraken {

Emitter::Emitter() : Emitter(EmitterType::kDefault) {
}

Emitter::Emitter(EmitterType type)
    : type_(type), initialized_(false), clients_(CompressType::kSnappy) {
}

void Emitter::UpdataRouter() {
  if (initialized_ == false) {
    return;
  }

  FetchRouterRequest req;
  FetchRouterResponse reply;

  if (s_connecter_->Call(RPCFuncType::kFetchRouterType, req, &reply) !=
      ErrorCode::kSuccess) {
    return;
  }

  Router old_router = router_;
  router_ = reply.router;

  // Remove old node id.
  for (const auto& node : old_router.nodes()) {
    if (router_.Contains(node.id) == false) {
      clients_.Remove(node.id);
    }
  }

  for (const auto& node : router_.nodes()) {
    clients_.Add(node.id, node.name);
  }

  LOG_INFO("Update router success.");
  LOG_INFO("Old router:" << old_router.Str());
  LOG_INFO("New router:" << router_.Str());
}

int32_t Emitter::PullDenseTableImpl(uint64_t table_id, Tensor* val) {
  uint64_t node_id = router_.Hit(utils::Hash(table_id));

  PullDenseTableRequest req;
  req.router_version = router_.version();
  req.table_id = table_id;

  PullDenseTableResponse reply;

  int32_t error_code =
      clients_.Call(node_id, RPCFuncType::kPullDenseTableType, req, &reply);
  if (error_code != ErrorCode::kSuccess) {
    return error_code;
  }

  *val = reply.val;

  return ErrorCode::kSuccess;
}

int32_t Emitter::CombinePullDenseTableImpl(
    const std::vector<uint64_t>& table_ids, std::vector<Tensor>* vals) {
  std::unordered_map<uint64_t, CombinePullDenseTableRequest> reqs;
  reqs.reserve(table_ids.size());

  std::vector<std::pair<uint64_t, size_t>> table_val_idx;
  table_val_idx.resize(table_ids.size());

  for (size_t i = 0; i < table_ids.size(); ++i) {
    uint64_t node_id = router_.Hit(utils::Hash(table_ids[i]));

    table_val_idx[i] = std::make_pair(node_id, reqs[node_id].table_ids.size());
    reqs[node_id].table_ids.emplace_back(table_ids[i]);
  }

  for (auto& [_, v] : reqs) {
    v.router_version = router_.version();
  }

  std::unordered_map<uint64_t, CombinePullDenseTableResponse> replies;

  auto error_code =
      clients_.Call(RPCFuncType::kCombinePullDenseTableType, reqs, &replies);
  if (error_code != ErrorCode::kSuccess) {
    return error_code;
  }

  vals->reserve(table_ids.size());

  for (size_t i = 0; i < table_ids.size(); ++i) {
    uint64_t node_id = table_val_idx[i].first;
    size_t idx = table_val_idx[i].second;

    vals->emplace_back(replies[node_id].vals.at(idx));
  }

  return ErrorCode::kSuccess;
}

int32_t Emitter::PullSparseTableImpl(uint64_t table_id, const Tensor& indices,
                                     Tensor* val) {
  // Maybe share memory with pytorch.
  Tensor indices_u64 = indices.Cast(ElementType::From<uint64_t>());

  int64_t row = indices_u64.Size();
  uint64_t* ptr = indices_u64.Data<uint64_t>();

  std::unordered_map<uint64_t, std::pair<uint64_t, size_t>> sparse_idx_map;
  sparse_idx_map.reserve(row);

  std::unordered_map<uint64_t /*node id*/, PullSparseTableRequest> reqs;
  reqs.reserve(router_.nodes().size());

  std::unordered_map<uint64_t, PullSparseTableResponse> replies;

  for (int64_t i = 0; i < row; ++i) {
    uint64_t sparse_id = ptr[i];

    if (sparse_idx_map.find(sparse_id) == sparse_idx_map.end()) {
      uint64_t node_id = router_.Hit(utils::Hash(table_id, sparse_id));

      sparse_idx_map[sparse_id] =
          std::make_pair(node_id, reqs[node_id].sparse_ids.size());
      reqs[node_id].sparse_ids.emplace_back(sparse_id);
    }
  }

  for (auto& [_, v] : reqs) {
    v.router_version = router_.version();
    v.table_id = table_id;
  }

  auto error_code =
      clients_.Call(RPCFuncType::kPullSparseTableType, reqs, &replies);
  if (error_code != ErrorCode::kSuccess) {
    return error_code;
  }

  std::vector<Tensor> vals;
  vals.reserve(row);

  for (int64_t i = 0; i < row; ++i) {
    uint64_t sparse_id = ptr[i];

    uint64_t node_id = sparse_idx_map[sparse_id].first;
    size_t val_i = sparse_idx_map[sparse_id].second;

    vals.emplace_back(replies[node_id].vals.at(val_i));
  }

  *val = indices_u64.ConcatVector(vals);

  std::vector<int64_t> dims = indices_u64.shape().dims();
  int64_t col = val->Size() / indices_u64.Size();
  dims.emplace_back(col);

  *val = val->Reshape(dims);

  return ErrorCode::kSuccess;
}

int32_t Emitter::CombinePullSparseTableImpl(
    const std::vector<uint64_t>& table_ids, const std::vector<Tensor>& indices,
    std::vector<Tensor>* vals) {
  ARGUMENT_CHECK(
      table_ids.size() == indices.size(),
      "CombinePullSparseTable need table_ids/indices has same size.");

  // Maybe share memory with pytorch.
  std::vector<Tensor> indice_u64s;
  indice_u64s.reserve(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    indice_u64s.emplace_back(indices[i].Cast(ElementType::From<uint64_t>()));
  }

  // <talbe_id, sparse_id>
  using KeyType = std::pair<uint64_t, uint64_t>;
  struct KeyHash : public std::unary_function<KeyType, std::size_t> {
    std::size_t operator()(const KeyType& k) const {
      return k.first ^ k.second;
    }
  };

  std::unordered_map<KeyType, std::pair<uint64_t, size_t>, KeyHash>
      sparse_val_idx;

  std::unordered_map<uint64_t, CombinePullSparseTableRequest> reqs;
  reqs.reserve(router_.nodes().size());

  std::unordered_map<uint64_t, CombinePullSparseTableResponse> replies;

  for (size_t t = 0; t < table_ids.size(); ++t) {
    uint64_t table_id = table_ids[t];

    int64_t row = indice_u64s[t].Size();
    uint64_t* ptr = indice_u64s[t].Data<uint64_t>();

    for (int64_t j = 0; j < row; ++j) {
      uint64_t sparse_id = ptr[j];

      uint64_t node_id = router_.Hit(utils::Hash(table_id, sparse_id));

      auto key = std::make_pair(table_id, sparse_id);

      if (sparse_val_idx.find(key) == sparse_val_idx.end()) {
        sparse_val_idx[key] = std::make_pair(
            node_id, reqs[node_id].table_sparse_ids[table_id].size());
        reqs[node_id].table_sparse_ids[table_id].emplace_back(sparse_id);
      }
    }
  }

  for (auto& [_, req] : reqs) {
    req.router_version = router_.version();
  }

  auto error_code =
      clients_.Call(RPCFuncType::kCombinePullSparseTableType, reqs, &replies);
  if (error_code != ErrorCode::kSuccess) {
    return error_code;
  }

  vals->reserve(indice_u64s.size());

  for (size_t t = 0; t < table_ids.size(); ++t) {
    uint64_t table_id = table_ids[t];

    int64_t row = indice_u64s[t].Size();
    uint64_t* ptr = indice_u64s[t].Data<uint64_t>();

    std::vector<Tensor> vecs;
    vecs.reserve(row);

    for (int64_t j = 0; j < row; ++j) {
      uint64_t sparse_id = ptr[j];

      auto key = std::make_pair(table_id, sparse_id);
      auto pos = sparse_val_idx[key];

      vecs.emplace_back(
          replies[pos.first].table_vals.at(table_id).at(pos.second));
    }

    Tensor val = indice_u64s[t].ConcatVector(vecs);

    std::vector<int64_t> dims = indice_u64s[t].shape().dims();
    int64_t col = val.Size() / indice_u64s[t].Size();

    dims.emplace_back(col);

    vals->emplace_back(val.Reshape(dims));
  }

  return ErrorCode::kSuccess;
}

void Emitter::Initialize(const std::string& s_addr) {
  if (initialized_) {
    return;
  }

  LOG_INFO("Try to connect scheduler:" << s_addr);
  s_connecter_.reset(new IndepConnecter(s_addr, CompressType::kNo));
  s_connecter_->Start();

  // Fetch router.
  FetchRouterRequest req;
  FetchRouterResponse reply;

  RPC_CALL(s_connecter_->Call(RPCFuncType::kFetchRouterType, req, &reply));
  router_ = reply.router;

  LOG_INFO("Fetch router:" << router_.Str());

  for (const auto& node : router_.nodes()) {
    clients_.Add(node.id, node.name);
  }

  // set flag.
  initialized_ = true;
}

void Emitter::Stop() {
  ARGUMENT_CHECK(initialized_, "Emitter not initialize.");

  clients_.RemoveAll();

  s_connecter_->Stop();
  s_connecter_.reset(nullptr);

  initialized_ = false;
}

void Emitter::InitModel(
    const std::string& model_name, OptimType optim_type,
    const std::unordered_map<std::string, std::string>& optim_conf) {
  ARGUMENT_CHECK(initialized_, "Emitter not initialize.");

  model_name_ = model_name;

  InitModelRequest req;
  req.name = model_name;
  req.optim_type = optim_type;
  req.optim_conf = optim_conf;
  InitModelResponse reply;

  RPC_CALL(s_connecter_->Call(RPCFuncType::kInitModelType, req, &reply));
}

void Emitter::UpdateLR(float lr) {
  lr_ = lr;
}

uint64_t Emitter::RegisterDenseTable(const std::string& name,
                                     const Tensor& val) {
  ARGUMENT_CHECK(initialized_, "Emitter not initialize.");

  RegisterDenseTableRequest req;
  req.name = name;
  req.val = val;
  RegisterDenseTableResponse reply;

  RPC_CALL(
      s_connecter_->Call(RPCFuncType::kRegisterDenseTableType, req, &reply));

  LOG_INFO("Register DenseTable:[" << name << "], id:[" << reply.table_id
                                   << "]");

  return reply.table_id;
}

uint64_t Emitter::RegisterSparseTable(
    const std::string& name, int64_t dimension, ElementType element_type,
    InitializerType init_type,
    const std::unordered_map<std::string, std::string>& init_conf) {
  RegisterSparseTableRequest req;
  req.name = name;
  req.dimension = dimension;
  req.element_type = element_type;
  req.init_type = init_type;
  req.init_conf = init_conf;

  RegisterSparseTableResponse reply;

  RPC_CALL(
      s_connecter_->Call(RPCFuncType::kRegisterSparseTableType, req, &reply));

  LOG_INFO("Register SparseTable:[" << name << "], id:[" << reply.table_id
                                    << "]");

  return reply.table_id;
}

Tensor Emitter::PullDenseTable(uint64_t table_id) {
  ARGUMENT_CHECK(initialized_, "Emitter not initialize.");

  Tensor val;

  auto error_code = PullDenseTableImpl(table_id, &val);
  if (error_code == ErrorCode::kRouterVersionError) {
    UpdataRouter();

    RPC_CALL(PullDenseTableImpl(table_id, &val));

    return val;
  } else {
    RPC_CALL(error_code);

    return val;
  }
}

std::vector<Tensor> Emitter::CombinePullDenseTable(
    const std::vector<uint64_t>& table_ids) {
  ARGUMENT_CHECK(initialized_, "Emitter not initialize.");

  std::vector<Tensor> vals;

  auto error_code = CombinePullDenseTableImpl(table_ids, &vals);
  if (error_code == ErrorCode::kRouterVersionError) {
    UpdataRouter();

    // Try agian.
    RPC_CALL(CombinePullDenseTableImpl(table_ids, &vals));

    return vals;
  } else {
    RPC_CALL(error_code);

    return vals;
  }
}

void Emitter::PushDenseTable(uint64_t table_id, const Tensor& grad) {
  ARGUMENT_CHECK(initialized_, "Emitter not initialize.");

  uint64_t node_id = router_.Hit(utils::Hash(table_id));

  PushDenseTableRequest req;
  req.router_version = router_.version();
  req.table_id = table_id;
  req.grad = grad;
  req.lr = lr_;

  // never use.
  // PushDenseTableResponse reply;

  auto callback = [](int32_t ecode, PushDenseTableResponse& /*not care*/) {
    // For perf the push always be async and never check the error code.
    // Even we get the Timeout/WrongRouter etc error. we still let it go.
    // WrongRouter can be fixed when call PullDenseTable. The timeout also fix
    // or get exception by other function. Even Push grad to Ps get fail it
    // still not affect the model (lost one step of gradient will not make the
    // DeepModel to be "wrong").
    if (ecode != ErrorCode::kSuccess) {
      LOG_WARNING("PushDenseTable got error code:"
                  << ecode << ", msg:" << ErrorCode::Msg(ecode)
                  << ", we not handle Push error!");
    }
  };

  clients_.CallAsync<PushDenseTableRequest, PushDenseTableResponse>(
      node_id, RPCFuncType::kPushDenseTableType, req, std::move(callback));
}

Tensor Emitter::PullSparseTable(uint64_t table_id, const Tensor& indices) {
  ARGUMENT_CHECK(initialized_, "Emitter not initialize.");

  Tensor val;

  auto error_code = PullSparseTableImpl(table_id, indices, &val);
  if (error_code == ErrorCode::kRouterVersionError) {
    UpdataRouter();

    // Try agian.
    RPC_CALL(PullSparseTableImpl(table_id, indices, &val));

    return val;
  } else {
    RPC_CALL(error_code);

    return val;
  }
}

std::vector<Tensor> Emitter::CombinePullSparseTable(
    const std::vector<uint64_t>& table_ids,
    const std::vector<Tensor>& indices) {
  ARGUMENT_CHECK(initialized_, "Emitter not initialize.");

  std::vector<Tensor> vals;
  auto error_code = CombinePullSparseTableImpl(table_ids, indices, &vals);

  if (error_code == ErrorCode::kRouterVersionError) {
    UpdataRouter();

    // Try agian.
    RPC_CALL(CombinePullSparseTableImpl(table_ids, indices, &vals));

    return vals;
  } else {
    RPC_CALL(error_code);

    return vals;
  }
}

void Emitter::PushSparseTable(uint64_t table_id, const Tensor& indices,
                              const Tensor& grads) {
  ARGUMENT_CHECK(initialized_, "Emitter not initialize.");

  // If the indices shape is [do, d1,...,dk] than grads shape must be [do,
  // d1,...,dk, dimension]
  std::vector<int64_t> dims = indices.shape().dims();
  int64_t dimension = grads.shape()[-1];
  dims.emplace_back(dimension);

  ARGUMENT_CHECK(Shape(dims) == grads.shape(),
                 "PushSparseTable indices and grads shape error.");

  Tensor indices_u64 = indices.Cast(ElementType::From<uint64_t>());
  int64_t row = indices_u64.Size();
  uint64_t* ptr = indices_u64.Data<uint64_t>();

  Tensor m_grads = grads.Reshape({row, dimension});

  std::unordered_map<uint64_t, std::pair<uint64_t, size_t>> sparse_idx_map;
  sparse_idx_map.reserve(row);

  std::unordered_map<uint64_t, PushSparseTableRequest> reqs;
  reqs.reserve(router_.nodes().size());

  for (int64_t i = 0; i < row; ++i) {
    uint64_t sparse_id = ptr[i];

    auto it = sparse_idx_map.find(sparse_id);
    if (it == sparse_idx_map.end()) {
      uint64_t node_id = router_.Hit(utils::Hash(table_id, sparse_id));

      sparse_idx_map[sparse_id] =
          std::make_pair(node_id, reqs[node_id].sparse_ids.size());

      reqs[node_id].sparse_ids.emplace_back(sparse_id);
      reqs[node_id].grads.emplace_back(m_grads.Vector(i).Clone());
    } else {
      reqs[it->second.first].grads[it->second.second] += m_grads.Vector(i);
    }
  }

  for (auto& [_, v] : reqs) {
    v.router_version = router_.version();
    v.table_id = table_id;
    v.lr = lr_;
  }

  auto callback = [](int32_t error_code,
                     PushSparseTableResponse& /*not care*/) {
    if (error_code != ErrorCode::kSuccess) {
      LOG_WARNING("PushSparseTable got error code:"
                  << error_code << ", msg:" << ErrorCode::Msg(error_code)
                  << ", we not handle Push error!");
    }
  };

  clients_.CallAsync<PushSparseTableRequest, PushSparseTableResponse>(
      RPCFuncType::kPushSparseTableType, reqs, callback);
}

void Emitter::CombinePushSparseTable(const std::vector<uint64_t>& table_ids,
                                     const std::vector<Tensor>& indices,
                                     const std::vector<Tensor>& grads) {
  ARGUMENT_CHECK(initialized_, "Emitter not initialize.");
  ARGUMENT_CHECK(
      table_ids.size() == indices.size() && table_ids.size() == grads.size(),
      "CombinePushSparseTable args error!");

  std::unordered_map<uint64_t, CombinePushSparseTableRequest> reqs;
  reqs.reserve(router_.nodes().size());

  // <talbe_id, sparse_id>
  using KeyType = std::pair<uint64_t, uint64_t>;
  struct KeyHash : public std::unary_function<KeyType, std::size_t> {
    std::size_t operator()(const KeyType& k) const {
      return k.first ^ k.second;
    }
  };

  // key:<talbe_id, sparse_id> value:<node_id, idx>
  std::unordered_map<KeyType, std::pair<uint64_t, size_t>, KeyHash>
      sparse_idx_map;

  for (size_t t = 0; t < table_ids.size(); ++t) {
    uint64_t table_id = table_ids[t];

    std::vector<int64_t> dims = indices[t].shape().dims();
    int64_t dimension = grads[t].shape()[-1];
    dims.emplace_back(dimension);

    ARGUMENT_CHECK(Shape(dims) == grads[t].shape(),
                   "PushSparseTable indices and grads shape error.");

    Tensor indices_u64 = indices[t].Cast(ElementType::From<uint64_t>());

    int64_t row = indices_u64.Size();
    uint64_t* ptr = indices_u64.Data<uint64_t>();

    Tensor m_grad = grads[t].Reshape({row, dimension});

    for (int64_t i = 0; i < row; ++i) {
      uint64_t sparse_id = ptr[i];

      KeyType key = std::make_pair(table_id, sparse_id);
      auto it = sparse_idx_map.find(key);
      if (it == sparse_idx_map.end()) {
        uint64_t node_id = router_.Hit(utils::Hash(table_id, sparse_id));

        sparse_idx_map[key] = std::make_pair(
            node_id, reqs[node_id].table_items[table_id].sparse_ids.size());

        reqs[node_id].table_items[table_id].sparse_ids.emplace_back(sparse_id);
        reqs[node_id].table_items[table_id].grads.emplace_back(
            m_grad.Vector(i).Clone());
      } else {
        uint64_t node_id = it->second.first;
        size_t idx = it->second.second;

        reqs[node_id].table_items[table_id].grads[idx] += m_grad.Vector(i);
      }
    }
  }

  for (auto& [_, req] : reqs) {
    req.router_version = router_.version();
    req.lr = lr_;
  }

  auto callback = [](int32_t error_code,
                     CombinePushSparseTableResponse& /*not care*/) {
    if (error_code != ErrorCode::kSuccess) {
      LOG_WARNING("CombinePushSparseTable got error code:"
                  << error_code << ", msg:" << ErrorCode::Msg(error_code)
                  << ", we not handle Push error!");
    }
  };

  clients_
      .CallAsync<CombinePushSparseTableRequest, CombinePushSparseTableResponse>(
          RPCFuncType::kCombinePushSparseTableType, reqs, callback);
}

bool Emitter::TrySaveModel() {
  TrySaveModelRequest req;
  TrySaveModelResponse reply;

  auto error_code =
      s_connecter_->Call(RPCFuncType::kTrySaveModelType, req, &reply);

  if (error_code != ErrorCode::kSuccess) {
    LOG_ERROR("Try save model get error:" << ErrorCode::Msg(error_code));
    return false;
  }

  return reply.success;
}

bool Emitter::TryLoadModelBlocked(const std::string& load_dir) {
  TryLoadModelRequest req;
  req.load_dir = load_dir;
  TryLoadModelResponse reply;

  auto error_code =
      s_connecter_->Call(RPCFuncType::kTryLoadModelType, req, &reply);

  if (error_code != ErrorCode::kSuccess) {
    LOG_ERROR("Try load model get error:" << ErrorCode::Msg(error_code));
    return false;
  }

  if (reply.success == false) {
    return false;
  }

  // Notify Ps load model success. will wait Ps finish load.
  int64_t sleep_s = 5;
  do {
    IsAllPsWorkingRequest req;
    IsAllPsWorkingResponse reply;

    auto error_code =
        s_connecter_->Call(RPCFuncType::kIsAllPsWorkingType, req, &reply);

    if (error_code != ErrorCode::kSuccess || reply.yes == false) {
      LOG_INFO("Some Ps still loading model, need wait:" << sleep_s << "s.");
      std::this_thread::sleep_for(std::chrono::seconds(sleep_s));
      sleep_s += sleep_s / 2;
    } else {
      return true;
    }
  } while (true);
}

}  // namespace kraken
