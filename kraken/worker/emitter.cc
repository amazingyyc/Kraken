#include "worker/emitter.h"

// #include <tuple>

#include "common/exception.h"
#include "common/log.h"
#include "protocol/combine_pull_dense_table_prot.h"
#include "protocol/fetch_router_prot.h"
#include "protocol/init_model_prot.h"
#include "protocol/pull_dense_table_prot.h"
#include "protocol/push_dense_table_prot.h"
#include "protocol/register_dense_table_prot.h"
#include "protocol/register_sparse_table_prot.h"
#include "protocol/rpc_func_type.h"

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
  for (const auto& [_, v] : old_router.nodes()) {
    if (router_.nodes().find(v.id) == router_.nodes().end()) {
      clients_.Remove(v.id);
    }
  }

  for (const auto& [_, v] : router_.nodes()) {
    clients_.Add(v.id, v.name);
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

  for (const auto& [_, v] : router_.nodes()) {
    clients_.Add(v.id, v.name);
  }

  // set flag.
  initialized_ = true;
}

void Emitter::Stop() {
  ARGUMENT_CHECK(initialized_, "Emitter not initialize.");

  s_connecter_->Stop();
  s_connecter_.reset(nullptr);

  initialized_ = false;
}

void Emitter::UpdateLR(float lr) {
  lr_ = lr;
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

uint64_t Emitter::RegisterDenseTable(const std::string& name,
                                     const Tensor& val) {
  ARGUMENT_CHECK(initialized_, "Emitter not initialize.");

  RegisterDenseTableRequest req;
  req.name = name;
  req.val = val;
  RegisterDenseTableResponse reply;

  RPC_CALL(
      s_connecter_->Call(RPCFuncType::kRegisterDenseTableType, req, &reply));

  LOG_INFO("Register DenseTable:" << name << ", id:" << reply.table_id);

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

  LOG_INFO("Register SparseTable: " << name << ", id: " << reply.table_id);

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
    LOG_WARNING("PushDenseTable got reply error code:"
                << ecode << ", msg:" << ErrorCode::Msg(ecode));
  };

  clients_.CallAsync<PushDenseTableRequest, PushDenseTableResponse>(
      node_id, RPCFuncType::kPushDenseTableType, req, std::move(callback));
}

// Tensor Emitter::PullDenseTable(uint64_t table_id) {
//   ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

//   size_t server_id = DenseTableRouter(model_id_, table_id);

//   PullDenseTableRequest req;
//   PullDenseTableResponse rsp;

//   req.model_id = model_id_;
//   req.table_id = table_id;

//   RPC_CALL(
//       clients_[server_id]->Call(RPCFuncType::kPullDenseTableType, req,
//       &rsp));

//   return rsp.val;
// }

// std::vector<Tensor> Emitter::CombinePullDenseTable(
//     const std::vector<uint64_t>& table_ids) {
//   ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

//   std::vector<size_t> table_val_idx;
//   table_val_idx.resize(table_ids.size());

//   std::vector<CombinePullDenseTableRequest> reqs;
//   reqs.resize(clients_.size());

//   std::vector<CombinePullDenseTableResponse> rsps;

//   std::vector<bool> mask(clients_.size(), false);

//   for (size_t i = 0; i < table_ids.size(); ++i) {
//     uint64_t table_id = table_ids[i];
//     size_t server_id = DenseTableRouter(model_id_, table_id);

//     mask[server_id] = true;

//     table_val_idx[i] = reqs[server_id].table_ids.size();
//     reqs[server_id].table_ids.emplace_back(table_id);
//   }

//   for (size_t i = 0; i < mask.size(); ++i) {
//     if (mask[i]) {
//       reqs[i].model_id = model_id_;
//     }
//   }

//   RPC_CALL(
//       ParallelCall(RPCFuncType::kCombinePullDenseTableType, mask, reqs,
//       &rsps));

//   std::vector<Tensor> vals;
//   vals.reserve(table_ids.size());

//   for (size_t i = 0; i < table_ids.size(); ++i) {
//     uint64_t table_id = table_ids[i];
//     size_t server_id = DenseTableRouter(model_id_, table_id);

//     vals.emplace_back(rsps[server_id].vals.at(table_val_idx[i]));
//   }

//   return vals;
// }

// void Emitter::PushDenseTable(uint64_t table_id, const Tensor& grad) {
//   ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

//   size_t server_id = DenseTableRouter(model_id_, table_id);

//   PushDenseTableRequest req;
//   req.model_id = model_id_;
//   req.table_id = table_id;
//   req.grad = grad;
//   req.lr = lr_.load();

//   auto callback = [](int32_t ecode, PushDenseTableResponse& /*not care*/) {
//     RPC_CALL(ecode);
//   };

//   clients_[server_id]->CallAsync<PushDenseTableRequest,
//   PushDenseTableResponse>(
//       RPCFuncType::kPushDenseTableType, req, std::move(callback));
// }

// Tensor Emitter::PushPullDenseTable(uint64_t table_id, const Tensor& grad) {
//   ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

//   size_t server_id = DenseTableRouter(model_id_, table_id);

//   PushPullDenseTableRequest req;
//   PushPullDenseTableResponse rsp;

//   req.model_id = model_id_;
//   req.table_id = table_id;
//   req.grad = grad;
//   req.lr = lr_.load();

//   RPC_CALL(clients_[server_id]->Call(RPCFuncType::kPushPullDenseTableType,
//   req,
//                                      &rsp));

//   return rsp.val;
// }

// Tensor Emitter::PullSparseTable(uint64_t table_id, const Tensor& indices) {
//   ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

//   // Maybe share memory with pytorch.
//   Tensor indices_u64 = indices.Cast(ElementType::From<uint64_t>());

//   int64_t row = indices_u64.Size();
//   uint64_t* ptr = indices_u64.Data<uint64_t>();

//   // The index in Req/Rsp for every sparse_id.
//   std::unordered_map<int64_t, size_t> sparse_indices;
//   sparse_indices.reserve(row);

//   std::vector<PullSparseTableRequest> reqs;
//   std::vector<PullSparseTableResponse> rsps;
//   reqs.resize(clients_.size());

//   std::vector<bool> mask(clients_.size(), false);

//   for (int64_t i = 0; i < row; ++i) {
//     uint64_t sparse_id = ptr[i];

//     if (sparse_indices.find(sparse_id) == sparse_indices.end()) {
//       size_t server_id = SparseTableRouter(model_id_, table_id, sparse_id);

//       mask[server_id] = true;

//       sparse_indices[sparse_id] = reqs[server_id].indices.size();
//       reqs[server_id].indices.emplace_back(sparse_id);
//     }
//   }

//   for (size_t i = 0; i < reqs.size(); ++i) {
//     if (mask[i]) {
//       reqs[i].model_id = model_id_;
//       reqs[i].table_id = table_id;
//     }
//   }

//   // Send to server.
//   RPC_CALL(ParallelCall(RPCFuncType::kPullSparseTableType, mask, reqs,
//   &rsps));

//   std::vector<Tensor> vals;
//   vals.reserve(row);

//   for (int64_t i = 0; i < row; ++i) {
//     uint64_t sparse_id = ptr[i];
//     size_t server_id = SparseTableRouter(model_id_, table_id, sparse_id);

//     size_t val_i = sparse_indices[sparse_id];

//     vals.emplace_back(rsps.at(server_id).vals.at(val_i));
//   }

//   Tensor val = indices_u64.ConcatVector(vals);

//   std::vector<int64_t> dims = indices_u64.shape().dims();
//   int64_t col = val.Size() / indices_u64.Size();

//   dims.emplace_back(col);

//   return val.Reshape(dims);
// }

// std::vector<Tensor> Emitter::CombinePullSparseTable(
//     const std::vector<uint64_t>& table_ids,
//     const std::vector<Tensor>& indices) {
//   ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");
//   ARGUMENT_CHECK(
//       table_ids.size() == indices.size(),
//       "CombinePullSparseTable need table_ids and indices has same size.");

//   // Maybe share memory with pytorch.
//   std::vector<Tensor> indice_u64s;
//   indice_u64s.reserve(indices.size());
//   for (size_t i = 0; i < indices.size(); ++i) {
//     indice_u64s.emplace_back(indices[i].Cast(ElementType::From<uint64_t>()));
//   }

//   std::vector<CombinePullSparseTableRequest> reqs;
//   std::vector<CombinePullSparseTableResponse> rsps;
//   reqs.resize(clients_.size());

//   std::vector<bool> mask(clients_.size(), false);

//   // <talbe_id, sparse_id>
//   using KeyT = std::pair<uint64_t, int64_t>;
//   struct KeyHash : public std::unary_function<KeyT, std::size_t> {
//     std::size_t operator()(const KeyT& k) const {
//       return k.first ^ k.second;
//     }
//   };

//   std::unordered_map<KeyT, size_t, KeyHash> sparse_indices;

//   for (size_t i = 0; i < table_ids.size(); ++i) {
//     uint64_t table_id = table_ids[i];

//     int64_t row = indice_u64s[i].Size();
//     uint64_t* ptr = indice_u64s[i].Data<uint64_t>();

//     for (int64_t j = 0; j < row; ++j) {
//       uint64_t sparse_id = ptr[j];
//       size_t server_id = SparseTableRouter(model_id_, table_id, sparse_id);

//       mask[server_id] = true;

//       auto key = std::make_pair(table_id, sparse_id);

//       if (sparse_indices.find(key) == sparse_indices.end()) {
//         sparse_indices[key] = reqs[server_id].indices[table_id].size();
//         reqs[server_id].indices[table_id].emplace_back(sparse_id);
//       }
//     }
//   }

//   for (size_t i = 0; i < reqs.size(); ++i) {
//     if (mask[i]) {
//       reqs[i].model_id = model_id_;
//     }
//   }

//   RPC_CALL(ParallelCall(RPCFuncType::kCombinePullSparseTableType, mask, reqs,
//                         &rsps));

//   std::vector<Tensor> vals;
//   vals.reserve(indice_u64s.size());

//   for (size_t i = 0; i < table_ids.size(); ++i) {
//     uint64_t table_id = table_ids[i];

//     int64_t row = indice_u64s[i].Size();
//     uint64_t* ptr = indice_u64s[i].Data<uint64_t>();

//     std::vector<Tensor> vecs;
//     vecs.reserve(row);

//     for (int64_t j = 0; j < row; ++j) {
//       uint64_t sparse_id = ptr[j];
//       size_t server_id = SparseTableRouter(model_id_, table_id, sparse_id);

//       KeyT key = std::make_pair(table_id, sparse_id);
//       size_t val_i = sparse_indices[key];

//       vecs.emplace_back(rsps[server_id].vals[table_id].at(val_i));
//     }

//     Tensor val = indice_u64s[i].ConcatVector(vecs);

//     std::vector<int64_t> dims = indice_u64s[i].shape().dims();
//     int64_t col = val.Size() / indice_u64s[i].Size();

//     dims.emplace_back(col);

//     vals.emplace_back(val.Reshape(dims));
//   }

//   return vals;
// }

// void Emitter::PushSparseTable(uint64_t table_id, const Tensor& indices,
//                               const Tensor& grads) {
//   ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

//   // If the indices shape is [do, d1,...,dk] than grads shape must be [do,
//   // d1,...,dk, dimension]
//   std::vector<int64_t> dims = indices.shape().dims();
//   int64_t dimension = grads.shape()[-1];
//   dims.emplace_back(dimension);

//   ARGUMENT_CHECK(Shape(dims) == grads.shape(),
//                  "PushSparseTable indices and grads shape error.");

//   Tensor indices_u64 = indices.Cast(ElementType::From<uint64_t>());
//   int64_t row = indices_u64.Size();
//   uint64_t* ptr = indices_u64.Data<uint64_t>();

//   Tensor m_grads = grads.Reshape({row, dimension});

//   std::unordered_map<int64_t, size_t> grad_indices;
//   grad_indices.reserve(row);

//   std::vector<PushSparseTableRequest> reqs;
//   reqs.resize(clients_.size());

//   std::vector<bool> mask(clients_.size(), false);

//   for (int64_t i = 0; i < row; ++i) {
//     uint64_t sparse_id = ptr[i];
//     size_t server_id = SparseTableRouter(model_id_, table_id, sparse_id);

//     mask[server_id] = true;

//     auto it = grad_indices.find(sparse_id);
//     if (it == grad_indices.end()) {
//       grad_indices[sparse_id] = reqs[server_id].indices.size();

//       reqs[server_id].indices.emplace_back(sparse_id);
//       reqs[server_id].grads.emplace_back(m_grads.Vector(i).Clone());
//     } else {
//       reqs[server_id].grads[it->second] += m_grads.Vector(i);
//     }
//   }

//   // Read Atomic value to local.
//   float l_lr = lr_.load();
//   for (size_t i = 0; i < mask.size(); ++i) {
//     if (mask[i]) {
//       reqs[i].model_id = model_id_;
//       reqs[i].table_id = table_id;
//       reqs[i].lr = l_lr;
//     }
//   }

//   ParallelCallAsync<PushSparseTableRequest, PushSparseTableResponse>(
//       RPCFuncType::kPushSparseTableType, mask, reqs);
// }

// void Emitter::SaveCheckPoint() {
//   SaveCheckPointRequest req;
//   std::vector<SaveCheckPointResponse> rsps;

//   req.model_id = model_id_;

//   RPC_CALL(ParallelCallAll(RPCFuncType::kSaveCheckPointType, req, &rsps));
// }

}  // namespace kraken
