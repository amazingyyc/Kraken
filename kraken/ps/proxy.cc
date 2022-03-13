#include "ps/proxy.h"

#include "common/utils.h"
#include "protocol/rpc_func_type.h"
#include "protocol/try_combine_fetch_dense_table_prot.h"
#include "protocol/try_fetch_dense_table_prot.h"

namespace kraken {

Proxy::Proxy(const std::unordered_set<uint64_t>& proxy_ids,
             const Router& router, CompressType compress_type)
    : proxy_ids_(proxy_ids),
      router_(router),
      compress_type_(compress_type),
      g_connecters_(compress_type) {
  for (auto id : proxy_ids) {
    const auto it = router_.nodes().find(id);
    ARGUMENT_CHECK(it != router_.nodes().end(), "Cannot find node:" << id);

    g_connecters_.Add(id, it->second.name);
  }
}

Proxy::~Proxy() {
  for (auto id : proxy_ids_) {
    g_connecters_.Remove(id);
  }
}

int32_t Proxy::TryFetchDenseTable(uint64_t table_id, std::string* name,
                                  Value* value) const {
  uint64_t node_id = router_.Hit(utils::Hash(table_id));

  auto it = proxy_ids_.find(node_id);
  if (it == proxy_ids_.end()) {
    return ErrorCode::kProxyNodeIdNotExistError;
  }

  TryFetchDenseTableRequest req;
  req.id = table_id;
  TryFetchDenseTableResponse reply;

  auto error_code = g_connecters_.Call(
      node_id, RPCFuncType::kTryFetchDenseTableType, req, &reply);
  if (error_code != ErrorCode::kSuccess) {
    return error_code;
  }

  *name = reply.name;
  *value = reply.value;

  return ErrorCode::kSuccess;
}

int32_t Proxy::TryCombineFetchDenseTable(const std::vector<uint64_t>& table_ids,
                                         std::vector<std::string>* names,
                                         std::vector<Value>* values) const {
  std::unordered_map<uint64_t, TryCombineFetchDenseTableRequest> reqs;
  reqs.reserve(table_ids.size());

  std::vector<std::pair<uint64_t, size_t>> table_val_idx;
  table_val_idx.resize(table_ids.size());

  for (size_t i = 0; i < table_ids.size(); ++i) {
    uint64_t node_id = router_.Hit(utils::Hash(table_ids[i]));

    if (proxy_ids_.find(node_id) == proxy_ids_.end()) {
      return ErrorCode::kProxyNodeIdNotExistError;
    }

    table_val_idx[i] = std::make_pair(node_id, reqs[node_id].ids.size());
    reqs[node_id].ids.emplace_back(table_ids[i]);
  }

  std::unordered_map<uint64_t, TryCombineFetchDenseTableResponse> replies;

  auto error_code = g_connecters_.Call<TryCombineFetchDenseTableRequest,
                                       TryCombineFetchDenseTableResponse>(
      RPCFuncType::kTryCombineFetchDenseTableType, reqs, &replies);
  if (error_code != ErrorCode::kSuccess) {
    return error_code;
  }

  names->reserve(table_ids.size());
  values->reserve(table_ids.size());

  for (size_t i = 0; i < table_ids.size(); ++i) {
    uint64_t node_id = table_val_idx[i].first;
    size_t idx = table_val_idx[i].second;

    names->emplace_back(replies[node_id].names.at(idx));
    values->emplace_back(replies[node_id].values.at(idx));
  }

  return ErrorCode::kSuccess;
}

// int32_t Proxy::FetchDenseTableValue(uint64_t model_id, uint64_t table_id,
//                                     Table::Value* vals) {
//   return ErrorCode::kSuccess;
// }

// int32_t Proxy::FetchAcceptSparseIds(uint64_t target_node_id,
//                                     const Router& new_router, uint64_t
//                                     model_id, uint64_t table_id,
//                                     std::vector<uint64_t>* sparse_ids) {
//   return ErrorCode::kSuccess;
// }

// int32_t Proxy::FetchSparseTableValues(uint64_t model_id, uint64_t table_id,
//                                       const std::vector<uint64_t>&
//                                       sparse_ids, std::vector<Table::Value>*
//                                       vals) {
//   return ErrorCode::kSuccess;
// }

// int32_t Proxy::DeleteRedundantData(const Router& new_router) {
//   return ErrorCode::kSuccess;
// }

}  // namespace kraken
