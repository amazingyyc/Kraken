#include "ps/proxy.h"

#include "common/utils.h"
#include "protocol/rpc_func_type.h"
#include "protocol/try_combine_fetch_dense_table_prot.h"
#include "protocol/try_fetch_dense_table_prot.h"
#include "protocol/try_fetch_sparse_meta_data_prot.h"
#include "protocol/try_fetch_sparse_values_prot.h"

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

  TryFetchDenseTableRequest req;
  req.table_id = table_id;
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
                                         std::vector<uint64_t>* exist_table_ids,
                                         std::vector<std::string>* names,
                                         std::vector<Value>* values) const {
  std::unordered_map<uint64_t, TryCombineFetchDenseTableRequest> reqs;
  reqs.reserve(table_ids.size());

  for (size_t i = 0; i < table_ids.size(); ++i) {
    uint64_t node_id = router_.Hit(utils::Hash(table_ids[i]));

    reqs[node_id].table_ids.emplace_back(table_ids[i]);
  }

  std::unordered_map<uint64_t, TryCombineFetchDenseTableResponse> replies;

  auto error_code = g_connecters_.Call<TryCombineFetchDenseTableRequest,
                                       TryCombineFetchDenseTableResponse>(
      RPCFuncType::kTryCombineFetchDenseTableType, reqs, &replies);
  if (error_code != ErrorCode::kSuccess) {
    return error_code;
  }

  exist_table_ids->reserve(table_ids.size());
  names->reserve(table_ids.size());
  values->reserve(table_ids.size());

  for (const auto& [k, v] : replies) {
    exist_table_ids->insert(exist_table_ids->begin(), v.exist_table_ids.begin(),
                            v.exist_table_ids.end());
    names->insert(names->begin(), v.names.begin(), v.names.end());
    values->insert(values->begin(), v.values.begin(), v.values.end());
  }

  return ErrorCode::kSuccess;
}

int32_t Proxy::TryFetchSparseMetaData(
    uint64_t table_id, std::string* name, int64_t* dimension,
    ElementType* element_type, InitializerType* init_type,
    std::unordered_map<std::string, std::string>* init_conf) {
  uint64_t node_id = router_.Hit(utils::Hash(table_id));

  TryFetchSparseMetaDataRequest req;
  req.table_id = table_id;
  TryFetchSparseMetaDataResponse reply;

  auto error_code = g_connecters_.Call(
      node_id, RPCFuncType::kTryFetchSparseMetaDataType, req, &reply);
  if (error_code != ErrorCode::kSuccess) {
    return error_code;
  }

  *name = reply.name;
  *dimension = reply.dimension;
  *element_type = reply.element_type;
  *init_type = reply.init_type;
  *init_conf = reply.init_conf;

  return ErrorCode::kSuccess;
}

int32_t Proxy::TryFetchSparseValues(uint64_t table_id,
                                    const std::vector<uint64_t>& sparse_ids,
                                    std::vector<uint64_t>* exist_sparse_ids,
                                    std::vector<Value>* values) {
  std::unordered_map<uint64_t, TryFetchSparseValuesRequest> reqs;

  for (size_t i = 0; i < sparse_ids.size(); ++i) {
    uint64_t node_id = router_.Hit(utils::Hash(table_id, sparse_ids[i]));

    reqs[node_id].sparse_ids.emplace_back(sparse_ids[i]);
  }

  for (auto& [_, v] : reqs) {
    v.table_id = table_id;
  }

  std::unordered_map<uint64_t, TryFetchSparseValuesResponse> replies;

  auto error_code = g_connecters_.Call(RPCFuncType::kTryFetchSparseValuesType,
                                       reqs, &replies);
  if (error_code != ErrorCode::kSuccess) {
    return error_code;
  }

  exist_sparse_ids->reserve(sparse_ids.size());
  values->reserve(sparse_ids.size());

  for (auto& [_, reply] : replies) {
    exist_sparse_ids->insert(exist_sparse_ids->end(),
                             reply.exist_sparse_ids.begin(),
                             reply.exist_sparse_ids.end());
    values->insert(values->end(), reply.values.begin(), reply.values.end());
  }

  return ErrorCode::kSuccess;
}

}  // namespace kraken
