#include "ps/proxy.h"

namespace kraken {

Proxy::Proxy(const Router& router, CompressType compress_type)
    : router_(router), compress_type_(compress_type) {
}

Proxy::~Proxy() {
  for (auto& [k, v] : connecters_) {
    v->Stop();
  }

  connecters_.clear();
}

bool Proxy::Add(const std::unordered_set<uint64_t>& proxy_ids) {
  for (auto id : proxy_ids) {
    if (connecters_.find(id) != connecters_.end()) {
      return false;
    }

    const auto& it = router_.nodes().find(id);
    if (it == router_.nodes().end()) {
      return false;
    }

    std::unique_ptr<IndepConnecter> connecter(
        new IndepConnecter(it->second.name, compress_type_));
    connecter->Start();

    connecters_.emplace(id, std::move(connecter));
  }

  return true;
}

bool Proxy::Add(uint64_t id) {
  if (connecters_.find(id) != connecters_.end()) {
    return false;
  }

  const auto& it = router_.nodes().find(id);
  if (it == router_.nodes().end()) {
    return false;
  }

  std::unique_ptr<IndepConnecter> connecter(
      new IndepConnecter(it->second.name, compress_type_));
  connecter->Start();

  connecters_.emplace(id, std::move(connecter));

  return true;
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
