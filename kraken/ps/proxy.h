#pragma once

#include <cinttypes>
#include <unordered_set>
#include <vector>

#include "common/router.h"
#include "rpc/indep_connecter.h"

namespace kraken {

class Proxy {
private:
  Router router_;
  CompressType compress_type_;

  std::unordered_map<uint64_t /*node id*/, std::unique_ptr<IndepConnecter>>
      connecters_;

public:
  Proxy(const Router& router, CompressType compress_type);

  ~Proxy();

public:
  bool Add(const std::unordered_set<uint64_t>& proxy_ids);

  bool Add(uint64_t id);

  // int32_t FetchDenseTableValue(uint64_t model_id, uint64_t table_id,
  //                              Table::Value* vals);

  // int32_t FetchAcceptSparseIds(uint64_t target_node_id, const Router&
  // new_router,
  //                              uint64_t model_id, uint64_t table_id,
  //                              std::vector<uint64_t>* sparse_ids);

  // int32_t FetchSparseTableValues(uint64_t model_id, uint64_t table_id,
  //                                const std::vector<uint64_t>& sparse_ids,
  //                                std::vector<Table::Value>* vals);

  // int32_t DeleteRedundantData(const Router& new_router);
};

}  // namespace kraken
