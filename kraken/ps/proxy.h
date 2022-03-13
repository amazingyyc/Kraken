#pragma once

#include <cinttypes>
#include <unordered_set>
#include <vector>

#include "common/router.h"
#include "rpc/group_connecters.h"
#include "rpc/indep_connecter.h"

namespace kraken {

class Proxy {
private:
  std::unordered_set<uint64_t> proxy_ids_;

  Router router_;
  CompressType compress_type_;

  GroupConnecters g_connecters_;

public:
  Proxy(const std::unordered_set<uint64_t>& proxy_ids, const Router& router,
        CompressType compress_type);

  ~Proxy();

public:
  int32_t TryFetchDenseTable(uint64_t table_id, std::string* name,
                             Value* value) const;

  int32_t TryCombineFetchDenseTable(const std::vector<uint64_t>& table_ids,
                                    std::vector<std::string>* names,
                                    std::vector<Value>* values) const;

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
