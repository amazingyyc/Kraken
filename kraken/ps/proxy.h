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
                                    std::vector<uint64_t>* exist_ids,
                                    std::vector<std::string>* names,
                                    std::vector<Value>* values) const;

  int32_t TryFetchSparseMetaData(
      uint64_t table_id, std::string* name, int64_t* dimension,
      ElementType* element_type, InitializerType* init_type,
      std::unordered_map<std::string, std::string>* init_conf);

  int32_t TryFetchSparseValues(uint64_t table_id,
                               const std::vector<uint64_t>& sparse_ids,
                               std::vector<uint64_t>* exist_sparse_ids,
                               std::vector<Value>* values);
};

}  // namespace kraken
