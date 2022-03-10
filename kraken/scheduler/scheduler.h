#pragma once

#include <cinttypes>
#include <unordered_map>
#include <unordered_set>

#include "common/info.h"
#include "common/router.h"
#include "rpc/combine_connecter.h"

namespace kraken {

class Scheduler {
private:
  // A connecter used to connect Ps/Worker.
  CombineConnecter connecter_;

  // All nodes.
  std::unordered_map<uint64_t, Node> nodes_;

  // Ps routing table.
  Router router_;

  bool model_init_;
  ModelMetaData model_mdata_;

public:
  Scheduler(CompressType compress_type);

  ~Scheduler();

public:
  void Start();

  void Stop();

  int32_t TryJoin(const std::string& addr, bool* allow, uint64_t* node_id,
                  Router* old_router, Router* new_router);

  int32_t FetchModelMetaData(bool* model_init, ModelMetaData* model_mdata);

  int32_t FetchRouter(Router *router);

  int32_t InitModel(
      const std::string& name, OptimType optim_type,
      const std::unordered_map<std::string, std::string>& optim_conf);

  int32_t RegisterDenseTable(std::string name, const Tensor& val,
                             uint64_t* table_id);

  int32_t RegisterSparseTable(
      std::string name, int64_t dimension, ElementType element_type,
      InitializerType init_type,
      const std::unordered_map<std::string, std::string>& init_conf,
      uint64_t* table_id);
};

}  // namespace kraken
