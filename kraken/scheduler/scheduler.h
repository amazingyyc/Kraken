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
  Scheduler();

  ~Scheduler();

private:
  bool LoadModelMetaData(const std::string& load_dir);

public:
  void Start();

  void Stop();

  // Call by Ps.
  int32_t TryJoin(const std::string& addr, bool* allow, uint64_t* node_id,
                  Router* old_router, Router* new_router, bool* model_init,
                  ModelMetaData* model_mdata);

  // Call by Worker.
  int32_t FetchRouter(Router* router);

  // Call by Worker.
  int32_t InitModel(
      const std::string& name, OptimType optim_type,
      const std::unordered_map<std::string, std::string>& optim_conf);

  // Call by Worker.
  int32_t RegisterDenseTable(std::string name, const Tensor& val,
                             uint64_t* table_id);

  // Call by Worker.
  int32_t RegisterSparseTable(
      std::string name, int64_t dimension, ElementType element_type,
      InitializerType init_type,
      const std::unordered_map<std::string, std::string>& init_conf,
      uint64_t* table_id);

  // Call by Worker.
  // Try to save model to disk maybe fail if s Ps is joining.
  int32_t TrySaveModel(bool* success);

  // Call by Worker
  // Try load model from disk.
  int32_t TryLoadModel(const std::string& load_dir, bool* success);

  // Call by Worker.
  // Check whethe all Ps is kWork status.
  int32_t IsAllPsWorking(bool *yes);
};

}  // namespace kraken
