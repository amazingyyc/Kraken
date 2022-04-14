#pragma once

#include <memory>

#include "worker/emitter.h"

namespace kraken {

class Worker {
private:
  std::unique_ptr<Emitter> emitter_;

public:
  Worker();

  void Initialize(const std::string& s_addr,
                  EmitterType emitter_type = EmitterType::kDefault,
                  uint64_t life_span = 1000, float eta = 0.75);

  void Stop();

  void InitModel(
      const std::string& model_name, OptimType optim_type,
      const std::unordered_map<std::string, std::string>& optim_conf);

  void UpdateLR(float lr);

  uint64_t RegisterDenseTable(const std::string& name, const Tensor& val);

  uint64_t RegisterSparseTable(
      const std::string& name, int64_t dimension, ElementType etype,
      InitializerType init_type,
      const std::unordered_map<std::string, std::string>& init_conf);

  Tensor PullDenseTable(uint64_t table_id);

  std::vector<Tensor> CombinePullDenseTable(
      const std::vector<uint64_t>& table_ids);

  void PushDenseTable(uint64_t table_id, const Tensor& grad);

  Tensor PullSparseTable(uint64_t table_id, const Tensor& indices);

  std::vector<Tensor> CombinePullSparseTable(
      const std::vector<uint64_t>& table_ids,
      const std::vector<Tensor>& indices);

  void PushSparseTable(uint64_t table_id, const Tensor& indices,
                       const Tensor& grads);

  void CombinePushSparseTable(const std::vector<uint64_t>& table_ids,
                              const std::vector<Tensor>& indices,
                              const std::vector<Tensor>& grads);
  
  bool TrySaveModel();

  bool TryLoadModelBlocked(const std::string& load_dir);
};

}  // namespace kraken
