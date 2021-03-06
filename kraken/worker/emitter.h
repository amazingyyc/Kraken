#pragma once

#include <cinttypes>
#include <memory>
#include <vector>

#include "common/info.h"
#include "common/router.h"
#include "rpc/group_connecters.h"
#include "rpc/indep_connecter.h"
#include "rpc/protocol.h"

namespace kraken {

// We assume that Emitter is thread-safe guarantee by caller.
class Emitter {
protected:
  EmitterType type_;
  bool initialized_;

  // Scheduler addr.
  std::string s_addr_;

  // Scheduler connecter.
  std::unique_ptr<IndepConnecter> s_connecter_;

  // connect to Ps node.
  GroupConnecters clients_;

  Router router_;

  std::string model_name_;
  float lr_;

public:
  Emitter();

protected:
  Emitter(EmitterType type);

protected:
  void UpdataRouter();

  int32_t PullDenseTableImpl(uint64_t table_id, Tensor* val);

  int32_t CombinePullDenseTableImpl(const std::vector<uint64_t>& table_ids,
                                    std::vector<Tensor>* vals);

  int32_t PullSparseTableImpl(uint64_t table_id, const Tensor& indices,
                              Tensor* val);

  int32_t CombinePullSparseTableImpl(const std::vector<uint64_t>& table_ids,
                                     const std::vector<Tensor>& indices,
                                     std::vector<Tensor>* vals);

public:
  void Initialize(const std::string& s_addr);

  void Stop();

  void InitModel(
      const std::string& model_name, OptimType optim_type,
      const std::unordered_map<std::string, std::string>& optim_conf);

  void UpdateLR(float lr);

  virtual uint64_t RegisterDenseTable(const std::string& name,
                                      const Tensor& val);

  uint64_t RegisterSparseTable(
      const std::string& name, int64_t dimension, ElementType element_type,
      InitializerType init_type,
      const std::unordered_map<std::string, std::string>& init_conf);

  Tensor PullDenseTable(uint64_t table_id);

  std::vector<Tensor> CombinePullDenseTable(
      const std::vector<uint64_t>& table_ids);

  virtual void PushDenseTable(uint64_t table_id, const Tensor& grad);

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
