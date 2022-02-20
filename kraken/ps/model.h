#pragma once

#include <memory>
#include <shared_mutex>

#include "ps/initializer/initializer.h"
#include "ps/optim/optim.h"
#include "ps/table.h"

namespace kraken {

namespace io {
class CheckpointExecutor;
class Checkpoint;
}  // namespace io

namespace watch {
class Watcher;
}

/**
 * \brief The model represent a DeepLearning model will contain the tarinable Dense/Sparse table.
 */
class Model {
  friend class io::CheckpointExecutor;
  friend class io::Checkpoint;
  friend class watch::Watcher;

private:
  std::shared_mutex mu_;

  uint64_t id_;
  std::string name_;

  std::unique_ptr<Optim> optim_;
  std::unordered_map<uint64_t, std::unique_ptr<Table>> tables_;

public:
  Model(uint64_t id, const std::string& name, std::unique_ptr<Optim>&& optim);

  ~Model() = default;

public:
  uint16_t id() const;

  const std::string& name() const;

  Optim* optim() const;

  int32_t RegisterDenseTable(uint64_t id, const std::string& name,
                             const Tensor& var);

  int32_t RegisterSparseTable(uint64_t id, const std::string& name,
                              int64_t dimension, ElementType etype,
                              std::unique_ptr<Initializer>&& initializer);

  int32_t PushDenseTable(uint64_t table_id, const Tensor& grad, float lr);

  int32_t PullDenseTable(uint64_t table_id, Tensor* val);

  int32_t CombinePullDenseTable(const std::vector<uint64_t>& table_ids,
                                std::vector<Tensor>* vals);

  int32_t PushPullDenseTable(uint64_t table_id, const Tensor& grad, float lr,
                             Tensor* val);

  int32_t PushSparseTable(uint64_t table_id,
                          const std::vector<int64_t>& indices,
                          const std::vector<Tensor>& grads, float lr);

  int32_t PullSparseTable(uint64_t table_id,
                          const std::vector<int64_t>& indices,
                          std::vector<Tensor>* vals);

  int32_t CombinePullSparseTable(
      const std::unordered_map<uint64_t, std::vector<int64_t>>& indices,
      std::unordered_map<uint64_t, std::vector<Tensor>>* vals);
};

}  // namespace kraken
