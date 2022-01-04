#include "ps/model.h"

#include "common/error_code.h"
#include "common/log.h"
#include "ps/dense_table.h"
#include "ps/sparse_table.h"

namespace kraken {

const size_t Model::kSparseTableSCount = 4;

Model::Model(uint64_t id, const std::string& name,
             std::unique_ptr<Optim>&& optim)
    : id_(id), name_(name), optim_(std::move(optim)) {
}

uint16_t Model::Id() const {
  return id_;
}

const std::string& Model::Name() const {
  return name_;
}

int32_t Model::RegisterDenseTable(uint64_t id, const std::string& name,
                                  const Tensor& val) {
  std::unique_lock<std::shared_mutex> lock(mu_);

  auto it = tables_.find(id);
  if (it != tables_.end()) {
    if (it->second->name() != name || it->second->type() != TableType::kDense) {
      return ErrorCode::kTableTypeUnCompatibleError;
    }

    const Tensor& exit_val = ((DenseTable*)(it->second.get()))->val();
    if (exit_val.shape() != val.shape() ||
        exit_val.element_type() != val.element_type()) {
      return ErrorCode::kDenseTabelError;
    }

    LOG_INFO("Registered DenseTable: "
             << name << ", id: " << id << ", shape: " << exit_val.shape().Str()
             << ", etype: " << exit_val.element_type().Name()
             << " already exist.");

    return ErrorCode::kSuccess;
  }

  std::unique_ptr<DenseTable> table(
      new DenseTable(optim_.get(), id, name, val));

  tables_.emplace(id, std::move(table));

  LOG_INFO("Register DenseTable:" << name << ", id:" << id
                                  << ", shape:" << val.shape().Str()
                                  << ", etype:" << val.element_type().Name());

  return ErrorCode::kSuccess;
}

int32_t Model::RegisterSparseTable(uint64_t id, const std::string& name,
                                   int64_t dimension, ElementType etype) {
  std::unique_lock<std::shared_mutex> lock(mu_);

  if (dimension <= 0) {
    return ErrorCode::kSparseDimensionError;
  }

  auto it = tables_.find(id);
  if (it != tables_.end()) {
    if (it->second->name() != name ||
        it->second->type() != TableType::kSparse) {
      return ErrorCode::kTableTypeUnCompatibleError;
    }

    SparseTable* table = (SparseTable*)(it->second.get());
    if (table->dimension() != dimension || table->etype() != etype) {
      return ErrorCode::kSparseTabelError;
    }

    LOG_INFO("Registered SparseTable:"
             << name << ", id:" << id << ", dimension:" << dimension
             << ", etype:" << etype.Name() << " already exist.");

    return ErrorCode::kSuccess;
  }

  std::unique_ptr<SparseTable> table(new SparseTable(
      optim_.get(), id, name, dimension, etype, kSparseTableSCount));

  tables_.emplace(id, std::move(table));

  LOG_INFO("Register SparseTable:" << name << ", id:" << id << ", dimension:"
                                   << dimension << ", etype:" << etype.Name());

  return ErrorCode::kSuccess;
}

int32_t Model::PushDenseTable(uint64_t table_id, const Tensor& grad, float lr) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = tables_.find(table_id);
  if (it == tables_.end()) {
    return ErrorCode::kUnRegisterTableError;
  }

  if (it->second->type() != TableType::kDense) {
    return ErrorCode::kTableTypeUnCompatibleError;
  }

  return it->second->Push(grad, lr);
}

int32_t Model::PullDenseTable(uint64_t table_id, Tensor* val) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = tables_.find(table_id);
  if (it == tables_.end()) {
    return ErrorCode::kUnRegisterTableError;
  }

  if (it->second->type() != TableType::kDense) {
    return ErrorCode::kTableTypeUnCompatibleError;
  }

  return it->second->Pull(val);
}

int32_t Model::PushPullDenseTable(uint64_t table_id, const Tensor& grad,
                                  float lr, Tensor* val) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = tables_.find(table_id);
  if (it == tables_.end()) {
    return ErrorCode::kUnRegisterTableError;
  }

  if (it->second->type() != TableType::kDense) {
    return ErrorCode::kTableTypeUnCompatibleError;
  }

  return it->second->PushPull(grad, lr, val);
}

int32_t Model::PushSparseTable(uint64_t table_id,
                               const std::vector<int64_t>& indices,
                               const std::vector<Tensor>& grads, float lr) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = tables_.find(table_id);
  if (it == tables_.end()) {
    return ErrorCode::kUnRegisterTableError;
  }

  if (it->second->type() != TableType::kSparse) {
    return ErrorCode::kTableTypeUnCompatibleError;
  }

  return it->second->Push(indices, grads, lr);
}

int32_t Model::PullSparseTable(uint64_t table_id,
                               const std::vector<int64_t>& indices,
                               std::vector<Tensor>* vars) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = tables_.find(table_id);
  if (it == tables_.end()) {
    return ErrorCode::kUnRegisterTableError;
  }

  if (it->second->type() != TableType::kSparse) {
    return ErrorCode::kTableTypeUnCompatibleError;
  }

  return it->second->Pull(indices, vars);
}

}  // namespace kraken
