#include "ps/model_manager.h"

#include "common/error_code.h"
#include "common/log.h"

namespace kraken {

int32_t ModelManager::ApplyModel(
    const std::string& name, OptimType optim_type,
    const std::unordered_map<std::string, std::string>& optim_conf,
    uint64_t* model_id) {
  std::unique_lock<std::shared_mutex> lock(mu_);

  auto it = model_id_map_.find(name);
  if (it != model_id_map_.end()) {
    *model_id = it->second;

    // (TODO) Whether need check the optimizer or replace it?
    // This maybe some potential error. If the different worker use different Optimizer.
    LOG_INFO("Applied mode:" << name << ", id:" << *model_id
                             << ", already exist.");

    return ErrorCode::kSuccess;
  }

  *model_id = (uint64_t)model_id_map_.size();

  ModelManager::Model model;
  model.id = *model_id;
  model.name = name;
  model.optim_type = optim_type;
  model.optim_conf = optim_conf;

  model_id_map_.emplace(name, *model_id);
  models_.emplace(*model_id, model);

  LOG_INFO("Apply a model name:" << name << ", id:" << *model_id
                                 << ", optim type:" << (int32_t)optim_type);

  return ErrorCode::kSuccess;
}

int32_t ModelManager::ApplyDenseTable(uint64_t model_id,
                                      const std::string& name,
                                      const Shape& shape,
                                      ElementType element_type,
                                      uint64_t* table_id) {
  std::unique_lock<std::shared_mutex> lock(mu_);

  auto it = models_.find(model_id);
  if (it == models_.end()) {
    return ErrorCode::kUnRegisterModelError;
  }

  auto& model = it->second;
  auto tit = model.table_id_map_.find(name);
  if (tit != model.table_id_map_.end()) {
    // Already register.
    uint64_t tid = tit->second;

    if (model.tables_[tid].table_type != TableType::kDense ||
        model.tables_[tid].shape != shape ||
        model.tables_[tid].element_type != element_type) {
      return ErrorCode::kDenseTableUnCompatibleError;
    }

    *table_id = tid;

    LOG_INFO("Applied DenseTable: " << name << ", id: " << *table_id
                                    << " already exist.");

    return ErrorCode::kSuccess;
  }

  *table_id = (uint64_t)model.table_id_map_.size();

  ModelManager::Table table;
  table.id = *table_id;
  table.name = name;
  table.table_type = TableType::kDense;
  table.element_type = element_type;
  table.shape = shape;

  model.table_id_map_.emplace(name, *table_id);
  model.tables_.emplace(*table_id, table);

  LOG_INFO("Apply a DenseTable name:"
           << name << ", id:" << *table_id << ", shape:" << shape.Str()
           << ", ElementTypeL:" << element_type.Name());

  return ErrorCode::kSuccess;
}

int32_t ModelManager::ApplySparseTable(
    uint64_t model_id, const std::string& name, int64_t dimension,
    ElementType element_type, InitializerType init_type,
    const std::unordered_map<std::string, std::string>& init_conf,
    uint64_t* table_id) {
  std::unique_lock<std::shared_mutex> lock(mu_);

  auto it = models_.find(model_id);
  if (it == models_.end()) {
    return ErrorCode::kUnRegisterModelError;
  }

  auto& model = it->second;
  auto tit = model.table_id_map_.find(name);
  if (tit != model.table_id_map_.end()) {
    // Already register.
    uint64_t tid = tit->second;

    if (model.tables_[tid].table_type != TableType::kSparse ||
        model.tables_[tid].element_type != element_type ||
        model.tables_[tid].dimension != dimension ||
        model.tables_[tid].init_type != init_type) {
      return ErrorCode::kSparseTableUnCompatibleError;
    }

    *table_id = tid;

    LOG_INFO("Applied SparseTable: " << name << ", id: " << *table_id
                                     << " already exist.");

    return ErrorCode::kSuccess;
  }

  *table_id = (uint64_t)model.table_id_map_.size();

  ModelManager::Table table;
  table.id = *table_id;
  table.name = name;
  table.table_type = TableType::kSparse;
  table.element_type = element_type;
  table.dimension = dimension;
  table.init_type = init_type;
  table.init_conf = init_conf;

  model.table_id_map_.emplace(name, *table_id);
  model.tables_.emplace(*table_id, table);

  LOG_INFO("Apply a SparseTable name:"
           << name << ", id:" << *table_id << ", dimension:" << dimension
           << ", ElementType:" << element_type.Name()
           << ", init type:" << (int32_t)init_type);

  return ErrorCode::kSuccess;
}

}  // namespace kraken
