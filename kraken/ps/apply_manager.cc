#include "ps/apply_manager.h"

#include "common/error_code.h"
#include "common/log.h"

namespace kraken {

int32_t ApplyManager::ApplyModel(const std::string& name, uint64_t* model_id) {
  std::unique_lock<std::shared_mutex> lock(mu_);

  auto it = model_id_map_.find(name);
  if (it != model_id_map_.end()) {
    *model_id = it->second;

    LOG_INFO("Applied mode:" << name << ", id:" << *model_id
                             << ", already exist.");

    return ErrorCode::kSuccess;
  }

  *model_id = (uint64_t)model_id_map_.size();

  ApplyManager::Model model;
  model.id = *model_id;
  model.name = name;

  model_id_map_.emplace(name, *model_id);
  models_.emplace(*model_id, model);

  LOG_INFO("Apply a model name:" << name << ", id:" << *model_id);

  return ErrorCode::kSuccess;
}

int32_t ApplyManager::ApplyTable(uint64_t model_id, const std::string& name,
                                 TableType type, uint64_t* table_id) {
  std::unique_lock<std::shared_mutex> lock(mu_);

  auto it = models_.find(model_id);
  if (it == models_.end()) {
    return ErrorCode::kUnRegisterModelError;
  }

  ApplyManager::Model& model = it->second;

  auto tit = model.table_id_map_.find(name);
  if (tit != model.table_id_map_.end()) {
    // Already register.
    uint64_t tid = tit->second;

    if (model.tables_[tid].table_type != type) {
      return ErrorCode::kTableTypeUnCompatibleError;
    }

    *table_id = tid;

    LOG_INFO("Applied table: " << name << ", id: " << *table_id
                               << " already exist.");

    return ErrorCode::kSuccess;
  }

  *table_id = (uint64_t)model.table_id_map_.size();

  // Create a table.
  ApplyManager::Table table;
  table.id = *table_id;
  table.name = name;
  table.table_type = type;

  model.table_id_map_.emplace(name, *table_id);
  model.tables_.emplace(*table_id, table);

  LOG_INFO("Apply a table for:" << name << ", id:" << *table_id);

  return ErrorCode::kSuccess;
}

}  // namespace kraken
