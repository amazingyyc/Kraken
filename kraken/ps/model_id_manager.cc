#include "ps/model_id_manager.h"

#include "common/error_code.h"
#include "common/log.h"

namespace kraken {

int32_t ModelIdManager::ApplyModelId(const std::string& model_name,
                                     uint64_t* model_id) {
  std::unique_lock<std::mutex> lock(mu_);

  auto it = model_infos_.find(model_name);
  if (it != model_infos_.end()) {
    *model_id = it->second.id;

    return ErrorCode::kSuccess;
  }

  *model_id = model_infos_.size();

  ModelIdManager::ModelInfo info;
  info.id = *model_id;
  info.name = model_name;

  model_infos_.emplace(model_name, info);

  return ErrorCode::kSuccess;
}

int32_t ModelIdManager::ApplyTableId(const std::string& model_name,
                                     const std::string& table_name,
                                     uint64_t* table_id) {
  std::unique_lock<std::mutex> lock(mu_);

  auto it = model_infos_.find(model_name);
  if (it != model_infos_.end()) {
    LOG_ERROR("Model:" << model_name << " not exist!");
    return ErrorCode::kUnRegisterModelError;
  }

  auto& model_info = it->second;
  auto tit = model_info.table_infos.find(table_name);
  if (tit != model_info.table_infos.end()) {
    *table_id = tit->second.id;

    return ErrorCode::kSuccess;
  }

  *table_id = model_info.table_infos.size();

  ModelIdManager::TableInfo info;
  info.id = *table_id;
  info.name = table_name;

  model_info.table_infos.emplace(table_name, info);

  return ErrorCode::kSuccess;
}

}  // namespace kraken