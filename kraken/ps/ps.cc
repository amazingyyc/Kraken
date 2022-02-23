#include "ps/ps.h"

#include "common/error_code.h"
#include "common/exception.h"
#include "common/log.h"
#include "ps/initializer/initializer.h"
#include "ps/optim/optim.h"
#include "rpc/protocol.h"

namespace kraken {

Ps::Ps(size_t shard_num, size_t shard_id, const std::string& save_dir,
       size_t max_save_count)
    : shard_num_(shard_num),
      shard_id_(shard_id),
      checkpoint_executor_(save_dir, max_save_count) {
}

size_t Ps::shard_num() const {
  return shard_num_;
}

size_t Ps::shard_id() const {
  return shard_id_;
}

void Ps::Load(const std::string& load_dir) {
  ARGUMENT_CHECK(checkpoint_executor_.Load(this, load_dir),
                 "Load model from:" << load_dir << " error!");
}

void Ps::Stop() {
  checkpoint_executor_.Stop();
}

int32_t Ps::ApplyModel(
    const std::string& name, OptimType optim_type,
    const std::unordered_map<std::string, std::string>& optim_conf,
    uint64_t* model_id) {
  std::unique_lock<std::shared_mutex> lock(mu_);

  for (const auto& [k, v] : model_infos_) {
    if (v.name == name) {
      // (TODO) check OptimType.
      *model_id = k;

      LOG_INFO("Apply model:" << name << ", OptimType:" << (int32_t)optim_type
                              << ", OptimConf:" << optim_conf
                              << ", already exist, id:" << *model_id);

      return ErrorCode::kSuccess;
    }
  }

  uint64_t id = model_infos_.size();
  while (model_infos_.find(id) != model_infos_.end()) {
    id++;
  }

  *model_id = id;

  // Create a new one.
  ModelInfo model_info;
  model_info.id = id;
  model_info.name = name;
  model_info.optim_type = optim_type;
  model_info.optim_conf = optim_conf;

  model_infos_.emplace(id, std::move(model_info));

  LOG_INFO("Applied model:" << name << ", id:" << id
                            << ", OptimType:" << (int32_t)optim_type
                            << ", OptimConf:" << optim_conf);

  return ErrorCode::kSuccess;
}

int32_t Ps::ApplyDenseTable(uint64_t model_id, const std::string& name,
                            const Shape& shape, ElementType element_type,
                            uint64_t* table_id) {
  std::unique_lock<std::shared_mutex> lock(mu_);

  auto it = model_infos_.find(model_id);
  if (it == model_infos_.end()) {
    LOG_ERROR("Can't find model:" << model_id);
    return ErrorCode::kUnRegisterModelError;
  }

  ModelInfo& model_info = it->second;

  for (const auto& [k, v] : model_info.table_infos) {
    if (v.name == name) {
      if (v.table_type != TableType::kDense || v.shape != shape ||
          v.element_type != element_type) {
        return ErrorCode::kDenseTableUnCompatibleError;
      }

      *table_id = k;

      LOG_INFO("Apply DenseTable:" << name << ", shape:" << shape.Str()
                                   << ", ElementType" << element_type.Name()
                                   << ", already exist, id:" << k);

      return ErrorCode::kSuccess;
    }
  }

  uint64_t id = model_info.table_infos.size();
  while (model_info.table_infos.find(id) != model_info.table_infos.end()) {
    id++;
  }

  *table_id = id;

  TableInfo table_info;
  table_info.id = id;
  table_info.name = name;
  table_info.table_type = TableType::kDense;
  table_info.element_type = element_type;
  table_info.shape = shape;

  model_info.table_infos.emplace(id, std::move(table_info));

  LOG_INFO("Applied DenseTable:" << name << ", id:" << id
                                 << ", shape:" << shape.Str()
                                 << ", ElementType:" << element_type.Name());

  return ErrorCode::kSuccess;
}

int32_t Ps::ApplySparseTable(
    uint64_t model_id, const std::string& name, int64_t dimension,
    ElementType element_type, InitializerType init_type,
    const std::unordered_map<std::string, std::string>& init_conf,
    uint64_t* table_id) {
  std::unique_lock<std::shared_mutex> lock(mu_);

  auto it = model_infos_.find(model_id);
  if (it == model_infos_.end()) {
    LOG_ERROR("Can't find model:" << model_id);
    return ErrorCode::kUnRegisterModelError;
  }

  ModelInfo& model_info = it->second;

  for (const auto& [k, v] : model_info.table_infos) {
    if (v.name == name) {
      if (v.table_type != TableType::kSparse || v.dimension != dimension ||
          v.element_type != element_type || v.init_type != init_type) {
        return ErrorCode::kSparseTableUnCompatibleError;
      }

      *table_id = k;

      LOG_INFO("Apply SparseTable:" << name << ", dimension:" << dimension
                                    << ", ElementType:" << element_type.Name()
                                    << ", InitType:" << (int32_t)init_type
                                    << ", InitConf:" << init_conf
                                    << ", already exist, id:" << k);

      return ErrorCode::kSuccess;
    }
  }

  uint64_t id = model_info.table_infos.size();
  while (model_info.table_infos.find(id) != model_info.table_infos.end()) {
    id++;
  }

  *table_id = id;

  TableInfo table_info;
  table_info.id = id;
  table_info.name = name;
  table_info.table_type = TableType::kSparse;
  table_info.element_type = element_type;
  table_info.dimension = dimension;
  table_info.init_type = init_type;
  table_info.init_conf = init_conf;

  model_info.table_infos.emplace(id, std::move(table_info));

  LOG_INFO("Applied SparseTable:" << name << ", id:" << id
                                  << ", dimension:" << dimension
                                  << ", ElementType:" << element_type.Name()
                                  << ", InitType:" << (int32_t)init_type
                                  << ", InitConf:" << init_conf);

  return ErrorCode::kSuccess;
}

int32_t Ps::RegisterModel(
    uint64_t id, const std::string& name, OptimType optim_type,
    const std::unordered_map<std::string, std::string>& optim_conf) {
  std::unique_lock<std::shared_mutex> lock(mu_);

  // Try to insert ModelInfo.
  {
    auto it = model_infos_.find(id);
    if (it == model_infos_.end()) {
      ModelInfo model_info;
      model_info.id = id;
      model_info.name = name;
      model_info.optim_type = optim_type;
      model_info.optim_conf = optim_conf;

      model_infos_.emplace(id, std::move(model_info));

      LOG_INFO("Register ModelInfo:" << name << ", id" << id
                                     << ", OptimType:" << (int32_t)optim_type
                                     << ", OptimConf:" << optim_conf);
    } else {
      LOG_INFO("Register ModelInfo:"
               << name << ", OptimType:" << (int32_t)optim_type
               << ", OptimConf:" << optim_conf << ", already exist, id:" << id);
    }
  }

  {
    auto it = models_.find(id);
    if (it == models_.end()) {
      std::unique_ptr<Optim> optim = Optim::Create(optim_type, optim_conf);
      if (optim == nullptr) {
        return ErrorCode::kUnSupportOptimTypeError;
      }

      std::unique_ptr<Model> model(new Model(id, name, std::move(optim)));
      models_.emplace(id, std::move(model));

      LOG_INFO("Register model:" << name << ", id:" << id
                                 << ", OptimType:" << (int32_t)optim_type
                                 << ", OptimConf:" << optim_conf);
    } else {
      LOG_INFO("Registerd model:"
               << name << ", OptimType:" << (int32_t)optim_type
               << ", OptimConf:" << optim_conf << ", already exist, id:" << id);
    }
  }

  return ErrorCode::kSuccess;
}

int32_t Ps::RegisterDenseTableInfo(uint64_t model_id, uint64_t id,
                                   const std::string& name, const Shape& shape,
                                   ElementType element_type) {
  std::unique_lock<std::shared_mutex> lock(mu_);

  auto it = model_infos_.find(model_id);
  if (it == model_infos_.end()) {
    return ErrorCode::kUnRegisterModelError;
  }

  auto& model_info = it->second;

  auto tit = model_info.table_infos.find(id);
  if (tit != model_info.table_infos.end()) {
    if (model_info.table_infos[id].name != name ||
        model_info.table_infos[id].table_type != TableType::kDense ||
        model_info.table_infos[id].shape != shape ||
        model_info.table_infos[id].element_type != element_type) {
      return ErrorCode::kDenseTableUnCompatibleError;
    }

    LOG_INFO("Register DenseTableInfo:"
             << name << ", shape:" << shape.Str() << ", ElementType"
             << element_type.Name() << ", already exist, id:" << id);

    return ErrorCode::kSuccess;
  }

  TableInfo table_info;
  table_info.id = id;
  table_info.name = name;
  table_info.table_type = TableType::kDense;
  table_info.element_type = element_type;
  table_info.shape = shape;

  model_info.table_infos.emplace(id, std::move(table_info));

  LOG_INFO("Register DenseTableInfo:"
           << name << ", id:" << id << ", shape:" << shape.Str()
           << ", ElementType:" << element_type.Name());

  return ErrorCode::kSuccess;
}

int32_t Ps::RegisterDenseTable(uint64_t model_id, uint64_t id,
                               const std::string& name, const Tensor& var) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = models_.find(model_id);
  if (it == models_.end()) {
    return ErrorCode::kUnRegisterModelError;
  }

  return it->second->RegisterDenseTable(id, name, var);
}

int32_t Ps::RegisterSparseTable(
    uint64_t model_id, uint64_t id, const std::string& name, int64_t dimension,
    ElementType element_type, InitializerType init_type,
    const std::unordered_map<std::string, std::string>& init_conf) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  {
    // Insert TableInfo.
    auto it = model_infos_.find(model_id);
    if (it == model_infos_.end()) {
      return ErrorCode::kUnRegisterModelError;
    }

    auto& model_info = it->second;

    auto tit = model_info.table_infos.find(id);
    if (tit != model_info.table_infos.end()) {
      if (model_info.table_infos[id].name != name ||
          model_info.table_infos[id].table_type != TableType::kSparse ||
          model_info.table_infos[id].element_type != element_type ||
          model_info.table_infos[id].dimension != dimension ||
          model_info.table_infos[id].init_type != init_type) {
        return ErrorCode::kSparseTableUnCompatibleError;
      }

      LOG_INFO("Register SparseTableInfo:"
               << name << ", dimension:" << dimension << ", ElementType:"
               << element_type.Name() << ", InitType:" << (int32_t)init_type
               << ", InitConf:" << init_conf << ", already exist, id:" << id);

    } else {
      TableInfo table_info;
      table_info.id = id;
      table_info.name = name;
      table_info.table_type = TableType::kSparse;
      table_info.element_type = element_type;
      table_info.dimension = dimension;
      table_info.init_type = init_type;
      table_info.init_conf = init_conf;

      model_info.table_infos.emplace(id, std::move(table_info));

      LOG_INFO("Apply SparseTableInfo:"
               << name << ", id:" << id << ", dimension:" << dimension
               << ", ElementType:" << element_type.Name() << ", InitType:"
               << (int32_t)init_type << ", InitConf:" << init_conf);
    }
  }

  {
    // Insert SparseTable.
    std::unique_ptr<Initializer> initializer =
        Initializer::Create(init_type, init_conf);

    if (initializer == nullptr) {
      return ErrorCode::kUnSupportInitializerTypeError;
    }

    auto it = models_.find(model_id);
    if (it == models_.end()) {
      return ErrorCode::kUnRegisterModelError;
    }

    auto ecode = it->second->RegisterSparseTable(
        id, name, dimension, element_type, std::move(initializer));

    if (ecode != ErrorCode::kSuccess) {
      return ecode;
    }
  }

  return ErrorCode::kSuccess;
}

int32_t Ps::PullDenseTable(uint64_t model_id, uint64_t table_id, Tensor* val) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = models_.find(model_id);
  if (it == models_.end()) {
    return ErrorCode::kUnRegisterModelError;
  }

  return it->second->PullDenseTable(table_id, val);
}

int32_t Ps::CombinePullDenseTable(uint64_t model_id,
                                  const std::vector<uint64_t>& table_ids,
                                  std::vector<Tensor>* vals) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = models_.find(model_id);
  if (it == models_.end()) {
    return ErrorCode::kUnRegisterModelError;
  }

  return it->second->CombinePullDenseTable(table_ids, vals);
}

int32_t Ps::PushPullDenseTable(uint64_t model_id, uint64_t table_id,
                               const Tensor& grad, float lr, Tensor* val) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = models_.find(model_id);
  if (it == models_.end()) {
    return ErrorCode::kUnRegisterModelError;
  }

  return it->second->PushPullDenseTable(table_id, grad, lr, val);
}

int32_t Ps::PushDenseTable(uint64_t model_id, uint64_t table_id,
                           const Tensor& grad, float lr) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = models_.find(model_id);
  if (it == models_.end()) {
    return ErrorCode::kUnRegisterModelError;
  }

  return it->second->PushDenseTable(table_id, grad, lr);
}

int32_t Ps::PullSparseTable(uint64_t model_id, uint64_t table_id,
                            const std::vector<uint64_t>& indices,
                            std::vector<Tensor>* vals) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = models_.find(model_id);
  if (it == models_.end()) {
    return ErrorCode::kUnRegisterModelError;
  }

  return it->second->PullSparseTable(table_id, indices, vals);
}

int32_t Ps::CombinePullSparseTable(
    uint64_t model_id,
    const std::unordered_map<uint64_t, std::vector<uint64_t>>& indices,
    std::unordered_map<uint64_t, std::vector<Tensor>>* vals) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = models_.find(model_id);
  if (it == models_.end()) {
    return ErrorCode::kUnRegisterModelError;
  }

  return it->second->CombinePullSparseTable(indices, vals);
}

int32_t Ps::PushSparseTable(uint64_t model_id, uint64_t table_id,
                            const std::vector<uint64_t>& indices,
                            const std::vector<Tensor>& grads, float lr) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = models_.find(model_id);
  if (it == models_.end()) {
    return ErrorCode::kUnRegisterModelError;
  }

  return it->second->PushSparseTable(table_id, indices, grads, lr);
}

int32_t Ps::SaveCheckPoint(uint64_t model_id) {
  auto done = [](uint64_t model_id, bool success) {
    if (success == false) {
      LOG_ERROR("Save check point for model:" << model_id << " error!");
    }
  };

  // use sub-thread to save check point.
  checkpoint_executor_.Save(this, model_id, std::move(done));

  return ErrorCode::kSuccess;
}

}  // namespace kraken
