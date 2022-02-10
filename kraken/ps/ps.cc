#include "ps/ps.h"

#include "common/error_code.h"
#include "common/log.h"
#include "ps/initializer/constant_initializer.h"
#include "ps/initializer/normal_initializer.h"
#include "ps/initializer/uniform_initializer.h"
#include "ps/initializer/xavier_normal_initializer.h"
#include "ps/initializer/xavier_uniform_initializer.h"
#include "ps/optim/adagrad.h"
#include "ps/optim/adam.h"
#include "ps/optim/rmsprop.h"
#include "ps/optim/sgd.h"
#include "rpc/protocol.h"

namespace kraken {

Ps::Ps(size_t shard_num, size_t shard_id)
    : shard_num_(shard_num), shard_id_(shard_id) {
}

size_t Ps::shard_num() const {
  return shard_num_;
}

size_t Ps::shard_id() const {
  return shard_id_;
}

int32_t Ps::ApplyModelId(const std::string& model_name, uint64_t* model_id) {
  return model_id_manager_.ApplyModelId(model_name, model_id);
}

int32_t Ps::ApplyTableId(const std::string& model_name,
                         const std::string& table_name, uint64_t* table_id) {
  return model_id_manager_.ApplyTableId(model_name, table_name, table_id);
}

int32_t Ps::RegisterModel(
    uint64_t id, const std::string& name, OptimType optim_type,
    const std::unordered_map<std::string, std::string>& optim_conf) {
  std::unique_lock<std::shared_mutex> lock(mu_);

  // Need insert into model_infos_/models_.
  {
    auto it = model_infos_.find(id);
    if (it == model_infos_.end()) {
      Ps::ModelInfo model_info;
      model_info.id = id;
      model_info.name = name;
      model_info.optim_type = optim_type;
      model_info.optim_conf = optim_conf;

      model_infos_.emplace(id, std::move(model_info));

      LOG_INFO("Register model info:" << name << ", id:" << id
                                      << ", name:" << name << ", optim_type:"
                                      << (uint32_t)optim_type);
    } else {
      LOG_INFO("Register model info:" << name << ", id:" << id
                                      << " already exist.");
    }
  }

  {
    auto it = models_.find(id);
    if (it == models_.end()) {
      std::unique_ptr<Optim> optim;
      if (optim_type == OptimType::kAdagrad) {
        optim.reset(new Adagrad(optim_conf));
      } else if (optim_type == OptimType::kAdam) {
        optim.reset(new Adam(optim_conf));
      } else if (optim_type == OptimType::kRMSprop) {
        optim.reset(new RMSprop(optim_conf));
      } else if (optim_type == OptimType::kSGD) {
        optim.reset(new SGD(optim_conf));
      } else {
        return ErrorCode::kUnSupportOptimTypeError;
      }

      std::unique_ptr<Model> model(new Model(id, name, std::move(optim)));
      models_.emplace(id, std::move(model));

      LOG_INFO("Register model:" << name << ", id:" << id << ", name:" << name
                                 << ", optim_type:" << (uint32_t)optim_type);
    } else {
      LOG_INFO("Registerd model: " << name << ", id: " << it->second->id()
                                   << " already exist.");
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

    LOG_INFO("Register DenseTableInfo: " << name << ", id: " << id
                                         << " already exist.");

    return ErrorCode::kSuccess;
  }

  Ps::TableInfo table_info;
  table_info.id = id;
  table_info.name = name;
  table_info.table_type = TableType::kDense;
  table_info.element_type = element_type;
  table_info.shape = shape;

  model_info.table_infos.emplace(id, std::move(table_info));

  LOG_INFO("Register DenseTableInfo name:"
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

int32_t Ps::RegisterSparseTableInfo(
    uint64_t model_id, uint64_t id, const std::string& name, int64_t dimension,
    ElementType element_type, InitializerType init_type,
    const std::unordered_map<std::string, std::string>& init_conf) {
  std::unique_lock<std::shared_mutex> lock(mu_);

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

    LOG_INFO("Register SparseTableInfo: " << name << ", id: " << id
                                          << " already exist.");

    return ErrorCode::kSuccess;
  }

  Ps::TableInfo table_info;
  table_info.id = id;
  table_info.name = name;
  table_info.table_type = TableType::kSparse;
  table_info.element_type = element_type;
  table_info.dimension = dimension;
  table_info.init_type = init_type;
  table_info.init_conf = init_conf;

  model_info.table_infos.emplace(id, std::move(table_info));

  LOG_INFO("Apply SparseTableInfo name:"
           << name << ", id:" << id << ", dimension:" << dimension
           << ", ElementType:" << element_type.Name()
           << ", init type:" << (int32_t)init_type);

  return ErrorCode::kSuccess;
}

int32_t Ps::RegisterSparseTable(
    uint64_t model_id, uint64_t id, const std::string& name, int64_t dimension,
    ElementType element_type, InitializerType init_type,
    const std::unordered_map<std::string, std::string>& init_conf) {
  std::unique_ptr<Initializer> initializer;
  if (init_type == InitializerType::kConstant) {
    initializer.reset(new ConstantInitializer(init_conf));
  } else if (init_type == InitializerType::kNormal) {
    initializer.reset(new NormalInitializer(init_conf));
  } else if (init_type == InitializerType::kUniform) {
    initializer.reset(new UniformInitializer(init_conf));
  } else if (init_type == InitializerType::kXavierNormal) {
    initializer.reset(new XavierNormalInitializer(init_conf));
  } else if (init_type == InitializerType::kXavierUniform) {
    initializer.reset(new XavierUniformInitializer(init_conf));
  } else {
    return ErrorCode::kUnSupportInitializerTypeError;
  }

  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = models_.find(model_id);
  if (it == models_.end()) {
    return ErrorCode::kUnRegisterModelError;
  }

  return it->second->RegisterSparseTable(id, name, dimension, element_type,
                                         std::move(initializer));
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
                            const std::vector<int64_t>& indices,
                            std::vector<Tensor>* vals) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = models_.find(model_id);
  if (it == models_.end()) {
    return ErrorCode::kUnRegisterModelError;
  }

  return it->second->PullSparseTable(table_id, indices, vals);
}

int32_t Ps::PushSparseTable(uint64_t model_id, uint64_t table_id,
                            const std::vector<int64_t>& indices,
                            const std::vector<Tensor>& grads, float lr) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = models_.find(model_id);
  if (it == models_.end()) {
    return ErrorCode::kUnRegisterModelError;
  }

  return it->second->PushSparseTable(table_id, indices, grads, lr);
}

}  // namespace kraken
