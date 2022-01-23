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

int32_t Ps::ApplyModel(const std::string& name, uint64_t* model_id) {
  return apply_mgr_.ApplyModel(name, model_id);
}

int32_t Ps::ApplyTable(uint64_t model_id, const std::string& name,
                       TableType type, uint64_t* table_id) {
  return apply_mgr_.ApplyTable(model_id, name, type, table_id);
}

int32_t Ps::RegisterModel(
    uint64_t id, const std::string& name, OptimType optim_type,
    const std::unordered_map<std::string, std::string>& optim_conf) {
  std::unique_lock<std::shared_mutex> lock(mu_);

  auto it = models_.find(id);
  if (it != models_.end()) {
    // (TODO) check name and optim type.
    LOG_INFO("Registerd model: " << name << ", id: " << it->second->id()
                                 << " already exist.");

    return ErrorCode::kSuccess;
  }

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

  LOG_INFO("Register model:" << name << ", id:" << id
                             << ", optim_type:" << (uint32_t)optim_type);

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

int32_t Ps::RegisterSparseTable(uint64_t model_id, uint64_t id,
                                const std::string& name, int64_t dimension,
                                ElementType etype) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = models_.find(model_id);
  if (it == models_.end()) {
    return ErrorCode::kUnRegisterModelError;
  }

  return it->second->RegisterSparseTable(id, name, dimension, etype);
}

int32_t Ps::RegisterSparseTableV2(
    uint64_t model_id, uint64_t id, const std::string& name, int64_t dimension,
    ElementType etype, InitializerType init_type,
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

  return it->second->RegisterSparseTable(id, name, dimension, etype,
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
