#include "worker/worker.h"

#include <string>
#include <unordered_map>

#include "common/exception.h"
#include "common/log.h"
#include "worker/dct_emitter.h"
#include "worker/emitter.h"

namespace kraken {

Worker::Worker() {
}

void Worker::Initialize(const std::string& s_addr, EmitterType emitter_type,
                        uint64_t life_span, float eta) {
  if (emitter_type == EmitterType::kDefault) {
    emitter_.reset(new Emitter());

    LOG_INFO("Initialize Emitter:" << emitter_type);
  } else if (emitter_type == EmitterType::kDCT) {
    emitter_.reset(new DCTEmitter(life_span, eta));

    LOG_INFO("Initialize Emitter:" << emitter_type << ", life_span:"
                                   << life_span << ", eta:" << eta);
  } else {
    RUNTIME_ERROR("Unsupport EmitterType:" << (uint32_t)emitter_type);
  }

  emitter_->Initialize(s_addr);
}

void Worker::Stop() {
  emitter_->Stop();
}

void Worker::InitModel(
    const std::string& model_name, OptimType optim_type,
    const std::unordered_map<std::string, std::string>& optim_conf) {
  emitter_->InitModel(model_name, optim_type, optim_conf);
}

void Worker::UpdateLR(float lr) {
  emitter_->UpdateLR(lr);
}

uint64_t Worker::RegisterDenseTable(const std::string& name,
                                    const Tensor& val) {
  return emitter_->RegisterDenseTable(name, val);
}

uint64_t Worker::RegisterSparseTable(
    const std::string& name, int64_t dimension, ElementType etype,
    InitializerType init_type,
    const std::unordered_map<std::string, std::string>& init_conf) {
  return emitter_->RegisterSparseTable(name, dimension, etype, init_type,
                                       init_conf);
}

Tensor Worker::PullDenseTable(uint64_t table_id) {
  return emitter_->PullDenseTable(table_id);
}

std::vector<Tensor> Worker::CombinePullDenseTable(
    const std::vector<uint64_t>& table_ids) {
  return emitter_->CombinePullDenseTable(table_ids);
}

void Worker::PushDenseTable(uint64_t table_id, const Tensor& grad) {
  emitter_->PushDenseTable(table_id, grad);
}

Tensor Worker::PullSparseTable(uint64_t table_id, const Tensor& indices) {
  return emitter_->PullSparseTable(table_id, indices);
}

std::vector<Tensor> Worker::CombinePullSparseTable(
    const std::vector<uint64_t>& table_ids,
    const std::vector<Tensor>& indices) {
  return emitter_->CombinePullSparseTable(table_ids, indices);
}

void Worker::PushSparseTable(uint64_t table_id, const Tensor& indices,
                             const Tensor& grads) {
  emitter_->PushSparseTable(table_id, indices, grads);
}

void Worker::CombinePushSparseTable(const std::vector<uint64_t>& table_ids,
                                    const std::vector<Tensor>& indices,
                                    const std::vector<Tensor>& grads) {
  emitter_->CombinePushSparseTable(table_ids, indices, grads);
}

}  // namespace kraken
