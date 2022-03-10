// #pragma once

// #include <torch/extension.h>
// #include <torch/torch.h>

// #include "ps/initializer/initializer.h"
// #include "ps/optim/optim.h"
// #include "worker/emitter.h"

// namespace kraken {
// namespace py {

// void Initialize(const std::string& addrs, EmitterType emitter_type,
//                 CompressType compress_type, uint64_t life_span, float eta);

// void Stop();

// uint64_t RegisterModel(
//     const std::string& model_name, OptimType optim_type,
//     const std::unordered_map<std::string, std::string>& optim_conf);

// void UpdateLR(float lr);

// uint64_t RegisterDenseTable(const std::string& name, torch::Tensor val);

// uint64_t RegisterSparseTable(
//     const std::string& name, int64_t dimension, pybind11::object dtype,
//     InitializerType init_type,
//     const std::unordered_map<std::string, std::string>& init_conf);

// torch::Tensor PullDenseTable(uint64_t table_id);

// std::vector<torch::Tensor> CombinePullDenseTable(
//     const std::vector<uint64_t>& table_ids);

// void PushDenseTable(uint64_t table_id, torch::Tensor grad);

// torch::Tensor PushPullDenseTable(uint64_t table_id, torch::Tensor grad);

// torch::Tensor PullSparseTable(uint64_t table_id, torch::Tensor indices);

// std::vector<torch::Tensor> CombinePullSparseTable(
//     const std::vector<uint64_t>& table_ids,
//     const std::vector<torch::Tensor>& indices);

// void PushSparseTable(uint64_t table_id, torch::Tensor indices,
//                      torch::Tensor grads);

// void SaveCheckPoint();

// }  // namespace py
// }  // namespace kraken
