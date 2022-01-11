#pragma once

#include <torch/extension.h>
#include <torch/torch.h>

#include "ps/optim/optim.h"

namespace kraken {
namespace py {

void Initialize(const std::string& addrs);

void Stop();

uint64_t RegisterModel(
    const std::string& model_name, OptimType optim_type,
    const std::unordered_map<std::string, std::string>& optim_conf);

void UpdateLR(float lr);

uint64_t RegisterDenseTable(const std::string& name, torch::Tensor val);

uint64_t RegisterSparseTable(const std::string& name, int64_t dimension,
                             pybind11::object dtype);

void PushDenseTable(uint64_t table_id, torch::Tensor grad);

torch::Tensor PullDenseTable(uint64_t table_id);

std::vector<torch::Tensor> PullListDenseTable(const std::vector<uint64_t>& table_ids);

torch::Tensor PushPullDenseTable(uint64_t table_id, torch::Tensor grad);

void PushSparseTable(uint64_t table_id, torch::Tensor indices,
                     torch::Tensor grads);

torch::Tensor PullSparseTable(uint64_t table_id, torch::Tensor indices);

}  // namespace py
}  // namespace kraken
