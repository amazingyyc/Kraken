#include "pytorch/py/pytorch.h"

#include <cinttypes>
#include <memory>
#include <vector>

#include "common/exception.h"
#include "pytorch/py/pytorch_utils.h"
#include "t/element_type.h"
#include "t/shape.h"
#include "t/storage.h"
#include "t/tensor.h"
#include "worker/worker.h"

namespace kraken {
namespace py {

std::once_flag flag;
Worker worker;

void Initialize(const std::string& s_addr, EmitterType emitter_type,
                uint64_t life_span, float eta) {
  std::call_once(flag, [&s_addr, emitter_type, life_span, eta]() {
    worker.Initialize(s_addr, emitter_type, life_span, eta);
  });
}

void Stop() {
  worker.Stop();
}

void InitModel(const std::string& model_name, OptimType optim_type,
               const std::unordered_map<std::string, std::string>& optim_conf) {
  worker.InitModel(model_name, optim_type, optim_conf);
}

void UpdateLR(float lr) {
  worker.UpdateLR(lr);
}

uint64_t RegisterDenseTable(const std::string& name, torch::Tensor val) {
  ARGUMENT_CHECK(!val.is_cuda(),
                 "RegisterDenseTable need torch::Tensor is CPU.");

  // Convert to contiguous tensor.
  torch::Tensor c_val = val;
  if (val.is_contiguous() == false) {
    c_val = val.contiguous();
  }

  Tensor k_val = TorchTensorToTensor(c_val);

  return worker.RegisterDenseTable(name, k_val);
}

uint64_t RegisterSparseTable(
    const std::string& name, int64_t dimension, pybind11::object dtype,
    InitializerType init_type,
    const std::unordered_map<std::string, std::string>& init_conf) {
  torch::Dtype ttype = torch::python::detail::py_object_to_dtype(dtype);
  ElementType etype = TorchDTypeToElementType(ttype);

  return worker.RegisterSparseTable(name, dimension, etype, init_type,
                                    init_conf);
}

torch::Tensor PullDenseTable(uint64_t table_id) {
  Tensor k_val = worker.PullDenseTable(table_id);

  torch::IntArrayRef sizes(k_val.shape().dims());
  torch::Dtype dtype = ElementTypeToTorchDType(k_val.element_type());

  torch::Tensor val = torch::zeros(sizes, dtype);

  // Copy memory.
  memcpy(val.data_ptr(), k_val.Ptr(), k_val.NumBytes());

  return val;
}

std::vector<torch::Tensor> CombinePullDenseTable(
    const std::vector<uint64_t>& table_ids) {
  std::vector<Tensor> k_vals = worker.CombinePullDenseTable(table_ids);
  std::vector<torch::Tensor> vals;

  for (auto& kv : k_vals) {
    // torch::IntArrayRef sizes = ShapeToTorchSizes(kv.shape());
    torch::IntArrayRef sizes(kv.shape().dims());
    torch::Dtype dtype = ElementTypeToTorchDType(kv.element_type());

    torch::Tensor v = torch::zeros(sizes, dtype);

    memcpy(v.data_ptr(), kv.Ptr(), kv.NumBytes());

    vals.emplace_back(v);
  }

  return vals;
}

void PushDenseTable(uint64_t table_id, torch::Tensor grad) {
  ARGUMENT_CHECK(!grad.is_cuda(), "PushDenseTable need torch::Tensor is CPU.");

  // Convert to contiguous tensor.
  torch::Tensor c_grad = grad;
  if (grad.is_contiguous() == false) {
    c_grad = grad.contiguous();
  }

  Tensor k_grad = TorchTensorToTensor(c_grad);

  worker.PushDenseTable(table_id, k_grad);
}

torch::Tensor PullSparseTable(uint64_t table_id, torch::Tensor indices) {
  ARGUMENT_CHECK(!indices.is_cuda(),
                 "PullSparseTable need torch::Tensor is CPU.");

  torch::Tensor c_indices = indices;
  if (indices.is_contiguous() == false) {
    c_indices = indices.contiguous();
  }

  Tensor k_indices = TorchTensorToTensor(c_indices);

  // The sparse embedding.
  Tensor k_val = worker.PullSparseTable(table_id, k_indices);

  // torch::IntArrayRef sizes = ShapeToTorchSizes(k_val.shape());
  torch::IntArrayRef sizes(k_val.shape().dims());
  torch::Dtype dtype = ElementTypeToTorchDType(k_val.element_type());
  torch::Tensor val = torch::zeros(sizes, dtype);

  // copy memory.
  memcpy(val.data_ptr(), k_val.Ptr(), k_val.NumBytes());

  return val;
}

std::vector<torch::Tensor> CombinePullSparseTable(
    const std::vector<uint64_t>& table_ids,
    const std::vector<torch::Tensor>& indices) {
  ARGUMENT_CHECK(table_ids.size() == indices.size(),
                 "CombinePullSparseTable args need same size!");

  size_t count = table_ids.size();

  std::vector<torch::Tensor> c_indices;
  c_indices.reserve(count);

  for (size_t i = 0; i < count; ++i) {
    if (indices[i].is_contiguous() == false) {
      c_indices.emplace_back(indices[i].contiguous());
    } else {
      c_indices.emplace_back(indices[i]);
    }
  }

  std::vector<Tensor> k_indices;
  k_indices.reserve(count);

  for (size_t i = 0; i < count; ++i) {
    k_indices.emplace_back(TorchTensorToTensor(c_indices[i]));
  }

  std::vector<Tensor> k_vals =
      worker.CombinePullSparseTable(table_ids, k_indices);

  std::vector<torch::Tensor> vals;

  for (size_t i = 0; i < k_vals.size(); ++i) {
    // torch::IntArrayRef sizes = ShapeToTorchSizes(k_vals[i].shape());
    torch::IntArrayRef sizes(k_vals[i].shape().dims());
    torch::Dtype dtype = ElementTypeToTorchDType(k_vals[i].element_type());
    torch::Tensor val = torch::zeros(sizes, dtype);

    // copy memory.
    memcpy(val.data_ptr(), k_vals[i].Ptr(), k_vals[i].NumBytes());

    vals.emplace_back(val);
  }

  return vals;
}

void PushSparseTable(uint64_t table_id, torch::Tensor indices,
                     torch::Tensor grad) {
  ARGUMENT_CHECK(!indices.is_cuda() && !grad.is_cuda(),
                 "PushSparseTable need torch::Tensor is CPU.");

  torch::Tensor c_indices = indices;
  if (indices.is_contiguous() == false) {
    c_indices = indices.contiguous();
  }

  torch::Tensor c_grad = grad;
  if (grad.is_contiguous() == false) {
    c_grad = grad.contiguous();
  }

  Tensor k_indices = TorchTensorToTensor(c_indices);
  Tensor k_grad = TorchTensorToTensor(c_grad);

  worker.PushSparseTable(table_id, k_indices, k_grad);
}

void CombinePushSparseTable(const std::vector<uint64_t>& table_ids,
                            const std::vector<torch::Tensor>& indices,
                            const std::vector<torch::Tensor>& grads) {
  ARGUMENT_CHECK(
      table_ids.size() == indices.size() && table_ids.size() == grads.size(),
      "CombinePushSparseTable args need same size!");

  size_t count = table_ids.size();

  // Avoid the torch::tensor be released we need store it in tmp.
  std::vector<torch::Tensor> c_indices;
  c_indices.reserve(count);

  std::vector<torch::Tensor> c_grads;
  c_grads.reserve(count);

  for (size_t i = 0; i < count; ++i) {
    if (indices[i].is_contiguous() == false) {
      c_indices.emplace_back(indices[i].contiguous());
    } else {
      c_indices.emplace_back(indices[i]);
    }

    if (grads[i].is_contiguous() == false) {
      c_grads.emplace_back(grads[i].contiguous());
    } else {
      c_grads.emplace_back(grads[i]);
    }
  }

  std::vector<Tensor> k_indices;
  k_indices.reserve(count);

  std::vector<Tensor> k_grads;
  k_grads.reserve(count);

  for (size_t i = 0; i < count; ++i) {
    k_indices.emplace_back(TorchTensorToTensor(c_indices[i]));
    k_grads.emplace_back(TorchTensorToTensor(c_grads[i]));
  }

  worker.CombinePushSparseTable(table_ids, k_indices, k_grads);
}

bool TrySaveModel() {
  return worker.TrySaveModel();
}

bool TryLoadModelBlocked(const std::string& load_dir) {
  return worker.TryLoadModelBlocked(load_dir);
}

}  // namespace py
}  // namespace kraken
