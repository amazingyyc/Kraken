#include "pybind11/pytorch.h"

#include <cinttypes>
#include <memory>
#include <vector>

#include "common/exception.h"
#include "pybind11/pytorch_utils.h"
#include "t/element_type.h"
#include "t/shape.h"
#include "t/storage.h"
#include "t/tensor.h"
#include "worker/emitter.h"

namespace kraken {
namespace py {

std::once_flag flag;
Emitter emitter;

void Initialize(const std::string& s_addr) {
  std::call_once(flag, [&s_addr]() { emitter.Initialize(s_addr); });
}

void Stop() {
  emitter.Stop();
}

void InitModel(const std::string& model_name, OptimType optim_type,
               const std::unordered_map<std::string, std::string>& optim_conf) {
  emitter.InitModel(model_name, optim_type, optim_conf);
}

void UpdateLR(float lr) {
  emitter.UpdateLR(lr);
}

uint64_t RegisterDenseTable(const std::string& name, torch::Tensor val) {
  ARGUMENT_CHECK(!val.is_cuda(),
                 "RegisterDenseTable need torch::Tensor is CPU.");

  // Convert to contiguous tensor.
  torch::Tensor cval = val;
  if (!val.is_contiguous()) {
    cval = val.contiguous();
  }

  Tensor kval = TorchTensorToTensor(cval);

  return emitter.RegisterDenseTable(name, kval);
}

uint64_t RegisterSparseTable(
    const std::string& name, int64_t dimension, pybind11::object dtype,
    InitializerType init_type,
    const std::unordered_map<std::string, std::string>& init_conf) {
  torch::Dtype ttype = torch::python::detail::py_object_to_dtype(dtype);
  ElementType etype = TorchDTypeToElementType(ttype);

  return emitter.RegisterSparseTable(name, dimension, etype, init_type,
                                     init_conf);
}

torch::Tensor PullDenseTable(uint64_t table_id) {
  Tensor kval = emitter.PullDenseTable(table_id);

  torch::IntArrayRef sizes = ShapeToTorchSizes(kval.shape());
  torch::Dtype dtype = ElementTypeToTorchDType(kval.element_type());

  torch::Tensor val = torch::zeros(sizes, dtype);

  // Copy memory.
  memcpy(val.data_ptr(), kval.Ptr(), kval.NumBytes());

  return val;
}

std::vector<torch::Tensor> CombinePullDenseTable(
    const std::vector<uint64_t>& table_ids) {
  std::vector<Tensor> kvals = emitter.CombinePullDenseTable(table_ids);
  std::vector<torch::Tensor> vals;

  for (auto& kv : kvals) {
    torch::IntArrayRef sizes = ShapeToTorchSizes(kv.shape());
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
  torch::Tensor cgrad = grad;
  if (!grad.is_contiguous()) {
    cgrad = grad.contiguous();
  }

  Tensor kgrad = TorchTensorToTensor(cgrad);

  emitter.PushDenseTable(table_id, kgrad);
}

torch::Tensor PullSparseTable(uint64_t table_id, torch::Tensor indices) {
  ARGUMENT_CHECK(!indices.is_cuda(),
                 "PullSparseTable need torch::Tensor is CPU.");

  torch::Tensor cindices = indices;
  if (!indices.is_contiguous()) {
    cindices = indices.contiguous();
  }

  Tensor kindices = TorchTensorToTensor(cindices);

  // The sparse embedding.
  Tensor kval = emitter.PullSparseTable(table_id, kindices);

  torch::IntArrayRef sizes = ShapeToTorchSizes(kval.shape());
  torch::Dtype dtype = ElementTypeToTorchDType(kval.element_type());
  torch::Tensor val = torch::zeros(sizes, dtype);

  // copy memory.
  memcpy(val.data_ptr(), kval.Ptr(), kval.NumBytes());

  return val;
}

std::vector<torch::Tensor> CombinePullSparseTable(
    const std::vector<uint64_t>& table_ids,
    const std::vector<torch::Tensor>& indices) {
  std::vector<Tensor> kindices;
  kindices.reserve(indices.size());

  for (size_t i = 0; i < indices.size(); ++i) {
    torch::Tensor cindice = indices[i];
    if (!indices[i].is_contiguous()) {
      cindice = indices[i].contiguous();
    }

    kindices.emplace_back(TorchTensorToTensor(cindice));
  }

  std::vector<Tensor> kvals =
      emitter.CombinePullSparseTable(table_ids, kindices);

  std::vector<torch::Tensor> vals;

  for (size_t i = 0; i < kvals.size(); ++i) {
    torch::IntArrayRef sizes = ShapeToTorchSizes(kvals[i].shape());
    torch::Dtype dtype = ElementTypeToTorchDType(kvals[i].element_type());
    torch::Tensor val = torch::zeros(sizes, dtype);

    // copy memory.
    memcpy(val.data_ptr(), kvals[i].Ptr(), kvals[i].NumBytes());

    vals.emplace_back(val);
  }

  return vals;
}

void PushSparseTable(uint64_t table_id, torch::Tensor indices,
                     torch::Tensor grad) {
  ARGUMENT_CHECK(!indices.is_cuda() && !grad.is_cuda(),
                 "PushSparseTable need torch::Tensor is CPU.");

  torch::Tensor cindices = indices;
  if (!cindices.is_contiguous()) {
    cindices = indices.contiguous();
  }

  torch::Tensor cgrad = grad;
  if (!cgrad.is_contiguous()) {
    cgrad = cgrad.contiguous();
  }

  Tensor kindices = TorchTensorToTensor(cindices);
  Tensor kgrad = TorchTensorToTensor(cgrad);

  emitter.PushSparseTable(table_id, kindices, kgrad);
}

void CombinePushSparseTable(const std::vector<uint64_t>& table_ids,
                            const std::vector<torch::Tensor>& indices,
                            const std::vector<torch::Tensor>& grads) {
  ARGUMENT_CHECK(
      table_ids.size() == indices.size() && table_ids.size() == grads.size(),
      "CombinePushSparseTable args need same size!");

  std::vector<Tensor> kindices;
  kindices.reserve(indices.size());

  std::vector<Tensor> kgrads;
  kgrads.reserve(indices.size());

  for (size_t i = 0; i < indices.size(); ++i) {
    torch::Tensor cindice = indices[i];
    if (!cindice.is_contiguous()) {
      cindice = cindice.contiguous();
    }

    kindices.emplace_back(TorchTensorToTensor(cindice));

    torch::Tensor cgrad = grads[i];
    if (!cgrad.is_contiguous()) {
      cgrad = cgrad.contiguous();
    }

    kgrads.emplace_back(TorchTensorToTensor(cgrad));
  }

  emitter.CombinePushSparseTable(table_ids, kindices, kgrads);
}

}  // namespace py
}  // namespace kraken
