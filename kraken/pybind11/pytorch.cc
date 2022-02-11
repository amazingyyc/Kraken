#include "pybind11/pytorch.h"

#include <cinttypes>
#include <memory>
#include <vector>

#include "common/exception.h"
#include "t/element_type.h"
#include "t/shape.h"
#include "t/storage.h"
#include "t/tensor.h"
#include "worker/worker.h"

namespace kraken {
namespace py {

std::once_flag flag;
Worker worker;

void Initialize(const std::string& addrs, EmitterType emitter_type,
                uint64_t life_span, float eta) {
  std::call_once(flag, [&addrs, emitter_type, life_span, eta]() {
    worker.Initialize(addrs, emitter_type, life_span, eta);
  });
}

void Stop() {
  worker.Stop();
}

Shape TorchSizesToShape(const torch::IntArrayRef& sizes) {
  std::vector<int64_t> dims;
  for (auto d : sizes) {
    dims.emplace_back(d);
  }

  return Shape(dims);
}

torch::IntArrayRef ShapeToTorchSizes(const Shape& shape) {
  const auto& dims = shape.dims();

  return torch::IntArrayRef(dims);
}

ElementType TorchDTypeToElementType(torch::Dtype dtype) {
  switch (dtype) {
    case torch::kUInt8:
      return ElementType::From<uint8_t>();
    case torch::kInt8:
      return ElementType::From<int8_t>();
    case torch::kInt16:
      return ElementType::From<int16_t>();
    case torch::kInt32:
      return ElementType::From<int32_t>();
    case torch::kInt64:
      return ElementType::From<int64_t>();
    case torch::kFloat16:
      return ElementType::From<half>();
    case torch::kFloat32:
      return ElementType::From<float>();
    case torch::kFloat64:
      return ElementType::From<double>();
    default:
      RUNTIME_ERROR("The Torch dtype does not support:" << dtype);
  }
}

torch::Dtype ElementTypeToTorchDType(ElementType etype) {
  switch (etype.dtype) {
    case DType::kUint8:
      return torch::kUInt8;
    case DType::kInt8:
      return torch::kInt8;
    case DType::kInt16:
      return torch::kInt16;
    case DType::kInt32:
      return torch::kInt32;
    case DType::kInt64:
      return torch::kInt64;
    case DType::kFloat16:
      return torch::kFloat16;
    case DType::kFloat32:
      return torch::kFloat32;
    case DType::kFloat64:
      return torch::kFloat64;
    default:
      RUNTIME_ERROR("The ElementType does not support:" << etype.Name());
  }
}

// Becareful the returned tensor will share memory with torch tensor.
Tensor TorchTensorToTensor(const torch::Tensor& tval) {
  ARGUMENT_CHECK(tval.is_contiguous(),
                 "TorchTensorToTensor need torch tensor is contiguous.");

  auto storage = Storage::From(tval.data_ptr(), tval.nbytes());

  Shape shape = TorchSizesToShape(tval.sizes());
  ElementType etype = TorchDTypeToElementType(tval.scalar_type());

  return Tensor::Dense(shape, storage, 0, etype);
}

uint64_t RegisterModel(
    const std::string& model_name, OptimType optim_type,
    const std::unordered_map<std::string, std::string>& optim_conf) {
  return worker.RegisterModel(model_name, optim_type, optim_conf);
}

void UpdateLR(float lr) {
  worker.UpdateLR(lr);
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

  return worker.RegisterDenseTable(name, kval);
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
  Tensor kval = worker.PullDenseTable(table_id);

  torch::IntArrayRef sizes = ShapeToTorchSizes(kval.shape());
  torch::Dtype dtype = ElementTypeToTorchDType(kval.element_type());

  torch::Tensor val = torch::zeros(sizes, dtype);

  // Copy memory.
  memcpy(val.data_ptr(), kval.Ptr(), kval.NumBytes());

  return val;
}

std::vector<torch::Tensor> CombinePullDenseTable(
    const std::vector<uint64_t>& table_ids) {
  std::vector<Tensor> kvals = worker.CombinePullDenseTable(table_ids);
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

  worker.PushDenseTable(table_id, kgrad);
}

torch::Tensor PushPullDenseTable(uint64_t table_id, torch::Tensor grad) {
  ARGUMENT_CHECK(!grad.is_cuda(), "PushDenseTable need torch::Tensor is CPU.");

  // Convert to contiguous tensor.
  torch::Tensor cgrad = grad;
  if (!grad.is_contiguous()) {
    cgrad = grad.contiguous();
  }

  Tensor kgrad = TorchTensorToTensor(cgrad);

  Tensor kval = worker.PushPullDenseTable(table_id, kgrad);

  torch::IntArrayRef sizes = ShapeToTorchSizes(kval.shape());
  torch::Dtype dtype = ElementTypeToTorchDType(kval.element_type());

  torch::Tensor val = torch::zeros(sizes, dtype);

  // Copy memory.
  memcpy(val.data_ptr(), kval.Ptr(), kval.NumBytes());

  return val;
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
  Tensor kval = worker.PullSparseTable(table_id, kindices);

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
      worker.CombinePullSparseTable(table_ids, kindices);

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
                     torch::Tensor grads) {
  ARGUMENT_CHECK(!indices.is_cuda() && !grads.is_cuda(),
                 "PushSparseTable need torch::Tensor is CPU.");

  torch::Tensor cindices = indices;
  if (!cindices.is_contiguous()) {
    cindices = indices.contiguous();
  }

  torch::Tensor cgrads = grads;
  if (!cgrads.is_contiguous()) {
    cgrads = grads.contiguous();
  }

  Tensor kindices = TorchTensorToTensor(cindices);
  Tensor kgrads = TorchTensorToTensor(cgrads);

  worker.PushSparseTable(table_id, kindices, kgrads);
}

void SaveCheckPoint() {
  worker.SaveCheckPoint();
}

}  // namespace py
}  // namespace kraken
