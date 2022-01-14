#include "pybind11/pytorch.h"

#include <cinttypes>
#include <memory>
#include <vector>

#include "common/element_type.h"
#include "common/exception.h"
#include "common/shape.h"
#include "common/tensor.h"
#include "worker/worker.h"

namespace kraken {
namespace py {

std::once_flag flag;
Worker worker;

void Initialize(const std::string& addrs) {
  std::call_once(flag, [&addrs]() { worker.Initialize(addrs); });
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

  std::shared_ptr<TensorStorage> storage =
      TensorStorage::From(tval.data_ptr(), tval.nbytes());

  Shape shape = TorchSizesToShape(tval.sizes());
  ElementType etype = TorchDTypeToElementType(tval.scalar_type());

  return Tensor::Create(storage, 0, shape, etype);
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

uint64_t RegisterSparseTable(const std::string& name, int64_t dimension,
                             pybind11::object dtype) {
  torch::Dtype ttype = torch::python::detail::py_object_to_dtype(dtype);
  ElementType etype = TorchDTypeToElementType(ttype);

  return worker.RegisterSparseTable(name, dimension, etype);
}

uint64_t RegisterSparseTableV2(
    const std::string& name, int64_t dimension, pybind11::object dtype,
    InitializerType init_type,
    const std::unordered_map<std::string, std::string>& init_conf) {
  torch::Dtype ttype = torch::python::detail::py_object_to_dtype(dtype);
  ElementType etype = TorchDTypeToElementType(ttype);

  return worker.RegisterSparseTableV2(name, dimension, etype, init_type,
                                      init_conf);
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

torch::Tensor PullDenseTable(uint64_t table_id) {
  Tensor kval = worker.PullDenseTable(table_id);

  torch::IntArrayRef sizes = ShapeToTorchSizes(kval.shape());
  torch::Dtype dtype = ElementTypeToTorchDType(kval.element_type());

  torch::Tensor val = torch::zeros(sizes, dtype);

  // Copy memory.
  memcpy(val.data_ptr(), kval.Ptr(), kval.NumBytes());

  return val;
}

std::vector<torch::Tensor> PullListDenseTable(
    const std::vector<uint64_t>& table_ids) {
  std::vector<Tensor> kvals = worker.PullListDenseTable(table_ids);
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

}  // namespace py
}  // namespace kraken
