#pragma once

#include <torch/extension.h>
#include <torch/torch.h>

#include "t/element_type.h"
#include "t/shape.h"
#include "t/tensor.h"

namespace kraken {
namespace py {

Shape TorchSizesToShape(const torch::IntArrayRef& sizes);

torch::IntArrayRef ShapeToTorchSizes(const Shape& shape);

ElementType TorchDTypeToElementType(torch::Dtype dtype);

torch::Dtype ElementTypeToTorchDType(ElementType etype);

// Becareful the returned tensor will share memory with torch tensor.
Tensor TorchTensorToTensor(const torch::Tensor& tval);

}  // namespace py
}  // namespace kraken
