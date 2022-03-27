#pragma once

#include <torch/extension.h>
#include <torch/torch.h>

namespace kraken {
namespace py {
namespace jagged {

// suppose the values shape is:[d0, d1, d2, ..., dn]
// The values will be split to a list of tensor:[v0, v1, ... , vk]
// And the shape will be:
// (offsets[1] - offsets[0], d1, d2, ..., dn),
// (offsets[2] - offsets[1], d1, d2, ..., dn)
// ...
// (offsets[k+1] - offsets[k], d1, d2, ..., dn)
// The SumForward output shape will be: [k, d1, d2, ..., dn]
// And output[i:] = sum(values[j:])[j=offsets[i]-offsets[i+1]]
torch::Tensor SumForward(torch::Tensor values, torch::Tensor offsets,
                         float patch_value);

torch::Tensor SumBackward(torch::Tensor offsets, torch::Tensor grads);

// Like Sum but output the mean.
torch::Tensor MeanForward(torch::Tensor values, torch::Tensor offsets,
                          float patch_value);

torch::Tensor MeanBackward(torch::Tensor offsets, torch::Tensor grads);

}  // namespace jagged
}  // namespace py
}  // namespace kraken
