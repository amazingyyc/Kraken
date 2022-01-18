#pragma once

#include <cinttypes>
#include <vector>

#include "t/tensor_impl.h"

namespace kraken {
namespace math {

// z = x + y
void Add(const TensorImpl& x, const TensorImpl& y, TensorImpl& z);
// y = x + v
void Add(float v, const TensorImpl& x, TensorImpl& y);

// z = x - y
void Sub(const TensorImpl& x, const TensorImpl& y, TensorImpl& z);
// y = v - x
void Sub(float v, const TensorImpl& x, TensorImpl& y);
// y = x - v
void Sub(const TensorImpl& x, float v, TensorImpl& y);

// z = x * y
void Mul(const TensorImpl& x, const TensorImpl& y, TensorImpl& z);
// y = v * x
void Mul(float v, const TensorImpl& x, TensorImpl& y);

// z = x / y
void Div(const TensorImpl& x, const TensorImpl& y, TensorImpl& z);
// y = v / x
void Div(float v, const TensorImpl& x, TensorImpl& y);
// y = x / v
void Div(const TensorImpl& x, float v, TensorImpl& y);

void Constant(TensorImpl& x, float v);

void Sqrt(const TensorImpl& x, TensorImpl& y);

void Max(const TensorImpl& x, const TensorImpl& y, TensorImpl& z);

void ConcatVector(const std::vector<std::shared_ptr<TensorImpl>>& xs,
                  TensorImpl& y);

void Normal(TensorImpl& x, float mean, float stddev);

void XavierNormal(TensorImpl& x, float gain);

void Uniform(TensorImpl& x, float lower, float upper);

void XavierUniform(TensorImpl& x, float gain);

// y = x >= v. y is bool.
void Ge(const TensorImpl& x, float v, TensorImpl& y);

void Abs(const TensorImpl& x, TensorImpl& y);

// Fetch topk value to y.
void TopK(const TensorImpl& x, TensorImpl& y);

// Get nonzero count.
void CountNonZero(const TensorImpl& x, float th, int64_t* count);

// Take element from x by indices to y
// indices store the element index from 0.
void Take(const TensorImpl& x, const TensorImpl& indices, TensorImpl& y);

// Get nonzero in flat index.
std::shared_ptr<TensorImpl> FlatNonZero(const TensorImpl& x, float th);

// Get nonzero index from x
// suppose nonzero count is nnz. Than the return shape NDims is 2: [nnz,
// x.Ndims()]
std::shared_ptr<TensorImpl> NonZero(const TensorImpl& x, float th);

// Transpose d0 and d1 dimension.
void Transpose(const TensorImpl& x, TensorImpl& y, int64_t d0, int64_t d1);

// Convert Coo to Dense.
void CooToDense(const TensorImpl& indices, const TensorImpl& values,
                TensorImpl& dense);

}  // namespace math
}  // namespace kraken
