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

}  // namespace math
}  // namespace kraken
