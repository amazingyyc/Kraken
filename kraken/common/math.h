#pragma once

#include "common/tensor.h"

namespace kraken {
namespace math {

// z = x + y
void Add(const Tensor& x, const Tensor& y, Tensor& z);
// y = x + v
void Add(float v, const Tensor& x, Tensor& y);

// z = x - y
void Sub(const Tensor& x, const Tensor& y, Tensor& z);
// y = v - x
void Sub(float v, const Tensor& x, Tensor& y);
// y = x - v
void Sub(const Tensor& x, float v, Tensor& y);

// z = x * y
void Mul(const Tensor& x, const Tensor& y, Tensor& z);
// y = v * x
void Mul(float v, const Tensor& x, Tensor& y);

// z = x / y
void Div(const Tensor& x, const Tensor& y, Tensor& z);
// y = v / x
void Div(float v, const Tensor& x, Tensor& y);
// y = x / v
void Div(const Tensor& x, float v, Tensor& y);

void Norm(Tensor& x, float lower, float upper);

/**
 * \brief This is a special concate for vector.
 * We suppose the tensor in xs is vector and concate on 0 dimension.
 *
 * \param xs Vector list
 * \param y Output.
 */
void ConcatVec(const std::vector<Tensor>& xs, Tensor& y);

void Sqrt(const Tensor& x, Tensor& y);

void Max(const Tensor& x, const Tensor& y, Tensor& z);

}  // namespace math
}  // namespace kraken
