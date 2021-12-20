#pragma once

#include "common/tensor.h"

namespace kraken {
namespace math {

// z = x + y
void add(const Tensor& x, const Tensor& y, Tensor& z);
// y = x + v
void add(float v, const Tensor& x, Tensor& y);

// z = x - y
void sub(const Tensor& x, const Tensor& y, Tensor& z);
// y = v - x
void sub(float v, const Tensor& x, Tensor& y);
// y = x - v
void sub(const Tensor& x, float v, Tensor& y);

// z = x * y
void mul(const Tensor& x, const Tensor& y, Tensor& z);
// y = v * x
void mul(float v, const Tensor& x, Tensor& y);

// z = x / y
void div(const Tensor& x, const Tensor& y, Tensor& z);
// y = v / x
void div(float v, const Tensor& x, Tensor& y);
// y = x / v
void div(const Tensor& x, float v, Tensor& y);

/**
 * \brief Initialize teh tensor norm randomly.
 * 
 * \param x The tensor.
 */
void initialize_norm(Tensor& x, float lower, float upper);

}  // namespace math
}  // namespace kraken
