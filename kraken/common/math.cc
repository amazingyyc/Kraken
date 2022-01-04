#include "common/math.h"

#include <eigen/Eigen/Dense>
#include <random>

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#include "common/exception.h"

namespace kraken {
namespace math {

template <typename T>
using EVector =
    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor>>;

template <typename T>
void AddImpl(T* x, T* y, T* z, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);
  EVector<T> zv(z, size);

  zv.noalias() = xv + yv;
}

void Add(const Tensor& x, const Tensor& y, Tensor& z) {
  ARGUMENT_CHECK(x.element_type() == y.element_type() &&
                     x.element_type() == z.element_type(),
                 "add need all tensor has same ElementType.");

  int64_t size = x.Size();

  ARGUMENT_CHECK(size == y.Size() && size == z.Size(),
                 "add need all tensor has same size.");

  if (x.element_type().Is<float>()) {
    AddImpl<float>(x.Data<float>(), y.Data<float>(), z.Data<float>(), size);
  } else if (x.element_type().Is<double>()) {
    AddImpl<double>(x.Data<double>(), y.Data<double>(), z.Data<double>(), size);
  } else {
    RUNTIME_ERROR("add not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void AddImpl(T v, T* x, T* y, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);

  yv.noalias() = (v + xv.array()).matrix();
}

void Add(float v, const Tensor& x, Tensor& y) {
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "add need all tensor has same ElementType.");

  ARGUMENT_CHECK(x.Size() == y.Size(), "add need all tensor has same size.");

  if (x.element_type().Is<float>()) {
    AddImpl<float>(v, x.Data<float>(), y.Data<float>(), x.Size());
  } else if (x.element_type().Is<double>()) {
    AddImpl<double>((double)v, x.Data<double>(), y.Data<double>(), x.Size());
  } else {
    RUNTIME_ERROR("add not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void SubImpl(T* x, T* y, T* z, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);
  EVector<T> zv(z, size);

  zv.noalias() = xv - yv;
}

void Sub(const Tensor& x, const Tensor& y, Tensor& z) {
  ARGUMENT_CHECK(x.element_type() == y.element_type() &&
                     x.element_type() == z.element_type(),
                 "sub need all tensor has same ElementType.");

  int64_t size = x.Size();

  ARGUMENT_CHECK(size == y.Size() && size == z.Size(),
                 "sub need all tensor has same size.");

  if (x.element_type().Is<float>()) {
    SubImpl<float>(x.Data<float>(), y.Data<float>(), z.Data<float>(), size);
  } else if (x.element_type().Is<double>()) {
    SubImpl<double>(x.Data<double>(), y.Data<double>(), z.Data<double>(), size);
  } else {
    RUNTIME_ERROR("sub not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void SubImpl(T v, T* x, T* y, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);

  yv.noalias() = (v - xv.array()).matrix();
}

void Sub(float v, const Tensor& x, Tensor& y) {
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "sub need all tensor has same ElementType.");

  ARGUMENT_CHECK(x.Size() == y.Size(), "sub need all tensor has same size.");

  if (x.element_type().Is<float>()) {
    SubImpl<float>(v, x.Data<float>(), y.Data<float>(), x.Size());
  } else if (x.element_type().Is<double>()) {
    SubImpl<double>((double)v, x.Data<double>(), y.Data<double>(), x.Size());
  } else {
    RUNTIME_ERROR("sub not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void SubImpl(T* x, T v, T* y, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);

  yv.noalias() = (xv.array() - v).matrix();
}

void Sub(const Tensor& x, float v, Tensor& y) {
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "sub need all tensor has same ElementType.");

  ARGUMENT_CHECK(x.Size() == y.Size(), "sub need all tensor has same size.");

  if (x.element_type().Is<float>()) {
    SubImpl<float>(x.Data<float>(), v, y.Data<float>(), x.Size());
  } else if (x.element_type().Is<double>()) {
    SubImpl<double>(x.Data<double>(), (double)v, y.Data<double>(), x.Size());
  } else {
    RUNTIME_ERROR("sub not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void MulImpl(T* x, T* y, T* z, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);
  EVector<T> zv(z, size);

  zv.noalias() = xv.cwiseProduct(yv);
}

void Mul(const Tensor& x, const Tensor& y, Tensor& z) {
  ARGUMENT_CHECK(x.element_type() == y.element_type() &&
                     x.element_type() == z.element_type(),
                 "mul need all tensor has same ElementType.");

  int64_t size = x.Size();

  ARGUMENT_CHECK(size == y.Size() && size == z.Size(),
                 "mul need all tensor has same size.");

  if (x.element_type().Is<float>()) {
    MulImpl<float>(x.Data<float>(), y.Data<float>(), z.Data<float>(), size);
  } else if (x.element_type().Is<double>()) {
    MulImpl<double>(x.Data<double>(), y.Data<double>(), z.Data<double>(), size);
  } else {
    RUNTIME_ERROR("mul not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void MulImpl(T v, T* x, T* y, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);

  yv.noalias() = v * xv;
}

void Mul(float v, const Tensor& x, Tensor& y) {
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "mul need all tensor has same ElementType.");

  ARGUMENT_CHECK(x.Size() == y.Size(), "mul need all tensor has same size.");

  if (x.element_type().Is<float>()) {
    MulImpl<float>(v, x.Data<float>(), y.Data<float>(), x.Size());
  } else if (x.element_type().Is<double>()) {
    MulImpl<double>((double)v, x.Data<double>(), y.Data<double>(), x.Size());
  } else {
    RUNTIME_ERROR("mul not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void DivImpl(T* x, T* y, T* z, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);
  EVector<T> zv(z, size);

  zv.noalias() = (xv.array() / yv.array()).matrix();
}

void Div(const Tensor& x, const Tensor& y, Tensor& z) {
  ARGUMENT_CHECK(x.element_type() == y.element_type() &&
                     x.element_type() == z.element_type(),
                 "div need all tensor has same ElementType.");

  int64_t size = x.Size();

  ARGUMENT_CHECK(size == y.Size() && size == z.Size(),
                 "div need all tensor has same size.");

  if (x.element_type().Is<float>()) {
    DivImpl<float>(x.Data<float>(), y.Data<float>(), z.Data<float>(), size);
  } else if (x.element_type().Is<double>()) {
    DivImpl<double>(x.Data<double>(), y.Data<double>(), z.Data<double>(), size);
  } else {
    RUNTIME_ERROR("div not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void DivImpl(T v, T* x, T* y, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);

  yv.noalias() = (v / xv.array()).matrix();
}

void Div(float v, const Tensor& x, Tensor& y) {
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "div need all tensor has same ElementType.");

  ARGUMENT_CHECK(x.Size() == y.Size(), "div need all tensor has same size.");

  if (x.element_type().Is<float>()) {
    DivImpl<float>(v, x.Data<float>(), y.Data<float>(), x.Size());
  } else if (x.element_type().Is<double>()) {
    DivImpl<double>((double)v, x.Data<double>(), y.Data<double>(), x.Size());
  } else {
    RUNTIME_ERROR("div not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void DivImpl(T* x, T v, T* y, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);

  yv.noalias() = xv / v;
}

void Div(const Tensor& x, float v, Tensor& y) {
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "div need all tensor has same ElementType.");

  ARGUMENT_CHECK(x.Size() == y.Size(), "div need all tensor has same size.");

  if (x.element_type().Is<float>()) {
    DivImpl<float>(x.Data<float>(), v, y.Data<float>(), x.Size());
  } else if (x.element_type().Is<double>()) {
    DivImpl<double>(x.Data<double>(), (double)v, y.Data<double>(), x.Size());
  } else {
    RUNTIME_ERROR("div not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void NormImpl(T* v, int64_t n, T lower, float upper) {
  static thread_local std::mt19937 generator;
  std::normal_distribution<T> distribution(lower, upper);

  for (int64_t i = 0; i < n; ++i) {
    v[i] = distribution(generator);
  }
}

void Norm(Tensor& x, float lower, float upper) {
  if (x.element_type().Is<float>()) {
    NormImpl<float>(x.Data<float>(), x.Size(), lower, upper);
  } else if (x.element_type().Is<double>()) {
    NormImpl<double>(x.Data<double>(), x.Size(), lower, upper);
  } else {
    RUNTIME_ERROR("norm not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void ConcatVecImpl(const std::vector<Tensor>& xs, T* y, int64_t row,
                   int64_t col) {
#pragma omp parallel for
  for (int64_t i = 0; i < row; ++i) {
    T* xp = xs[i].Data<T>();
    T* yp = y + i * col;

    memcpy(yp, xp, col * sizeof(T));
  }
}

void ConcatVec(const std::vector<Tensor>& xs, Tensor& y) {
  // The tensor in xs must be vector and y must be a matrix.
  int64_t row = (int64_t)xs.size();
  int64_t col = xs[0].Size();

  ARGUMENT_CHECK(
      y.shape().NDims() == 2 && row == y.shape()[0] && col == y.shape()[1],
      "concat_vec need xs is a vector list and y must be a matrix.");

  for (const auto& v : xs) {
    ARGUMENT_CHECK(v.Size() == col, "concat_vec's parameter error.");
    ARGUMENT_CHECK(v.element_type() == y.element_type(),
                   "concat_vec need parameter has same type.");
  }

  if (y.element_type().Is<float>()) {
    ConcatVecImpl<float>(xs, y.Data<float>(), row, col);
  } else if (y.element_type().Is<double>()) {
    ConcatVecImpl<double>(xs, y.Data<double>(), row, col);
  } else {
    RUNTIME_ERROR(
        "concat_vec not support ElementType:" << y.element_type().Name());
  }
}

template <typename T>
void SqrtImpl(T* x, T* y, int64_t n) {
  EVector<T> xv(x, n);
  EVector<T> yv(y, n);

  yv.noalias() = xv.array().sqrt().matrix();
}

void Sqrt(const Tensor& x, Tensor& y) {
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "Sqrt need all tensor has same ElementType.");
  ARGUMENT_CHECK(x.Size() == y.Size(), "Sqrt need all tensor has same size.");

  if (x.element_type().Is<float>()) {
    SqrtImpl<float>(x.Data<float>(), y.Data<float>(), x.Size());
  } else if (x.element_type().Is<double>()) {
    SqrtImpl<double>(x.Data<double>(), y.Data<double>(), x.Size());
  } else {
    RUNTIME_ERROR("Sqrt not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void MaxImpl(T* x, T* y, T* z, int64_t n) {
  EVector<T> xv(x, n);
  EVector<T> yv(y, n);
  EVector<T> zv(z, n);

  zv.noalias() = xv.cwiseMax(yv);
}

void Max(const Tensor& x, const Tensor& y, Tensor& z) {
  ARGUMENT_CHECK(x.element_type() == y.element_type() &&
                     x.element_type() == z.element_type(),
                 "Max need all tensor has same ElementType.");

  int64_t size = x.Size();

  ARGUMENT_CHECK(size == y.Size() && size == z.Size(),
                 "Max need all tensor has same size.");

  if (x.element_type().Is<float>()) {
    MaxImpl<float>(x.Data<float>(), y.Data<float>(), z.Data<float>(), size);
  } else if (x.element_type().Is<double>()) {
    MaxImpl<double>(x.Data<double>(), y.Data<double>(), z.Data<double>(), size);
  } else {
    RUNTIME_ERROR("Max not support ElementType:" << x.element_type().Name());
  }
}

}  // namespace math
}  // namespace kraken
