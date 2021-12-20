#include "common/math.h"

#include <eigen/Eigen/Dense>
#include <eigen/unsupported/Eigen/CXX11/Tensor>

#include "common/exception.h"

namespace kraken {

template <typename T>
using EVector =
    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor>>;

template <typename T>
void add_impl(T* x, T* y, T* z, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);
  EVector<T> zv(z, size);

  zv.noalias() = xv + yv;
}

void add(const Tensor& x, const Tensor& y, Tensor& z) {
  ARGUMENT_CHECK(x.element_type() == y.element_type() &&
                     x.element_type() == z.element_type(),
                 "add need all tensor has same ElementType.");

  int64_t size = x.size();

  ARGUMENT_CHECK(size == y.size() && size == z.size(),
                 "add need all tensor has same size.");

  if (x.element_type().is<float>()) {
    add_impl<float>(x.data<float>(), y.data<float>(), z.data<float>(), size);
  } else if (x.element_type().is<double>()) {
    add_impl<double>(x.data<double>(), y.data<double>(), z.data<double>(),
                     size);
  } else {
    RUNTIME_ERROR("add not support ElementType:" << x.element_type().name());
  }
}

template <typename T>
void add_impl(T v, T* x, T* y, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);

  yv.noalias() = (v + xv.array()).matrix();
}

void add(float v, const Tensor& x, Tensor& y) {
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "add need all tensor has same ElementType.");

  ARGUMENT_CHECK(x.size() == y.size(), "add need all tensor has same size.");

  if (x.element_type().is<float>()) {
    add_impl<float>(v, x.data<float>(), y.data<float>(), x.size());
  } else if (x.element_type().is<double>()) {
    add_impl<double>((double)v, x.data<double>(), y.data<double>(), x.size());
  } else {
    RUNTIME_ERROR("add not support ElementType:" << x.element_type().name());
  }
}

template <typename T>
void sub_impl(T* x, T* y, T* z, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);
  EVector<T> zv(z, size);

  zv.noalias() = xv - yv;
}

void sub(const Tensor& x, const Tensor& y, Tensor& z) {
  ARGUMENT_CHECK(x.element_type() == y.element_type() &&
                     x.element_type() == z.element_type(),
                 "sub need all tensor has same ElementType.");

  int64_t size = x.size();

  ARGUMENT_CHECK(size == y.size() && size == z.size(),
                 "sub need all tensor has same size.");

  if (x.element_type().is<float>()) {
    sub_impl<float>(x.data<float>(), y.data<float>(), z.data<float>(), size);
  } else if (x.element_type().is<double>()) {
    sub_impl<double>(x.data<double>(), y.data<double>(), z.data<double>(),
                     size);
  } else {
    RUNTIME_ERROR("sub not support ElementType:" << x.element_type().name());
  }
}

template <typename T>
void sub_impl(T v, T* x, T* y, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);

  yv.noalias() = (v - xv.array()).matrix();
}

void sub(float v, const Tensor& x, Tensor& y) {
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "sub need all tensor has same ElementType.");

  ARGUMENT_CHECK(x.size() == y.size(), "sub need all tensor has same size.");

  if (x.element_type().is<float>()) {
    sub_impl<float>(v, x.data<float>(), y.data<float>(), x.size());
  } else if (x.element_type().is<double>()) {
    sub_impl<double>((double)v, x.data<double>(), y.data<double>(), x.size());
  } else {
    RUNTIME_ERROR("sub not support ElementType:" << x.element_type().name());
  }
}

template <typename T>
void sub_impl(T* x, T v, T* y, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);

  yv.noalias() = (xv.array() - v).matrix();
}

void sub(const Tensor& x, float v, Tensor& y) {
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "sub need all tensor has same ElementType.");

  ARGUMENT_CHECK(x.size() == y.size(), "sub need all tensor has same size.");

  if (x.element_type().is<float>()) {
    sub_impl<float>(x.data<float>(), v, y.data<float>(), x.size());
  } else if (x.element_type().is<double>()) {
    sub_impl<double>(x.data<double>(), (double)v, y.data<double>(), x.size());
  } else {
    RUNTIME_ERROR("sub not support ElementType:" << x.element_type().name());
  }
}

template <typename T>
void mul_impl(T* x, T* y, T* z, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);
  EVector<T> zv(z, size);

  zv.noalias() = xv.cwiseProduct(yv);
}

void mul(const Tensor& x, const Tensor& y, Tensor& z) {
  ARGUMENT_CHECK(x.element_type() == y.element_type() &&
                     x.element_type() == z.element_type(),
                 "mul need all tensor has same ElementType.");

  int64_t size = x.size();

  ARGUMENT_CHECK(size == y.size() && size == z.size(),
                 "mul need all tensor has same size.");

  if (x.element_type().is<float>()) {
    mul_impl<float>(x.data<float>(), y.data<float>(), z.data<float>(), size);
  } else if (x.element_type().is<double>()) {
    mul_impl<double>(x.data<double>(), y.data<double>(), z.data<double>(),
                     size);
  } else {
    RUNTIME_ERROR("mul not support ElementType:" << x.element_type().name());
  }
}

template <typename T>
void mul_impl(T v, T* x, T* y, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);

  yv.noalias() = v * xv;
}

void mul(float v, const Tensor& x, Tensor& y) {
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "mul need all tensor has same ElementType.");

  ARGUMENT_CHECK(x.size() == y.size(), "mul need all tensor has same size.");

  if (x.element_type().is<float>()) {
    mul_impl<float>(v, x.data<float>(), y.data<float>(), x.size());
  } else if (x.element_type().is<double>()) {
    mul_impl<double>((double)v, x.data<double>(), y.data<double>(), x.size());
  } else {
    RUNTIME_ERROR("mul not support ElementType:" << x.element_type().name());
  }
}

template <typename T>
void div_impl(T* x, T* y, T* z, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);
  EVector<T> zv(z, size);

  zv.noalias() = (xv.array() / yv.array()).matrix();
}

void div(const Tensor& x, const Tensor& y, Tensor& z) {
  ARGUMENT_CHECK(x.element_type() == y.element_type() &&
                     x.element_type() == z.element_type(),
                 "div need all tensor has same ElementType.");

  int64_t size = x.size();

  ARGUMENT_CHECK(size == y.size() && size == z.size(),
                 "div need all tensor has same size.");

  if (x.element_type().is<float>()) {
    div_impl<float>(x.data<float>(), y.data<float>(), z.data<float>(), size);
  } else if (x.element_type().is<double>()) {
    div_impl<double>(x.data<double>(), y.data<double>(), z.data<double>(),
                     size);
  } else {
    RUNTIME_ERROR("div not support ElementType:" << x.element_type().name());
  }
}

template <typename T>
void div_impl(T v, T* x, T* y, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);

  yv.noalias() = (v / xv.array()).matrix();
}

void div(float v, const Tensor& x, Tensor& y) {
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "div need all tensor has same ElementType.");

  ARGUMENT_CHECK(x.size() == y.size(), "div need all tensor has same size.");

  if (x.element_type().is<float>()) {
    div_impl<float>(v, x.data<float>(), y.data<float>(), x.size());
  } else if (x.element_type().is<double>()) {
    div_impl<double>((double)v, x.data<double>(), y.data<double>(), x.size());
  } else {
    RUNTIME_ERROR("div not support ElementType:" << x.element_type().name());
  }
}

template <typename T>
void div_impl(T* x, T v, T* y, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);

  yv.noalias() = xv / v;
}

void div(const Tensor& x, float v, Tensor& y) {
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "div need all tensor has same ElementType.");

  ARGUMENT_CHECK(x.size() == y.size(), "div need all tensor has same size.");

  if (x.element_type().is<float>()) {
    div_impl<float>(x.data<float>(), v, y.data<float>(), x.size());
  } else if (x.element_type().is<double>()) {
    div_impl<double>(x.data<double>(), (double)v, y.data<double>(), x.size());
  } else {
    RUNTIME_ERROR("div not support ElementType:" << x.element_type().name());
  }
}

template <typename T>
void initialize_norm_impl(T* v, int64_t n, T lower, float upper) {
  static thread_local std::mt19937 generator;
  std::normal_distribution<T> distribution(lower, upper);

  for (int64_t i = 0; i < n; ++i) {
    v[i] = distribution(generator);
  }
}

void initialize_norm(Tensor& x, float lower, float upper) {
  if (x.element_type().is<float>()) {
    initialize_norm_impl<float>(x.data<float>(), x.size(), lower, upper);
  } else if (x.element_type().is<double>()) {
    initialize_norm_impl<double>(x.data<double>(), x.size(), lower, upper);
  } else {
    RUNTIME_ERROR(
        "initialize_norm not support ElementType:" << x.element_type().name());
  }
}

}  // namespace kraken
