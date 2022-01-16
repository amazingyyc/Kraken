#include "t/math.h"

#include <eigen/Eigen/Dense>
#include <random>

#include "common/utils.h"

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#include "common/exception.h"

namespace kraken {
namespace math {

template <typename T>
using EVector =
    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor>>;

std::vector<int64_t> CalFanInAndFanOut(const TensorImpl& t) {
  ARGUMENT_CHECK(t.IsDense(), "CalFanInAndFanOut need Dense Tensor.");

  int64_t ndims = t.shape().NDims();

  int64_t num_input_fmaps = 1;
  int64_t num_output_fmaps = 1;
  int64_t receptive_field_size = 1;

  // This maybe different with pytorch.
  if (ndims >= 1) {
    num_output_fmaps = t.shape().Dim(0);
  }

  if (ndims >= 2) {
    num_input_fmaps = t.shape().Dim(1);
  }

  for (int64_t i = 2; i < ndims; ++i) {
    receptive_field_size *= t.shape().Dim(i);
  }

  int64_t fan_in = num_input_fmaps * receptive_field_size;
  int64_t fan_out = num_output_fmaps * receptive_field_size;

  return {fan_in, fan_out};
}

int64_t CalculateCorrectFan(const TensorImpl& t, const std::string& mode) {
  ARGUMENT_CHECK(t.IsDense(), "CalculateCorrectFan need Dense Tensor.");

  std::string l_mode = utils::ToLower(mode);

  ARGUMENT_CHECK(l_mode == "fan_in" || l_mode == "fan_out",
                 "CalculateCorrectFan need mode is fan_in/fan_out.");

  auto fan_in_out = CalFanInAndFanOut(t);

  if (l_mode == "fan_in") {
    return fan_in_out[0];
  } else {
    return fan_in_out[1];
  }
}

template <typename T>
void AddImpl(T* x, T* y, T* z, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);
  EVector<T> zv(z, size);

  zv.noalias() = xv + yv;
}

void Add(const TensorImpl& x, const TensorImpl& y, TensorImpl& z) {
  ARGUMENT_CHECK(x.IsDense() && y.IsDense() && z.IsDense(),
                 "Add need Dense TensorImpl.");
  ARGUMENT_CHECK(x.element_type() == y.element_type() &&
                     x.element_type() == z.element_type(),
                 "Add need all TensorImpl has same ElementType.");

  int64_t size = x.Size();

  ARGUMENT_CHECK(size == y.Size() && size == z.Size(),
                 "Add need all TensorImpl has same size.");

  if (x.element_type().Is<float>()) {
    AddImpl<float>(x.Data<float>(), y.Data<float>(), z.Data<float>(), size);
  } else if (x.element_type().Is<double>()) {
    AddImpl<double>(x.Data<double>(), y.Data<double>(), z.Data<double>(), size);
  } else {
    RUNTIME_ERROR("Add not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void AddImpl(T v, T* x, T* y, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);

  yv.noalias() = (v + xv.array()).matrix();
}

void Add(float v, const TensorImpl& x, TensorImpl& y) {
  ARGUMENT_CHECK(x.IsDense() && y.IsDense(), "Add need TensorImpl is Dense.");
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "Add need all tensor has same ElementType.");
  ARGUMENT_CHECK(x.Size() == y.Size(), "Add need all tensor has same size.");

  if (x.element_type().Is<float>()) {
    AddImpl<float>(v, x.Data<float>(), y.Data<float>(), x.Size());
  } else if (x.element_type().Is<double>()) {
    AddImpl<double>((double)v, x.Data<double>(), y.Data<double>(), x.Size());
  } else {
    RUNTIME_ERROR("Add not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void SubImpl(T* x, T* y, T* z, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);
  EVector<T> zv(z, size);

  zv.noalias() = xv - yv;
}

void Sub(const TensorImpl& x, const TensorImpl& y, TensorImpl& z) {
  ARGUMENT_CHECK(x.IsDense() && y.IsDense() && z.IsDense(),
                 "Sub need Dense TensorImpl.");
  ARGUMENT_CHECK(x.element_type() == y.element_type() &&
                     x.element_type() == z.element_type(),
                 "Sub need all TensorImpl has same ElementType.");

  int64_t size = x.Size();

  ARGUMENT_CHECK(size == y.Size() && size == z.Size(),
                 "Sub need all TensorImpl has same size.");

  if (x.element_type().Is<float>()) {
    SubImpl<float>(x.Data<float>(), y.Data<float>(), z.Data<float>(), size);
  } else if (x.element_type().Is<double>()) {
    SubImpl<double>(x.Data<double>(), y.Data<double>(), z.Data<double>(), size);
  } else {
    RUNTIME_ERROR("Sub not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void SubImpl(T v, T* x, T* y, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);

  yv.noalias() = (v - xv.array()).matrix();
}

void Sub(float v, const TensorImpl& x, TensorImpl& y) {
  ARGUMENT_CHECK(x.IsDense() && y.IsDense(), "Sub need TensorImpl is Dense.");
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "Sub need all TensorImpl has same ElementType.");
  ARGUMENT_CHECK(x.Size() == y.Size(),
                 "Sub need all TensorImpl has same size.");

  if (x.element_type().Is<float>()) {
    SubImpl<float>(v, x.Data<float>(), y.Data<float>(), x.Size());
  } else if (x.element_type().Is<double>()) {
    SubImpl<double>((double)v, x.Data<double>(), y.Data<double>(), x.Size());
  } else {
    RUNTIME_ERROR("Sub not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void SubImpl(T* x, T v, T* y, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);

  yv.noalias() = (xv.array() - v).matrix();
}

void Sub(const TensorImpl& x, float v, TensorImpl& y) {
  ARGUMENT_CHECK(x.IsDense() && y.IsDense(), "Sub need TensorImpl is Dense.");
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "Sub need all TensorImpl has same ElementType.");
  ARGUMENT_CHECK(x.Size() == y.Size(),
                 "Sub need all TensorImpl has same size.");

  if (x.element_type().Is<float>()) {
    SubImpl<float>(x.Data<float>(), v, y.Data<float>(), x.Size());
  } else if (x.element_type().Is<double>()) {
    SubImpl<double>(x.Data<double>(), (double)v, y.Data<double>(), x.Size());
  } else {
    RUNTIME_ERROR("Sub not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void MulImpl(T* x, T* y, T* z, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);
  EVector<T> zv(z, size);

  zv.noalias() = xv.cwiseProduct(yv);
}

void Mul(const TensorImpl& x, const TensorImpl& y, TensorImpl& z) {
  ARGUMENT_CHECK(x.IsDense() && y.IsDense() && z.IsDense(),
                 "Mul need Dense TensorImpl.");
  ARGUMENT_CHECK(x.element_type() == y.element_type() &&
                     x.element_type() == z.element_type(),
                 "Mul need all TensorImpl has same ElementType.");

  int64_t size = x.Size();

  ARGUMENT_CHECK(size == y.Size() && size == z.Size(),
                 "Mul need all TensorImpl has same size.");

  if (x.element_type().Is<float>()) {
    MulImpl<float>(x.Data<float>(), y.Data<float>(), z.Data<float>(), size);
  } else if (x.element_type().Is<double>()) {
    MulImpl<double>(x.Data<double>(), y.Data<double>(), z.Data<double>(), size);
  } else {
    RUNTIME_ERROR("Mul not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void MulImpl(T v, T* x, T* y, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);

  yv.noalias() = v * xv;
}

void Mul(float v, const TensorImpl& x, TensorImpl& y) {
  ARGUMENT_CHECK(x.IsDense() && y.IsDense(), "Mul need Dense TensorImpl.");
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "Mul need all TensorImpl has same ElementType.");

  ARGUMENT_CHECK(x.Size() == y.Size(),
                 "Mul need all TensorImpl has same size.");

  if (x.element_type().Is<float>()) {
    MulImpl<float>(v, x.Data<float>(), y.Data<float>(), x.Size());
  } else if (x.element_type().Is<double>()) {
    MulImpl<double>((double)v, x.Data<double>(), y.Data<double>(), x.Size());
  } else {
    RUNTIME_ERROR("Mul not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void DivImpl(T* x, T* y, T* z, int64_t n) {
  EVector<T> xv(x, n);
  EVector<T> yv(y, n);
  EVector<T> zv(z, n);

  zv.noalias() = (xv.array() / yv.array()).matrix();
}

void Div(const TensorImpl& x, const TensorImpl& y, TensorImpl& z) {
  ARGUMENT_CHECK(x.IsDense() && y.IsDense() && z.IsDense(),
                 "Div need Dense TensorImpl.");
  ARGUMENT_CHECK(x.element_type() == y.element_type() &&
                     x.element_type() == z.element_type(),
                 "Div need all TensorImpl has same ElementType.");

  int64_t size = x.Size();

  ARGUMENT_CHECK(size == y.Size() && size == z.Size(),
                 "Div need all TensorImpl has same size.");

  if (x.element_type().Is<float>()) {
    DivImpl<float>(x.Data<float>(), y.Data<float>(), z.Data<float>(), size);
  } else if (x.element_type().Is<double>()) {
    DivImpl<double>(x.Data<double>(), y.Data<double>(), z.Data<double>(), size);
  } else {
    RUNTIME_ERROR("Div not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void DivImpl(T v, T* x, T* y, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);

  yv.noalias() = (v / xv.array()).matrix();
}

void Div(float v, const TensorImpl& x, TensorImpl& y) {
  ARGUMENT_CHECK(x.IsDense() && y.IsDense(), "Div need Dense TensorImpl.");
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "Div need all TensorImpl has same ElementType.");
  ARGUMENT_CHECK(x.Size() == y.Size(),
                 "Div need all TensorImpl has same size.");

  if (x.element_type().Is<float>()) {
    DivImpl<float>(v, x.Data<float>(), y.Data<float>(), x.Size());
  } else if (x.element_type().Is<double>()) {
    DivImpl<double>((double)v, x.Data<double>(), y.Data<double>(), x.Size());
  } else {
    RUNTIME_ERROR("Div not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void DivImpl(T* x, T v, T* y, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);

  yv.noalias() = xv / v;
}

void Div(const TensorImpl& x, float v, TensorImpl& y) {
  ARGUMENT_CHECK(x.IsDense() && y.IsDense(), "Div need Dense TensorImpl.");
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "Div need all TensorImpl has same ElementType.");
  ARGUMENT_CHECK(x.Size() == y.Size(),
                 "Div need all TensorImpl has same size.");

  if (x.element_type().Is<float>()) {
    DivImpl<float>(x.Data<float>(), v, y.Data<float>(), x.Size());
  } else if (x.element_type().Is<double>()) {
    DivImpl<double>(x.Data<double>(), (double)v, y.Data<double>(), x.Size());
  } else {
    RUNTIME_ERROR("Div not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void ConstantImpl(T* p, int64_t n, T v) {
  int64_t limit = n / 8 * 8;
  int64_t i = 0;
  for (; i < limit; i += 8) {
    p[i] = v;
    p[i + 1] = v;
    p[i + 2] = v;
    p[i + 3] = v;
    p[i + 4] = v;
    p[i + 5] = v;
    p[i + 6] = v;
    p[i + 7] = v;
  }

  for (; i < n; ++i) {
    p[i] = v;
  }
}

void Constant(TensorImpl& x, float v) {
  if (x.element_type().Is<float>()) {
    ConstantImpl<float>(x.Data<float>(), x.Size(), v);
  } else if (x.element_type().Is<double>()) {
    ConstantImpl<double>(x.Data<double>(), x.Size(), v);
  } else {
    RUNTIME_ERROR(
        "Constant not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void SqrtImpl(T* x, T* y, int64_t n) {
  EVector<T> xv(x, n);
  EVector<T> yv(y, n);

  yv.noalias() = xv.array().sqrt().matrix();
}

void Sqrt(const TensorImpl& x, TensorImpl& y) {
  ARGUMENT_CHECK(x.IsDense() && y.IsDense(), "Sqrt need Dense TensorImpl.");
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "Sqrt need all TensorImpl has same ElementType.");
  ARGUMENT_CHECK(x.Size() == y.Size(),
                 "Sqrt need all TensorImpl has same size.");

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

void Max(const TensorImpl& x, const TensorImpl& y, TensorImpl& z) {
  ARGUMENT_CHECK(x.IsDense() && y.IsDense() && z.IsDense(),
                 "Max need Dense TensorImpl.");
  ARGUMENT_CHECK(x.element_type() == y.element_type() &&
                     x.element_type() == z.element_type(),
                 "Max need all TensorImpl has same ElementType.");

  int64_t size = x.Size();

  ARGUMENT_CHECK(size == y.Size() && size == z.Size(),
                 "Max need all TensorImpl has same size.");

  if (x.element_type().Is<float>()) {
    MaxImpl<float>(x.Data<float>(), y.Data<float>(), z.Data<float>(), size);
  } else if (x.element_type().Is<double>()) {
    MaxImpl<double>(x.Data<double>(), y.Data<double>(), z.Data<double>(), size);
  } else {
    RUNTIME_ERROR("Max not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void ConcatVectorImpl(const std::vector<std::shared_ptr<TensorImpl>>& xs, T* y,
                      int64_t row, int64_t col) {
#pragma omp parallel for
  for (int64_t i = 0; i < row; ++i) {
    T* xp = xs[i]->Data<T>();
    T* yp = y + i * col;

    memcpy(yp, xp, col * sizeof(T));
  }
}

void ConcatVector(const std::vector<std::shared_ptr<TensorImpl>>& xs,
                  TensorImpl& y) {
  int64_t row = (int64_t)xs.size();
  int64_t col = xs[0]->Size();

  ARGUMENT_CHECK(
      y.shape().NDims() == 2 && row == y.shape()[0] && col == y.shape()[1],
      "ConcatVector need xs is a vector list and y must be a matrix.");

  for (const auto& v : xs) {
    ARGUMENT_CHECK(v->Size() == col, "ConcatVector's parameter error.");
    ARGUMENT_CHECK(v->element_type() == y.element_type(),
                   "ConcatVector need parameter has same type.");
  }

  if (y.element_type().Is<float>()) {
    ConcatVectorImpl<float>(xs, y.Data<float>(), row, col);
  } else if (y.element_type().Is<double>()) {
    ConcatVectorImpl<double>(xs, y.Data<double>(), row, col);
  } else {
    RUNTIME_ERROR(
        "ConcatVector not support ElementType:" << y.element_type().Name());
  }
}

template <typename T>
void NormalImpl(T* v, int64_t n, T mean, T stddev) {
  static thread_local std::mt19937 generator;
  std::normal_distribution<T> distribution(mean, stddev);

  for (int64_t i = 0; i < n; ++i) {
    v[i] = distribution(generator);
  }
}

void Normal(TensorImpl& x, float mean, float stddev) {
  if (x.element_type().Is<float>()) {
    NormalImpl<float>(x.Data<float>(), x.Size(), mean, stddev);
  } else if (x.element_type().Is<double>()) {
    NormalImpl<double>(x.Data<double>(), x.Size(), mean, stddev);
  } else {
    RUNTIME_ERROR("Normal not support ElementType:" << x.element_type().Name());
  }
}

void XavierNormal(TensorImpl& x, float gain) {
  std::vector<int64_t> fan_in_out = CalFanInAndFanOut(x);
  int64_t fan_in = fan_in_out[0];
  int64_t fan_out = fan_in_out[1];

  float std = gain * std::sqrt(2.0 / float(fan_in + fan_out));

  Normal(x, 0, std);
}

template <typename T>
void UniformImpl(T* v, int64_t n, T lower, T upper) {
  static thread_local std::mt19937 gen;
  std::uniform_real_distribution<T> dis(lower, upper);

  for (int64_t i = 0; i < n; ++i) {
    v[i] = dis(gen);
  }
}

void Uniform(TensorImpl& x, float lower, float upper) {
  if (x.element_type().Is<float>()) {
    UniformImpl<float>(x.Data<float>(), x.Size(), lower, upper);
  } else if (x.element_type().Is<double>()) {
    UniformImpl<double>(x.Data<double>(), x.Size(), lower, upper);
  } else {
    RUNTIME_ERROR(
        "Uniform not support ElementType:" << x.element_type().Name());
  }
}

void XavierUniform(TensorImpl& x, float gain) {
  std::vector<int64_t> fan_in_out = CalFanInAndFanOut(x);
  int64_t fan_in = fan_in_out[0];
  int64_t fan_out = fan_in_out[1];

  float std = gain * std::sqrt(2.0 / float(fan_in + fan_out));
  float a = std::sqrt(3.0) * std;

  Uniform(x, -a, a);
}

template <typename T>
void GeImpl(T* x, T v, bool* y, int64_t n) {
  EVector<T> xv(x, n);
  EVector<bool> yv(y, n);

  yv.noalias() = (xv.array() >= v).template cast<bool>().matrix();
}

void Ge(const TensorImpl& x, float v, TensorImpl& y) {
  ARGUMENT_CHECK(x.Size() == y.Size(), "Ge need parameter has same size.");
  ARGUMENT_CHECK(y.element_type().Is<bool>(), "Ge need y is bool.");

  if (x.element_type().Is<float>()) {
    GeImpl<float>(x.Data<float>(), v, y.Data<bool>(), x.Size());
  } else if (x.element_type().Is<double>()) {
    GeImpl<double>(x.Data<double>(), v, y.Data<bool>(), x.Size());
  } else {
    RUNTIME_ERROR("Ge not support ElementType:" << x.element_type().Name());
  }
}

}  // namespace math
}  // namespace kraken
