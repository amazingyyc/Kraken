#include "t/math.h"

#include <eigen/Eigen/Dense>
#include <queue>
#include <random>
#include <thread>
#include <vector>

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
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<T> dist{mean, stddev};

  for (int64_t i = 0; i < n; ++i) {
    v[i] = dist(gen);
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
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::uniform_real_distribution<T> dist{lower, upper};

  for (int64_t i = 0; i < n; ++i) {
    v[i] = dist(gen);
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

template <typename T>
void AbsImpl(T* x, T* y, int64_t size) {
  EVector<T> xv(x, size);
  EVector<T> yv(y, size);

  yv.noalias() = xv.cwiseAbs();
}

void Abs(const TensorImpl& x, TensorImpl& y) {
  ARGUMENT_CHECK(x.IsDense() && y.IsDense(), "Abs need Dense TensorImpl.");
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "Abs need all TensorImpl has same ElementType.");
  ARGUMENT_CHECK(x.Size() == y.Size(),
                 "Abs need all TensorImpl has same size.");

  if (x.element_type().Is<float>()) {
    AbsImpl<float>(x.Data<float>(), y.Data<float>(), x.Size());
  } else if (x.element_type().Is<double>()) {
    AbsImpl<double>(x.Data<double>(), y.Data<double>(), x.Size());
  } else {
    RUNTIME_ERROR("Abs not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
void TopKParallelImpl(Device* d, T* x, int64_t n, T* y, int64_t k) {
  ARGUMENT_CHECK(n >= k, "TopKParallelImpl need n >= k.");

  int64_t max_t_num = n / k;

  int64_t t_num = (int64_t)std::thread::hardware_concurrency();
  t_num = std::min(max_t_num, t_num);

  int64_t stride = n / t_num;

  T* tp = (T*)d->Malloc(sizeof(T) * t_num * k);
  std::vector<T*> tps;
  tps.resize(t_num);

  for (int64_t i = 0; i < t_num; ++i) {
    tps[i] = tp + i * k;
  }

#pragma omp parallel for
  for (int64_t i = 0; i < t_num; ++i) {
    int64_t start = i * stride;
    int64_t end = std::min(start + stride, n);

    std::priority_queue<T, std::vector<T>, std::greater<T>> q;

    for (int64_t j = start; j < end; ++j) {
      if (q.size() < (size_t)k) {
        q.push(x[j]);
      } else if (q.top() < x[j]) {
        q.pop();
        q.push(x[j]);
      }
    }

    int64_t l = k;
    while (!q.empty()) {
      (tps[i])[--l] = q.top();
      q.pop();
    }
  }

  std::priority_queue<T, std::vector<T>, std::greater<T>> q;

  for (int64_t j = 0; j < t_num * k; ++j) {
    if (q.size() < (size_t)k) {
      q.push(tp[j]);
    } else if (q.top() < tp[j]) {
      q.pop();
      q.push(tp[j]);
    }
  }

  int64_t l = k;
  while (!q.empty()) {
    y[--l] = q.top();
    q.pop();
  }

  tps.clear();

  d->Free(tp);
}

template <typename T>
void TopKImpl(T* x, int64_t n, T* y, int64_t k) {
  std::priority_queue<T, std::vector<T>, std::greater<T>> q;

  for (int64_t i = 0; i < n; ++i) {
    if (q.size() < k) {
      q.push(x[i]);
    } else if (q.top() < x[i]) {
      q.pop();
      q.push(x[i]);
    }
  }

  int64_t i = k;
  while (!q.empty()) {
    y[--i] = q.top();
    q.pop();
  }
}

void TopK(const TensorImpl& x, TensorImpl& y) {
  ARGUMENT_CHECK(x.IsDense() && y.IsDense(), "TopK need Dense TensorImpl.");
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "TopK need all TensorImpl has same ElementType.");
  ARGUMENT_CHECK(y.Size() > 0, "TopK need y size > 0.");
  ARGUMENT_CHECK(x.Size() >= y.Size(), "TopK x size >= y size.");

  if (x.element_type().Is<float>()) {
    TopKParallelImpl<float>(x.Device(), x.Data<float>(), x.Size(),
                            y.Data<float>(), y.Size());
  } else if (x.element_type().Is<double>()) {
    TopKParallelImpl<double>(x.Device(), x.Data<double>(), x.Size(),
                             y.Data<double>(), y.Size());
  } else {
    RUNTIME_ERROR("TopK not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
int64_t CountNonZeroImpl(T* x, int64_t n, T th) {
  int64_t t_num = (int64_t)std::thread::hardware_concurrency();
  if (n <= 256) {
    t_num = 1;
  }

  int64_t stride = (n + t_num - 1) / t_num;
  stride = std::max<int64_t>(256, stride);

  t_num = (n + stride - 1) / stride;

  std::vector<int64_t> counts(t_num, 0);

  th = std::abs(th);

#pragma omp parallel for
  for (int64_t i = 0; i < t_num; ++i) {
    int64_t start = i * stride;
    int64_t end = std::min(start + stride, n);

    for (int64_t j = start; j < end; ++j) {
      if (std::abs(x[j]) >= th) {
        counts[i] += 1;
      }
    }
  }

  int64_t count = 0;
  for (auto c : counts) {
    count += c;
  }

  return count;
}

int64_t CountNonZero(const TensorImpl& x, float th) {
  ARGUMENT_CHECK(x.IsDense(), "CountNonZero need Dense TensorImpl.");

  if (x.Size() == 0) {
    return 0;
  }

  if (x.element_type().Is<float>()) {
    return CountNonZeroImpl<float>(x.Data<float>(), x.Size(), th);
  } else if (x.element_type().Is<double>()) {
    return CountNonZeroImpl<double>(x.Data<double>(), x.Size(), th);
  } else {
    RUNTIME_ERROR(
        "CountNonZero not support ElementType:" << x.element_type().Name());
  }
}

template <typename T, typename IT>
void TakeImpl(T* x, int64_t n, IT* id, T* y, int64_t k) {
  int64_t t_num = (int64_t)std::thread::hardware_concurrency();
  if (k <= 256) {
    t_num = 1;
  }

  int64_t stride = (k + t_num - 1) / t_num;
  stride = std::max<int64_t>(256, stride);

  t_num = (k + stride - 1) / stride;

#pragma omp parallel for
  for (int64_t i = 0; i < t_num; ++i) {
    int64_t start = i * stride;
    int64_t end = std::min(start + stride, k);

    for (int64_t j = start; j < end; ++j) {
      IT rid = id[j];

      if (rid < 0 || (int64_t)rid >= n) {
        RUNTIME_ERROR("Take outof range.");
      }

      y[j] = x[rid];
    }
  }
}

void Take(const TensorImpl& x, const TensorImpl& indices, TensorImpl& y) {
  ARGUMENT_CHECK(x.IsDense() && indices.IsDense() && y.IsDense(),
                 "Take need TensorImpl is Dense.");
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "Take need x/y has same element_type.");
  ARGUMENT_CHECK(!indices.IsEmpty(), "Take need indices size > 0.");
  ARGUMENT_CHECK(indices.shape() == y.shape(),
                 "Take need indices/y has same shape.");

  if (x.element_type().Is<float>()) {
    if (indices.element_type().Is<uint32_t>()) {
      TakeImpl<float, uint32_t>(x.Data<float>(), x.Size(),
                                indices.Data<uint32_t>(), y.Data<float>(),
                                y.Size());
    } else if (indices.element_type().Is<int32_t>()) {
      TakeImpl<float, int32_t>(x.Data<float>(), x.Size(),
                               indices.Data<int32_t>(), y.Data<float>(),
                               y.Size());
    } else if (indices.element_type().Is<uint64_t>()) {
      TakeImpl<float, uint64_t>(x.Data<float>(), x.Size(),
                                indices.Data<uint64_t>(), y.Data<float>(),
                                y.Size());
    } else if (indices.element_type().Is<int64_t>()) {
      TakeImpl<float, int64_t>(x.Data<float>(), x.Size(),
                               indices.Data<int64_t>(), y.Data<float>(),
                               y.Size());
    } else {
      RUNTIME_ERROR(
          "Take not support ElementType:" << indices.element_type().Name());
    }
  } else if (x.element_type().Is<double>()) {
    if (indices.element_type().Is<uint32_t>()) {
      TakeImpl<double, uint32_t>(x.Data<double>(), x.Size(),
                                 indices.Data<uint32_t>(), y.Data<double>(),
                                 y.Size());
    } else if (indices.element_type().Is<int32_t>()) {
      TakeImpl<double, int32_t>(x.Data<double>(), x.Size(),
                                indices.Data<int32_t>(), y.Data<double>(),
                                y.Size());
    } else if (indices.element_type().Is<uint64_t>()) {
      TakeImpl<double, uint64_t>(x.Data<double>(), x.Size(),
                                 indices.Data<uint64_t>(), y.Data<double>(),
                                 y.Size());
    } else if (indices.element_type().Is<int64_t>()) {
      TakeImpl<double, int64_t>(x.Data<double>(), x.Size(),
                                indices.Data<int64_t>(), y.Data<double>(),
                                y.Size());
    } else {
      RUNTIME_ERROR(
          "Take not support ElementType:" << indices.element_type().Name());
    }
  } else {
    RUNTIME_ERROR("Take not support ElementType:" << x.element_type().Name());
  }
}

template <typename T>
std::shared_ptr<TensorImpl> FlatNonZeroImpl(T* x, int64_t n, T th) {
  int64_t t_num = (int64_t)std::thread::hardware_concurrency();
  if (n <= 256) {
    t_num = 1;
  }

  int64_t stride = (n + t_num - 1) / t_num;
  stride = std::max<int64_t>(256, stride);

  t_num = (n + stride - 1) / stride;

  std::vector<std::vector<int64_t>> ids;
  ids.resize(t_num);

#pragma omp parallel for
  for (int64_t i = 0; i < t_num; ++i) {
    int64_t start = i * stride;
    int64_t end = std::min(start + stride, n);

    for (int64_t j = start; j < end; ++j) {
      if (std::abs(x[j]) >= th) {
        ids[i].emplace_back(j);
      }
    }
  }

  int64_t num = 0;
  for (int64_t i = 0; i < t_num; ++i) {
    num += (int64_t)ids[i].size();
  }

  if (num <= 0) {
    return TensorImpl::Empty(ElementType::From<int64_t>());
  }

  auto storage = Storage::Create(sizeof(int64_t) * num);

  int64_t* ptr = (int64_t*)storage->ptr();
  size_t offset = 0;
  for (int64_t i = 0; i < t_num; ++i) {
    storage->device()->Memcpy(ptr + offset, ids[i].data(),
                              sizeof(int64_t) * ids[i].size());
    offset += ids[i].size();
  }

  return TensorImpl::Dense(Shape({num}), storage, 0,
                           ElementType::From<int64_t>());
}

std::shared_ptr<TensorImpl> FlatNonZero(const TensorImpl& x, float th) {
  ARGUMENT_CHECK(x.IsDense(), "FlatNonZero need TensorImpl is Dense.");

  if (x.IsEmpty()) {
    return TensorImpl::Empty(ElementType::From<int64_t>());
  }

  if (x.element_type().Is<float>()) {
    return FlatNonZeroImpl<float>(x.Data<float>(), x.Size(), th);
  } else if (x.element_type().Is<double>()) {
    return FlatNonZeroImpl<double>(x.Data<double>(), x.Size(), th);
  } else {
    RUNTIME_ERROR(
        "FlatNonZero not support ElementType:" << x.element_type().Name());
  }
}

// fid shape: [nnz]
// id shape: [nnz, shape.NDims()]
void NonZeroImpl(int64_t* fid, int64_t nnz, int64_t* id, const Shape& shape) {
  int64_t t_num = (int64_t)std::thread::hardware_concurrency();
  if (nnz <= 256) {
    t_num = 1;
  }

  int64_t stride = (nnz + t_num - 1) / t_num;
  stride = std::max<int64_t>(256, stride);

  t_num = (nnz + stride - 1) / stride;

  int64_t ndims = shape.NDims();

#pragma omp parallel for
  for (int64_t i = 0; i < t_num; ++i) {
    int64_t start = i * stride;
    int64_t end = std::min(start + stride, nnz);

    for (int64_t j = start; j < end; ++j) {
      int64_t* rid = id + j * ndims;
      int64_t fi = fid[j];

      for (int64_t k = 0; k < ndims; ++k) {
        rid[k] = fi / shape.Stride(k);
        fi %= shape.Stride(k);
      }
    }
  }
}

std::shared_ptr<TensorImpl> NonZero(const TensorImpl& x, float th) {
  ARGUMENT_CHECK(x.IsDense(), "NonZero need TensorImpl is Dense.");

  if (x.IsEmpty()) {
    return TensorImpl::Empty(ElementType::From<int64_t>());
  }

  std::shared_ptr<TensorImpl> f_indices = FlatNonZero(x, th);
  if (f_indices->IsEmpty()) {
    return f_indices;
  }

  int64_t nnz = f_indices->Size();
  int64_t ndims = x.shape().NDims();

  if (ndims == 1) {
    return f_indices->Reshape({nnz, 1});
  }

  auto storage = Storage::Create(sizeof(int64_t) * nnz * ndims);

  NonZeroImpl(f_indices->Data<int64_t>(), nnz, (int64_t*)storage->ptr(),
              x.shape());

  return TensorImpl::Dense(Shape({nnz, ndims}), storage, 0,
                           ElementType::From<int64_t>());
}

template <typename T>
void TransposeImpl(T* x, const Shape& xshape, T* y, const Shape& yshape,
                   int64_t n, int64_t d0, int64_t d1) {
  int64_t t_num = (int64_t)std::thread::hardware_concurrency();
  if (n <= 256) {
    t_num = 1;
  }

  int64_t stride = (n + t_num - 1) / t_num;
  stride = std::max<int64_t>(256, stride);

  t_num = (n + stride - 1) / stride;

  int64_t ndims = xshape.NDims();

#pragma omp parallel for
  for (int64_t i = 0; i < t_num; ++i) {
    int64_t start = i * stride;
    int64_t end = std::min(start + stride, n);

    std::vector<int64_t> dims;
    dims.resize(ndims);

    for (int64_t j = start; j < end; ++j) {
      int64_t o = j;
      for (int64_t l = 0; l < ndims; ++l) {
        dims[l] = o / yshape.Stride(l);
        o %= yshape.Stride(l);
      }

      std::swap(dims[d0], dims[d1]);

      o = 0;
      for (int64_t l = 0; l < ndims; ++l) {
        o += dims[l] * xshape.Stride(l);
      }

      y[j] = x[o];
    }
  }
}

void Transpose(const TensorImpl& x, TensorImpl& y, int64_t d0, int64_t d1) {
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "Transpose need x/y has same ElementType.");
  ARGUMENT_CHECK(x.shape().Size() == y.shape().Size(),
                 "Transpose need x/y has same size.");
  ARGUMENT_CHECK(x.shape().NDims() == y.shape().NDims(),
                 "Transpose need x/y has same ndims.")

  while (d0 < 0) {
    d0 += x.shape().NDims();
  }

  while (d1 < 0) {
    d1 += x.shape().NDims();
  }

  ARGUMENT_CHECK(d0 < x.shape().NDims() && d1 < x.shape().NDims() && d0 != d1,
                 "Transpose d0/d1 error.");

  if (x.element_type().Is<float>()) {
    TransposeImpl<float>(x.Data<float>(), x.shape(), y.Data<float>(), y.shape(),
                         x.shape().Size(), d0, d1);
  } else if (x.element_type().Is<double>()) {
    TransposeImpl<double>(x.Data<double>(), x.shape(), y.Data<double>(),
                          y.shape(), x.shape().Size(), d0, d1);
  } else {
    RUNTIME_ERROR(
        "Transpose not support ElementType:" << x.element_type().Name());
  }
}

template <typename IT, typename T>
void CooToDenseImpl(Device* device, IT* indices, T* values, T* out,
                    int64_t sparse_dim, int64_t nnz, const Shape& shape) {
  // indices shape: [sparse_dim, nnz]
  // values shape: [nnz, shape[sparse_dim:]]
  std::vector<int64_t> s_dims;
  for (int64_t i = 0; i < sparse_dim; ++i) {
    s_dims.emplace_back(shape[i]);
  }

  Shape s_shape(s_dims);

  int64_t row = s_shape.Size();
  int64_t col = shape.Size() / row;

  int64_t t_num = (int64_t)std::thread::hardware_concurrency();
  if (nnz <= 256) {
    t_num = 1;
  }

  int64_t stride = (nnz + t_num - 1) / t_num;
  stride = std::max<int64_t>(256, stride);

  t_num = (nnz + stride - 1) / stride;

#pragma omp parallel for
  for (int64_t i = 0; i < t_num; ++i) {
    int64_t start = i * stride;
    int64_t end = std::min(start + stride, nnz);

    for (int64_t j = start; j < end; ++j) {
      int64_t r = 0;
      for (int64_t l = 0; l < sparse_dim; ++l) {
        r += indices[j + l * nnz] * s_shape.Stride(l);
      }

      // Should add the r maybe same.
      AddImpl<T>(values + j * col, out + r * col, out + r * col, col);
    }
  }
}

void CooToDense(const TensorImpl& indices, const TensorImpl& values,
                TensorImpl& dense) {
  ARGUMENT_CHECK(indices.IsDense() && values.IsDense() && dense.IsDense(),
                 "CooToDense need indices/values/dense is dense.");
  ARGUMENT_CHECK(!indices.IsEmpty() && !values.IsEmpty(),
                 "CooToDense need indices/values not empty.");
  ARGUMENT_CHECK(indices.shape().NDims() == 2,
                 "CooToDense need indices's NDim is 2.");
  ARGUMENT_CHECK(indices.shape()[-1] == values.shape()[0],
                 "CooToDense need indices's last dimension same with values's "
                 "first dimension.");
  ARGUMENT_CHECK(
      dense.shape().NDims() + 1 == indices.shape()[0] + values.shape().NDims(),
      "CooToDense shape error!");

  // make it zero.
  dense.Zero();

  int64_t sparse_dim = indices.shape()[0];
  for (int64_t i = sparse_dim, j = 1; i < dense.shape().NDims(); ++i, ++j) {
    ARGUMENT_CHECK(dense.shape()[i] == values.shape()[j],
                   "CooToDense shape error!");
  }

  int64_t nnz = indices.shape()[1];

  if (values.element_type().Is<float>()) {
    if (indices.element_type().Is<int64_t>()) {
      CooToDenseImpl<int64_t, float>(values.Device(), indices.Data<int64_t>(),
                                     values.Data<float>(), dense.Data<float>(),
                                     sparse_dim, nnz, dense.shape());
    } else {
      RUNTIME_ERROR("CooToDense not support ElementType:"
                    << indices.element_type().Name());
    }
  } else if (values.element_type().Is<double>()) {
    if (indices.element_type().Is<int64_t>()) {
      CooToDenseImpl<int64_t, double>(
          values.Device(), indices.Data<int64_t>(), values.Data<double>(),
          dense.Data<double>(), sparse_dim, nnz, dense.shape());
    } else {
      RUNTIME_ERROR("CooToDense not support ElementType:"
                    << indices.element_type().Name());
    }
  } else {
    RUNTIME_ERROR(
        "CooToDense not support ElementType:" << values.element_type().Name());
  }
}

template <typename T>
void LtKeepImpl(T* x, T th, T* y, int64_t n) {
  int64_t t_num = (int64_t)std::thread::hardware_concurrency();
  if (n <= 256) {
    t_num = 1;
  }

  int64_t stride = (n + t_num - 1) / t_num;
  stride = std::max<int64_t>(256, stride);

  t_num = (n + stride - 1) / stride;

#pragma omp parallel for
  for (int64_t i = 0; i < t_num; ++i) {
    int64_t start = i * stride;
    int64_t end = std::min(start + stride, n);

    for (int64_t j = start; j < end; ++j) {
      if (std::abs(x[j]) < th) {
        y[j] = x[j];
      } else {
        y[j] = 0;
      }
    }
  }
}

void LtKeep(const TensorImpl& x, float th, TensorImpl& y) {
  ARGUMENT_CHECK(x.IsDense() && y.IsDense(), "LtKeep need dense.");
  ARGUMENT_CHECK(x.element_type() == y.element_type(),
                 "LtKeep need input has same element type.");
  ARGUMENT_CHECK(x.Size() == y.Size(), "LtKeep need input has same size.");

  if (x.element_type().Is<float>()) {
    LtKeepImpl<float>(x.Data<float>(), th, y.Data<float>(), x.Size());
  } else if (x.element_type().Is<double>()) {
    LtKeepImpl<double>(x.Data<double>(), th, y.Data<double>(), x.Size());
  } else {
    RUNTIME_ERROR("LtKeep not support ElementType:" << x.element_type().Name());
  }
}

template <typename From, typename To>
void CastImpl(From* x, To* y, int64_t n) {
  EVector<From> xv(x, n);
  EVector<To> yv(y, n);

  yv.noalias() = xv.template cast<To>();
}

void Cast(const TensorImpl& x, TensorImpl& y) {
  ARGUMENT_CHECK(x.IsDense() && y.IsDense(), "Cast need TensorImpl.");
  ARGUMENT_CHECK(x.Size() == y.Size(), "Cast need input has same size.");

#define CONVERT_FUNC(From, To) \
  if (x.element_type().Is<From>() && y.element_type().Is<To>()) { \
    CastImpl<From, To>(x.Data<From>(), y.Data<To>(), x.Size()); \
    return; \
  }

  CONVERT_FUNC(uint32_t, int64_t);
  CONVERT_FUNC(int32_t, int64_t);
  CONVERT_FUNC(uint32_t, uint64_t);
  CONVERT_FUNC(int32_t, uint64_t);
  CONVERT_FUNC(int64_t, uint64_t);

  RUNTIME_ERROR("Unsupport convert from:" << x.element_type().Name()
                                          << " to:" << y.element_type().Name());
}

}  // namespace math
}  // namespace kraken
