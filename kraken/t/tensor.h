#pragma once

#include "t/tensor_impl.h"

namespace kraken {

class Tensor {
private:
  std::shared_ptr<TensorImpl> impl_;

public:
  Tensor() = default;

  explicit Tensor(std::shared_ptr<TensorImpl> impl);

  ~Tensor() = default;

public:
  std::shared_ptr<TensorImpl> impl() const;

  Layout layout() const;

  const Shape& shape() const;

  const Tensor& indices() const;

  const Tensor& values() const;

  ElementType element_type() const;

  kraken::Device* Device() const;

  bool IsCoo() const;

  bool IsDense() const;

  int64_t Size() const;

  int64_t NumBytes() const;

  void* Ptr() const;

  bool IsEmpty() const;

  template <typename T>
  T* Data() const {
    return impl_->Data<T>();
  }

  std::string Str() const;

public:
  static Tensor Dense(const std::vector<int64_t>& dims,
                      ElementType element_type);

  static Tensor Dense(const Shape& shape, ElementType element_type);

  static Tensor Dense(const Shape& shape, std::shared_ptr<Storage> storage,
                      size_t offset, ElementType element_type);

  static Tensor Empty(const Shape& shape, ElementType element_type);

public:
  Tensor operator+(const Tensor& other) const;
  Tensor operator-(const Tensor& other) const;
  Tensor operator*(const Tensor& other) const;
  Tensor operator/(const Tensor& other) const;

  Tensor operator+=(const Tensor& other);
  Tensor operator-=(const Tensor& other);
  Tensor operator*=(const Tensor& other);
  Tensor operator/=(const Tensor& other);

  Tensor operator+(float v) const;
  Tensor operator-(float v) const;
  Tensor operator*(float v) const;
  Tensor operator/(float v) const;

  Tensor operator+=(float v);
  Tensor operator-=(float v);
  Tensor operator*=(float v);
  Tensor operator/=(float v);

  float operator[](int64_t i) const;

  Tensor Reshape(const Shape& nshape) const;

  Tensor Reshape(const std::vector<int64_t>& dims) const;

  Tensor Zero();

  Tensor Like() const;

  Tensor Clone() const;

  Tensor Constant(float v);

  Tensor Square(bool in_place = false);

  Tensor Sqrt(bool in_place = false);

  Tensor Max(const Tensor& other) const;

  Tensor Vector(int64_t idx) const;

  Tensor ConcatVector(const std::vector<Tensor>& vecs) const;

  Tensor Normal(float mean, float stddev);

  Tensor XavierNormal(float gain);

  Tensor Uniform(float lower, float upper);

  Tensor XavierUniform(float gain);

  Tensor Abs(bool in_place = false);

  Tensor TopK(int64_t k) const;

  Tensor Take(const Tensor& indices) const;

  Tensor FlatNonZero(float th) const;

  Tensor NonZero(float th) const;

  Tensor Transpose(int64_t d0 = 0, int64_t d1 = 1) const;

  Tensor ToDense() const;

  Tensor ToCoo(float th) const;

  Tensor LtKeep(float th) const;

  Tensor Cast(ElementType to_type) const;
};

// operator override
Tensor operator+(float v, const Tensor& t);
Tensor operator-(float v, const Tensor& t);
Tensor operator*(float v, const Tensor& t);
Tensor operator/(float v, const Tensor& t);

}  // namespace kraken
