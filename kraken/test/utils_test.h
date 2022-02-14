#pragma once

#include <gtest/gtest.h>

#include <vector>

#include "t/shape.h"
#include "t/tensor.h"

namespace kraken {
namespace test {

template <typename T>
Tensor VectorToTensor(const std::vector<T>& vec) {
  Tensor t =
      Tensor::Dense(Shape({(int64_t)vec.size()}), ElementType::From<T>());

  t.Device()->Memcpy(t.Ptr(), vec.data(), sizeof(T) * vec.size());

  return t;
}

template <typename T>
std::vector<T> TensorToVector(const Tensor& t) {
  size_t size = (size_t)t.Size();

  std::vector<T> vec;
  vec.resize(size);

  t.Device()->Memcpy(vec.data(), t.Ptr(), sizeof(T) * size);

  return vec;
}

template <typename T>
Tensor RandomTensor(const Shape& shape) {
  int64_t size = shape.Size();

  std::vector<T> vals;
  for (int64_t i = 0; i < size; ++i) {
    vals.emplace_back(utils::ThreadLocalRandom<T>(-1000, 1000));
  }

  return VectorToTensor<T>(vals).Reshape(shape);
}

inline void AssertVectorF32(const std::vector<float>& v1,
                            const std::vector<float>& v2) {
  EXPECT_EQ(v1.size(), v2.size());

  for (size_t i = 0; i < v1.size(); ++i) {
    EXPECT_FLOAT_EQ(v1[i], v2[i]);
  }
}

inline void AssertTensorEQ(const Tensor& t1, const Tensor& t2) {
  EXPECT_EQ(t1.layout(), t2.layout());

  if (t1.layout() == Layout::kStride) {
    EXPECT_EQ(t1.shape(), t2.shape());
    EXPECT_EQ(t1.element_type(), t2.element_type());
    EXPECT_EQ(0, memcmp(t1.Ptr(), t2.Ptr(), t1.NumBytes()));
  } else {
    EXPECT_EQ(t1.shape(), t2.shape());

    AssertTensorEQ(t1.indices(), t2.indices());
    AssertTensorEQ(t1.values(), t2.values());
  }
}

inline void AssertTensorFloatEQ(const Tensor& t1, const Tensor& t2) {
  EXPECT_EQ(t1.layout(), Layout::kStride);
  EXPECT_EQ(t2.layout(), Layout::kStride);

  EXPECT_EQ(t1.shape(), t2.shape());
  EXPECT_EQ(t1.element_type(), ElementType::From<float>());
  EXPECT_EQ(t1.element_type(), ElementType::From<float>());

  for (int64_t i = 0; i < t1.Size(); ++i) {
    EXPECT_FLOAT_EQ(t1.Data<float>()[i], t2.Data<float>()[i]);
  }
}

}  // namespace test
}  // namespace kraken
