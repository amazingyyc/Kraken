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

void AssertVectorF32(const std::vector<float>& v1,
                     const std::vector<float>& v2) {
  EXPECT_EQ(v1.size(), v2.size());

  for (size_t i = 0; i < v1.size(); ++i) {
    EXPECT_FLOAT_EQ(v1[i], v2[i]);
  }
}

}  // namespace test
}  // namespace kraken
