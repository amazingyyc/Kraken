#include <gtest/gtest.h>

#include <cinttypes>
#include <vector>

#include "test/utils_test.h"

namespace kraken {
namespace test {

TEST(Math, Abs) {
  std::vector<float> vec{-1.0, 1.0, 2.0, 3.0, -4.0, -0.1};
  std::vector<float> expect{1.0, 1.0, 2.0, 3.0, 4.0, 0.1};

  Tensor t_vec = VectorToTensor(vec);
  std::vector<float> real = TensorToVector<float>(t_vec.Abs());

  AssertVectorF32(expect, real);
}

TEST(Math, TopK) {
  std::vector<float> vec{-1.0, 1.0, 2.0, 3.0, -4.0, -0.1};
  std::vector<float> expect{3.0, 2.0, 1.0, -0.1};

  Tensor t_vec = VectorToTensor(vec);
  std::vector<float> real = TensorToVector<float>(t_vec.TopK(4));

  AssertVectorF32(expect, real);
}

TEST(Math, Take) {
  std::vector<float> values{-1.0, 1.0, 2.0, 3.0, -4.0, -0.1};
  std::vector<int64_t> indices{0, 1, 3, 5};
  std::vector<float> expect{-1.0, 1.0, 3.0, -0.1};

  Tensor t_values = VectorToTensor(values);
  Tensor t_indices = VectorToTensor(indices);

  std::vector<float> real = TensorToVector<float>(t_values.Take(t_indices));

  AssertVectorF32(expect, real);
}

TEST(Math, ToCooToDense) {
  std::vector<float> values{-1.0, 1.0,   2.0,     3.0,      -4.0,
                            -0.1, 0.001, 0.00001, 0.000001, 128};
  std::vector<float> expect{-1.0, 1.0, 2.0, 3.0, -4.0,
                            -0.1, 0.0, 0.0, 0.0, 128};

  Tensor t_values = VectorToTensor(values);

  Tensor t_coo = t_values.ToCoo(0.01);
  Tensor t_dense = t_coo.ToDense();

  std::vector<float> real = TensorToVector<float>(t_dense);

  AssertVectorF32(expect, real);
}

TEST(Math, LtKeep) {
  std::vector<float> values{-1.0, 1.0,   2.0,     3.0,      -4.0,
                            -0.1, 0.001, 0.00001, 0.000001, 128};
  std::vector<float> expect{0, 0, 0, 0, 0, 0, 0.001, 0.00001, 0.000001, 0};

  Tensor t_values = VectorToTensor(values);

  std::vector<float> real = TensorToVector<float>(t_values.LtKeep(0.01));

  AssertVectorF32(expect, real);
}

}  // namespace test
}  // namespace kraken
