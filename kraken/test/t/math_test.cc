#include <gtest/gtest.h>

#include <cinttypes>
#include <vector>

#include "common/utils.h"
#include "test/utils_test.h"

namespace kraken {
namespace test {

TEST(Math, Add) {
  {
    int64_t c = utils::ThreadLocalRandom<int64_t>(1, 1000);

    std::vector<float> v0;
    std::vector<float> v1;
    std::vector<float> expect;

    for (int64_t i = 0; i < c; ++i) {
      v0.emplace_back(utils::ThreadLocalRandom<float>(-1000, 1000));
      v1.emplace_back(utils::ThreadLocalRandom<float>(-1000, 1000));

      expect.emplace_back(v0.back() + v1.back());
    }

    Tensor t0 = VectorToTensor(v0);
    Tensor t1 = VectorToTensor(v1);
    Tensor t2 = t0 + t1;

    std::vector<float> real = TensorToVector<float>(t2);

    AssertVectorF32(expect, real);
  }

  {
    int64_t c = utils::ThreadLocalRandom<int64_t>(1, 1000);

    float a = utils::ThreadLocalRandom<float>(-1000, 1000);

    std::vector<float> v0;
    std::vector<float> expect;

    for (int64_t i = 0; i < c; ++i) {
      v0.emplace_back(utils::ThreadLocalRandom<float>(-1000, 1000));

      expect.emplace_back(v0.back() + a);
    }

    Tensor t0 = VectorToTensor(v0);

    std::vector<float> real = TensorToVector<float>(t0 + a);

    AssertVectorF32(expect, real);
  }
}

TEST(Math, Sub) {
  {
    int64_t c = utils::ThreadLocalRandom<int64_t>(1, 1000);

    std::vector<float> v0;
    std::vector<float> v1;
    std::vector<float> expect;

    for (int64_t i = 0; i < c; ++i) {
      v0.emplace_back(utils::ThreadLocalRandom<float>(-1000, 1000));
      v1.emplace_back(utils::ThreadLocalRandom<float>(-1000, 1000));

      expect.emplace_back(v0.back() - v1.back());
    }

    Tensor t0 = VectorToTensor(v0);
    Tensor t1 = VectorToTensor(v1);
    Tensor t2 = t0 - t1;

    std::vector<float> real = TensorToVector<float>(t2);

    AssertVectorF32(expect, real);
  }

  {
    int64_t c = utils::ThreadLocalRandom<int64_t>(1, 1000);

    float a = utils::ThreadLocalRandom<float>(-1000, 1000);

    std::vector<float> v0;
    std::vector<float> expect;

    for (int64_t i = 0; i < c; ++i) {
      v0.emplace_back(utils::ThreadLocalRandom<float>(-1000, 1000));

      expect.emplace_back(v0.back() - a);
    }

    Tensor t0 = VectorToTensor(v0);

    std::vector<float> real = TensorToVector<float>(t0 - a);

    AssertVectorF32(expect, real);
  }

  {
    int64_t c = utils::ThreadLocalRandom<int64_t>(1, 1000);

    float a = utils::ThreadLocalRandom<float>(-1000, 1000);

    std::vector<float> v0;
    std::vector<float> expect;

    for (int64_t i = 0; i < c; ++i) {
      v0.emplace_back(utils::ThreadLocalRandom<float>(-1000, 1000));

      expect.emplace_back(a - v0.back());
    }

    Tensor t0 = VectorToTensor(v0);

    std::vector<float> real = TensorToVector<float>(a - t0);

    AssertVectorF32(expect, real);
  }
}

TEST(Math, Mul) {
  {
    int64_t c = utils::ThreadLocalRandom<int64_t>(1, 1000);

    std::vector<float> v0;
    std::vector<float> v1;
    std::vector<float> expect;

    for (int64_t i = 0; i < c; ++i) {
      v0.emplace_back(utils::ThreadLocalRandom<float>(-1000, 1000));
      v1.emplace_back(utils::ThreadLocalRandom<float>(-1000, 1000));

      expect.emplace_back(v0.back() * v1.back());
    }

    Tensor t0 = VectorToTensor(v0);
    Tensor t1 = VectorToTensor(v1);
    Tensor t2 = t0 * t1;

    std::vector<float> real = TensorToVector<float>(t2);

    AssertVectorF32(expect, real);
  }

  {
    int64_t c = utils::ThreadLocalRandom<int64_t>(1, 1000);

    float a = utils::ThreadLocalRandom<float>(-1000, 1000);

    std::vector<float> v0;
    std::vector<float> expect;

    for (int64_t i = 0; i < c; ++i) {
      v0.emplace_back(utils::ThreadLocalRandom<float>(-1000, 1000));

      expect.emplace_back(v0.back() * a);
    }

    Tensor t0 = VectorToTensor(v0);

    std::vector<float> real = TensorToVector<float>(t0 * a);

    AssertVectorF32(expect, real);
  }
}

TEST(Math, Div) {
  {
    int64_t c = utils::ThreadLocalRandom<int64_t>(1, 1000);

    std::vector<float> v0;
    std::vector<float> v1;
    std::vector<float> expect;

    for (int64_t i = 0; i < c; ++i) {
      v0.emplace_back(utils::ThreadLocalRandom<float>(1, 1000));
      v1.emplace_back(utils::ThreadLocalRandom<float>(1, 1000));

      expect.emplace_back(v0.back() / v1.back());
    }

    Tensor t0 = VectorToTensor(v0);
    Tensor t1 = VectorToTensor(v1);
    Tensor t2 = t0 / t1;

    std::vector<float> real = TensorToVector<float>(t2);

    AssertVectorF32(expect, real);
  }

  {
    int64_t c = utils::ThreadLocalRandom<int64_t>(1, 1000);

    float a = utils::ThreadLocalRandom<float>(1, 1000);

    std::vector<float> v0;
    std::vector<float> expect;

    for (int64_t i = 0; i < c; ++i) {
      v0.emplace_back(utils::ThreadLocalRandom<float>(1, 1000));

      expect.emplace_back(v0.back() / a);
    }

    Tensor t0 = VectorToTensor(v0);

    std::vector<float> real = TensorToVector<float>(t0 / a);

    AssertVectorF32(expect, real);
  }

  {
    int64_t c = utils::ThreadLocalRandom<int64_t>(1, 1000);

    float a = utils::ThreadLocalRandom<float>(1, 1000);

    std::vector<float> v0;
    std::vector<float> expect;

    for (int64_t i = 0; i < c; ++i) {
      v0.emplace_back(utils::ThreadLocalRandom<float>(1, 1000));

      expect.emplace_back(a / v0.back());
    }

    Tensor t0 = VectorToTensor(v0);

    std::vector<float> real = TensorToVector<float>(a / t0);

    AssertVectorF32(expect, real);
  }
}

TEST(Math, Constant) {
  int64_t c = utils::ThreadLocalRandom<int64_t>(1, 1000);

  std::vector<float> v0(c, 0);
  Tensor t0 = VectorToTensor(v0);

  float v = utils::ThreadLocalRandom<float>(-1000, 1000);

  t0.Constant(v);

  std::vector<float> real = TensorToVector<float>(t0);

  for (auto i : real) {
    EXPECT_FLOAT_EQ(v, i);
  }
}

TEST(Math, Sqrt) {
  int64_t c = utils::ThreadLocalRandom<int64_t>(1, 1000);

  std::vector<float> v0;
  std::vector<float> expect;

  for (int64_t i = 0; i < c; ++i) {
    v0.emplace_back(utils::ThreadLocalRandom<float>(0, 1000));

    expect.emplace_back(std::sqrt(v0.back()));
  }

  Tensor t0 = VectorToTensor(v0);

  std::vector<float> real = TensorToVector<float>(t0.Sqrt());

  AssertVectorF32(expect, real);
}

TEST(Math, Max) {
  int64_t c = utils::ThreadLocalRandom<int64_t>(1, 1000);

  std::vector<float> v0;
  std::vector<float> v1;
  std::vector<float> expect;

  for (int64_t i = 0; i < c; ++i) {
    v0.emplace_back(utils::ThreadLocalRandom<float>(-1000, 1000));
    v1.emplace_back(utils::ThreadLocalRandom<float>(-1000, 1000));

    expect.emplace_back(std::max(v0.back(), v1.back()));
  }

  Tensor t0 = VectorToTensor(v0);
  Tensor t1 = VectorToTensor(v1);
  Tensor t2 = t0.Max(t1);

  std::vector<float> real = TensorToVector<float>(t2);

  AssertVectorF32(expect, real);
}

TEST(Math, ConcatVector) {
  int64_t row = utils::ThreadLocalRandom<int64_t>(1, 1000);
  int64_t col = utils::ThreadLocalRandom<int64_t>(1, 1000);

  std::vector<Tensor> vecs;
  std::vector<float> expect;

  for (int64_t r = 0; r < row; ++r) {
    std::vector<float> vals;

    for (int64_t c = 0; c < col; ++c) {
      vals.emplace_back(utils::ThreadLocalRandom<float>(-1000, 1000));

      expect.emplace_back(vals.back());
    }

    vecs.emplace_back(VectorToTensor(vals));
  }

  Tensor t = vecs[0].ConcatVector(vecs);

  std::vector<float> real = TensorToVector<float>(t);

  AssertVectorF32(expect, real);
}

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
