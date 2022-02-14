#include "worker/emitter.h"

#include <gtest/gtest.h>

#include "test/utils_test.h"

namespace kraken {
namespace test {

TEST(Emitter, CompressTypekNo) {
  std::unique_ptr<Emitter> emitter(new Emitter());
  emitter->Initialize("127.0.0.1:50000,127.0.0.1:50001", CompressType::kNo);

  EXPECT_EQ(0, emitter->RegisterModel("Emitter.CompressType.kNo",
                                      OptimType::kSGD, {}));

  Tensor d0 = RandomTensor<float>(Shape({10, 10}));
  Tensor d1 = RandomTensor<float>(Shape({10, 10}));
  Tensor g0 = RandomTensor<float>(Shape({10, 10}));
  Tensor g1 = RandomTensor<float>(Shape({10, 10}));

  float v0 = utils::ThreadLocalRandom<float>(-1000, 1000);
  float v1 = utils::ThreadLocalRandom<float>(-1000, 1000);

  EXPECT_EQ(0, emitter->RegisterDenseTable("DensTable0", d0));
  EXPECT_EQ(1, emitter->RegisterDenseTable("DensTable1", d1));
  EXPECT_EQ(2, emitter->RegisterSparseTable("SparseTable0", 100,
                                            ElementType::From<float>(),
                                            InitializerType::kConstant,
                                            {{"value", std::to_string(v0)}}));
  EXPECT_EQ(3, emitter->RegisterSparseTable("SparseTable1", 100,
                                            ElementType::From<float>(),
                                            InitializerType::kConstant,
                                            {{"value", std::to_string(v1)}}));

  {
    Tensor r0 = emitter->PullDenseTable(0);
    Tensor r1 = emitter->PullDenseTable(1);

    AssertTensorEQ(d0, r0);
    AssertTensorEQ(d1, r1);
  }

  {
    std::vector<Tensor> rs = emitter->CombinePullDenseTable({0, 1});

    AssertTensorEQ(d0, rs[0]);
    AssertTensorEQ(d1, rs[1]);
  }

  {
    float lr = utils::ThreadLocalRandom<float>(0.1, 1.0);

    emitter->UpdateLR(lr);
    emitter->PushDenseTable(0, g0);
    Tensor r0 = emitter->PullDenseTable(0);

    AssertTensorFloatEQ(r0, d0 - lr * g0);
  }

  {
    float lr = utils::ThreadLocalRandom<float>(0.1, 1.0);

    emitter->UpdateLR(lr);
    Tensor r1 = emitter->PushPullDenseTable(1, g1);

    AssertTensorFloatEQ(r1, (d1 - lr * g1));
  }

  {
    Tensor indices = VectorToTensor<int64_t>(std::vector<int64_t>({0, 1}));

    Tensor real = emitter->PullSparseTable(2, indices);

    AssertTensorFloatEQ(
        real, Tensor::Dense({2, 100}, ElementType::From<float>()).Constant(v0));
  }

  {
    Tensor indices0 = VectorToTensor<int64_t>(std::vector<int64_t>({0, 1}));
    Tensor indices1 = VectorToTensor<int64_t>(std::vector<int64_t>({0, 1}));

    std::vector<Tensor> reals =
        emitter->CombinePullSparseTable({2, 3}, {indices0, indices1});

    AssertTensorFloatEQ(
        reals[0],
        Tensor::Dense({2, 100}, ElementType::From<float>()).Constant(v0));
    AssertTensorFloatEQ(
        reals[1],
        Tensor::Dense({2, 100}, ElementType::From<float>()).Constant(v1));
  }

  {
    Tensor indices = VectorToTensor<int64_t>(std::vector<int64_t>({0, 1}));
    Tensor grad = RandomTensor<float>(Shape({2, 100}));

    float lr = utils::ThreadLocalRandom<float>(0.1, 1.0);

    emitter->UpdateLR(lr);
    emitter->PushSparseTable(2, indices, grad);
    Tensor real = emitter->PullSparseTable(2, indices);

    AssertTensorFloatEQ(
        real, Tensor::Dense({2, 100}, ElementType::From<float>()).Constant(v0) -
                  lr * grad);
  }

  emitter->Stop();
}

TEST(Emitter, CompressTypekSnappy) {
  std::unique_ptr<Emitter> emitter(new Emitter());
  emitter->Initialize("127.0.0.1:50000,127.0.0.1:50001", CompressType::kSnappy);

  EXPECT_EQ(1, emitter->RegisterModel("Emitter.CompressType.kSnappy",
                                      OptimType::kSGD, {}));

  Tensor d0 = RandomTensor<float>(Shape({10, 10}));
  Tensor d1 = RandomTensor<float>(Shape({10, 10}));
  Tensor g0 = RandomTensor<float>(Shape({10, 10}));
  Tensor g1 = RandomTensor<float>(Shape({10, 10}));

  float v0 = utils::ThreadLocalRandom<float>(-1000, 1000);
  float v1 = utils::ThreadLocalRandom<float>(-1000, 1000);

  EXPECT_EQ(0, emitter->RegisterDenseTable("DensTable0", d0));
  EXPECT_EQ(1, emitter->RegisterDenseTable("DensTable1", d1));
  EXPECT_EQ(2, emitter->RegisterSparseTable("SparseTable0", 100,
                                            ElementType::From<float>(),
                                            InitializerType::kConstant,
                                            {{"value", std::to_string(v0)}}));
  EXPECT_EQ(3, emitter->RegisterSparseTable("SparseTable1", 100,
                                            ElementType::From<float>(),
                                            InitializerType::kConstant,
                                            {{"value", std::to_string(v1)}}));

  {
    Tensor r0 = emitter->PullDenseTable(0);
    Tensor r1 = emitter->PullDenseTable(1);

    AssertTensorEQ(d0, r0);
    AssertTensorEQ(d1, r1);
  }

  {
    std::vector<Tensor> rs = emitter->CombinePullDenseTable({0, 1});

    AssertTensorEQ(d0, rs[0]);
    AssertTensorEQ(d1, rs[1]);
  }

  {
    float lr = utils::ThreadLocalRandom<float>(0.1, 1.0);

    emitter->UpdateLR(lr);
    emitter->PushDenseTable(0, g0);
    Tensor r0 = emitter->PullDenseTable(0);

    AssertTensorFloatEQ(r0, d0 - lr * g0);
  }

  {
    float lr = utils::ThreadLocalRandom<float>(0.1, 1.0);

    emitter->UpdateLR(lr);
    Tensor r1 = emitter->PushPullDenseTable(1, g1);

    AssertTensorFloatEQ(r1, (d1 - lr * g1));
  }

  {
    Tensor indices = VectorToTensor<int64_t>(std::vector<int64_t>({0, 1}));

    Tensor real = emitter->PullSparseTable(2, indices);

    AssertTensorFloatEQ(
        real, Tensor::Dense({2, 100}, ElementType::From<float>()).Constant(v0));
  }

  {
    Tensor indices0 = VectorToTensor<int64_t>(std::vector<int64_t>({0, 1}));
    Tensor indices1 = VectorToTensor<int64_t>(std::vector<int64_t>({0, 1}));

    std::vector<Tensor> reals =
        emitter->CombinePullSparseTable({2, 3}, {indices0, indices1});

    AssertTensorFloatEQ(
        reals[0],
        Tensor::Dense({2, 100}, ElementType::From<float>()).Constant(v0));
    AssertTensorFloatEQ(
        reals[1],
        Tensor::Dense({2, 100}, ElementType::From<float>()).Constant(v1));
  }

  {
    Tensor indices = VectorToTensor<int64_t>(std::vector<int64_t>({0, 1}));
    Tensor grad = RandomTensor<float>(Shape({2, 100}));

    float lr = utils::ThreadLocalRandom<float>(0.1, 1.0);

    emitter->UpdateLR(lr);
    emitter->PushSparseTable(2, indices, grad);
    Tensor real = emitter->PullSparseTable(2, indices);

    AssertTensorFloatEQ(
        real, Tensor::Dense({2, 100}, ElementType::From<float>()).Constant(v0) -
                  lr * grad);
  }

  emitter->Stop();
}

}  // namespace test
}  // namespace kraken
