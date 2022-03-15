#include "worker/emitter.h"

#include <gtest/gtest.h>

#include "common/utils.h"
#include "test/utils_test.h"

namespace kraken {
namespace test {

TEST(Emitter, Test) {
  std::unique_ptr<Emitter> emitter(new Emitter());
  emitter->Initialize("127.0.0.1:50000");

  emitter->InitModel("Emitter.Test", OptimType::kSGD, {});

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

    // Wait push finish.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    Tensor r0 = emitter->PullDenseTable(0);

    AssertTensorFloatEQ(r0, d0 - lr * g0);
  }

  {
    Tensor indices = VectorToTensor<uint64_t>(std::vector<uint64_t>({0, 1}));

    Tensor real = emitter->PullSparseTable(2, indices);

    AssertTensorFloatEQ(
        real, Tensor::Dense({2, 100}, ElementType::From<float>()).Constant(v0));
  }

  {
    Tensor indices = VectorToTensor<uint64_t>(std::vector<uint64_t>({0, 1}));
    Tensor grad = RandomTensor<float>(Shape({2, 100}));

    float lr = utils::ThreadLocalRandom<float>(0.1, 1.0);

    emitter->UpdateLR(lr);
    emitter->PushSparseTable(2, indices, grad);

    // Wait push finish.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    Tensor real = emitter->PullSparseTable(2, indices);

    AssertTensorFloatEQ(
        real, Tensor::Dense({2, 100}, ElementType::From<float>()).Constant(v0) -
                  lr * grad);
  }

  emitter->Stop();
}

}  // namespace test
}  // namespace kraken
