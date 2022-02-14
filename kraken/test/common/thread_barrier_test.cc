#include "common/thread_barrier.h"

#include <gtest/gtest.h>

#include <thread>

namespace kraken {
namespace test {

TEST(ThreadBarrier, Funcs) {
  uint32_t loop = 1000;

  std::vector<uint32_t> data(loop, 0);

  ThreadBarrier barrier(loop);

  std::vector<std::thread> threads;
  for (uint32_t i = 0; i < loop; ++i) {
    threads.emplace_back(std::thread([i, &data, &barrier]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));

      data[i] = i;
      barrier.Release();
    }));
  }

  barrier.Wait();

  for (uint32_t i = 0; i < loop; ++i) {
    EXPECT_EQ(data[i], i);
  }

  for (auto& t : threads) {
    t.join();
  }
}

}  // namespace test
}  // namespace kraken
