#include "common/consistent_hasher.h"

#include <gtest/gtest.h>

namespace kraken {
namespace test {

TEST(ConsistentHasher, Funcs) {
  {
    ConsistentHasher hasher(1);

    auto boundary = hasher.Boundary(0);

    EXPECT_EQ(std::get<0>(boundary), 0);
    EXPECT_EQ(std::get<1>(boundary), std::numeric_limits<uint64_t>::max());
    EXPECT_EQ(hasher.Hit(0), 0);
    EXPECT_EQ(hasher.Hit(std::numeric_limits<uint64_t>::max()), 0);
  }

  {
    ConsistentHasher hasher(2);

    auto b0 = hasher.Boundary(0);
    auto b1 = hasher.Boundary(1);

    uint64_t max_v = std::numeric_limits<uint64_t>::max();
    uint64_t stride = max_v / 2;

    if ((max_v - 2 + 1) % 2 == 0) {
      stride++;
    }

    EXPECT_EQ(std::get<0>(b0), 0);
    EXPECT_EQ(std::get<1>(b0), stride - 1);
    EXPECT_EQ(std::get<0>(b1), stride);
    EXPECT_EQ(std::get<1>(b1), max_v);

    EXPECT_EQ(hasher.Hit(0), 0);
    EXPECT_EQ(hasher.Hit(stride - 1), 0);
    EXPECT_EQ(hasher.Hit(stride), 1);
    EXPECT_EQ(hasher.Hit(max_v), 1);
  }
}

}  // namespace test
}  // namespace kraken
