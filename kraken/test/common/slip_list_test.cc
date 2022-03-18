#include <gtest/gtest.h>

#include <string>

#include "common/skip_list.h"
#include "common/snappy.h"
#include "common/utils.h"

namespace kraken {
namespace test {

TEST(SkipList, Test) {
  {
    SkipList<uint64_t, int> list;

    EXPECT_EQ(0, list.Size());

    list.Insert(1, 1);
    EXPECT_EQ(1, list.Size());

    list.Insert(2, 2);
    EXPECT_EQ(2, list.Size());

    list.Insert(3, 3);
    EXPECT_EQ(3, list.Size());

    EXPECT_TRUE(list.Contains(1));
    EXPECT_TRUE(list.Contains(2));
    EXPECT_TRUE(list.Contains(3));
    EXPECT_FALSE(list.Contains(4));
    EXPECT_FALSE(list.Contains(5));
    EXPECT_FALSE(list.Contains(6));

    list.Insert(6, 6);
    list.Insert(4, 4);
    list.Insert(5, 5);
    EXPECT_EQ(6, list.Size());

    EXPECT_FALSE(list.Insert(6, 6));
    EXPECT_EQ(6, list.Size());

    auto it = list.Begin();
    for (int i = 1; i <= 6; ++i) {
      EXPECT_TRUE(it.Valid());
      EXPECT_EQ(it.value(), i);

      it.Next();
    }
  }

  {
    SkipList<int, int> list;
    int size = utils::ThreadLocalRandom<int>(1, 1000);

    for (int i = 0; i < size; ++i) {
      list.Insert(i, i);
    }

    EXPECT_EQ(size, list.Size());

    auto it = list.Begin();
    while (it.Valid()) {
      if (it.key() % 2 != 0) {
        it = list.Remove(it);
      } else {
        it.Next();
      }
    }

    EXPECT_EQ(size / 2 + size % 2, list.Size());

    int expect = 0;
    it = list.Begin();
    while (it.Valid()) {
      EXPECT_EQ(expect, it.key());

      expect += 2;
      it.Next();
    }

    list.Clear();

    EXPECT_EQ(0, list.Size());
  }
}

}  // namespace test
}  // namespace kraken
