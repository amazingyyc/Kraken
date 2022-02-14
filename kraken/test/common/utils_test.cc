#include "common/utils.h"

#include <gtest/gtest.h>

#include <string>

namespace kraken {
namespace test {

TEST(utils, Split) {
  std::string str = ",a,b,c,d,,";
  std::string delim = ",";

  std::vector<std::string> tokens;

  utils::Split(str, delim, &tokens);

  EXPECT_EQ(tokens.size(), 7);
  EXPECT_EQ(tokens[0], "");
  EXPECT_EQ(tokens[1], "a");
  EXPECT_EQ(tokens[2], "b");
  EXPECT_EQ(tokens[3], "c");
  EXPECT_EQ(tokens[4], "d");
  EXPECT_EQ(tokens[5], "");
  EXPECT_EQ(tokens[6], "");
}

TEST(utils, ToLower) {
  auto lower = utils::ToLower("AbC");

  EXPECT_EQ(lower, "abc");
}

TEST(utils, EndWith) {
  EXPECT_TRUE(utils::EndWith("value ending", "ending"));
  EXPECT_FALSE(utils::EndWith("value endin", "ending"));
}

}  // namespace test
}  // namespace kraken
