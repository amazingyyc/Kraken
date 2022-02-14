#include "common/snappy.h"

#include <gtest/gtest.h>

#include <string>

#include "common/utils.h"
#include "snappy.h"

namespace kraken {
namespace test {

TEST(Snappy, CompressUnCompress) {
  size_t size = utils::ThreadLocalRandom<size_t>(1, 10000);

  std::string str;
  for (size_t i = 0; i < size; ++i) {
    str.push_back(utils::ThreadLocalRandom<char>(-128, 127));
  }

  SnappySource c_source(str.data(), str.size());
  SnappySink c_sink;
  EXPECT_TRUE(snappy::Compress(&c_source, &c_sink) > 0);

  SnappySource uc_source(c_sink.ptr(), c_sink.offset());
  SnappySink uc_sink;
  EXPECT_TRUE(snappy::Uncompress(&uc_source, &uc_sink));

  EXPECT_EQ(uc_sink.offset(), str.size());
  EXPECT_EQ(0, memcmp(uc_sink.ptr(), str.data(), str.size()));
}

}  // namespace test
}  // namespace kraken
