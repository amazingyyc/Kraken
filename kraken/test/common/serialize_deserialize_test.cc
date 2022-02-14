#include "common/serialize.h"

#include <gtest/gtest.h>

#include "common/deserialize.h"
#include "common/mem_buffer.h"
#include "common/mem_reader.h"
#include "test/utils_test.h"

namespace kraken {
namespace test {

TEST(SerializeDeserialize, BaseType) {
#define BASE_TYPE_TEST(T, v) \
  { \
    MemBuffer mem_buf; \
    T expect = v; \
    Serialize serialize(&mem_buf); \
    EXPECT_TRUE(serialize << expect); \
    T val; \
    MemReader mem_reader(mem_buf.ptr(), mem_buf.offset()); \
    Deserialize deserialize(&mem_reader); \
    EXPECT_TRUE(deserialize >> val); \
    EXPECT_TRUE(expect == val); \
  }

  BASE_TYPE_TEST(bool, true);
  BASE_TYPE_TEST(bool, false);
  BASE_TYPE_TEST(uint8_t, 0);
  BASE_TYPE_TEST(uint8_t, 101);
  BASE_TYPE_TEST(uint8_t, 255);
  BASE_TYPE_TEST(int8_t, -128);
  BASE_TYPE_TEST(int8_t, 0);
  BASE_TYPE_TEST(int8_t, 127);

  BASE_TYPE_TEST(uint16_t, std::numeric_limits<uint16_t>::lowest());
  BASE_TYPE_TEST(uint16_t, std::numeric_limits<uint16_t>::min());
  BASE_TYPE_TEST(uint16_t, std::numeric_limits<uint16_t>::max());
  BASE_TYPE_TEST(uint16_t, 101);

  BASE_TYPE_TEST(int16_t, std::numeric_limits<int16_t>::lowest());
  BASE_TYPE_TEST(int16_t, std::numeric_limits<int16_t>::min());
  BASE_TYPE_TEST(int16_t, std::numeric_limits<int16_t>::max());
  BASE_TYPE_TEST(int16_t, 101);
  BASE_TYPE_TEST(int16_t, -101);

  BASE_TYPE_TEST(uint32_t, std::numeric_limits<uint32_t>::lowest());
  BASE_TYPE_TEST(uint32_t, std::numeric_limits<uint32_t>::min());
  BASE_TYPE_TEST(uint32_t, std::numeric_limits<uint32_t>::max());
  BASE_TYPE_TEST(uint32_t, 101);

  BASE_TYPE_TEST(int32_t, std::numeric_limits<int32_t>::lowest());
  BASE_TYPE_TEST(int32_t, std::numeric_limits<int32_t>::min());
  BASE_TYPE_TEST(int32_t, std::numeric_limits<int32_t>::max());
  BASE_TYPE_TEST(int32_t, 101);
  BASE_TYPE_TEST(int32_t, -101);

  BASE_TYPE_TEST(uint64_t, std::numeric_limits<uint64_t>::lowest());
  BASE_TYPE_TEST(uint64_t, std::numeric_limits<uint64_t>::min());
  BASE_TYPE_TEST(uint64_t, std::numeric_limits<uint64_t>::max());
  BASE_TYPE_TEST(uint64_t, 101);

  BASE_TYPE_TEST(int64_t, std::numeric_limits<int64_t>::lowest());
  BASE_TYPE_TEST(int64_t, std::numeric_limits<int64_t>::min());
  BASE_TYPE_TEST(int64_t, std::numeric_limits<int64_t>::max());
  BASE_TYPE_TEST(int64_t, 101);
  BASE_TYPE_TEST(int64_t, -101);

  BASE_TYPE_TEST(float, std::numeric_limits<float>::lowest());
  BASE_TYPE_TEST(float, std::numeric_limits<float>::min());
  BASE_TYPE_TEST(float, std::numeric_limits<float>::max());
  BASE_TYPE_TEST(float, 101.1);
  BASE_TYPE_TEST(float, -101.2);

  BASE_TYPE_TEST(double, std::numeric_limits<double>::lowest());
  BASE_TYPE_TEST(double, std::numeric_limits<double>::min());
  BASE_TYPE_TEST(double, std::numeric_limits<double>::max());
  BASE_TYPE_TEST(double, 101.1);
  BASE_TYPE_TEST(double, -101.2);
}

TEST(SerializeDeserializeTest, VecBaseType) {
#define VEC_BASE_TYPE_TEST(T, v) \
  { \
    MemBuffer mem_buf; \
    std::vector<T> expect = std::vector<T> v; \
    Serialize serialize(&mem_buf); \
    EXPECT_TRUE(serialize << expect); \
    std::vector<T> val; \
    MemReader mem_reader(mem_buf.ptr(), mem_buf.offset()); \
    Deserialize deserialize(&mem_reader); \
    EXPECT_TRUE(deserialize >> val); \
    EXPECT_TRUE(expect.size() == val.size()); \
    for (size_t i = 0; i < expect.size(); ++i) { \
      EXPECT_TRUE(expect[i] == val[i]); \
    } \
  }

  VEC_BASE_TYPE_TEST(uint8_t, ({1, 2, 3, 4}));
  VEC_BASE_TYPE_TEST(int8_t, ({-1, -2, 3, 4}));
  VEC_BASE_TYPE_TEST(uint16_t, ({1, 2, 3, 4}));
  VEC_BASE_TYPE_TEST(int16_t, ({-10, -20, 3, 4, 100, 200}));
  VEC_BASE_TYPE_TEST(uint32_t, ({0, 1, 2, 3, 4, 909}));
  VEC_BASE_TYPE_TEST(int32_t, ({-101, -2, 3, 4, 45, 599, -100}));
  VEC_BASE_TYPE_TEST(uint64_t, ({1, 0, 2, 3, 4, 990, 87, 45}));
  VEC_BASE_TYPE_TEST(int64_t, ({-1, -2, -90, -57, 3, 467}));
  VEC_BASE_TYPE_TEST(float, ({-1940.09, 1.0, 2.0, 3, 4, 689.0}));
  VEC_BASE_TYPE_TEST(double, ({-1940.09, 1.0, 2.0, 3, 4, 689.0}));
}

TEST(SerializeDeserializeTest, String) {
  {
    MemBuffer mem_buf;

    std::string expect = "";

    Serialize serialize(&mem_buf);
    EXPECT_TRUE(serialize << expect);

    std::string val;

    MemReader mem_reader(mem_buf.ptr(), mem_buf.offset());
    Deserialize deserialize(&mem_reader);
    EXPECT_TRUE(deserialize >> val);

    EXPECT_EQ(expect, val);
  }

  {
    MemBuffer mem_buf;

    std::string expect = "This a string.";

    Serialize serialize(&mem_buf);
    EXPECT_TRUE(serialize << expect);

    std::string val;

    MemReader mem_reader(mem_buf.ptr(), mem_buf.offset());
    Deserialize deserialize(&mem_reader);
    EXPECT_TRUE(deserialize >> val);

    EXPECT_EQ(expect, val);
  }

  {
    MemBuffer mem_buf;

    std::string expect = "中文";

    Serialize serialize(&mem_buf);
    EXPECT_TRUE(serialize << expect);

    std::string val;

    MemReader mem_reader(mem_buf.ptr(), mem_buf.offset());
    Deserialize deserialize(&mem_reader);
    EXPECT_TRUE(deserialize >> val);

    EXPECT_EQ(expect, val);
  }
}

TEST(SerializeDeserializeTest, NotBaseType) {
  {
    MemBuffer mem_buf;

    Serialize serialize(&mem_buf);
    EXPECT_TRUE(serialize << CompressType::kNo);
    EXPECT_TRUE(serialize << CompressType::kSnappy);

    CompressType v0;
    CompressType v1;

    MemReader mem_reader(mem_buf.ptr(), mem_buf.offset());
    Deserialize deserialize(&mem_reader);
    EXPECT_TRUE(deserialize >> v0);
    EXPECT_TRUE(deserialize >> v1);

    EXPECT_EQ(v0, CompressType::kNo);
    EXPECT_EQ(v1, CompressType::kSnappy);
  }

  {
    RequestHeader expect;
    expect.timestamp = utils::ThreadLocalRandom<uint64_t>(0, 1000);
    expect.type = utils::ThreadLocalRandom<uint32_t>(0, 10000);
    expect.compress_type = CompressType::kSnappy;

    RequestHeader val;

    MemBuffer mem_buf;

    Serialize serialize(&mem_buf);
    EXPECT_TRUE(serialize << expect);

    MemReader mem_reader(mem_buf.ptr(), mem_buf.offset());
    Deserialize deserialize(&mem_reader);
    EXPECT_TRUE(deserialize >> val);

    EXPECT_EQ(expect.timestamp, val.timestamp);
    EXPECT_EQ(expect.type, val.type);
    EXPECT_EQ(expect.compress_type, val.compress_type);
  }

  {
    ReplyHeader expect;
    expect.timestamp = utils::ThreadLocalRandom<uint64_t>(0, 1000);
    expect.error_code = utils::ThreadLocalRandom<int32_t>(-1000, 10000);
    expect.compress_type = CompressType::kSnappy;

    ReplyHeader val;

    MemBuffer mem_buf;

    Serialize serialize(&mem_buf);
    EXPECT_TRUE(serialize << expect);

    MemReader mem_reader(mem_buf.ptr(), mem_buf.offset());
    Deserialize deserialize(&mem_reader);
    EXPECT_TRUE(deserialize >> val);

    EXPECT_EQ(expect.timestamp, val.timestamp);
    EXPECT_EQ(expect.error_code, val.error_code);
    EXPECT_EQ(expect.compress_type, val.compress_type);
  }

  {
    MemBuffer mem_buf;

    Serialize serialize(&mem_buf);
    EXPECT_TRUE(serialize << InitializerType::kConstant);
    EXPECT_TRUE(serialize << InitializerType::kUniform);
    EXPECT_TRUE(serialize << InitializerType::kNormal);
    EXPECT_TRUE(serialize << InitializerType::kXavierUniform);
    EXPECT_TRUE(serialize << InitializerType::kXavierNormal);

    InitializerType v0;
    InitializerType v1;
    InitializerType v2;
    InitializerType v3;
    InitializerType v4;

    MemReader mem_reader(mem_buf.ptr(), mem_buf.offset());
    Deserialize deserialize(&mem_reader);
    EXPECT_TRUE(deserialize >> v0);
    EXPECT_TRUE(deserialize >> v1);
    EXPECT_TRUE(deserialize >> v2);
    EXPECT_TRUE(deserialize >> v3);
    EXPECT_TRUE(deserialize >> v4);

    EXPECT_EQ(v0, InitializerType::kConstant);
    EXPECT_EQ(v1, InitializerType::kUniform);
    EXPECT_EQ(v2, InitializerType::kNormal);
    EXPECT_EQ(v3, InitializerType::kXavierUniform);
    EXPECT_EQ(v4, InitializerType::kXavierNormal);
  }

  {
    MemBuffer mem_buf;

    Serialize serialize(&mem_buf);
    EXPECT_TRUE(serialize << OptimType::kAdagrad);
    EXPECT_TRUE(serialize << OptimType::kAdam);
    EXPECT_TRUE(serialize << OptimType::kRMSprop);
    EXPECT_TRUE(serialize << OptimType::kSGD);

    OptimType v0;
    OptimType v1;
    OptimType v2;
    OptimType v3;

    MemReader mem_reader(mem_buf.ptr(), mem_buf.offset());
    Deserialize deserialize(&mem_reader);
    EXPECT_TRUE(deserialize >> v0);
    EXPECT_TRUE(deserialize >> v1);
    EXPECT_TRUE(deserialize >> v2);
    EXPECT_TRUE(deserialize >> v3);

    EXPECT_EQ(v0, OptimType::kAdagrad);
    EXPECT_EQ(v1, OptimType::kAdam);
    EXPECT_EQ(v2, OptimType::kRMSprop);
    EXPECT_EQ(v3, OptimType::kSGD);
  }

  {
    MemBuffer mem_buf;

    Serialize serialize(&mem_buf);
    EXPECT_TRUE(serialize << StateType::kSteps);
    EXPECT_TRUE(serialize << StateType::kMomentumBuffer);
    EXPECT_TRUE(serialize << StateType::kStateSum);
    EXPECT_TRUE(serialize << StateType::kFirstMoment);
    EXPECT_TRUE(serialize << StateType::kSecondMoment);
    EXPECT_TRUE(serialize << StateType::kSecondMomentMax);
    EXPECT_TRUE(serialize << StateType::kSquareAverage);
    EXPECT_TRUE(serialize << StateType::kGAve);

    StateType v0;
    StateType v1;
    StateType v2;
    StateType v3;
    StateType v4;
    StateType v5;
    StateType v6;
    StateType v7;

    MemReader mem_reader(mem_buf.ptr(), mem_buf.offset());
    Deserialize deserialize(&mem_reader);
    EXPECT_TRUE(deserialize >> v0);
    EXPECT_TRUE(deserialize >> v1);
    EXPECT_TRUE(deserialize >> v2);
    EXPECT_TRUE(deserialize >> v3);
    EXPECT_TRUE(deserialize >> v4);
    EXPECT_TRUE(deserialize >> v5);
    EXPECT_TRUE(deserialize >> v6);
    EXPECT_TRUE(deserialize >> v7);

    EXPECT_EQ(v0, StateType::kSteps);
    EXPECT_EQ(v1, StateType::kMomentumBuffer);
    EXPECT_EQ(v2, StateType::kStateSum);
    EXPECT_EQ(v3, StateType::kFirstMoment);
    EXPECT_EQ(v4, StateType::kSecondMoment);
    EXPECT_EQ(v5, StateType::kSecondMomentMax);
    EXPECT_EQ(v6, StateType::kSquareAverage);
    EXPECT_EQ(v7, StateType::kGAve);
  }

  {
    MemBuffer mem_buf;

    Serialize serialize(&mem_buf);
    EXPECT_TRUE(serialize << TableType::kDense);
    EXPECT_TRUE(serialize << TableType::kSparse);

    TableType v0;
    TableType v1;

    MemReader mem_reader(mem_buf.ptr(), mem_buf.offset());
    Deserialize deserialize(&mem_reader);
    EXPECT_TRUE(deserialize >> v0);
    EXPECT_TRUE(deserialize >> v1);

    EXPECT_EQ(v0, TableType::kDense);
    EXPECT_EQ(v1, TableType::kSparse);
  }

  {
    for (uint8_t i = 0; i <= 12; ++i) {
      DType expect = (DType)i;
      DType val;

      MemBuffer mem_buf;

      Serialize serialize(&mem_buf);
      EXPECT_TRUE(serialize << expect);

      MemReader mem_reader(mem_buf.ptr(), mem_buf.offset());
      Deserialize deserialize(&mem_reader);
      EXPECT_TRUE(deserialize >> val);

      EXPECT_EQ(expect, val);
    }
  }

  {
    for (int i = 0; i < 10; ++i) {
      std::vector<int64_t> dims;

      for (size_t j = 0; j < 10; ++j) {
        dims.emplace_back(utils::ThreadLocalRandom<int64_t>(1, 12345));
      }

      Shape expect(dims);
      Shape val;

      MemBuffer mem_buf;

      Serialize serialize(&mem_buf);
      EXPECT_TRUE(serialize << expect);

      MemReader mem_reader(mem_buf.ptr(), mem_buf.offset());
      Deserialize deserialize(&mem_reader);
      EXPECT_TRUE(deserialize >> val);

      EXPECT_TRUE(expect == val);
    }
  }

  {
    MemBuffer mem_buf;

    Serialize serialize(&mem_buf);
    EXPECT_TRUE(serialize << Layout::kStride);
    EXPECT_TRUE(serialize << Layout::kCoo);

    Layout v0;
    Layout v1;

    MemReader mem_reader(mem_buf.ptr(), mem_buf.offset());
    Deserialize deserialize(&mem_reader);
    EXPECT_TRUE(deserialize >> v0);
    EXPECT_TRUE(deserialize >> v1);

    EXPECT_EQ(v0, Layout::kStride);
    EXPECT_EQ(v1, Layout::kCoo);
  }

  {
    // Tensor.
    size_t c = utils::ThreadLocalRandom<size_t>(1, 1000);

    Tensor expect;
    Tensor val;

    std::vector<float> vs;
    for (size_t i = 0; i < c; ++i) {
      vs.emplace_back(utils::ThreadLocalRandom<float>(-1000, 1000));
    }

    expect = VectorToTensor(vs);

    MemBuffer mem_buf;
    Serialize serialize(&mem_buf);
    EXPECT_TRUE(serialize << expect);

    MemReader mem_reader(mem_buf.ptr(), mem_buf.offset());
    Deserialize deserialize(&mem_reader);
    EXPECT_TRUE(deserialize >> val);

    EXPECT_EQ(expect.layout(), val.layout());
    EXPECT_EQ(expect.shape(), val.shape());
    EXPECT_EQ(expect.element_type(), val.element_type());

    AssertTensorEQ(expect, val);
  }

  {
    // COO tensor.
    size_t c = utils::ThreadLocalRandom<size_t>(1, 1000);

    Tensor dense;
    Tensor val;

    std::vector<float> vs;
    for (size_t i = 0; i < c; ++i) {
      vs.emplace_back(utils::ThreadLocalRandom<float>(-1000, 1000));
    }

    dense = VectorToTensor(vs);

    Tensor expect = dense.ToCoo(10);

    EXPECT_TRUE(expect.IsCoo());

    MemBuffer mem_buf;
    Serialize serialize(&mem_buf);
    EXPECT_TRUE(serialize << expect);

    MemReader mem_reader(mem_buf.ptr(), mem_buf.offset());
    Deserialize deserialize(&mem_reader);
    EXPECT_TRUE(deserialize >> val);

    EXPECT_TRUE(val.IsCoo());
    AssertTensorEQ(expect, val);
  }

  {
    std::unordered_map<std::string, std::string> expect = {
        {"123", "345"},
        {"abc", "cde"},
        {"@#$%", "*^&)"},
    };

    std::unordered_map<std::string, std::string> val;

    MemBuffer mem_buf;
    Serialize serialize(&mem_buf);
    EXPECT_TRUE(serialize << expect);

    MemReader mem_reader(mem_buf.ptr(), mem_buf.offset());
    Deserialize deserialize(&mem_reader);
    EXPECT_TRUE(deserialize >> val);

    EXPECT_EQ(expect.size(), val.size());

    for (const auto& [k, v] : expect) {
      EXPECT_TRUE(val.find(k) != val.end());
      EXPECT_EQ(val[k], v);
    }
  }
}

}  // namespace test
}  // namespace kraken
