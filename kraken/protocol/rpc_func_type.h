#pragma once

#include <cinttypes>

namespace kraken {

struct RPCFuncType {
  static constexpr uint32_t kApplyModelType = 0;
  static constexpr uint32_t kApplyTableType = 1;
  static constexpr uint32_t kRegisterModelType = 2;
  static constexpr uint32_t kRegisterDenseTableType = 3;
  static constexpr uint32_t kRegisterSparseTableType = 4;
  static constexpr uint32_t kRegisterSparseTableV2Type = 5;
  static constexpr uint32_t kPushDenseTableType = 6;
  static constexpr uint32_t kPullDenseTableType = 7;
  static constexpr uint32_t kPullListDenseTableType = 8;
  static constexpr uint32_t kPushPullDenseTableType = 9;
  static constexpr uint32_t kPushSparseTableType = 10;
  static constexpr uint32_t kPullSparseTableType = 11;
  static constexpr uint32_t kCombinePullSparseTableType = 12;
};

}  // namespace kraken
