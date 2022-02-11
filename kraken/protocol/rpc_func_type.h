#pragma once

#include <cinttypes>

namespace kraken {

struct RPCFuncType {
  static constexpr uint32_t kApplyModelType = 0;
  static constexpr uint32_t kApplyDenseTableType = 1;
  static constexpr uint32_t kApplySparseTableType = 2;
  static constexpr uint32_t kRegisterModelType = 3;
  static constexpr uint32_t kRegisterDenseTableInfoType = 4;
  static constexpr uint32_t kRegisterDenseTableType = 5;
  static constexpr uint32_t kRegisterSparseTableType = 6;
  static constexpr uint32_t kPullDenseTableType = 7;
  static constexpr uint32_t kCombinePullDenseTableType = 8;
  static constexpr uint32_t kPushPullDenseTableType = 9;
  static constexpr uint32_t kPushDenseTableType = 10;
  static constexpr uint32_t kPullSparseTableType = 11;
  static constexpr uint32_t kCombinePullSparseTableType = 12;
  static constexpr uint32_t kPushSparseTableType = 13;
  static constexpr uint32_t kSaveCheckPointType = 14;
};

}  // namespace kraken
