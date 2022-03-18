#pragma once

#include <cinttypes>

namespace kraken {

struct RPCFuncType {
  static constexpr uint32_t kHeartbeatType = 0;
  static constexpr uint32_t kTryJoinType = 1;
  static constexpr uint32_t kNotifyNodeJoinType = 2;
  static constexpr uint32_t kInitModelType = 3;
  static constexpr uint32_t kRegisterDenseTableType = 4;
  static constexpr uint32_t kRegisterSparseTableType = 5;
  static constexpr uint32_t kCreateModelType = 6;
  static constexpr uint32_t kCreateDenseTableType = 7;
  static constexpr uint32_t kCreateSparseTableType = 8;
  static constexpr uint32_t kFetchModelMetaDataType = 9;
  static constexpr uint32_t kNotifyFinishTransferType = 10;
  static constexpr uint32_t kFetchRouterType = 11;
  static constexpr uint32_t kTransferDenseTableType = 12;
  static constexpr uint32_t kTransferSparseMetaDataType = 13;
  static constexpr uint32_t kTransferSparseValuesType = 14;
  static constexpr uint32_t kTryFetchDenseTableType = 15;
  static constexpr uint32_t kTryCombineFetchDenseTableType = 16;
  static constexpr uint32_t kTryFetchSparseMetaDataType = 17;
  static constexpr uint32_t kTryFetchSparseValuesType = 18;
  static constexpr uint32_t kPullDenseTableType = 19;
  static constexpr uint32_t kCombinePullDenseTableType = 20;
  static constexpr uint32_t kPushDenseTableType = 21;
  static constexpr uint32_t kPullSparseTableType = 22;
  static constexpr uint32_t kCombinePullSparseTableType = 23;
  static constexpr uint32_t kPushSparseTableType = 24;
  static constexpr uint32_t kCombinePushSparseTableType = 25;
};

}  // namespace kraken
