#pragma once

#include <cinttypes>

namespace kraken {

struct RPCFuncType {
  static constexpr uint32_t kHeartbeatType = 0;
  static constexpr uint32_t kTryJoinType = 1;
  static constexpr uint32_t kNotifyNodeJoinType = 2;
  static constexpr uint32_t kNotifyRouterChangeType = 3;
  static constexpr uint32_t kInitModelType = 4;
  static constexpr uint32_t kRegisterDenseTableType = 5;
  static constexpr uint32_t kRegisterSparseTableType = 6;
  static constexpr uint32_t kCreateModelType = 7;
  static constexpr uint32_t kCreateDenseTableType = 8;
  static constexpr uint32_t kCreateSparseTableType = 9;
  static constexpr uint32_t kFetchModelMetaDataType = 10;
  static constexpr uint32_t kNotifyFinishTransferType = 11;
  static constexpr uint32_t kFetchRouterType = 12;
  static constexpr uint32_t kTransferDenseTableType = 13;
  static constexpr uint32_t kTransferSparseMetaDataType = 14;
  static constexpr uint32_t kTransferSparseValuesType = 15;
  static constexpr uint32_t kTryFetchDenseTableType = 16;
  static constexpr uint32_t kTryCombineFetchDenseTableType = 17;
  static constexpr uint32_t kTryFetchSparseMetaDataType = 18;
  static constexpr uint32_t kTryFetchSparseValuesType = 19;
  static constexpr uint32_t kPullDenseTableType = 20;
  static constexpr uint32_t kCombinePullDenseTableType = 21;
  static constexpr uint32_t kPushDenseTableType = 22;
  static constexpr uint32_t kPullSparseTableType = 23;
  static constexpr uint32_t kPushSparseTableType = 24;
};

}  // namespace kraken
