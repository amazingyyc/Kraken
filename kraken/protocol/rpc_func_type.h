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

  // static constexpr uint32_t kApplyModelType = 0;
  // static constexpr uint32_t kApplyDenseTableType = 1;
  // static constexpr uint32_t kApplySparseTableType = 2;
  // static constexpr uint32_t kRegisterModelType = 3;
  // static constexpr uint32_t kRegisterDenseTableInfoType = 4;
  // static constexpr uint32_t kRegisterDenseTableType = 5;
  // static constexpr uint32_t kRegisterSparseTableType = 6;
  // static constexpr uint32_t kPullDenseTableType = 7;
  // static constexpr uint32_t kCombinePullDenseTableType = 8;
  // static constexpr uint32_t kPushPullDenseTableType = 9;
  // static constexpr uint32_t kPushDenseTableType = 10;
  // static constexpr uint32_t kPullSparseTableType = 11;
  // static constexpr uint32_t kCombinePullSparseTableType = 12;
  // static constexpr uint32_t kPushSparseTableType = 13;
  // static constexpr uint32_t kSaveCheckPointType = 14;
  // static constexpr uint32_t kHeartBeatType = 15;
  // static constexpr uint32_t kNotifyPsTopolChangeType = 16;
  // static constexpr uint32_t kNotifyPsRouterChangeType = 17;
  // static constexpr uint32_t kNotifyPsDropDataType = 18;
  // static constexpr uint32_t kNotifyPsLeavingType = 20;
  // static constexpr uint32_t kNotifyPsCanLeaveType = 21;
  // static constexpr uint32_t kCreateModelType = 22;
  // static constexpr uint32_t kCreateDenseTableType = 23;
  // static constexpr uint32_t kCreateSparseTableType = 24;
  // static constexpr uint32_t kPsTryRegisterType = 25;
  // static constexpr uint32_t kPsJoinType = 26;
  // static constexpr uint32_t kTryJoinType = 27;
  // static constexpr uint32_t kHeartbeatType = 28;
  // static constexpr uint32_t kNotifyNodeJoinType = 29;
  // static constexpr uint32_t kNotifyRouterChangeType = 30;
  // static constexpr uint32_t kInitModelType = 31;
};

}  // namespace kraken
