#pragma once

#include <cinttypes>

namespace kraken {

struct ErrorCode {
  static constexpr int32_t kUnknowError = -1;
  static constexpr int32_t kSuccess = 0;
  static constexpr int32_t kRequestHeaderError = 1;
  static constexpr int32_t kUnRegisterFuncError = 2;
  static constexpr int32_t kSerializeRequestError = 3;
  static constexpr int32_t kSerializeReplyError = 4;
  static constexpr int32_t kDeserializeRequestError = 5;
  static constexpr int32_t kDeserializeReplyError = 6;
  static constexpr int32_t kUnSupportOptimTypeError = 7;
  static constexpr int32_t kOptimTypeUnCompatibleError = 8;
  static constexpr int32_t kTableTypeUnCompatibleError = 9;
  static constexpr int32_t kDenseTabelError = 10;
  static constexpr int32_t kSparseDimensionError = 11;
  static constexpr int32_t kSparseTabelError = 12;
  static constexpr int32_t kUnRegisterModelError = 13;
  static constexpr int32_t kUnRegisterTableError = 14;
  static constexpr int32_t kUnImplementError = 15;
  static constexpr int32_t kGradientUnCompatibleError = 16;
  static constexpr int32_t kSparseTableIdError = 17;
  static constexpr int32_t kSparseTableIdNotExistError = 18;
  static constexpr int32_t kSparseTableUnSupportIndicesTypeError = 19;
  static constexpr int32_t kPushSparseTableParameterError = 20;
};

}  // namespace kraken
