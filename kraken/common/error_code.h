#pragma once

#include <cinttypes>

namespace kraken {

struct ErrorCode {
  static constexpr int32_t kUnknowError = -1;
  static constexpr int32_t kSuccess = 0;
  static constexpr int32_t kUnRegisterFuncError = 1;
  static constexpr int32_t kSerializeRequestError = 2;
  static constexpr int32_t kSerializeReplyError = 3;
  static constexpr int32_t kDeserializeRequestError = 4;
  static constexpr int32_t kDeserializeReplyError = 5;
  static constexpr int32_t kUnSupportOptimTypeError = 6;
  static constexpr int32_t kTableTypeUnCompatibleError = 7;
  static constexpr int32_t kDenseTableUnCompatibleError = 8;
  static constexpr int32_t kSparseDimensionError = 9;
  static constexpr int32_t kSparseTableUnCompatibleError = 10;
  static constexpr int32_t kUnRegisterModelError = 11;
  static constexpr int32_t kUnRegisterTableError = 12;
  static constexpr int32_t kInterfaceUnImplementError = 13;
  static constexpr int32_t kGradientUnCompatibleError = 14;
  static constexpr int32_t kSparseTableIdError = 15;
  static constexpr int32_t kSparseTableIdNotExistError = 16;
  static constexpr int32_t kPushSparseTableParameterError = 17;
  static constexpr int32_t kUnSupportInitializerTypeError = 18;
  static constexpr int32_t kUnSupportCompressTypeError = 19;
  static constexpr int32_t kSnappyUncompressError = 20;
  static constexpr int32_t kSnappyCompressError = 21;

  static const char* Msg(int32_t code) {
    switch (code) {
      case ErrorCode::kUnknowError:
        return "Unknow error.";
      case ErrorCode::kSuccess:
        return "Success.";
      case ErrorCode::kUnRegisterFuncError:
        return "UnRegister RPC function.";
      case ErrorCode::kSerializeRequestError:
        return "Serialize request error.";
      case ErrorCode::kSerializeReplyError:
        return "Serialize reply error.";
      case ErrorCode::kDeserializeRequestError:
        return "Deserialize request error.";
      case ErrorCode::kDeserializeReplyError:
        return "Deserialize reply error.";
      case ErrorCode::kUnSupportOptimTypeError:
        return "UnSupport optim type.";
      case ErrorCode::kTableTypeUnCompatibleError:
        return "Uncompatible table type.";
      case ErrorCode::kDenseTableUnCompatibleError:
        return "Uncompatible Dense table";
      case ErrorCode::kUnRegisterModelError:
        return "Unregister model.";
      case ErrorCode::kUnRegisterTableError:
        return "Unregister table.";
      case ErrorCode::kInterfaceUnImplementError:
        return "Unimplement interface.";
      case ErrorCode::kGradientUnCompatibleError:
        return "Uncompatible gradient.";
      case ErrorCode::kSparseTableIdError:
        return "Sparse id error.";
      case ErrorCode::kSparseTableIdNotExistError:
        return "Sparse table id not exist.";
      case ErrorCode::kPushSparseTableParameterError:
        return "Push sparse table parameter error.";
      case ErrorCode::kUnSupportInitializerTypeError:
        return "UnSupport initializer type.";
      default:
        return "Unknow error.";
    }
  }
};

}  // namespace kraken
