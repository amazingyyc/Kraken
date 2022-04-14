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
  static constexpr int32_t kDenseTableUnCompatibleError = 7;
  static constexpr int32_t kSparseDimensionError = 8;
  static constexpr int32_t kSparseTableUnCompatibleError = 9;
  static constexpr int32_t kInterfaceUnImplementError = 10;
  static constexpr int32_t kGradientUnCompatibleError = 11;
  static constexpr int32_t kSparseIdNotExistError = 12;
  static constexpr int32_t kUnSupportInitializerTypeError = 13;
  static constexpr int32_t kUnSupportCompressTypeError = 14;
  static constexpr int32_t kSocketNotExistError = 15;
  static constexpr int32_t kNodeStatusError = 16;
  static constexpr int32_t kTimeoutError = 17;
  static constexpr int32_t kUnSupportEventError = 18;
  static constexpr int32_t kModelNotInitializedError = 19;
  static constexpr int32_t kConnectNodeError = 20;
  static constexpr int32_t kTableNotExistError = 21;
  static constexpr int32_t kRouterVersionError = 22;
  static constexpr int32_t kModelAlreadyInitializedError = 23;
  static constexpr int32_t kLoadModelError = 24;

  static const char* Msg(int32_t code) {
    switch (code) {
      case ErrorCode::kUnknowError:
        return "Unknow error";
      case ErrorCode::kSuccess:
        return "Success";
      case ErrorCode::kUnRegisterFuncError:
        return "UnRegister RPC function";
      case ErrorCode::kSerializeRequestError:
        return "Serialize request error";
      case ErrorCode::kSerializeReplyError:
        return "Serialize reply error";
      case ErrorCode::kDeserializeRequestError:
        return "Deserialize request error";
      case ErrorCode::kDeserializeReplyError:
        return "Deserialize reply error";
      case ErrorCode::kUnSupportOptimTypeError:
        return "UnSupport optim type";
      case ErrorCode::kDenseTableUnCompatibleError:
        return "Uncompatible Dense table";
      case ErrorCode::kSparseDimensionError:
        return "Sparse Dimension error";
      case ErrorCode::kSparseTableUnCompatibleError:
        return "SparseTable Uncompatible Error";
      case ErrorCode::kInterfaceUnImplementError:
        return "UnImplement Interface";
      case ErrorCode::kGradientUnCompatibleError:
        return "Uncompatible gradient";
      case ErrorCode::kSparseIdNotExistError:
        return "SparseId not exist";
      case ErrorCode::kUnSupportInitializerTypeError:
        return "UnSupport initializer type";
      case ErrorCode::kUnSupportCompressTypeError:
        return "UnSupportCompress type";
      case ErrorCode::kSocketNotExistError:
        return "Socket not exist";
      case ErrorCode::kNodeStatusError:
        return "NodeStatus error";
      case ErrorCode::kTimeoutError:
        return "Timeout error";
      case ErrorCode::kUnSupportEventError:
        return "UnSupport event";
      case ErrorCode::kModelNotInitializedError:
        return "Model not initialized";
      case ErrorCode::kConnectNodeError:
        return "Connect node error";
      case ErrorCode::kTableNotExistError:
        return "Table not exist";
      case ErrorCode::kRouterVersionError:
        return "Router version error";
      case ErrorCode::kModelAlreadyInitializedError:
        return "Model already initialized error";
      case ErrorCode::kLoadModelError:
        return "Load model error";
      default:
        return "Unrecognized error";
    }
  }
};

}  // namespace kraken
