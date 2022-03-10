#include "ps/transfer.h"

#include "protocol/notify_finish_transfer_prot.h"
#include "protocol/rpc_func_type.h"

namespace kraken {

Transfer::Transfer(uint64_t target_id, const std::string& target_addr,
                   CompressType compress_type, uint32_t try_num)
    : target_id_(target_id),
      target_addr_(target_addr),
      connecter_(target_addr, compress_type),
      try_num_(try_num) {
  connecter_.Start();
}

Transfer::~Transfer() {
  connecter_.Stop();
}

int32_t Transfer::TransferDenseValue(uint64_t table_id, const Value& val) {
  return ErrorCode::kSuccess;
}

int32_t Transfer::TransferSparseValues(uint64_t table_id,
                                       const std::vector<uint64_t>& sparse_ids,
                                       const std::vector<Value>& vals) {
  return ErrorCode::kSuccess;
}

int32_t Transfer::NotifyFinishTransfer(uint64_t node_id) {
  uint32_t try_n = try_num_;

  NotifyFinishTransferRequest req;
  req.node_id = node_id;
  NotifyFinishTransferResponse reply;

  int32_t error_code = ErrorCode::kSuccess;

  while (try_n-- > 0) {
    error_code =
        connecter_.Call(RPCFuncType::kNotifyFinishTransferType, req, &reply);

    if (error_code == ErrorCode::kSuccess) {
      return ErrorCode::kSuccess;
    }
  }

  return error_code;
}

}  // namespace kraken
