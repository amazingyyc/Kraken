#include "ps/transfer.h"

#include "protocol/notify_finish_transfer_prot.h"
#include "protocol/rpc_func_type.h"
#include "protocol/transfer_dense_table_prot.h"
#include "protocol/transfer_sparse_meta_data_prot.h"
#include "protocol/transfer_sparse_values_prot.h"

namespace kraken {

Transfer::Transfer(uint64_t target_id, const std::string& target_addr,
                   CompressType compress_type, uint32_t try_num)
    : target_id_(target_id),
      target_addr_(target_addr),
      connecter_(new IndepConnecter(target_addr, compress_type)),
      try_num_(try_num) {
  connecter_->Start();
}

Transfer::~Transfer() {
  connecter_->Stop();
}

int32_t Transfer::TransferDenseTable(uint64_t from_node_id, uint64_t table_id,
                                     const std::string& name,
                                     Value& val) const {
  uint32_t try_n = try_num_;

  TransferDenseTableRequest req;
  req.from_node_id = from_node_id;
  req.table_id = table_id;
  req.name = name;
  req.value = val;

  TransferDenseTableResponse reply;

  int32_t error_code = ErrorCode::kSuccess;
  while (try_n-- > 0) {
    error_code =
        connecter_->Call(RPCFuncType::kTransferDenseTableType, req, &reply);

    if (error_code == ErrorCode::kSuccess) {
      return ErrorCode::kSuccess;
    }
  }

  return error_code;
}

int32_t Transfer::TransferSparseMetaData(
    uint64_t from_node_id, uint64_t table_id, std::string name,
    int64_t dimension, ElementType element_type, InitializerType init_type,
    const std::unordered_map<std::string, std::string>& init_conf) const {
  uint32_t try_n = try_num_;

  TransferSparseMetaDataRequest req;
  req.from_node_id = from_node_id;
  req.table_id = table_id;
  req.name = name;
  req.dimension = dimension;
  req.element_type = element_type;
  req.init_type = init_type;
  req.init_conf = init_conf;

  TransferSparseMetaDataResponse reply;

  int32_t error_code = ErrorCode::kSuccess;
  while (try_n-- > 0) {
    error_code =
        connecter_->Call(RPCFuncType::kTransferSparseMetaDataType, req, &reply);

    if (error_code == ErrorCode::kSuccess) {
      return ErrorCode::kSuccess;
    }
  }

  return error_code;
}

int32_t Transfer::TransferSparseValues(uint64_t from_node_id, uint64_t table_id,
                                       const std::vector<uint64_t>& sparse_ids,
                                       const std::vector<Value>& vals) const {
  if (sparse_ids.empty()) {
    return ErrorCode::kSuccess;
  }

  uint32_t try_n = try_num_;

  TransferSparseValuesRequest req;
  req.from_node_id = from_node_id;
  req.table_id = table_id;
  req.sparse_ids = sparse_ids;
  req.values = vals;
  TransferSparseValuesResponse reply;

  int32_t error_code = ErrorCode::kSuccess;
  while (try_n-- > 0) {
    error_code =
        connecter_->Call(RPCFuncType::kTransferSparseValuesType, req, &reply);

    if (error_code == ErrorCode::kSuccess) {
      return ErrorCode::kSuccess;
    }
  }

  return error_code;
}

int32_t Transfer::NotifyFinishTransfer(uint64_t from_node_id) const {
  uint32_t try_n = try_num_;

  NotifyFinishTransferRequest req;
  req.from_node_id = from_node_id;
  NotifyFinishTransferResponse reply;

  int32_t error_code = ErrorCode::kSuccess;

  while (try_n-- > 0) {
    error_code =
        connecter_->Call(RPCFuncType::kNotifyFinishTransferType, req, &reply);

    if (error_code == ErrorCode::kSuccess) {
      return ErrorCode::kSuccess;
    }
  }

  return error_code;
}

}  // namespace kraken
