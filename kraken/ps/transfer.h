#pragma once

#include <cinttypes>
#include <string>

#include "rpc/indep_connecter.h"

namespace kraken {

class Transfer {
private:
  uint64_t target_id_;
  std::string target_addr_;

  IndepConnecter connecter_;

  uint32_t try_num_;

public:
  // (TODO) really need 3 times?
  Transfer(uint64_t target_id, const std::string& target_addr,
           CompressType compress_type, uint32_t try_num = 3);

  ~Transfer();

public:
  int32_t TransferDenseTable(uint64_t from_node_id, uint64_t table_id,
                             const std::string& name, Value& val);

  int32_t TransferSparseMetaData(
      uint64_t table_id, std::string name, int64_t dimension,
      ElementType element_type, InitializerType init_type,
      const std::unordered_map<std::string, std::string>& init_conf);

  int32_t TransferSparseValues(uint64_t table_id,
                               const std::vector<uint64_t>& sparse_ids,
                               const std::vector<Value>& vals);

  // Notify the target Node we already finish trasfer data.
  int32_t NotifyFinishTransfer(uint64_t node_id);
};

}  // namespace kraken
