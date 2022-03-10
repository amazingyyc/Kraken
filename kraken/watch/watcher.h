// #pragma once

// #include <cinttypes>
// #include <memory>
// #include <string>
// #include <vector>

// #include "io/checkpoint.h"
// #include "ps/info.h"
// #include "ps/model.h"
// #include "t/tensor.h"

// namespace kraken {
// namespace watch {

// /**
//  * \brief This is a simple class used to deserialize the Model from file used to debug.
//  */
// class Watcher : public io::Checkpoint {
// private:
//   ModelInfo model_info_;

//   std::unique_ptr<Model> model_;

//   bool loaded_;

// public:
//   Watcher();

// public:
//   void Load(const std::string& shard_dir);

//   const ModelInfo& model_info() const;

//   std::vector<TableInfo> DenseTableInfos() const;
//   std::vector<TableInfo> ExistDenseTableInfos() const;

//   std::vector<TableInfo> SparseTableInfos() const;
//   std::vector<TableInfo> ExistSparseTableInfos() const;

//   std::vector<int64_t> ExistSparseTableIds(uint64_t table_id) const;

//   bool IsTableExist(uint64_t table_id) const;
//   bool IsTableExist(const std::string& table_name) const;

//   bool IsDenseTableValExist(uint64_t table_id);
//   bool IsSparseTableValExist(uint64_t table_id, uint64_t sparse_id);

//   Tensor DenseTableVal(uint64_t table_id) const;
//   Tensor SparseTableVal(uint64_t table_id, uint64_t sparse_id) const;
// };

// }  // namespace watch
// }  // namespace kraken
