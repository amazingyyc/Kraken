#pragma once

#include <memory>
#include <string>

#include "common/consistent_hasher.h"
#include "ps/dense_table.h"
#include "ps/model_manager.h"
#include "ps/ps.h"
#include "ps/sparse_table.h"
#include "ps/table.h"

namespace kraken {
namespace io {

class Saver {
private:
  static const std::string kModelInfoName;
  static const std::string kModelBinaryName;
  static const std::string kDenseTableSuffix;
  static const std::string kSparseTableSuffix;
  static const std::string kPartitionFolderPrefix;

private:
  bool IsDirExist(const std::string& dir) const;

  bool IsFileExist(const std::string& p) const;

  // Create a dir. exist_delete whether delete it if exist.
  bool CreateDir(const std::string& dir, bool exist_delete) const;

  bool GetSortedPartitionFolder(
      const std::string& dir,
      std::vector<std::string>* partition_folders) const;

  // Generate model info path.
  std::string GenModelInfoPath(const std::string& dir) const;

  // Get binary path.
  std::string GenModelBinaryPath(const std::string& dir) const;

  bool Save(const std::string& model_info_path,
            const std::string& model_binary_path,
            ModelManager::Model& model) const;

  bool Save(const std::string& dir, DenseTable* table) const;

  bool Save(const std::string& dir, SparseTable* table) const;

  bool Load(const std::string& model_binary_path,
            ModelManager::Model* model) const;

  bool Load(const std::string& path, DenseTable* table) const;

  bool Load(const std::vector<std::string>& paths, size_t shard_id,
            uint64_t model_id, const ConsistentHasher& hasher,
            SparseTable* table);

public:
  bool Save(Ps* ps, const std::string& dir, uint64_t model_id);

  bool Load(Ps* ps, const std::string& dir);
};

}  // namespace io
}  // namespace kraken
