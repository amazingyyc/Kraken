#pragma once

#include <filesystem>
#include <functional>
#include <list>
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>

#include "common/consistent_hasher.h"
#include "ps/dense_table.h"
#include "ps/model_manager.h"
#include "ps/ps.h"
#include "ps/sparse_table.h"
#include "ps/table.h"

namespace kraken {
namespace io {

class CheckPoint {
private:
  static const std::string kModelInfoName;
  static const std::string kModelBinaryName;
  static const std::string kDenseTableSuffix;
  static const std::string kSparseTableSuffix;
  static const std::string kPartitionFolderPrefix;

  struct Task {
    uint64_t model_id;

    std::function<void(bool)> done;
  };

private:
  Ps* ps_;

  // The directory to store the model file.
  // The real model path is: models_dir_ + model_name + timestamp.
  std::string save_dir_;

  // How many checkpoint will be saved.
  size_t max_save_count_;

  // Store the saved model path will delelte the oldest if the count >
  // max_save_count_.
  std::unordered_map<uint64_t, std::list<std::string>> model_check_points_;

  // Will use a separate thread to dump model.
  std::thread woker_;

  std::atomic_bool stop_;

  std::mutex mu_;
  std::condition_variable cond_var_;
  std::queue<Task> task_que_;

public:
  CheckPoint(Ps* ps, const std::string& save_dir, size_t max_save_count = 3);

private:
  bool IsDirExist(const std::string& dir) const;

  bool IsDirExist(const std::filesystem::path& path) const;

  bool IsFileExist(const std::string& p) const;

  bool IsFileExist(const std::filesystem::path& path) const;

  bool DeleteDir(const std::string& dir) const;

  // Create a dir. exist_delete whether delete it if exist.
  bool CreateDir(const std::string& dir, bool exist_delete) const;

  bool CreateDir(const std::filesystem::path& path, bool exist_delete) const;

  bool GetSortedPartitionFolder(
      const std::string& dir,
      std::vector<std::string>* partition_folders) const;

  // Get dense table file path from dir.
  bool GetDenseTablePaths(const std::string& dir,
                          std::vector<std::filesystem::path>* paths);

  // Get Sparse table file path from dir.
  bool GetSparseTablePaths(const std::string& dir,
                           std::vector<std::filesystem::path>* paths);

  // Generate a sub model dir from current dir by time.
  std::string GenModelDirByTime() const;

  // Generate model info path.
  std::string GenModelInfoPath(const std::string& dir) const;

  // Get model binary path.
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

  bool Load(const std::string& path, size_t shard_id, uint64_t model_id,
            const ConsistentHasher& hasher, SparseTable* table);

  bool Save(Ps* ps, const std::string& save_dir, uint64_t model_id);

  bool Load(Ps* ps, const std::string& dir);

  void Run();

public:
  void Stop();

  void Save(uint64_t model_id, std::function<void(bool)>&& done);

  bool Load(const std::string& model_dir);
};

}  // namespace io
}  // namespace kraken
