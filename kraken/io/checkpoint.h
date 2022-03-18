// #pragma once

// #include <filesystem>
// #include <memory>
// #include <string>

// #include "common/consistent_hasher.h"
// #include "ps/initializer/initializer.h"
// #include "ps/optim/optim.h"

// namespace kraken {

// class ModelInfo;
// class DenseTable;
// class SparseTable;

// namespace io {

// class Checkpoint {
// protected:
//   static const std::string kModelInfoName;
//   static const std::string kModelBinaryName;
//   static const std::string kDenseTableSuffix;
//   static const std::string kSparseTableSuffix;
//   static const std::string kShardFolderPrefix;

// protected:
//   const char* OptimTypeName(OptimType type) const;

//   const char* InitializerTypeName(InitializerType type) const;

//   bool IsDirExist(const std::string& dir) const;
//   bool IsDirExist(const std::filesystem::path& path) const;

//   bool IsFileExist(const std::string& p) const;
//   bool IsFileExist(const std::filesystem::path& path) const;

//   bool DeleteDir(const std::string& dir) const;
//   bool DeleteDir(const std::filesystem::path& path) const;

//   // Create a dir. exist_delete whether delete it if exist.
//   bool CreateDir(const std::string& dir, bool exist_delete) const;
//   bool CreateDir(const std::filesystem::path& path, bool exist_delete) const;

//   // Get dense table file path from dir.
//   bool GetDenseTablePaths(const std::string& dir,
//                           std::vector<std::filesystem::path>* paths);

//   // Get Sparse table file path from dir.
//   bool GetSparseTablePaths(const std::string& dir,
//                            std::vector<std::filesystem::path>* paths);

//   // Generate model info path.
//   std::string GenModelInfoPath(const std::string& dir) const;

//   // Get model binary path.
//   std::string GenModelBinaryPath(const std::string& dir) const;

//   bool SaveModelInfo(const std::string& path,
//                      const ModelInfo& model_info) const;

//   bool SaveModelBinaryInfo(const std::string& path,
//                            const ModelInfo& model_info) const;

//   bool SaveDenseTable(const std::string& path, DenseTable* table) const;

//   bool SaveSparseTable(const std::string& path, SparseTable* table) const;

//   bool LoadModelBinaryInfo(const std::string& path,
//                            ModelInfo* model_info) const;

//   bool LoadDenseTable(const std::string& path, DenseTable* table) const;

//   // Load the SaprseTable from file.
//   bool LoadSparseTable(const std::string& path, SparseTable* table) const;

//   // Load SparseTable from file and check whether it blongs to this shard.
//   bool LoadSparseTable(const std::string& path, size_t shard_id,
//                        uint64_t model_id, const ConsistentHasher& hasher,
//                        SparseTable* table) const;
// };

// }  // namespace io
// }  // namespace kraken
