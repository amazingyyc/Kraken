#pragma once

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>

#include "common/info.h"
#include "common/router.h"
#include "ps/dense_table.h"
#include "ps/sparse_table.h"

namespace kraken {
namespace io {

class Checkpoint {
public:
  static const std::string kRouterName;
  static const std::string kRouterBinaryName;
  static const std::string kModelMetaDataName;
  static const std::string kModelMetaDataBinaryName;
  static const std::string kDenseTableSuffix;
  static const std::string kSparseTableSuffix;
  static const std::string kShardFolderPrefix;

  static const char* OptimTypeName(OptimType type);

  static const char* InitializerTypeName(InitializerType type);

  // Return a folder name by current time.
  // format: y-m-d-h-m-s
  static std::string FolderNameByTime();

  // Convert fail will return std::chrono::system_clock::from_time_t(0).
  // Format must be: y-m-d-h-m-s
  static std::chrono::time_point<std::chrono::system_clock> StrToTimePoint(
      const std::string& str);

  static bool StrToTimePoint(
      const std::string& str,
      std::chrono::time_point<std::chrono::system_clock>* time_p);

  static bool IsDirExist(const std::string& dir);
  static bool IsDirExist(const std::filesystem::path& path);

  static bool IsFileExist(const std::string& path);
  static bool IsFileExist(const std::filesystem::path& path);

  static bool DeleteDir(const std::string& dir);
  static bool DeleteDir(const std::filesystem::path& path);

  // Create a dir. delete_exist whether delete it if exist.
  static bool CreateDir(const std::string& dir, bool delete_exist);
  static bool CreateDir(const std::filesystem::path& path, bool delete_exist);

  static std::string GenRouterPath(const std::string& dir);
  static std::string GenRouterBinaryPath(const std::string& dir);

  static std::string GenModelMetaDataPath(const std::string& dir);
  static std::string GenModelMetaDataBinaryPath(const std::string& dir);

  // Get dense table file path from dir.
  static bool GetDenseTablePaths(const std::string& dir,
                                 std::vector<std::filesystem::path>* paths);

  // Get Sparse table file path from dir.
  static bool GetSparseTablePaths(const std::string& dir,
                                  std::vector<std::filesystem::path>* paths);

  // Save router
  static bool SaveRouter(const std::string& path, const Router& router);
  static bool SaveRouterBinary(const std::string& path, const Router& router);

  // Load router.
  static bool LoadRouterBinary(const std::string& path, Router* router);

  // Save
  static bool SaveModelMetaData(const std::string& path,
                                const ModelMetaData& model_mdata);
  static bool SaveModelMetaDataBinary(const std::string& path,
                                      const ModelMetaData& model_mdata);

  static bool LoadModelMetaDataBinary(const std::string& path,
                                      ModelMetaData* model_mdata);

  // Save DenseTable to file not thread-safe.
  static bool SaveDenseTable(const std::string& path, DenseTable* table);

  // Save SparseTable to file.
  static bool SaveSparseTable(const std::string& path, SparseTable* table);

  static std::unique_ptr<DenseTable> LoadDenseTable(const std::string& path);

  static bool LoadSparseTable(const std::string& path, SparseTable* table,
                              uint64_t node_id, const Router& router);
};

}  // namespace io
}  // namespace kraken
