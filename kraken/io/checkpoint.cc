#include "io/checkpoint.h"

#include "common/deserialize.h"
#include "common/log.h"
#include "common/serialize.h"
#include "configor/json.hpp"
#include "io/file_reader.h"
#include "io/file_writer.h"
#include "ps/dense_table.h"
#include "ps/info.h"
#include "ps/sparse_table.h"
#include "ps/table.h"

namespace kraken {
namespace io {

const std::string Checkpoint::kModelInfoName = "model.json";
const std::string Checkpoint::kModelBinaryName = "model.binary";
const std::string Checkpoint::kDenseTableSuffix = ".dense";
const std::string Checkpoint::kSparseTableSuffix = ".sparse";
const std::string Checkpoint::kShardFolderPrefix = "shard_";

const char* Checkpoint::OptimTypeName(OptimType type) const {
  if (type == OptimType::kAdagrad) {
    return "Adagrad";
  } else if (type == OptimType::kAdam) {
    return "Adam";
  } else if (type == OptimType::kRMSprop) {
    return "RMSprop";
  } else if (type == OptimType::kSGD) {
    return "SGD";
  } else {
    return "UnKnow";
  }
}

const char* Checkpoint::InitializerTypeName(InitializerType type) const {
  if (type == InitializerType::kConstant) {
    return "Constant";
  } else if (type == InitializerType::kUniform) {
    return "Uniform";
  } else if (type == InitializerType::kNormal) {
    return "Normal";
  } else if (type == InitializerType::kXavierUniform) {
    return "XavierUniform";
  } else if (type == InitializerType::kXavierNormal) {
    return "XavierNormal";
  } else {
    return "UnKnow";
  }
}

bool Checkpoint::IsDirExist(const std::string& dir) const {
  std::filesystem::path path(dir);

  return IsDirExist(path);
}

bool Checkpoint::IsDirExist(const std::filesystem::path& path) const {
  std::error_code error_code;
  auto status = std::filesystem::status(path, error_code);

  if (error_code) {
    return false;
  }

  if (std::filesystem::exists(status) == false) {
    return false;
  }

  return std::filesystem::is_directory(status);
}

bool Checkpoint::IsFileExist(const std::string& p) const {
  std::filesystem::path path(p);

  return IsFileExist(path);
}

bool Checkpoint::IsFileExist(const std::filesystem::path& path) const {
  std::error_code error_code;
  auto status = std::filesystem::status(path, error_code);

  if (error_code) {
    return false;
  }

  if (std::filesystem::exists(status) == false) {
    return false;
  }

  return std::filesystem::is_directory(status) == false;
}

bool Checkpoint::DeleteDir(const std::string& dir) const {
  std::filesystem::path path(dir);

  return DeleteDir(path);
}

bool Checkpoint::DeleteDir(const std::filesystem::path& path) const {
  if (IsDirExist(path)) {
    std::error_code error_code;
    std::filesystem::remove_all(path, error_code);

    if (error_code) {
      return false;
    }
  }

  return true;
}

bool Checkpoint::CreateDir(const std::string& dir, bool exist_delete) const {
  std::filesystem::path path(dir);

  return CreateDir(path, exist_delete);
}

bool Checkpoint::CreateDir(const std::filesystem::path& path,
                           bool exist_delete) const {
  if (IsDirExist(path)) {
    if (exist_delete) {
      std::error_code error_code;
      std::filesystem::remove_all(path, error_code);

      if (error_code) {
        return false;
      }
    } else {
      return true;
    }
  }

  // Create dir.
  std::error_code error_code;
  std::filesystem::create_directories(path, error_code);

  if (error_code) {
    return false;
  }

  return true;
}

bool Checkpoint::GetDenseTablePaths(const std::string& dir,
                                    std::vector<std::filesystem::path>* paths) {
  std::filesystem::path path(dir);
  for (auto const& entry : std::filesystem::directory_iterator{path}) {
    const std::filesystem::path& f_path = entry.path();

    if (IsFileExist(f_path)) {
      auto filename = f_path.filename().string();
      if (utils::EndWith(filename, kDenseTableSuffix)) {
        paths->emplace_back(f_path);
      }
    }
  }

  return true;
}

bool Checkpoint::GetSparseTablePaths(
    const std::string& dir, std::vector<std::filesystem::path>* paths) {
  std::filesystem::path path(dir);
  for (auto const& entry : std::filesystem::directory_iterator{path}) {
    const std::filesystem::path& f_path = entry.path();

    if (IsFileExist(f_path)) {
      auto filename = f_path.filename().string();
      if (utils::EndWith(filename, kSparseTableSuffix)) {
        paths->emplace_back(f_path);
      }
    }
  }

  return true;
}

std::string Checkpoint::GenModelInfoPath(const std::string& dir) const {
  std::filesystem::path path(dir);
  path /= kModelInfoName;

  return path.string();
}

std::string Checkpoint::GenModelBinaryPath(const std::string& dir) const {
  std::filesystem::path path(dir);
  path /= kModelBinaryName;

  return path.string();
}

bool Checkpoint::SaveModelInfo(const std::string& path,
                               const ModelInfo& model_info) const {
  // Readable.
  configor::json j;
  j["id"] = model_info.id;
  j["name"] = model_info.name;
  j["optim_type"] = OptimTypeName(model_info.optim_type);
  j["optim_conf"] = model_info.optim_conf;

  configor::json tables_j = configor::json::array({});
  for (auto& [table_name, table_info] : model_info.table_infos) {
    configor::json t_j;

    t_j["id"] = table_info.id;
    t_j["name"] = table_info.name;
    t_j["table_type"] =
        (table_info.table_type == TableType::kDense ? "Dense" : "Sparse");
    t_j["element_type"] = table_info.element_type.Name();

    if (table_info.table_type == TableType::kDense) {
      t_j["shape"] = table_info.shape.dims();
    } else if (table_info.table_type == TableType::kSparse) {
      t_j["dimension"] = table_info.dimension;
      t_j["init_type"] = InitializerTypeName(table_info.init_type);
      t_j["init_conf"] = table_info.init_conf;
    }

    tables_j.push_back(std::move(t_j));
  }

  j["tables"] = tables_j;

  std::string pretty_str = j.dump(4, ' ');

  // Dump json string to file.
  std::ofstream out_f(path);
  if (!out_f.is_open()) {
    LOG_ERROR("Open file:" << path << " error!");
    return false;
  }

  // Write to file.
  out_f << pretty_str << std::endl;
  out_f.close();

  return true;
}

bool Checkpoint::SaveModelBinaryInfo(const std::string& path,
                                     const ModelInfo& model_info) const {
  // Binary.
  FileWriter writer(path);
  if (writer.IsOpen() == false) {
    LOG_ERROR("Open file:" << path << " error!");
    return false;
  }

  Serialize serialize(&writer);

  if ((serialize << model_info.id) == false ||
      (serialize << model_info.name) == false ||
      (serialize << model_info.optim_type) == false ||
      (serialize << model_info.optim_conf) == false) {
    return false;
  }

  uint64_t table_size = model_info.table_infos.size();
  if ((serialize << table_size) == false) {
    return false;
  }

  for (const auto& [k, v] : model_info.table_infos) {
    // table id.
    if ((serialize << k) == false) {
      return false;
    }

    if ((serialize << v.id) == false || (serialize << v.name) == false ||
        (serialize << v.table_type) == false ||
        (serialize << v.element_type) == false ||
        (serialize << v.shape) == false ||
        (serialize << v.dimension) == false ||
        (serialize << v.init_type) == false ||
        (serialize << v.init_conf) == false) {
      return false;
    }
  }

  return true;
}

bool Checkpoint::SaveDenseTable(const std::string& path,
                                DenseTable* table) const {
  // Open a file to serialize the DenseTable.
  FileWriter writer(path);
  if (writer.IsOpen() == false) {
    LOG_ERROR("Open file:" << path << " error!");
    return false;
  }

  Serialize serialize(&writer);

  if ((serialize << table->type_) == false ||
      (serialize << table->id_) == false ||
      (serialize << table->name_) == false ||
      (serialize << table->val_) == false) {
    return false;
  }

  return true;
}

bool Checkpoint::SaveSparseTable(const std::string& path,
                                 SparseTable* table) const {
  FileWriter writer(path);
  if (writer.IsOpen() == false) {
    LOG_ERROR("Open file:" << path << " error!");
    return false;
  }

  Serialize serialize(&writer);

  if ((serialize << table->type_) == false ||
      (serialize << table->id_) == false ||
      (serialize << table->name_) == false ||
      (serialize << table->dimension_) == false ||
      (serialize << table->element_type_) == false ||
      (serialize << table->initializer_->type()) == false ||
      (serialize << table->initializer_->conf()) == false) {
    return false;
  }

  {
    // At here lock all hash table of SparseTable.
    auto lt = table->vals_.lock_table();

    uint64_t size = lt.size();
    if ((serialize << size) == false) {
      return false;
    }

    // At here it's thread-safe to read the vals.
    for (auto it = lt.begin(); it != lt.end(); ++it) {
      // key-value
      if ((serialize << it->first) == false ||
          (serialize << it->second) == false) {
        return false;
      }
    }
  }

  return true;
}

bool Checkpoint::LoadModelBinaryInfo(const std::string& path,
                                     ModelInfo* model_info) const {
  // Binary.
  FileReader reader(path);
  if (reader.IsOpen() == false) {
    LOG_ERROR("Open file:" << path << " error!");
    return false;
  }

  Deserialize deserialize(&reader);

  if ((deserialize >> model_info->id) == false ||
      (deserialize >> model_info->name) == false ||
      (deserialize >> model_info->optim_type) == false ||
      (deserialize >> model_info->optim_conf) == false) {
    return false;
  }

  uint64_t table_size;
  if ((deserialize >> table_size) == false) {
    return false;
  }

  model_info->table_infos.reserve(table_size);

  for (uint64_t i = 0; i < table_size; ++i) {
    uint64_t table_id;
    TableInfo table_info;

    if ((deserialize >> table_id) == false) {
      return false;
    }

    if ((deserialize >> table_info.id) == false ||
        (deserialize >> table_info.name) == false ||
        (deserialize >> table_info.table_type) == false ||
        (deserialize >> table_info.element_type) == false ||
        (deserialize >> table_info.shape) == false ||
        (deserialize >> table_info.dimension) == false ||
        (deserialize >> table_info.init_type) == false ||
        (deserialize >> table_info.init_conf) == false) {
      return false;
    }

    model_info->table_infos.emplace(table_id, std::move(table_info));
  }

  return true;
}

bool Checkpoint::LoadDenseTable(const std::string& path,
                                DenseTable* table) const {
  FileReader reader(path);
  if (reader.IsOpen() == false) {
    LOG_ERROR("Open file:" << path << " error!");
    return false;
  }

  Deserialize deserialize(&reader);

  if ((deserialize >> table->type_) == false ||
      (deserialize >> table->id_) == false ||
      (deserialize >> table->name_) == false ||
      (deserialize >> table->val_) == false) {
    return false;
  }

  return true;
}

// Load the SaprseTable from file.
bool Checkpoint::LoadSparseTable(const std::string& path,
                                 SparseTable* table) const {
  FileReader reader(path);
  if (reader.IsOpen() == false) {
    LOG_ERROR("Open file:" << path << " error!");
    return false;
  }

  Deserialize deserialize(&reader);

  // Just deserialize the information the initializer should be initited
  // already.
  InitializerType init_type;
  std::unordered_map<std::string, std::string> init_conf;

  if ((deserialize >> table->type_) == false ||
      (deserialize >> table->id_) == false ||
      (deserialize >> table->name_) == false ||
      (deserialize >> table->dimension_) == false ||
      (deserialize >> table->element_type_) == false ||
      (deserialize >> init_type) == false ||
      (deserialize >> init_conf) == false) {
    return false;
  }

  uint64_t val_size;
  if ((deserialize >> val_size) == false) {
    return false;
  }

  for (uint64_t i = 0; i < val_size; ++i) {
    uint64_t sparse_id;
    Table::Value val;

    if ((deserialize >> sparse_id) == false || (deserialize >> val) == false) {
      return false;
    }

    table->vals_.insert(sparse_id, val);
  }

  return true;
}

bool Checkpoint::LoadSparseTable(const std::string& path, size_t shard_id,
                                 uint64_t model_id,
                                 const ConsistentHasher& hasher,
                                 SparseTable* table) const {
  FileReader reader(path);
  if (reader.IsOpen() == false) {
    LOG_ERROR("Open file:" << path << " error!");
    return false;
  }

  Deserialize deserialize(&reader);

  TableType type;
  uint64_t id;
  std::string name;
  int64_t dimension;
  ElementType etype;
  InitializerType init_type;
  std::unordered_map<std::string, std::string> init_conf;

  if ((deserialize >> type) == false || (deserialize >> id) == false ||
      (deserialize >> name) == false || (deserialize >> dimension) == false ||
      (deserialize >> etype) == false || (deserialize >> init_type) == false ||
      (deserialize >> init_conf) == false) {
    return false;
  }

  if (type != TableType::kSparse || id != table->id_ || name != table->name_ ||
      dimension != table->dimension_ || etype != table->element_type_ ||
      init_type != table->initializer_->type()) {
    return false;
  }

  uint64_t val_size;
  if ((deserialize >> val_size) == false) {
    return false;
  }

  for (uint64_t i = 0; i < val_size; ++i) {
    uint64_t sparse_id;
    Table::Value val;

    if ((deserialize >> sparse_id) == false || (deserialize >> val) == false) {
      return false;
    }

    if (shard_id == hasher(model_id, id, sparse_id)) {
      table->vals_.insert(sparse_id, val);
    }
  }

  return true;
}

}  // namespace io
}  // namespace kraken
