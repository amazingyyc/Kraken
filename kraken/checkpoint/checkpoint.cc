#include "checkpoint/checkpoint.h"

#include <fstream>

#include "checkpoint/file_reader.h"
#include "checkpoint/file_writer.h"
#include "common/deserialize.h"
#include "common/exception.h"
#include "common/log.h"
#include "common/serialize.h"
#include "common/utils.h"
#include "configor/json.hpp"

namespace kraken {
namespace io {

const std::string Checkpoint::kRouterName = "router.json";
const std::string Checkpoint::kRouterBinaryName = "router.binary";
const std::string Checkpoint::kModelMetaDataName = "model.json";
const std::string Checkpoint::kModelMetaDataBinaryName = "model.binary";
const std::string Checkpoint::kDenseTableSuffix = ".dense";
const std::string Checkpoint::kSparseTableSuffix = ".sparse";
const std::string Checkpoint::kShardFolderPrefix = "shard_";

const char* Checkpoint::OptimTypeName(OptimType type) {
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

const char* Checkpoint::InitializerTypeName(InitializerType type) {
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

std::string Checkpoint::FolderNameByTime() {
  std::time_t tt = time(NULL);
  auto local = localtime(&tt);

  char buf[80];
  strftime(buf, 80, "%Y-%m-%d-%H-%M-%S", local);

  return std::string(buf);
}

std::chrono::time_point<std::chrono::system_clock> Checkpoint::StrToTimePoint(
    const std::string& str) {
  std::vector<std::string> items;
  utils::Split(str, "-", &items);

  if (items.size() != 6) {
    return std::chrono::system_clock::from_time_t(0);
  }

  int year, month, day, hour, minute, second;

  try {
    year = std::stoi(items[0]);
    month = std::stoi(items[1]);
    day = std::stoi(items[2]);
    hour = std::stoi(items[3]);
    minute = std::stoi(items[4]);
    second = std::stoi(items[5]);
  } catch (...) {
    return std::chrono::system_clock::from_time_t(0);
  }

  std::tm local = std::tm();
  local.tm_year = year - 1900;
  local.tm_mon = month - 1;
  local.tm_mday = day;
  local.tm_hour = hour;
  local.tm_min = minute;
  local.tm_sec = second;

  return std::chrono::system_clock::from_time_t(std::mktime(&local));
}

bool Checkpoint::StrToTimePoint(
    const std::string& str,
    std::chrono::time_point<std::chrono::system_clock>* time_p) {
  std::vector<std::string> items;
  utils::Split(str, "-", &items);

  if (items.size() != 6) {
    return false;
  }

  int year, month, day, hour, minute, second;

  try {
    year = std::stoi(items[0]);
    month = std::stoi(items[1]);
    day = std::stoi(items[2]);
    hour = std::stoi(items[3]);
    minute = std::stoi(items[4]);
    second = std::stoi(items[5]);
  } catch (...) {
    return false;
  }

  std::tm local = std::tm();
  local.tm_year = year - 1900;
  local.tm_mon = month - 1;
  local.tm_mday = day;
  local.tm_hour = hour;
  local.tm_min = minute;
  local.tm_sec = second;

  *time_p = std::chrono::system_clock::from_time_t(std::mktime(&local));

  return true;
}

bool Checkpoint::IsDirExist(const std::string& dir) {
  std::filesystem::path path(dir);

  return IsDirExist(path);
}

bool Checkpoint::IsDirExist(const std::filesystem::path& path) {
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

bool Checkpoint::IsFileExist(const std::string& path) {
  return IsFileExist(std::filesystem::path(path));
}

bool Checkpoint::IsFileExist(const std::filesystem::path& path) {
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

bool Checkpoint::DeleteDir(const std::string& dir) {
  std::filesystem::path path(dir);

  return DeleteDir(path);
}

bool Checkpoint::DeleteDir(const std::filesystem::path& path) {
  if (IsDirExist(path)) {
    std::error_code error_code;
    std::filesystem::remove_all(path, error_code);

    if (error_code) {
      return false;
    }
  }

  return true;
}

// Create a dir. exist_delete whether delete it if exist.
bool Checkpoint::CreateDir(const std::string& dir, bool exist_delete) {
  std::filesystem::path path(dir);

  return CreateDir(path, exist_delete);
}

bool Checkpoint::CreateDir(const std::filesystem::path& path,
                           bool exist_delete) {
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

std::string Checkpoint::GenRouterPath(const std::string& dir) {
  std::filesystem::path path(dir);
  path /= kRouterName;

  return path.string();
}

std::string Checkpoint::GenRouterBinaryPath(const std::string& dir) {
  std::filesystem::path path(dir);
  path /= kRouterBinaryName;

  return path.string();
}

std::string Checkpoint::GenModelMetaDataPath(const std::string& dir) {
  std::filesystem::path path(dir);
  path /= kModelMetaDataName;

  return path.string();
}

std::string Checkpoint::GenModelMetaDataBinaryPath(const std::string& dir) {
  std::filesystem::path path(dir);
  path /= kModelMetaDataBinaryName;

  return path.string();
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

bool Checkpoint::SaveRouter(const std::string& path, const Router& router) {
  configor::json nodes_array_j = configor::json::array({});
  for (const auto& node : router.nodes()) {
    configor::json node_j;
    node_j["id"] = node.id;
    node_j["name"] = node.name;

    configor::json vnodes_array_j = configor::json::array({});

    for (auto hash_v : node.vnode_list) {
      const auto& virtaul_node = router.virtual_node(hash_v);

      configor::json vnode_j;
      vnode_j["hash_v"] = virtaul_node.hash_v;
      vnode_j["name"] = virtaul_node.name;

      vnodes_array_j.push_back(std::move(vnode_j));
    }

    node_j["virtual_nodes"] = vnodes_array_j;

    nodes_array_j.push_back(std::move(node_j));
  }

  std::string pretty_str = nodes_array_j.dump(4, ' ');

  std::ofstream out_f(path);
  if (!out_f.is_open()) {
    LOG_ERROR("Open file:[" << path << "] error!");
    return false;
  }

  out_f << pretty_str << std::endl;
  out_f.close();

  return true;
}

bool Checkpoint::SaveRouterBinary(const std::string& path,
                                  const Router& router) {
  FileWriter writer(path);
  if (writer.IsOpen() == false) {
    LOG_ERROR("Open file:[" << path << "] error!");
    return false;
  }

  Serialize serialize(&writer);

  if (serialize << router == false) {
    return false;
  }

  return true;
}

bool Checkpoint::LoadRouterBinary(const std::string& path, Router* router) {
  FileReader reader(path);
  if (reader.IsOpen() == false) {
    LOG_ERROR("Open file:[" << path << "] error!");
    return false;
  }

  Deserialize deserialize(&reader);
  if (deserialize >> *router == false) {
    return false;
  }

  return true;
}

bool Checkpoint::SaveModelMetaData(const std::string& path,
                                   const ModelMetaData& model_mdata) {
  // Readable information.
  configor::json j;
  j["name"] = model_mdata.name;
  j["optim_type"] = OptimTypeName(model_mdata.optim_type);
  j["optim_conf"] = model_mdata.optim_conf;

  configor::json tables_j = configor::json::array({});
  for (const auto& [table_name, table_mdata] : model_mdata.table_mdatas) {
    configor::json t_j;

    t_j["id"] = table_mdata.id;
    t_j["name"] = table_mdata.name;
    t_j["table_type"] =
        (table_mdata.table_type == TableType::kDense ? "Dense" : "Sparse");
    t_j["element_type"] = table_mdata.element_type.Name();

    if (table_mdata.table_type == TableType::kDense) {
      t_j["shape"] = table_mdata.shape.dims();
    } else if (table_mdata.table_type == TableType::kSparse) {
      t_j["dimension"] = table_mdata.dimension;
      t_j["init_type"] = InitializerTypeName(table_mdata.init_type);
      t_j["init_conf"] = table_mdata.init_conf;
    }

    tables_j.push_back(std::move(t_j));
  }

  j["tables"] = tables_j;

  std::string pretty_str = j.dump(4, ' ');

  // Dump json string to file.
  std::ofstream out_f(path);
  if (!out_f.is_open()) {
    LOG_ERROR("Open file:[" << path << "] error!");
    return false;
  }

  // Write to file.
  out_f << pretty_str << std::endl;
  out_f.close();

  return true;
}

bool Checkpoint::SaveModelMetaDataBinary(const std::string& path,
                                         const ModelMetaData& model_mdata) {
  FileWriter writer(path);
  if (writer.IsOpen() == false) {
    LOG_ERROR("Open file:[" << path << "] error!");
    return false;
  }

  Serialize serialize(&writer);

  if ((serialize << model_mdata.name) == false ||
      (serialize << model_mdata.optim_type) == false ||
      (serialize << model_mdata.optim_conf) == false) {
    return false;
  }

  uint64_t table_size = model_mdata.table_mdatas.size();
  if ((serialize << table_size) == false) {
    return false;
  }

  for (const auto& [k, v] : model_mdata.table_mdatas) {
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

bool Checkpoint::SaveDenseTable(const std::string& path, DenseTable* table) {
  FileWriter writer(path);
  if (writer.IsOpen() == false) {
    LOG_ERROR("Open file:[" << path << "] error!");
    return false;
  }

  Serialize serialize(&writer);

  if ((serialize << table->type()) == false ||
      (serialize << table->id()) == false ||
      (serialize << table->name()) == false ||
      (serialize << table->val()) == false) {
    return false;
  }

  return true;
}

bool Checkpoint::SaveSparseTable(const std::string& path, SparseTable* table) {
  FileWriter writer(path);
  if (writer.IsOpen() == false) {
    LOG_ERROR("Open file:[" << path << "] error!");
    return false;
  }

  Serialize serialize(&writer);

  if ((serialize << table->type()) == false ||
      (serialize << table->id()) == false ||
      (serialize << table->name()) == false ||
      (serialize << table->dimension()) == false ||
      (serialize << table->element_type()) == false ||
      (serialize << table->initializer()->type()) == false ||
      (serialize << table->initializer()->conf()) == false) {
    return false;
  }

  auto* parallel_vals = table->mutable_vals();
  uint64_t slot_count = parallel_vals->slot_count();

  // Serialize SlotCount.
  // ParallelSipList is special so we need serialize slot by slot.
  if (serialize << slot_count == false) {
    return false;
  }

  // Below is thread-safe.
  for (uint64_t slot = 0; slot < slot_count; ++slot) {
    auto h = parallel_vals->SharedSkipListHandler(slot);
    uint64_t skip_list_size = h.skip_list.Size();

    if (serialize << skip_list_size == false) {
      return false;
    }

    for (auto it = h.skip_list.Begin(); it.Valid(); it.Next()) {
      if (serialize << it.key() == false || serialize << it.value() == false) {
        return false;
      }
    }
  }

  return true;
}

bool Checkpoint::LoadModelMetaDataBinary(const std::string& path,
                                         ModelMetaData* model_mdata) {
  FileReader reader(path);
  if (reader.IsOpen() == false) {
    LOG_ERROR("Open file:[" << path << "] error!");
    return false;
  }

  Deserialize deserialize(&reader);
  if ((deserialize >> model_mdata->name) == false ||
      (deserialize >> model_mdata->optim_type) == false ||
      (deserialize >> model_mdata->optim_conf) == false) {
    return false;
  }

  uint64_t table_size;
  if ((deserialize >> table_size) == false) {
    return false;
  }

  model_mdata->table_mdatas.reserve(table_size);

  for (uint64_t i = 0; i < table_size; ++i) {
    uint64_t table_id;
    TableMetaData table_mdata;

    if ((deserialize >> table_id) == false) {
      return false;
    }

    if ((deserialize >> table_mdata.id) == false ||
        (deserialize >> table_mdata.name) == false ||
        (deserialize >> table_mdata.table_type) == false ||
        (deserialize >> table_mdata.element_type) == false ||
        (deserialize >> table_mdata.shape) == false ||
        (deserialize >> table_mdata.dimension) == false ||
        (deserialize >> table_mdata.init_type) == false ||
        (deserialize >> table_mdata.init_conf) == false) {
      return false;
    }

    model_mdata->table_mdatas.emplace(table_id, std::move(table_mdata));
  }

  return true;
}

std::unique_ptr<DenseTable> Checkpoint::LoadDenseTable(
    const std::string& path) {
  FileReader reader(path);
  if (reader.IsOpen() == false) {
    LOG_ERROR("Open file:[" << path << "] error!");
    return nullptr;
  }

  TableType table_type;
  uint64_t table_id;
  std::string table_name;
  Value table_val;

  Deserialize deserialize(&reader);
  if ((deserialize >> table_type) == false ||
      (deserialize >> table_id) == false ||
      (deserialize >> table_name) == false ||
      (deserialize >> table_val) == false) {
    return nullptr;
  }

  if (table_type != TableType::kDense) {
    return nullptr;
  }

  return std::make_unique<DenseTable>(table_id, table_name, table_val);
}

bool Checkpoint::LoadSparseTable(const std::string& path, SparseTable* table,
                                 uint64_t node_id, const Router& router) {
  FileReader reader(path);
  if (reader.IsOpen() == false) {
    LOG_ERROR("Open file:[" << path << "] error!");
    return false;
  }

  Deserialize deserialize(&reader);

  TableType type;
  uint64_t table_id;
  std::string table_name;
  int64_t dimension;
  ElementType element_type;
  InitializerType init_type;
  std::unordered_map<std::string, std::string> init_conf;

  if ((deserialize >> type) == false || (deserialize >> table_id) == false ||
      (deserialize >> table_name) == false ||
      (deserialize >> dimension) == false ||
      (deserialize >> element_type) == false ||
      (deserialize >> init_type) == false ||
      (deserialize >> init_conf) == false) {
    return false;
  }

  if (type != TableType::kSparse || table_id != table->id() ||
      table_name != table->name() || dimension != table->dimension() ||
      element_type != table->element_type() ||
      init_type != table->initializer()->type()) {
    return false;
  }

  auto* parallel_vals = table->mutable_vals();

  uint64_t slot_count;
  if ((deserialize >> slot_count) == false) {
    return false;
  }

  for (uint64_t slot = 0; slot < slot_count; ++slot) {
    uint64_t size;
    if (deserialize >> size == false) {
      return false;
    }

    for (uint64_t i = 0; i < size; ++i) {
      uint64_t sparse_id;
      Value value;
      if ((deserialize >> sparse_id) == false ||
          (deserialize >> value) == false) {
        return false;
      }

      if (node_id == router.Hit(utils::Hash(table_id, sparse_id))) {
        parallel_vals->Insert(sparse_id, value);
      }
    }
  }

  return true;
}

}  // namespace io
}  // namespace kraken
