#include "io/check_point.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>

#include "common/deserialize.h"
#include "common/exception.h"
#include "common/log.h"
#include "common/serialize.h"
#include "configor/json.hpp"
#include "io/file_reader.h"
#include "io/file_writer.h"
#include "ps/dense_table.h"
#include "ps/info.h"
#include "ps/initializer/constant_initializer.h"
#include "ps/initializer/normal_initializer.h"
#include "ps/initializer/uniform_initializer.h"
#include "ps/initializer/xavier_normal_initializer.h"
#include "ps/initializer/xavier_uniform_initializer.h"
#include "ps/optim/adagrad.h"
#include "ps/optim/adam.h"
#include "ps/optim/rmsprop.h"
#include "ps/optim/sgd.h"
#include "ps/ps.h"
#include "ps/sparse_table.h"
#include "ps/table.h"

namespace kraken {
namespace io {

const std::string CheckPoint::kModelInfoName = "model.json";
const std::string CheckPoint::kModelBinaryName = "model.binary";
const std::string CheckPoint::kDenseTableSuffix = ".dense";
const std::string CheckPoint::kSparseTableSuffix = ".sparse";
const std::string CheckPoint::kShardFolderPrefix = "shard_";

CheckPoint::CheckPoint(Ps* ps, const std::string& save_dir,
                       size_t max_save_count)
    : ps_(ps),
      save_dir_(save_dir),
      max_save_count_(max_save_count),
      stop_(false) {
  woker_ = std::thread(&CheckPoint::Run, this);

  LOG_INFO("Save dir:" << save_dir_ << ", max_save_count:" << max_save_count_);
}

const char* CheckPoint::OptimTypeName(OptimType type) const {
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

const char* CheckPoint::InitializerTypeName(InitializerType type) const {
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

bool CheckPoint::IsDirExist(const std::string& dir) const {
  std::filesystem::path path(dir);

  return IsDirExist(path);
}

bool CheckPoint::IsDirExist(const std::filesystem::path& path) const {
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

bool CheckPoint::IsFileExist(const std::string& p) const {
  std::filesystem::path path(p);

  return IsFileExist(path);
}

bool CheckPoint::IsFileExist(const std::filesystem::path& path) const {
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

bool CheckPoint::DeleteDir(const std::string& dir) const {
  std::filesystem::path path(dir);

  return DeleteDir(path);
}

bool CheckPoint::DeleteDir(const std::filesystem::path& path) const {
  if (IsDirExist(path)) {
    std::error_code error_code;
    std::filesystem::remove_all(path, error_code);

    if (error_code) {
      return false;
    }
  }

  return true;
}

bool CheckPoint::CreateDir(const std::string& dir, bool exist_delete) const {
  std::filesystem::path path(dir);

  return CreateDir(path, exist_delete);
}

bool CheckPoint::CreateDir(const std::filesystem::path& path,
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

bool CheckPoint::GetSortedShardFolders(
    const std::string& dir, std::vector<std::string>* shard_folders) const {
  std::vector<std::pair<std::string, size_t>> shards;

  std::filesystem::path path(dir);
  for (auto const& entry : std::filesystem::directory_iterator{path}) {
    if (entry.is_directory()) {
      auto filename = entry.path().filename().string();
      if (filename.rfind(kShardFolderPrefix, 0) != 0) {
        continue;
      }

      auto num_s = filename.substr(kShardFolderPrefix.size(),
                                   filename.size() - kShardFolderPrefix.size());

      try {
        size_t num = std::stoull(num_s);

        shards.emplace_back(std::make_pair(entry.path().string(), num));
      } catch (...) {
        return false;
      }
    }
  }

  std::sort(shards.begin(), shards.end(),
            [](const std::pair<std::string, size_t>& p1,
               const std::pair<std::string, size_t>& p2) -> bool {
              return p1.second < p2.second;
            });

  shard_folders->clear();
  shard_folders->reserve(shards.size());

  for (const auto& i : shards) {
    shard_folders->emplace_back(i.first);
  }

  return true;
}

bool CheckPoint::GetLatestCheckPointFolderPath(const std::string& shard_dir,
                                               std::string* path) {
  std::filesystem::path shard_path(shard_dir);

  std::vector<std::pair<std::string, uint64_t>> folders;
  for (auto const& entry : std::filesystem::directory_iterator{shard_path}) {
    if (entry.is_directory()) {
      auto filename = entry.path().filename().string();

      try {
        uint64_t num = std::stoull(filename);

        folders.emplace_back(std::make_pair(entry.path().string(), num));
      } catch (...) {
        return false;
      }
    }
  }

  if (folders.empty()) {
    return false;
  }

  std::sort(folders.begin(), folders.end(),
            [](const std::pair<std::string, uint64_t>& p1,
               const std::pair<std::string, uint64_t>& p2) -> bool {
              return p1.second > p2.second;
            });

  *path = folders.front().first;

  return true;
}

bool CheckPoint::GetDenseTablePaths(const std::string& dir,
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

bool CheckPoint::GetSparseTablePaths(
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

std::string CheckPoint::GenModelDirByTime() const {
  auto now = std::chrono::system_clock::now();
  time_t tt = std::chrono::system_clock::to_time_t(now);
  tm utc_tm = *gmtime(&tt);

  // Name format: year-month-day-hour-min-sec.
  return std::to_string(utc_tm.tm_year + 1900) + "-" +
         std::to_string(utc_tm.tm_mon + 1) + "-" +
         std::to_string(utc_tm.tm_mday) + "-" + std::to_string(utc_tm.tm_hour) +
         "-" + std::to_string(utc_tm.tm_min) + "-" +
         std::to_string(utc_tm.tm_sec);
}

std::string CheckPoint::GenModelInfoPath(const std::string& dir) const {
  std::filesystem::path path(dir);
  path /= kModelInfoName;

  return path.string();
}

std::string CheckPoint::GenModelBinaryPath(const std::string& dir) const {
  std::filesystem::path path(dir);
  path /= kModelBinaryName;

  return path.string();
}

bool CheckPoint::Save(const std::string& model_info_path,
                      const std::string& model_binary_path,
                      const ModelInfo& model_info) const {
  // We will dump 2 file. one is readable another is binary serialize.
  {
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
    std::ofstream out_f(model_info_path);
    if (!out_f.is_open()) {
      LOG_ERROR("Open file:" << model_info_path << " error!");
      return false;
    }

    // Write to file.
    out_f << pretty_str << std::endl;
    out_f.close();
  }

  {
    // Binary.
    FileWriter writer(model_binary_path);
    if (writer.IsOpen() == false) {
      LOG_ERROR("Open file:" << model_binary_path << " error!");
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

    for (auto& t : model_info.table_infos) {
      // table id.
      if ((serialize << t.first) == false) {
        return false;
      }

      if ((serialize << t.second.id) == false ||
          (serialize << t.second.name) == false ||
          (serialize << t.second.table_type) == false ||
          (serialize << t.second.element_type) == false ||
          (serialize << t.second.shape) == false ||
          (serialize << t.second.dimension) == false ||
          (serialize << t.second.init_type) == false ||
          (serialize << t.second.init_conf) == false) {
        return false;
      }
    }
  }

  return true;
}

bool CheckPoint::Save(const std::string& dir, DenseTable* table) const {
  std::filesystem::path path(dir);
  path /= (table->name() + kDenseTableSuffix);

  // Open a file to serialize the DenseTable.
  FileWriter writer(path.string());
  if (writer.IsOpen() == false) {
    LOG_ERROR("Open file:" << path.string() << " error!");
    return false;
  }

  Serialize serialize(&writer);

  // table type | id | name | value
  if ((serialize << table->type_) == false ||
      (serialize << table->id_) == false ||
      (serialize << table->name_) == false ||
      (serialize << table->val_) == false) {
    return false;
  }

  return true;
}

bool CheckPoint::Save(const std::string& dir, SparseTable* table) const {
  std::filesystem::path path(dir);
  path /= (table->name() + kSparseTableSuffix);

  FileWriter writer(path.string());
  if (writer.IsOpen() == false) {
    LOG_ERROR("Open file:" << path.string() << " error!");
    return false;
  }

  Serialize serialize(&writer);

  // At here we just read the variable of this table.
  // The table's variable never change. So it safe.
  if ((serialize << table->type_) == false ||
      (serialize << table->id_) == false ||
      (serialize << table->name_) == false) {
    return false;
  }

  // dimension | element type.
  if ((serialize << table->dimension_) == false ||
      (serialize << table->element_type_) == false) {
    return false;
  }

  // Initializer.
  if ((serialize << table->initializer_->type()) == false ||
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

    // The lt will release lock automatically.
  }

  return true;
}

bool CheckPoint::Load(const std::string& model_binary_path,
                      ModelInfo* model_info) const {
  // Binary.
  FileReader reader(model_binary_path);
  if (reader.IsOpen() == false) {
    LOG_ERROR("Open file:" << model_binary_path << " error!");
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

bool CheckPoint::Load(const std::string& path, DenseTable* table) const {
  FileReader reader(path);
  if (reader.IsOpen() == false) {
    LOG_ERROR("Open file:" << path << " error!");
    return false;
  }

  Deserialize deserialize(&reader);

  TableType type;
  uint64_t id;
  std::string name;
  Table::Value val;

  if ((deserialize >> type) == false || (deserialize >> id) == false ||
      (deserialize >> name) == false || (deserialize >> val) == false) {
    return false;
  }

  if (type != TableType::kDense || id != table->id_ || name != table->name_) {
    return false;
  }

  table->val_ = val;

  return true;
}

bool CheckPoint::Load(const std::string& path, size_t shard_id,
                      uint64_t model_id, const ConsistentHasher& hasher,
                      SparseTable* table) {
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

  // Parse embedding and insert to hash-table it's thread-safe.
  uint64_t val_size;
  if ((deserialize >> val_size) == false) {
    return false;
  }

  for (uint64_t i = 0; i < val_size; ++i) {
    int64_t sparse_id;
    Table::Value val;

    if ((deserialize >> sparse_id) == false || (deserialize >> val) == false) {
      return false;
    }

    // If this val belong this shard insert it.
    if (shard_id == hasher(model_id, id, sparse_id)) {
      table->vals_.insert(sparse_id, val);
    }
  }

  return true;
}

bool CheckPoint::Save(Ps* ps, const std::string& save_dir, uint64_t model_id) {
  LOG_INFO("Try to save model:" << model_id << " to dir:" << save_dir);

  if (model_save_infos_[model_id].save_paths.size() >= max_save_count_) {
    const auto& delete_path = model_save_infos_[model_id].save_paths.front();

    if (DeleteDir(delete_path) == false) {
      LOG_ERROR("Delete dir:" << delete_path << " error!");
      return false;
    }

    model_save_infos_[model_id].save_paths.pop_front();
  }

  // try to get model name.
  std::shared_lock<std::shared_mutex> lock_ps(ps->mu_);

  // Shard 0 is leader.
  size_t shard_id = ps->shard_id();

  if (ps->model_infos_.find(model_id) == ps->model_infos_.end() ||
      ps->models_.find(model_id) == ps->models_.end()) {
    LOG_ERROR("Model id:" << model_id << " not exist.");
    return false;
  }

  const auto& model_info = ps->model_infos_[model_id];
  Model* model = ps->models_[model_id].get();

  // Create model dump dir.
  std::filesystem::path shard_path(save_dir);

  // (TODO) the model name maybe error when as a folder name.
  shard_path /= model_info.name;
  shard_path /= (kShardFolderPrefix + std::to_string(shard_id));
  shard_path /= std::to_string(model_save_infos_[model_id].index++);

  // Try to create folder.
  if (CreateDir(shard_path, false) == false) {
    LOG_ERROR("Create dir:" << shard_path << " error.");
    return false;
  }

  {
    // For every shard we need dump model info.
    auto model_info_path = GenModelInfoPath(shard_path.string());
    auto model_binary_path = GenModelBinaryPath(shard_path.string());

    if (Save(model_info_path, model_binary_path, model_info) == false) {
      return false;
    }
  }

  // Try to dump tables.
  std::shared_lock<std::shared_mutex> lock_model(model->mu_);

  for (auto& [k, v] : model->tables_) {
    Table* table = v.get();

    if (table->type_ == TableType::kDense) {
      DenseTable* dense_table = (DenseTable*)table;
      std::shared_lock<std::shared_mutex> lock_table(dense_table->mu_);

      if (Save(shard_path.string(), dense_table) == false) {
        LOG_ERROR("Dump DenseTable:" << table->name_ << " error.");
        return false;
      }
    } else if (table->type_ == TableType::kSparse) {
      if (Save(shard_path.string(), (SparseTable*)table) == false) {
        LOG_ERROR("Dump SparseTable:" << table->name_ << " error.");
        return false;
      }
    }
  }

  // Store the path.
  model_save_infos_[model_id].save_paths.emplace_back(shard_path.string());

  LOG_INFO("Finish save model:" << model_info.name << " to dir:" << shard_path);

  return true;
}

bool CheckPoint::Load(Ps* ps, const std::string& model_dir) {
  LOG_INFO("Try to load model from:" << model_dir);

  if (IsDirExist(model_dir) == false) {
    LOG_ERROR("Dir:" << model_dir << " not exist.");
    return false;
  }

  // model_dir contain the shard folder and shard folder may contain
  // multi-model.
  std::vector<std::string> shard_folders;
  if (GetSortedShardFolders(model_dir, &shard_folders) == false ||
      shard_folders.empty()) {
    LOG_ERROR("Get shard folder error, from dir:" << model_dir);
    return false;
  }

  size_t shard_num = ps->shard_num_;
  size_t shard_id = ps->shard_id_;

  ConsistentHasher hasher(shard_num);
  ConsistentHasher old_hasher(shard_folders.size());

  std::vector<size_t> include_shards;
  {
    auto boundary = hasher.Boundary(shard_id);

    // boundary: [lower, upper]
    uint64_t lower = boundary.first;
    uint64_t upper = boundary.second;

    for (size_t i = 0; i < old_hasher.bucket_count(); ++i) {
      auto cur_b = old_hasher.Boundary(i);

      if (cur_b.first <= upper && cur_b.second >= lower) {
        include_shards.emplace_back(i);
      }
    }
  }

  // Every shard may has multi-checkpoint so we need the latest one.
  std::vector<std::string> checkpoint_paths;
  for (auto i : include_shards) {
    std::string path;
    if (GetLatestCheckPointFolderPath(shard_folders[i], &path) == false) {
      LOG_ERROR("Get latest checkpoint folder from:" << shard_folders[i]
                                                     << " error!");
      return false;
    }

    LOG_INFO("Get latest checkpoint dir:" << path);

    checkpoint_paths.emplace_back(std::move(path));
  }

  if (checkpoint_paths.empty()) {
    LOG_ERROR("Get latest checkpoints folder error!");
    return false;
  }

  // Try to load ModelInfo.
  auto model_binary_path = GenModelBinaryPath(checkpoint_paths[0]);
  ModelInfo model_info;

  LOG_INFO("Try to load ModelInfo from:" << model_binary_path);
  if (Load(model_binary_path, &model_info) == false) {
    LOG_ERROR(
        "Load model binary error, model_binary_path:" << model_binary_path);
    return false;
  }

  // Create model.
  std::unique_ptr<Model> model;
  {
    std::unique_ptr<Optim> optim =
        Optim::Create(model_info.optim_type, model_info.optim_conf);

    if (optim == nullptr) {
      LOG_ERROR("Unsupported optim type:" << (int32_t)model_info.optim_type);
      return false;
    }

    model.reset(new Model(model_info.id, model_info.name, std::move(optim)));
  }

  // Because the sparse table will load for every shard.
  // So we firstly initialize the SparseTable.
  for (auto& [k, v] : model_info.table_infos) {
    if (v.table_type != TableType::kSparse) {
      continue;
    }

    if (model->tables_.find(v.id) != model->tables_.end()) {
      LOG_ERROR("SparseTable name:" << v.name << ", id:" << v.id
                                    << " already exist.");
      return false;
    }

    std::unique_ptr<Initializer> initializer =
        Initializer::Create(v.init_type, v.init_conf);

    if (initializer == nullptr) {
      LOG_ERROR("Unrecognized initialize type:" << (int32_t)v.init_type);
      return false;
    }

    std::unique_ptr<SparseTable> table(
        new SparseTable(model->optim_.get(), v.id, v.name, v.dimension,
                        v.element_type, std::move(initializer)));

    model->tables_.emplace(v.id, std::move(table));
  }

  std::unordered_map<std::string, uint64_t> table_name_id_map;
  for (const auto& [k, v] : model_info.table_infos) {
    table_name_id_map[v.name] = v.id;
  }

  for (auto& dir : checkpoint_paths) {
    std::vector<std::filesystem::path> dense_paths;
    std::vector<std::filesystem::path> sparse_paths;

    if (GetDenseTablePaths(dir, &dense_paths) == false) {
      LOG_ERROR("Try to get dense table paths error, dir:" << dir);
      return false;
    }

    if (GetSparseTablePaths(dir, &sparse_paths) == false) {
      LOG_ERROR("Try to get sparse table paths error, dir:" << dir);
      return false;
    }

    // DenseTable.
    for (const auto& d_path : dense_paths) {
      LOG_INFO("Try to load DenseTable from:" << d_path);

      // Get dense table name and check whether it belong to this shard.
      auto table_name = d_path.stem().string();

      if (table_name_id_map.find(table_name) == table_name_id_map.end()) {
        LOG_ERROR("Dense table:" << table_name << " not recognized.");
        return false;
      }

      uint64_t table_id = table_name_id_map[table_name];

      if (hasher(model_info.id, table_id) != shard_id) {
        // Not belong this shard.
        continue;
      }

      const auto& table_info = model_info.table_infos[table_id];

      // this val is dummy, will replaced when deserialize.
      auto val = Tensor::Dense(table_info.shape, table_info.element_type);

      std::unique_ptr<DenseTable> table(
          new DenseTable(model->optim_.get(), table_id, table_name, val));

      if (Load(d_path.string(), table.get()) == false) {
        LOG_ERROR("Load DenseTabel:" << table_name << " error!");
        return false;
      }

      // Insert to model.
      model->tables_.emplace(table_id, std::move(table));
    }

    // SparseTable
    for (const auto& s_path : sparse_paths) {
      LOG_INFO("Try to load SparseTable from:" << s_path);

      auto table_name = s_path.stem().string();

      if (table_name_id_map.find(table_name) == table_name_id_map.end()) {
        LOG_ERROR("Sparse table:" << table_name << " not recognized.");
        return false;
      }

      uint64_t table_id = table_name_id_map[table_name];

      if (model->tables_.find(table_id) == model->tables_.end()) {
        LOG_ERROR("Sparse table:" << table_name << " id:" << table_id
                                  << " not recognized.");
        return false;
      }

      if (model->tables_[table_id]->type_ != TableType::kSparse) {
        LOG_ERROR("Sparse table:" << table_name << " id:" << table_id
                                  << " not recognized.");
        return false;
      }

      SparseTable* table = (SparseTable*)(model->tables_[table_id].get());

      if (Load(s_path.string(), shard_id, model_info.id, hasher, table) ==
          false) {
        LOG_ERROR("Load SparseTable:" << table_name << " error!");
        return false;
      }
    }
  }

  LOG_INFO("Load Model:" << model_info.name << " finish.");

  std::unique_lock<std::shared_mutex> lock(ps->mu_);
  ps->model_infos_.emplace(model_info.id, std::move(model_info));
  ps->models_.emplace(model->id_, std::move(model));

  return true;
}

void CheckPoint::Run() {
  while (true) {
    std::unique_lock<std::mutex> lock(mu_);

    if (task_que_.empty() == false) {
      CheckPointTask task = std::move(task_que_.front());
      task_que_.pop();

      bool success = Save(ps_, save_dir_, task.model_id);

      if (task.done) {
        task.done(task.model_id, success);
      }
    } else if (stop_.load()) {
      break;
    } else {
      cond_var_.wait(lock, [this] {
        return this->stop_.load() || false == this->task_que_.empty();
      });
    }
  }
}

void CheckPoint::Stop() {
  stop_.store(false);

  if (woker_.joinable()) {
    woker_.join();
  }
}

void CheckPoint::Save(uint64_t model_id,
                      std::function<void(uint64_t, bool)>&& done) {
  if (save_dir_.empty()) {
    LOG_ERROR("Save dir is empty, save checkpoint error!");

    if (done) {
      done(model_id, false);
    }

    return;
  }

  CheckPointTask task;
  task.model_id = model_id;
  task.done = std::move(done);

  {
    std::unique_lock<std::mutex> lock(mu_);
    task_que_.emplace(std::move(task));
  }

  cond_var_.notify_one();
}

bool CheckPoint::Load(const std::string& model_dir) {
  return Load(ps_, model_dir);
}

}  // namespace io
}  // namespace kraken
