#include "io/saver.h"

#include <filesystem>
#include <fstream>

#include "common/deserialize.h"
#include "common/exception.h"
#include "common/log.h"
#include "common/serialize.h"
#include "configor/json.hpp"
#include "io/file_reader.h"
#include "io/file_writer.h"
#include "ps/optim/adagrad.h"
#include "ps/optim/adam.h"
#include "ps/optim/rmsprop.h"
#include "ps/optim/sgd.h"

namespace kraken {
namespace io {

const std::string Saver::kModelInfoName = "model.json";
const std::string Saver::kModelBinaryName = "model.binary";
const std::string Saver::kDenseTableSuffix = ".dense";
const std::string Saver::kSparseTableSuffix = ".sparse";
const std::string Saver::kPartitionFolderPrefix = "partition_";

bool Saver::IsDirExist(const std::string& dir) const {
  std::filesystem::path path(dir);

  std::error_code error_code;
  auto status = std::filesystem::status(dir, error_code);

  if (error_code) {
    return false;
  }

  if (std::filesystem::exists(status) == false) {
    return false;
  }

  return std::filesystem::is_directory(status);
}

bool Saver::IsFileExist(const std::string& p) const {
  std::filesystem::path path(p);

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

bool Saver::CreateDir(const std::string& dir, bool exist_delete) const {
  std::filesystem::path path(dir);

  if (IsDirExist(dir)) {
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

bool Saver::GetSortedPartitionFolder(
    const std::string& dir, std::vector<std::string>* partition_folders) const {
  std::vector<std::pair<std::string, size_t>> partitions;

  std::filesystem::path path(dir);
  for (auto const& entry : std::filesystem::directory_iterator{path}) {
    if (entry.is_directory()) {
      auto filename = entry.path().filename().string();
      if (filename.rfind(kPartitionFolderPrefix, 0) != 0) {
        continue;
      }

      auto num_s =
          filename.substr(kPartitionFolderPrefix.size(),
                          filename.size() - kPartitionFolderPrefix.size());

      try {
        size_t num = std::stoull(num_s);

        partitions.emplace_back(std::make_pair(filename, num));
      } catch (...) {
        return false;
      }
    }
  }

  std::sort(partitions.begin(), partitions.end(),
            [](const std::pair<std::string, size_t>& p1,
               const std::pair<std::string, size_t>& p2) -> bool {
              return p1.second < p2.second;
            });

  partition_folders->clear();
  partition_folders->reserve(partitions.size());

  for (const auto& i : partitions) {
    partition_folders->emplace_back(i.first);
  }

  return true;
}

std::string Saver::GenModelInfoPath(const std::string& dir) const {
  std::filesystem::path path(dir);
  path /= kModelInfoName;

  return path.string();
}

std::string Saver::GenModelBinaryPath(const std::string& dir) const {
  std::filesystem::path path(dir);
  path /= kModelBinaryName;

  return path.string();
}

bool Saver::Save(const std::string& model_info_path,
                 const std::string& model_binary_path,
                 ModelManager::Model& model) const {
  // We will dump 2 file. one is readable another is binary serialize.
  {
    // Readable.
    configor::json j;
    j["id"] = model.id;
    j["name"] = model.name;
    j["optim_type"] = (int32_t)model.optim_type;
    j["optim_conf"] = model.optim_conf;

    configor::json tables_j = configor::json::array({});
    for (auto& table : model.tables_) {
      configor::json t_j;

      t_j["id"] = table.second.id;
      t_j["name"] = table.second.name;
      t_j["table_type"] = (int32_t)table.second.table_type;
      t_j["element_type"] = (int32_t)table.second.element_type.dtype;

      if (table.second.table_type == TableType::kDense) {
        t_j["shape"] = table.second.shape.dims();
      } else if (table.second.table_type == TableType::kSparse) {
        t_j["dimension"] = table.second.dimension;
        t_j["init_type"] = (int32_t)table.second.init_type;
        t_j["init_conf"] = table.second.init_conf;
      }

      tables_j.push_back(std::move(t_j));
    }

    j["tables"] = tables_j;

    std::string pretty_str = j.dump(4, ' ');

    // Dump json string to file.
    std::ofstream out_f(model_info_path.c_str());
    if (!out_f.is_open()) {
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
      return false;
    }

    Serialize serialize(&writer);

    if ((serialize << model.id) == false ||
        (serialize << model.name) == false ||
        (serialize << model.optim_type) == false ||
        (serialize << model.optim_conf) == false) {
      return false;
    }

    if ((serialize << model.table_id_map_) == false) {
      return false;
    }

    uint64_t table_size = model.tables_.size();
    if ((serialize << table_size) == false) {
      return false;
    }

    for (auto& t : model.tables_) {
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

bool Saver::Save(const std::string& dir, DenseTable* table) const {
  std::filesystem::path path(dir);
  path /= (table->name() + kDenseTableSuffix);

  // Open a file to serialize the DenseTable.
  FileWriter writer(path.string());
  if (writer.IsOpen() == false) {
    return false;
  }

  Serialize serialize(&writer);

  // table type | id | name | value (val | bag (state|state_i))
  if ((serialize << table->type()) == false ||
      (serialize << table->id()) == false ||
      (serialize << table->name()) == false) {
    return false;
  }

  // Value.
  if ((serialize << table->val_.val) == false ||
      (serialize << table->val_.bag.state) == false ||
      (serialize << table->val_.bag.state_i) == false) {
    return false;
  }

  return true;
}

bool Saver::Save(const std::string& dir, SparseTable* table) const {
  std::filesystem::path path(dir);
  path /= (table->name() + kSparseTableSuffix);

  FileWriter writer(path.string());
  if (writer.IsOpen() == false) {
    return false;
  }

  Serialize serialize(&writer);

  // At here we just read the variable of this table.
  // The table's variable never change. So it safe.
  if ((serialize << table->type()) == false ||
      (serialize << table->id()) == false ||
      (serialize << table->name()) == false) {
    return false;
  }

  // dimension | element type.
  if ((serialize << table->dimension()) == false ||
      (serialize << table->etype()) == false) {
    return false;
  }

  // Initializer.
  if ((serialize << table->initializer()->type()) == false ||
      (serialize << table->initializer()->conf()) == false) {
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
      // key
      if ((serialize << it->first) == false) {
        return false;
      }

      // Value.
      if ((serialize << it->second.val) == false ||
          (serialize << it->second.bag.state) == false ||
          (serialize << it->second.bag.state_i) == false) {
        return false;
      }
    }

    // The lt will release lock automatically.
  }

  return true;
}

bool Saver::Load(const std::string& model_binary_path,
                 ModelManager::Model* model) const {
  // Binary.
  FileReader reader(model_binary_path);
  if (reader.IsOpen() == false) {
    return false;
  }

  Deserialize deserialize(&reader);

  if ((deserialize >> model->id) == false ||
      (deserialize >> model->name) == false ||
      (deserialize >> model->optim_type) == false ||
      (deserialize >> model->optim_conf) == false) {
    return false;
  }

  if ((deserialize >> model->table_id_map_) == false) {
    return false;
  }

  uint64_t table_size;
  if ((deserialize >> table_size) == false) {
    return false;
  }

  model->tables_.reserve(table_size);

  for (uint64_t i = 0; i < table_size; ++i) {
    uint64_t table_id;
    ModelManager::Table table;

    if ((deserialize >> table_id) == false) {
      return false;
    }

    if ((deserialize >> table.id) == false ||
        (deserialize >> table.name) == false ||
        (deserialize >> table.table_type) == false ||
        (deserialize >> table.element_type) == false ||
        (deserialize >> table.shape) == false ||
        (deserialize >> table.dimension) == false ||
        (deserialize >> table.init_type) == false ||
        (deserialize >> table.init_conf) == false) {
      return false;
    }

    model->tables_.emplace(table_id, std::move(table));
  }

  return true;
}

bool Saver::Load(const std::string& path, DenseTable* table) const {
  FileReader reader(path);
  if (reader.IsOpen() == false) {
    return false;
  }

  Deserialize deserialize(&reader);

  TableType type;
  uint64_t id;
  std::string name;
  Table::Value val;

  if ((deserialize >> type) == false || (deserialize >> id) == false ||
      (deserialize >> name) == false) {
    return false;
  }

  if ((deserialize >> val.val) == false ||
      (deserialize >> val.bag.state) == false ||
      (deserialize >> val.bag.state_i) == false) {
    return false;
  }

  if (type != TableType::kDense || id != table->id_ || name == table->name_) {
    return false;
  }

  table->val_ = val;

  return true;
}

bool Saver::Load(const std::vector<std::string>& paths, size_t shard_id,
                 uint64_t model_id, const ConsistentHasher& hasher,
                 SparseTable* table) {
  // The sparse table maybe store in mulit-files so we need load all of them and
  // check the hash whehter it belong this shard.
  size_t size = paths.size();

  std::vector<bool> has_error(size, false);

  uint64_t table_id = table->id_;

#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    do {
      FileReader reader(paths[i]);
      if (reader.IsOpen() == false) {
        has_error[i] = true;
        break;
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
          (deserialize >> name) == false ||
          (deserialize >> dimension) == false ||
          (deserialize >> etype) == false ||
          (deserialize >> init_type) == false ||
          (deserialize >> init_conf) == false) {
        has_error[i] = true;
        break;
      }

      if (type != TableType::kSparse || id != table->id_ ||
          name != table->name_ || dimension != table->dimension_ ||
          etype == table->etype_ || init_type != table->initializer_->type()) {
        has_error[i] = true;
        break;
      }

      // Parse embedding and insert to hash-table it's thread-safe.
      uint64_t val_size;
      if ((deserialize >> val_size) == false) {
        has_error[i] = true;
        break;
      }

      for (uint64_t j = 0; j < val_size; ++j) {
        int64_t sparse_id;
        Table::Value val;

        if ((deserialize >> sparse_id) == false) {
          has_error[i] = true;
          break;
        }

        if ((deserialize >> val.val) == false ||
            (deserialize >> val.bag.state) == false ||
            (deserialize >> val.bag.state_i) == false) {
          has_error[i] = true;
          break;
        }

        // If this val belong this shard insert it.
        if (shard_id == hasher(model_id, table_id, sparse_id)) {
          table->vals_.insert(sparse_id, val);
        }
      }
    } while (false);
  }

  for (size_t i = 0; i < has_error.size(); ++i) {
    if (has_error[i]) {
      return false;
    }
  }

  return true;
}

bool Saver::Save(Ps* ps, const std::string& dir, uint64_t model_id) {
  // Shard 0 is leader.
  size_t shard_id = ps->shard_id();
  bool is_leader = (shard_id == 0);

  // Try to create folder.
  if (CreateDir(dir, false) == false) {
    LOG_ERROR("Create dir:" << dir << " error.");
    return false;
  }

  if (is_leader) {
    std::shared_lock<std::shared_mutex> _(ps->model_manager_.mu_);

    // If this is a leader we need dump model info.
    auto it = ps->model_manager_.models_.find(model_id);
    if (it == ps->model_manager_.models_.end()) {
      LOG_ERROR("Model id:" << model_id << " not exist in ModelManager.");
      return false;
    }

    // Generate path.
    auto model_info_path = GenModelInfoPath(dir);
    auto model_binary_path = GenModelBinaryPath(dir);

    if (Save(model_info_path, model_binary_path,
             ps->model_manager_.models_[model_id]) == false) {
      return false;
    }
  }

  // We need create a partition folder.
  std::filesystem::path partition_dir(dir);
  partition_dir /= (kPartitionFolderPrefix + std::to_string(shard_id));

  auto partition_dir_s = partition_dir.string();
  if (CreateDir(partition_dir_s, true) == false) {
    LOG_ERROR("Create dir:" << partition_dir_s << " error.");
    return false;
  }

  // Dump model's table.
  std::shared_lock<std::shared_mutex> lock_ps(ps->mu_);
  auto it = ps->models_.find(model_id);
  if (it == ps->models_.end()) {
    LOG_ERROR("Model id:" << model_id << " not exist.");
    return false;
  }

  Model* model = it->second.get();

  std::shared_lock<std::shared_mutex> lock_model(model->mu_);
  for (auto it = model->tables_.begin(); it != model->tables_.end(); ++it) {
    Table* table = it->second.get();
    if (table->type_ == TableType::kDense) {
      DenseTable* dense_table = (DenseTable*)table;
      std::shared_lock<std::shared_mutex> lock_table(dense_table->mu_);
      if (Save(partition_dir_s, dense_table) == false) {
        LOG_ERROR("Dump DenseTable:" << table->name_ << " error.");
        return false;
      }
    } else if (table->type_ == TableType::kSparse) {
      if (Save(partition_dir_s, (SparseTable*)table) == false) {
        LOG_ERROR("Dump SparseTable:" << table->name_ << " error.");
        return false;
      }
    }
  }

  return true;
}

bool Saver::Load(Ps* ps, const std::string& dir) {
  if (IsDirExist(dir) == false) {
    LOG_ERROR("Dir:" << dir << " not exist.");
    return false;
  }

  std::vector<std::string> partition_folders;
  if (GetSortedPartitionFolder(dir, &partition_folders) == false ||
      partition_folders.empty()) {
    LOG_ERROR("Get partition folder error, from dir:" << dir);
    return false;
  }

  size_t shard_num = ps->shard_num_;
  size_t shard_id = ps->shard_id_;

  ConsistentHasher haser(shard_num);
  ConsistentHasher old_haser(partition_folders.size());

  std::vector<size_t> include_partitions;
  {
    // Get the shared boundary.
    auto boundary = haser.ShardBoundary(shard_id);

    // boundary: [lower, upper)
    uint64_t lower = boundary.first;
    uint64_t upper = boundary.second;

    for (size_t i = 0; i < old_haser.bucket_count(); ++i) {
      auto cur_b = old_haser.ShardBoundary(i);

      if (cur_b.first < upper && cur_b.second > lower) {
        include_partitions.emplace_back(i);
      }
    }
  }

  // Try to load ModelManager::Model.
  auto model_binary_path = GenModelBinaryPath(dir);
  ModelManager::Model m_model;
  if (Load(model_binary_path, &m_model) == false) {
    LOG_ERROR(
        "Load model binary error, model_binary_path:" << model_binary_path);
    return false;
  }

  // Create model.
  std::unique_ptr<Model> model;
  {
    std::unique_ptr<Optim> optim;
    if (m_model.optim_type == OptimType::kAdagrad) {
      optim.reset(new Adagrad(m_model.optim_conf));
    } else if (m_model.optim_type == OptimType::kAdam) {
      optim.reset(new Adam(m_model.optim_conf));
    } else if (m_model.optim_type == OptimType::kRMSprop) {
      optim.reset(new RMSprop(m_model.optim_conf));
    } else if (m_model.optim_type == OptimType::kSGD) {
      optim.reset(new SGD(m_model.optim_conf));
    } else {
      LOG_ERROR("Unsupported optim type:" << (int32_t)m_model.optim_type);
      return false;
    }

    model.reset(new Model(m_model.id, m_model.name, std::move(optim)));
  }

  // Load table.
}

}  // namespace io
}  // namespace kraken
