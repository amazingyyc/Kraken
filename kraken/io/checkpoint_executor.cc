#include "io/checkpoint_executor.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>

#include "common/log.h"
#include "ps/dense_table.h"
#include "ps/ps.h"
#include "ps/sparse_table.h"
#include "ps/table.h"

namespace kraken {
namespace io {

CheckpointExecutor::CheckpointExecutor(const std::string& save_dir,
                                       size_t max_save_count)
    : save_dir_(save_dir), max_save_count_(max_save_count), stop_(false) {
  worker_ = std::thread(&CheckpointExecutor::Run, this);
}

bool CheckpointExecutor::GetSortedShardFolders(
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

bool CheckpointExecutor::GetLatestCheckPointFolderPath(
    const std::string& shard_dir, std::string* path) {
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

bool CheckpointExecutor::Save(Ps* ps, const std::string& save_dir,
                              uint64_t model_id) {
  LOG_INFO("Try to save model id:" << model_id << " to dir:" << save_dir);

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

    if (SaveModelInfo(model_info_path, model_info) == false ||
        SaveModelBinaryInfo(model_binary_path, model_info) == false) {
      LOG_ERROR("Save ModelInfo error!");
      return false;
    }
  }

  // Try to dump tables.
  std::shared_lock<std::shared_mutex> lock_model(model->mu_);

  for (auto& [k, v] : model->tables_) {
    Table* table = v.get();

    std::filesystem::path table_path = shard_path;

    if (table->type_ == TableType::kDense) {
      table_path /= table->name_ + kDenseTableSuffix;

      DenseTable* dense_table = (DenseTable*)table;
      std::shared_lock<std::shared_mutex> lock_table(dense_table->mu_);

      if (SaveDenseTable(table_path.string(), dense_table) == false) {
        LOG_ERROR("Dump DenseTable:" << table->name_ << " error.");
        return false;
      }
    } else if (table->type_ == TableType::kSparse) {
      table_path /= table->name_ + kSparseTableSuffix;

      if (SaveSparseTable(table_path.string(), (SparseTable*)table) == false) {
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

void CheckpointExecutor::Run() {
  while (true) {
    std::unique_lock<std::mutex> lock(mu_);

    if (task_que_.empty() == false) {
      CheckPointTask task = std::move(task_que_.front());
      task_que_.pop();

      bool success = Save(task.ps, save_dir_, task.model_id);

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

void CheckpointExecutor::Stop() {
  stop_.store(false);

  if (worker_.joinable()) {
    worker_.join();
  }
}

void CheckpointExecutor::Save(Ps* ps, uint64_t model_id,
                              std::function<void(uint64_t, bool)>&& done) {
  if (save_dir_.empty()) {
    LOG_ERROR("Save dir is empty, save checkpoint error!");

    if (done) {
      done(model_id, false);
    }

    return;
  }

  CheckPointTask task;
  task.ps = ps;
  task.model_id = model_id;
  task.done = std::move(done);

  {
    std::unique_lock<std::mutex> lock(mu_);
    task_que_.emplace(std::move(task));
  }

  cond_var_.notify_one();
}

bool CheckpointExecutor::Load(Ps* ps, const std::string& model_dir) {
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
  if (LoadModelBinaryInfo(model_binary_path, &model_info) == false) {
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

  std::unordered_map<std::string, uint64_t> table_name_id;
  for (const auto& [k, v] : model_info.table_infos) {
    table_name_id[v.name] = v.id;
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

      if (table_name_id.find(table_name) == table_name_id.end()) {
        LOG_ERROR("Dense table:" << table_name << " not recognized.");
        return false;
      }

      uint64_t table_id = table_name_id[table_name];

      if (hasher(model_info.id, table_id) != shard_id) {
        // Not belong this shard.
        continue;
      }

      const auto& table_info = model_info.table_infos[table_id];

      // this val is dummy, will replaced when deserialize.
      auto val = Tensor::Dense(table_info.shape, table_info.element_type);

      std::unique_ptr<DenseTable> table(
          new DenseTable(model->optim_.get(), table_id, table_name, val));

      if (LoadDenseTable(d_path.string(), table.get()) == false) {
        LOG_ERROR("Load DenseTabel:" << table_name << " error!");
        return false;
      }

      // Check.
      if (table->type() != table_info.table_type ||
          table->id() != table_info.id || table->name() != table_info.name ||
          table->val().shape() != table_info.shape ||
          table->val().element_type() != table_info.element_type) {
        return false;
      }

      // Insert to model.
      model->tables_.emplace(table_id, std::move(table));
    }

    // SparseTable
    for (const auto& s_path : sparse_paths) {
      LOG_INFO("Try to load SparseTable from:" << s_path);

      auto table_name = s_path.stem().string();

      if (table_name_id.find(table_name) == table_name_id.end()) {
        LOG_ERROR("Sparse table:" << table_name << " not recognized.");
        return false;
      }

      uint64_t table_id = table_name_id[table_name];

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

      if (LoadSparseTable(s_path.string(), shard_id, model_info.id, hasher,
                          table) == false) {
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

}  // namespace io
}  // namespace kraken
