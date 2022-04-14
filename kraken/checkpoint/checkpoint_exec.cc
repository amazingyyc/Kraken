#include "checkpoint/checkpoint_exec.h"

#include "checkpoint/checkpoint.h"
#include "common/log.h"
#include "ps/ps.h"

namespace kraken {
namespace io {

CheckpointExec::CheckpointExec() : saved_dir_(""), max_save_count_(0) {
}

CheckpointExec::CheckpointExec(const std::string& saved_dir,
                               size_t max_save_count)
    : saved_dir_(saved_dir), max_save_count_(max_save_count) {
}

bool CheckpointExec::GetSortedShardSavedDirs(
    const std::string& dir, std::vector<std::string>* saved_dirs) const {
  using TimePoint = std::chrono::time_point<std::chrono::system_clock>;

  if (Checkpoint::IsDirExist(dir) == false) {
    return true;
  }

  std::vector<std::pair<std::string, TimePoint>> shards;

  std::filesystem::path path(dir);
  for (auto const& entry : std::filesystem::directory_iterator{path}) {
    if (entry.is_directory()) {
      auto filename = entry.path().filename().string();

      try {
        auto time_p = Checkpoint::StrToTimePoint(filename);

        shards.emplace_back(std::make_pair(entry.path().string(), time_p));
      } catch (...) {
        return false;
      }
    }
  }

  std::sort(shards.begin(), shards.end(),
            [](const std::pair<std::string, TimePoint>& p1,
               const std::pair<std::string, TimePoint>& p2) -> bool {
              return p1.second < p2.second;
            });

  saved_dirs->clear();
  saved_dirs->reserve(shards.size());

  for (const auto& i : shards) {
    saved_dirs->emplace_back(i.first);
  }

  return true;
}

bool CheckpointExec::GetModelShardDirs(
    const std::string& dir,
    std::unordered_map<uint64_t, std::string>* shard_dirs) {
  if (Checkpoint::IsDirExist(dir) == false) {
    return false;
  }

  std::filesystem::path path(dir);
  for (auto const& entry : std::filesystem::directory_iterator{path}) {
    if (entry.is_directory()) {
      auto filename = entry.path().filename().string();

      if (utils::StartWith(filename, Checkpoint::kShardFolderPrefix)) {
        try {
          uint64_t node_id = std::stoul(filename.substr(
              Checkpoint::kShardFolderPrefix.size(),
              filename.size() - Checkpoint::kShardFolderPrefix.size()));

          shard_dirs->emplace(node_id, entry.path().string());
        } catch (...) {
          return false;
        }
      }
    }
  }

  return true;
}

bool CheckpointExec::GetLatestShardDir(const std::string& dir,
                                       std::string* latest_dir) {
  if (Checkpoint::IsDirExist(dir) == false) {
    return false;
  }

  auto latest_time_p = std::chrono::system_clock::from_time_t(0);

  std::filesystem::path path(dir);
  for (auto const& entry : std::filesystem::directory_iterator{path}) {
    if (entry.is_directory()) {
      auto filename = entry.path().filename().string();
      auto time_p = std::chrono::system_clock::from_time_t(0);

      if (Checkpoint::StrToTimePoint(filename, &time_p) == false) {
        return false;
      }

      if (time_p > latest_time_p) {
        latest_time_p = time_p;
        *latest_dir = entry.path().string();
      }
    }
  }

  return true;
}

bool CheckpointExec::LoadDenseTables(
    Ps* ps, const ModelMetaData& model_mdata,
    const std::vector<std::string>& dirs) const {
  std::unordered_map<std::string, uint64_t> table_name_id;
  for (const auto& [id, table] : model_mdata.table_mdatas) {
    table_name_id[table.name] = id;
  }

  for (const auto& dir : dirs) {
    std::vector<std::filesystem::path> dense_paths;
    if (Checkpoint::GetDenseTablePaths(dir, &dense_paths) == false) {
      LOG_ERROR("Try to get dense table paths error, dir:[" << dir << "]");
      return false;
    }

    for (const auto& d_path : dense_paths) {
      LOG_INFO("Try to load DenseTable from:[" << d_path << "]");

      // Get dense table name and check whether it belong to this shard.
      auto table_name = d_path.stem().string();

      auto it = table_name_id.find(table_name);
      if (it == table_name_id.end()) {
        LOG_ERROR("Dense table:[" << table_name << "] not recognized.");
        return false;
      }

      uint64_t table_id = it->second;
      if (ps->router_.Hit(utils::Hash(table_id)) != ps->node_id_) {
        continue;
      }

      auto table = Checkpoint::LoadDenseTable(d_path.string());
      if (table == nullptr) {
        LOG_ERROR("Load DenseTable from:[" << d_path << "] error!");
        return false;
      }

      // (TODO) check DenseTable attributes.

      std::unique_lock<std::shared_mutex> _(ps->model_mu_);
      ps->tables_.Insert(table_id, std::move(table));
    }
  }

  return true;
}

bool CheckpointExec::LoadSparseTables(
    Ps* ps, const ModelMetaData& model_mdata,
    const std::vector<std::string>& dirs) const {
  std::unordered_map<std::string, uint64_t> table_name_id;
  for (const auto& [id, table] : model_mdata.table_mdatas) {
    table_name_id[table.name] = id;
  }

  for (const auto& dir : dirs) {
    std::vector<std::filesystem::path> sparse_paths;
    if (Checkpoint::GetSparseTablePaths(dir, &sparse_paths) == false) {
      LOG_ERROR("Try to get sparse table paths error, dir:[" << dir << "]");
      return false;
    }

    for (const auto& s_path : sparse_paths) {
      LOG_INFO("Try to load SparseTable from:[" << s_path << "]");

      auto table_name = s_path.stem().string();
      auto it = table_name_id.find(table_name);
      if (it == table_name_id.end()) {
        LOG_ERROR("Sparse table:[" << table_name << "] not recognized.");
        return false;
      }

      uint64_t table_id = it->second;

      std::shared_lock<std::shared_mutex> _(ps->model_mu_);
      auto tit = ps->tables_.Find(table_id);
      if (tit.Valid() == false) {
        LOG_ERROR("Sparse table:[" << table_name << "] id:[" << table_id
                                   << "] not recognized.");
        return false;
      }

      if (tit.value()->type() != TableType::kSparse) {
        LOG_ERROR("Table:[" << table_name << "] is not SparseTable!");
        return false;
      }

      SparseTable* table = (SparseTable*)(tit.value().get());
      if (Checkpoint::LoadSparseTable(s_path.string(), table, ps->node_id_,
                                      ps->router_) == false) {
        LOG_ERROR("Load SparseTable:[" << table_name << "] error!");
        return false;
      }
    }
  }

  return true;
}

bool CheckpointExec::GetLatestModelMetaDataBinaryPath(
    const std::string& load_dir, std::string* path) {
  // Read shard dir for folder.
  std::unordered_map<uint64_t, std::string> shard_dirs;
  if (GetModelShardDirs(load_dir, &shard_dirs) == false) {
    return false;
  }

  if (shard_dirs.empty()) {
    return false;
  }

  std::string latest_shard_dir;
  if (GetLatestShardDir(shard_dirs.begin()->second, &latest_shard_dir) ==
      false) {
    return false;
  }

  *path = Checkpoint::GenModelMetaDataBinaryPath(latest_shard_dir);

  return true;
}

// - save_dir
//   - shard0
//     - time stamp folder name
//     - time stamp folder name
//   - shard1
//     - time stamp folder name
//     - time stamp folder name
bool CheckpointExec::Save(Ps* ps, const ModelMetaData& model_mdata) {
  if (saved_dir_.empty()) {
    LOG_ERROR("Saved dir is empty!");
    return false;
  }

  LOG_INFO("Try save model into:" << saved_dir_);

  // Lock the Ps.
  std::shared_lock<std::shared_mutex> lock_ps(ps->mu_);
  uint64_t node_id = ps->node_id_;
  Router router = ps->router_;

  std::filesystem::path shard_path(saved_dir_);
  shard_path /= Checkpoint::kShardFolderPrefix + std::to_string(node_id);

  std::vector<std::string> shard_saved_dirs;
  if (GetSortedShardSavedDirs(shard_path.string(), &shard_saved_dirs) ==
      false) {
    LOG_ERROR("Get sorted shard dirs from:[" << shard_path << "] error!");
    return false;
  }

  while (shard_saved_dirs.size() >= max_save_count_) {
    // Remove the oldest shard folder.
    if (Checkpoint::DeleteDir(shard_saved_dirs[0]) == false) {
      LOG_ERROR("Delete shard dir:[" << shard_saved_dirs[0] << "] error!");
      return false;
    }

    LOG_INFO("Delete old saved dir:[" << shard_saved_dirs[0] << "]");

    shard_saved_dirs.erase(shard_saved_dirs.begin());
  }

  std::filesystem::path cur_save_path = shard_path;
  cur_save_path /= Checkpoint::FolderNameByTime();

  if (Checkpoint::CreateDir(cur_save_path, true) == false) {
    LOG_ERROR("Delete folder:[" << cur_save_path.string() << "] error");
    return false;
  }

  {
    // Save Router.
    std::string router_path = Checkpoint::GenRouterPath(cur_save_path.string());

    std::string router_binary_path =
        Checkpoint::GenRouterBinaryPath(cur_save_path.string());

    if (Checkpoint::SaveRouter(router_path, router) == false) {
      LOG_ERROR("Save router to:[" << router_path << "] error!");
      return false;
    } else {
      LOG_INFO("Save router to:[" << router_path << "]");
    }

    if (Checkpoint::SaveRouterBinary(router_binary_path, router) == false) {
      LOG_ERROR("Save router binary to:[" << router_binary_path << "] error!");
      return false;
    } else {
      LOG_INFO("Save router binary to:[" << router_binary_path << "]");
    }
  }

  // Save model meta data.
  {
    std::string mdata_path =
        Checkpoint::GenModelMetaDataPath(cur_save_path.string());

    std::string mdata_binary_path =
        Checkpoint::GenModelMetaDataBinaryPath(cur_save_path.string());

    if (Checkpoint::SaveModelMetaData(mdata_path, model_mdata) == false) {
      LOG_ERROR("Save model meta data to:[" << mdata_path << "] error!");
      return false;
    } else {
      LOG_INFO("Save model meta data to:[" << mdata_path << "]");
    }

    if (Checkpoint::SaveModelMetaDataBinary(mdata_binary_path, model_mdata) ==
        false) {
      LOG_ERROR("Save model meta data binary to:[" << mdata_binary_path
                                                   << "] error!");
      return false;
    } else {
      LOG_INFO("Save model meta data binary to:[" << mdata_binary_path << "]");
    }
  }

  // Save Table.
  std::shared_lock<std::shared_mutex> lock_model(ps->model_mu_);
  for (auto it = ps->tables_.Begin(); it.Valid(); it.Next()) {
    Table* table = it.value().get();

    std::filesystem::path table_path = cur_save_path;

    if (table->type() == TableType::kDense) {
      table_path /= table->name() + Checkpoint::kDenseTableSuffix;

      DenseTable* dense_table = (DenseTable*)table;

      auto h = dense_table->shared_handler();
      if (Checkpoint::SaveDenseTable(table_path.string(), dense_table) ==
          false) {
        LOG_ERROR("Save DenseTable:[" << dense_table->name() << "] error");
        return false;
      } else {
        LOG_INFO("Save DenseTable:[" << dense_table->name() << "] to:["
                                     << table_path << "]");
      }
    } else if (table->type() == TableType::kSparse) {
      table_path /= table->name() + Checkpoint::kSparseTableSuffix;

      SparseTable* sparse_table = (SparseTable*)table;

      if (Checkpoint::SaveSparseTable(table_path.string(), sparse_table) ==
          false) {
        LOG_ERROR("Save SparseTable:[" << sparse_table->name() << "] error.");
        return false;
      } else {
        LOG_INFO("Save SparseTable:[" << sparse_table->name() << "] to:["
                                      << table_path << "]");
      }
    }
  }

  LOG_INFO("Save model:[" << ps->model_name_ << "] to:[" << cur_save_path
                          << "] success");

  return true;
}

bool CheckpointExec::Load(Ps* ps, const std::string& load_dir) {
  std::shared_lock<std::shared_mutex> lock_ps(ps->mu_);

  // Read shard dir for folder.
  std::unordered_map<uint64_t, std::string> shard_dirs;
  if (GetModelShardDirs(load_dir, &shard_dirs) == false) {
    LOG_ERROR("Get shard dir from:[" << load_dir << "] error!");
    return false;
  }

  if (shard_dirs.empty()) {
    return false;
  }

  std::unordered_map<uint64_t, std::string> shard_latest_dirs;
  for (const auto& [node_id, shard_dir] : shard_dirs) {
    std::string shard_latest_dir;

    if (GetLatestShardDir(shard_dir, &shard_latest_dir) == false) {
      LOG_ERROR("Get latest shard dir error from:[" << shard_dir << "]");
      return false;
    }

    shard_latest_dirs.emplace(node_id, std::move(shard_latest_dir));
  }

  Router old_router;
  {
    // Load Router.
    auto router_binary_path =
        Checkpoint::GenRouterBinaryPath(shard_latest_dirs.begin()->second);

    if (Checkpoint::LoadRouterBinary(router_binary_path, &old_router) ==
        false) {
      LOG_ERROR("Load router from:[" << router_binary_path << "] error!");
      return false;
    }
  }

  LOG_INFO("The saved model router is:" << old_router.Str());
  LOG_INFO("Current router is:" << ps->router_.Str());

  ModelMetaData model_mdata;
  {
    // Load model meta data.
    auto model_binary_path = Checkpoint::GenModelMetaDataBinaryPath(
        shard_latest_dirs.begin()->second);

    if (Checkpoint::LoadModelMetaDataBinary(model_binary_path, &model_mdata) ==
        false) {
      LOG_ERROR("Load model meta data from:[" << model_binary_path
                                              << "] error!");
      return false;
    }
  }

  std::unordered_set<uint64_t> intersect_nodes;
  {
    auto hash_ranges = ps->router_.NodeHashRanges(ps->node_id_);
    for (const auto& range : hash_ranges) {
      auto node_ids = old_router.IntersectNodes(range);

      intersect_nodes.insert(node_ids.begin(), node_ids.end());
    }
  }

  LOG_INFO("Current node:[" << ps->node_id_ << "] will load saved model from:"
                            << intersect_nodes);

  std::vector<std::string> intersect_dirs;
  for (const auto& node_id : intersect_nodes) {
    auto it = shard_latest_dirs.find(node_id);
    if (it == shard_latest_dirs.end()) {
      LOG_ERROR("Can't find shard dir for old rotuer node:[" << node_id
                                                             << "]!");
      return false;
    }

    intersect_dirs.emplace_back(it->second);
  }

  // Load dense table.
  if (LoadDenseTables(ps, model_mdata, intersect_dirs) == false) {
    LOG_ERROR("Load DenseTables error!");
    return false;
  }

  // Before load sparse table we need create sparse table instance.
  for (const auto& [id, table_mdata] : model_mdata.table_mdatas) {
    if (table_mdata.table_type != TableType::kSparse) {
      continue;
    }

    std::unique_lock<std::shared_mutex> lock_model(ps->model_mu_);
    auto it = ps->tables_.Find(table_mdata.id);
    if (it.Valid()) {
      LOG_ERROR("SparseTable name:" << table_mdata.name << ", id:"
                                    << table_mdata.id << " already exist.");
      return false;
    }

    std::unique_ptr<Initializer> initializer =
        Initializer::Create(table_mdata.init_type, table_mdata.init_conf);

    if (initializer == nullptr) {
      LOG_ERROR("Unrecognized initialize type:" << table_mdata.init_type);
      return false;
    }

    std::unique_ptr<SparseTable> table(
        new SparseTable(table_mdata.id, table_mdata.name, table_mdata.dimension,
                        table_mdata.element_type, std::move(initializer)));

    ps->tables_.Insert(table_mdata.id, std::move(table));
  }

  // Load SparseTable.
  if (LoadSparseTables(ps, model_mdata, intersect_dirs) == false) {
    LOG_ERROR("Load SparseTables error!");
    return false;
  }

  std::unique_lock<std::shared_mutex> lock_model(ps->model_mu_);
  if (ps->model_init_ == true) {
    LOG_ERROR("Model already inited!");
    return false;
  }

  ps->optim_ = Optim::Create(model_mdata.optim_type, model_mdata.optim_conf);
  if (ps->optim_ == nullptr) {
    LOG_ERROR("Unsupported optim, type:["
              << model_mdata.optim_type << "], conf:[" << model_mdata.optim_conf
              << "]");
    return false;
  }

  ps->model_name_ = model_mdata.name;
  ps->model_init_ = true;

  LOG_INFO("Load saved model success!");

  return true;
}

}  // namespace io
}  // namespace kraken
