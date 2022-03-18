// #include "watch/watcher.h"

// #include <filesystem>

// #include "common/exception.h"
// #include "common/log.h"
// #include "ps/dense_table.h"
// #include "ps/model.h"
// #include "ps/sparse_table.h"

// namespace kraken {
// namespace watch {

// Watcher::Watcher() : loaded_(false) {
// }

// void Watcher::Load(const std::string& shard_dir) {
//   if (loaded_) {
//     LOG_INFO("Watcher has been loaded!");
//     return;
//   }

//   ARGUMENT_CHECK(IsDirExist(shard_dir),
//                  "Shard dir:" << shard_dir << " not exist!");

//   std::string model_binary_path = GenModelBinaryPath(shard_dir);
//   ARGUMENT_CHECK(
//       LoadModelBinaryInfo(model_binary_path, &model_info_),
//       "Load model binary error, model_binary_path:" << model_binary_path);

//   // Create model.
//   {
//     std::unique_ptr<Optim> optim =
//         Optim::Create(model_info_.optim_type, model_info_.optim_conf);

//     ARGUMENT_CHECK(optim != nullptr, "Unsupported optim type:"
//                                          << (int32_t)model_info_.optim_type);

//     model_.reset(new Model(model_info_.id, model_info_.name, std::move(optim)));
//   }

//   std::unordered_map<std::string, uint64_t> table_name_id;
//   for (const auto& [k, v] : model_info_.table_infos) {
//     table_name_id[v.name] = v.id;
//   }

//   std::vector<std::filesystem::path> dense_paths;
//   std::vector<std::filesystem::path> sparse_paths;

//   ARGUMENT_CHECK(GetDenseTablePaths(shard_dir, &dense_paths),
//                  "Try to get dense table paths error, dir:" << shard_dir);

//   ARGUMENT_CHECK(GetSparseTablePaths(shard_dir, &sparse_paths),
//                  "Try to get sparse table paths error, dir:" << shard_dir);

//   // DenseTable.
//   for (const auto& d_path : dense_paths) {
//     LOG_INFO("Try to load DenseTable from:" << d_path);

//     auto table_name = d_path.stem().string();

//     ARGUMENT_CHECK(table_name_id.find(table_name) != table_name_id.end(),
//                    "DenseTable:" << table_name << " not recognized.");

//     uint64_t table_id = table_name_id[table_name];
//     const auto& table_info = model_info_.table_infos[table_id];

//     ARGUMENT_CHECK(table_info.table_type == TableType::kDense,
//                    "DenseTable:" << table_name << " id:" << table_id
//                                  << " not recognized.");

//     // this val is dummy, will replaced when deserialize.
//     auto val = Tensor::Dense(table_info.shape, table_info.element_type);

//     std::unique_ptr<DenseTable> table(
//         new DenseTable(model_->optim(), table_id, table_name, val));

//     ARGUMENT_CHECK(LoadDenseTable(d_path.string(), table.get()),
//                    "Load DenseTable:" << table_name << " error!");

//     // Check the loaded DenseTable whether "equal" the TableInfo.
//     if (table->type_ != table_info.table_type || table->id_ != table_info.id ||
//         table->name_ != table_info.name ||
//         table->val_.val.shape() != table_info.shape ||
//         table->val_.val.element_type() != table_info.element_type) {
//       RUNTIME_ERROR("DenseTableInfo is not same with the Deserialized.");
//     }

//     // Insert to model.
//     model_->tables_.emplace(table_id, std::move(table));
//   }

//   // SparseTable.
//   for (const auto& s_path : sparse_paths) {
//     LOG_INFO("Try to load SparseTable from:" << s_path);

//     auto table_name = s_path.stem().string();

//     ARGUMENT_CHECK(table_name_id.find(table_name) != table_name_id.end(),
//                    "SparseTable:" << table_name << " not recognized.");

//     uint64_t table_id = table_name_id[table_name];
//     const auto& table_info = model_info_.table_infos[table_id];

//     ARGUMENT_CHECK(table_info.table_type == TableType::kSparse,
//                    "SparseTable:" << table_name << " id:" << table_id
//                                   << " not recognized.");

//     std::unique_ptr<Initializer> initializer =
//         Initializer::Create(table_info.init_type, table_info.init_conf);

//     ARGUMENT_CHECK(
//         initializer != nullptr,
//         "Unrecognized initialize type:" << (int32_t)table_info.init_type);

//     std::unique_ptr<SparseTable> table(new SparseTable(
//         model_->optim_.get(), table_info.id, table_info.name,
//         table_info.dimension, table_info.element_type, std::move(initializer)));

//     ARGUMENT_CHECK(LoadSparseTable(s_path.string(), table.get()),
//                    "Load SparseTable:" << table_name << " error!");

//     if (table->type_ != table_info.table_type || table->id_ != table_info.id ||
//         table->name_ != table_info.name ||
//         table->dimension_ != table_info.dimension ||
//         table->element_type_ != table_info.element_type) {
//       RUNTIME_ERROR("SparseTableInfo is not same with the Deserialized.");
//     }

//     // Insert to model.
//     model_->tables_.emplace(table_id, std::move(table));
//   }

//   loaded_ = true;
// }

// const ModelInfo& Watcher::model_info() const {
//   return model_info_;
// }

// std::vector<TableInfo> Watcher::DenseTableInfos() const {
//   std::vector<TableInfo> table_infos;

//   for (const auto& [k, v] : model_info_.table_infos) {
//     if (v.table_type == TableType::kDense) {
//       table_infos.emplace_back(v);
//     }
//   }

//   return table_infos;
// }

// std::vector<TableInfo> Watcher::ExistDenseTableInfos() const {
//   std::vector<TableInfo> table_infos;

//   for (const auto& [k, v] : model_info_.table_infos) {
//     if (v.table_type == TableType::kDense) {
//       if (model_->tables_.find(v.id) != model_->tables_.end()) {
//         table_infos.emplace_back(v);
//       }
//     }
//   }

//   return table_infos;
// }

// std::vector<TableInfo> Watcher::SparseTableInfos() const {
//   std::vector<TableInfo> table_infos;

//   for (const auto& [k, v] : model_info_.table_infos) {
//     if (v.table_type == TableType::kSparse) {
//       table_infos.emplace_back(v);
//     }
//   }

//   return table_infos;
// }

// std::vector<TableInfo> Watcher::ExistSparseTableInfos() const {
//   std::vector<TableInfo> table_infos;

//   for (const auto& [k, v] : model_info_.table_infos) {
//     if (v.table_type == TableType::kSparse) {
//       if (model_->tables_.find(v.id) != model_->tables_.end()) {
//         table_infos.emplace_back(v);
//       }
//     }
//   }

//   return table_infos;
// }

// std::vector<int64_t> Watcher::ExistSparseTableIds(uint64_t table_id) const {
//   auto it = model_->tables_.find(table_id);

//   ARGUMENT_CHECK(it != model_->tables_.end(),
//                  "table_id:" << table_id << " not exist!");
//   ARGUMENT_CHECK(it->second->type_ == TableType::kSparse,
//                  "table_id:" << table_id << " is not SparseTable!");

//   SparseTable* table = (SparseTable*)it->second.get();

//   auto lt = table->vals_.lock_table();

//   std::vector<int64_t> sparse_ids;
//   sparse_ids.reserve(lt.size());

//   for (auto it = lt.begin(); it != lt.end(); ++it) {
//     sparse_ids.emplace_back(it->first);
//   }

//   return sparse_ids;
// }

// bool Watcher::IsTableExist(uint64_t table_id) const {
//   return model_->tables_.find(table_id) != model_->tables_.end();
// }
// bool Watcher::IsTableExist(const std::string& table_name) const {
//   for (const auto& [k, v] : model_->tables_) {
//     if (v->name_ == table_name) {
//       return true;
//     }
//   }

//   return false;
// }

// bool Watcher::IsDenseTableValExist(uint64_t table_id) {
//   auto it = model_->tables_.find(table_id);
//   if (it == model_->tables_.end()) {
//     return false;
//   }

//   return it->second->type_ == TableType::kDense;
// }

// bool Watcher::IsSparseTableValExist(uint64_t table_id, uint64_t sparse_id) {
//   auto it = model_->tables_.find(table_id);
//   if (it == model_->tables_.end()) {
//     return false;
//   }

//   if (it->second->type_ != TableType::kSparse) {
//     return false;
//   }

//   SparseTable* table = (SparseTable*)it->second.get();

//   return table->vals_.contains(sparse_id);
// }

// Tensor Watcher::DenseTableVal(uint64_t table_id) const {
//   auto it = model_->tables_.find(table_id);

//   ARGUMENT_CHECK(it != model_->tables_.end(),
//                  "table_id:" << table_id << " not exist!");
//   ARGUMENT_CHECK(it->second->type_ == TableType::kDense,
//                  "table_id:" << table_id << " is not DenseTable!");

//   DenseTable* table = (DenseTable*)it->second.get();

//   return table->val_.val;
// }

// Tensor Watcher::SparseTableVal(uint64_t table_id, uint64_t sparse_id) const {
//   auto it = model_->tables_.find(table_id);

//   ARGUMENT_CHECK(it != model_->tables_.end(),
//                  "table_id:" << table_id << " not exist!");
//   ARGUMENT_CHECK(it->second->type_ == TableType::kSparse,
//                  "table_id:" << table_id << " is not SparseTable!");

//   SparseTable* table = (SparseTable*)it->second.get();

//   Tensor val;

//   bool exist = table->vals_.find_fn(
//       sparse_id, [&val](const Table::Value& v) { val = v.val; });

//   ARGUMENT_CHECK(exist, "sparse_id:" << sparse_id << " not exist!");

//   return val;
// }

// }  // namespace watch
// }  // namespace kraken
