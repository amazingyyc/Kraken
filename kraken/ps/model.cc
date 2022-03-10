// #include "ps/model.h"

// #include "common/error_code.h"
// #include "common/exception.h"
// #include "common/log.h"
// #include "ps/dense_table.h"
// #include "ps/initializer/normal_initializer.h"
// #include "ps/sparse_table.h"

// namespace kraken {

// Model::Model(uint64_t id, const std::string& name,
//              std::unique_ptr<Optim>&& optim)
//     : id_(id), name_(name), optim_(std::move(optim)) {
// }

// uint16_t Model::id() const {
//   return id_;
// }

// const std::string& Model::name() const {
//   return name_;
// }

// Optim* Model::optim() const {
//   return optim_.get();
// }

// bool Model::IsTableExist(uint64_t table_id) {
//   std::shared_lock<std::shared_mutex> lock(mu_);

//   return tables_.find(table_id) != tables_.end();
// }

// void Model::TryCreateDenseTable(uint64_t id, std::string name,
//                                 const Table::Value& val) {
//   std::unique_lock<std::shared_mutex> lock(mu_);

//   if (tables_.find(id) == tables_.end()) {
//     std::unique_ptr<DenseTable> table(
//         new DenseTable(optim_.get(), id, name, val));

//     tables_.emplace(id, std::move(table));
//   }
// }

// void Model::TryCreateSparseTable(uint64_t id, const std::string& name,
//                                  int64_t dimension, ElementType etype,
//                                  std::unique_ptr<Initializer>&& initializer) {
//   std::unique_lock<std::shared_mutex> lock(mu_);

//   if (tables_.find(id) == tables_.end()) {
//     ARGUMENT_CHECK(dimension > 0, "SparseTable dimension must > 0.");

//     std::unique_ptr<SparseTable> table(new SparseTable(
//         optim_.get(), id, name, dimension, etype, std::move(initializer)));

//     tables_.emplace(id, std::move(table));
//   }
// }

// void Model::TryInsertIntoSparseTable(uint64_t id,
//                                      const std::vector<uint64_t>& sparse_ids,
//                                      const std::vector<Table::Value>& vals) {
//   std::shared_lock<std::shared_mutex> lock(mu_);

//   auto it = tables_.find(id);
//   if (it != tables_.end()) {
//     ARGUMENT_CHECK(it->second->type() == TableType::kSparse,
//                    "Not a SparseTable!");

//     SparseTable* table = (SparseTable*)(it->second.get());
//     table->TryInsert(sparse_ids, vals);
//   }
// }

// int32_t Model::CreateDenseTable(uint64_t id, std::string name,
//                                 const Tensor& val) {
//   std::unique_lock<std::shared_mutex> lock(mu_);

//   auto it = tables_.find(id);
//   if (it != tables_.end()) {
//     return ErrorCode::kTableAlreadyCreateError;
//   }

//   std::unique_ptr<DenseTable> table(
//       new DenseTable(optim_.get(), id, name, val));

//   tables_.emplace(id, std::move(table));

//   return ErrorCode::kSuccess;
// }

// int32_t Model::CreateSparseTable(uint64_t id, const std::string& name,
//                                  int64_t dimension, ElementType etype,
//                                  std::unique_ptr<Initializer>&& initializer) {
//   std::unique_lock<std::shared_mutex> lock(mu_);

//   auto it = tables_.find(id);
//   if (it != tables_.end()) {
//     return ErrorCode::kTableAlreadyCreateError;
//   }

//   if (dimension <= 0) {
//     return ErrorCode::kSparseDimensionError;
//   }

//   std::unique_ptr<SparseTable> table(new SparseTable(
//       optim_.get(), id, name, dimension, etype, std::move(initializer)));

//   tables_.emplace(id, std::move(table));

//   return ErrorCode::kSuccess;
// }

// int32_t Model::RegisterDenseTable(uint64_t id, const std::string& name,
//                                   const Tensor& val) {
//   std::unique_lock<std::shared_mutex> lock(mu_);

//   auto it = tables_.find(id);
//   if (it != tables_.end()) {
//     if (it->second->name() != name || it->second->type() != TableType::kDense) {
//       return ErrorCode::kSparseTableUnCompatibleError;
//     }

//     const Tensor& exit_val = ((DenseTable*)(it->second.get()))->val();
//     if (exit_val.shape() != val.shape() ||
//         exit_val.element_type() != val.element_type()) {
//       return ErrorCode::kSparseTableUnCompatibleError;
//     }

//     LOG_INFO("Register DenseTable:"
//              << name << ", shape: " << exit_val.shape().Str()
//              << ", ElementType: " << exit_val.element_type().Name()
//              << " already exist, id: " << id);

//     return ErrorCode::kSuccess;
//   }

//   std::unique_ptr<DenseTable> table(
//       new DenseTable(optim_.get(), id, name, val));

//   tables_.emplace(id, std::move(table));

//   LOG_INFO("Registered DenseTable:"
//            << name << ", id:" << id << ", shape:" << val.shape().Str()
//            << ", ElementType:" << val.element_type().Name());

//   return ErrorCode::kSuccess;
// }

// int32_t Model::RegisterSparseTable(uint64_t id, const std::string& name,
//                                    int64_t dimension, ElementType etype,
//                                    std::unique_ptr<Initializer>&& initializer) {
//   std::unique_lock<std::shared_mutex> lock(mu_);

//   if (dimension <= 0) {
//     return ErrorCode::kSparseDimensionError;
//   }

//   auto it = tables_.find(id);
//   if (it != tables_.end()) {
//     if (it->second->name() != name ||
//         it->second->type() != TableType::kSparse) {
//       return ErrorCode::kSparseTableUnCompatibleError;
//     }

//     SparseTable* table = (SparseTable*)(it->second.get());
//     if (table->dimension() != dimension || table->element_type() != etype) {
//       return ErrorCode::kSparseTableUnCompatibleError;
//     }

//     LOG_INFO("Registered SparseTable:" << name << ", dimension:" << dimension
//                                        << ", ElementType:" << etype.Name()
//                                        << ", alrady exist, id:" << id);

//     return ErrorCode::kSuccess;
//   }

//   std::unique_ptr<SparseTable> table(new SparseTable(
//       optim_.get(), id, name, dimension, etype, std::move(initializer)));

//   tables_.emplace(id, std::move(table));

//   LOG_INFO("Register SparseTable:" << name << ", id:" << id
//                                    << ", dimension:" << dimension
//                                    << ", ElementType:" << etype.Name());

//   return ErrorCode::kSuccess;
// }

// int32_t Model::PushDenseTable(uint64_t table_id, const Tensor& grad, float lr) {
//   std::shared_lock<std::shared_mutex> lock(mu_);

//   auto it = tables_.find(table_id);
//   if (it == tables_.end()) {
//     return ErrorCode::kUnRegisterTableError;
//   }

//   if (it->second->type() != TableType::kDense) {
//     return ErrorCode::kSparseTableUnCompatibleError;
//   }

//   return it->second->Push(grad, lr);
// }

// int32_t Model::PullDenseTable(uint64_t table_id, Tensor* val) {
//   std::shared_lock<std::shared_mutex> lock(mu_);

//   auto it = tables_.find(table_id);
//   if (it == tables_.end()) {
//     return ErrorCode::kUnRegisterTableError;
//   }

//   if (it->second->type() != TableType::kDense) {
//     return ErrorCode::kSparseTableUnCompatibleError;
//   }

//   return it->second->Pull(val);
// }

// int32_t Model::CombinePullDenseTable(const std::vector<uint64_t>& table_ids,
//                                      std::vector<Tensor>* vals) {
//   vals->reserve(table_ids.size());

//   std::shared_lock<std::shared_mutex> lock(mu_);

//   for (size_t i = 0; i < table_ids.size(); ++i) {
//     auto it = tables_.find(table_ids[i]);
//     if (it == tables_.end()) {
//       return ErrorCode::kUnRegisterTableError;
//     }

//     if (it->second->type() != TableType::kDense) {
//       return ErrorCode::kSparseTableUnCompatibleError;
//     }

//     Tensor val;
//     int32_t ecode = it->second->Pull(&val);
//     if (ecode != ErrorCode::kSuccess) {
//       return ecode;
//     }

//     vals->emplace_back(val);
//   }

//   return ErrorCode::kSuccess;
// }

// int32_t Model::PushPullDenseTable(uint64_t table_id, const Tensor& grad,
//                                   float lr, Tensor* val) {
//   std::shared_lock<std::shared_mutex> lock(mu_);

//   auto it = tables_.find(table_id);
//   if (it == tables_.end()) {
//     return ErrorCode::kUnRegisterTableError;
//   }

//   if (it->second->type() != TableType::kDense) {
//     return ErrorCode::kSparseTableUnCompatibleError;
//   }

//   return it->second->PushPull(grad, lr, val);
// }

// int32_t Model::PushSparseTable(uint64_t table_id,
//                                const std::vector<uint64_t>& indices,
//                                const std::vector<Tensor>& grads, float lr) {
//   std::shared_lock<std::shared_mutex> lock(mu_);

//   auto it = tables_.find(table_id);
//   if (it == tables_.end()) {
//     return ErrorCode::kUnRegisterTableError;
//   }

//   if (it->second->type() != TableType::kSparse) {
//     return ErrorCode::kSparseTableUnCompatibleError;
//   }

//   return it->second->Push(indices, grads, lr);
// }

// int32_t Model::PullSparseTable(uint64_t table_id,
//                                const std::vector<uint64_t>& indices,
//                                std::vector<Tensor>* vals) {
//   std::shared_lock<std::shared_mutex> lock(mu_);

//   auto it = tables_.find(table_id);
//   if (it == tables_.end()) {
//     return ErrorCode::kUnRegisterTableError;
//   }

//   if (it->second->type() != TableType::kSparse) {
//     return ErrorCode::kSparseTableUnCompatibleError;
//   }

//   return it->second->Pull(indices, vals);
// }

// int32_t Model::CombinePullSparseTable(
//     const std::unordered_map<uint64_t, std::vector<uint64_t>>& indices,
//     std::unordered_map<uint64_t, std::vector<Tensor>>* vals) {
//   std::shared_lock<std::shared_mutex> lock(mu_);

//   vals->clear();
//   vals->reserve(indices.size());

//   for (const auto& [table_id, indice] : indices) {
//     auto it = tables_.find(table_id);
//     if (it == tables_.end()) {
//       return ErrorCode::kUnRegisterTableError;
//     }

//     if (it->second->type() != TableType::kSparse) {
//       return ErrorCode::kSparseTableUnCompatibleError;
//     }

//     std::vector<Tensor> val;

//     int32_t ecode = it->second->Pull(indice, &val);
//     if (ecode != ErrorCode::kSuccess) {
//       return ecode;
//     }

//     vals->emplace(table_id, std::move(val));
//   }

//   return ErrorCode::kSuccess;
// }

// }  // namespace kraken
