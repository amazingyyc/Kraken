#include "ps/model.h"

#include "ps/dense_table.h"
#include "ps/sparse_table.h"

namespace kraken {

const size_t Model::kSparseTableSCount = 4;

Model::Model(uint64_t id, const std::string& name,
             std::unique_ptr<Optim>&& optim)
    : id_(id), name_(name), optim_(std::move(optim)), table_id_gen_(0) {
}

bool Model::RegisterDenseTable(const std::string& name, const Shape& shape,
                               ElementType etype, uint64_t* table_id) {
  std::unique_lock<std::shared_mutex> lock(mu_);

  auto it = table_name_id_.find(name);
  if (it != table_name_id_.end()) {
    *table_id = it->second;

    const Tensor& exit_var = ((DenseTable*)(tables_[*table_id].get()))->Var();

    if (exit_var.shape() != shape || exit_var.element_type() != etype) {
      return false;
    }

    return true;
  }

  *table_id = table_id_gen_++;
  table_name_id_[name] = *table_id;

  std::unique_ptr<DenseTable> table(
      new DenseTable(optim_.get(), *table_id, name, shape, etype));

  tables_.emplace(*table_id, std::move(table));

  return true;
}

bool Model::RegisterSparseTable(const std::string& name, int64_t dimension,
                                ElementType etype, uint64_t* table_id) {
  std::unique_lock<std::shared_mutex> lock(mu_);

  if (dimension <= 0) {
    return false;
  }

  auto it = table_name_id_.find(name);
  if (it != table_name_id_.end()) {
    *table_id = it->second;

    SparseTable* table = (SparseTable*)(tables_[*table_id].get());
    if (table->Dimension() != dimension || table->EType() != etype) {
      return false;
    }

    return true;
  }

  *table_id = table_id_gen_++;
  table_name_id_[name] = *table_id;

  std::unique_ptr<SparseTable> table(new SparseTable(
      optim_.get(), *table_id, name, dimension, etype, kSparseTableSCount));

  tables_.emplace(*table_id, std::move(table));

  return true;
}

}  // namespace kraken
