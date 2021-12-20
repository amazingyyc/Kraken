#include "ps/dense_table.h"

#include <mutex>
#include <shared_mutex>

namespace kraken {

DenseTable::DenseTable(Optim* optim, uint64_t id, const std::string& name,
                       const Shape& shape, ElementType etype)
    : Table(optim, id, name) {
  var_ = Tensor::create(shape, etype);

  // (TODO) initialize the tensor.
}

const Tensor& DenseTable::Var() const {
  return var_;
}

bool DenseTable::Push(const Tensor& grad, float lr) {
  std::unique_lock<std::shared_mutex> lock(mu_);

  return optim_->Update(grad, lr, &var_);
}

bool DenseTable::Pull(Tensor* var) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  *var = var_.clone();

  return true;
}

}  // namespace kraken
