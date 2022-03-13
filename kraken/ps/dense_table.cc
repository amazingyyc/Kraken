#include "ps/dense_table.h"

#include <mutex>
#include <shared_mutex>

#include "common/error_code.h"

namespace kraken {

DenseTable::DenseTable(uint64_t id, const std::string& name, const Tensor& val)
    : Table(TableType::kDense, id, name) {
  val_.val = val;
}

DenseTable::DenseTable(uint64_t id, const std::string& name, const Value& val)
    : Table(TableType::kDense, id, name), val_(val) {
}

DenseTable::UniqueHandler DenseTable::unique_handler() {
  return DenseTable::UniqueHandler(mu_);
}

DenseTable::SharedHandler DenseTable::shared_handler() {
  return DenseTable::SharedHandler(mu_);
}

const Value& DenseTable::val() const {
  return val_;
}

int32_t DenseTable::Pull(Tensor* val) {
  std::shared_lock<std::shared_mutex> lock(mu_);

  *val = val_.val.Clone();

  return ErrorCode::kSuccess;
}

int32_t DenseTable::Push(Optim* optim, const Tensor& grad, float lr) {
  std::unique_lock<std::shared_mutex> lock(mu_);

  return optim->Update(grad, lr, &val_);
}

int32_t DenseTable::PushPull(const Tensor& grad, float lr, Tensor* val) {
  // std::unique_lock<std::shared_mutex> lock(mu_);

  // int32_t ecode = optim_->Update(grad, lr, &val_.val, &val_.bag);
  // if (ecode != ErrorCode::kSuccess) {
  //   return ecode;
  // }

  // *val = val_.val.Clone();

  return ErrorCode::kSuccess;
}

}  // namespace kraken
