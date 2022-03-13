#include "ps/table.h"

#include "common/error_code.h"
#include "common/exception.h"

namespace kraken {

Table::Table(TableType type, uint64_t id, const std::string& name)
    : type_(type), id_(id), name_(name) {
}

TableType Table::type() const {
  return type_;
}

uint64_t Table::id() const {
  return id_;
}

const std::string& Table::name() const {
  return name_;
}

int32_t Table::Pull(Tensor* var) {
  return ErrorCode::kInterfaceUnImplementError;
}

int32_t Table::Push(Optim* optim, const Tensor& grad, float lr) {
  return ErrorCode::kInterfaceUnImplementError;
}

int32_t Table::PushPull(const Tensor& grad, float lr, Tensor* val) {
  return ErrorCode::kInterfaceUnImplementError;
}

int32_t Table::Pull(const std::vector<uint64_t>& indices,
                    std::vector<Tensor>* vals) {
  return ErrorCode::kInterfaceUnImplementError;
}
int32_t Table::Push(const std::vector<uint64_t>& indices,
                    const std::vector<Tensor>& grads, float lr) {
  return ErrorCode::kInterfaceUnImplementError;
}

}  // namespace kraken
