#include "ps/table.h"

#include "common/exception.h"

namespace kraken {

Table::Table(Optim* optim, uint64_t id, const std::string& name)
    : optim_(optim), id_(id), name_(name) {
}

uint64_t Table::Id() const {
  return id_;
}

const std::string Table::Name() const {
  return name_;
}

bool Table::Push(const Tensor& grad, float lr) {
  RUNTIME_ERROR("The subclass must implement Push function.")
}

bool Table::Pull(Tensor* var) {
  RUNTIME_ERROR("The subclass must implement Pull function.")
}

bool Table::Push(const std::vector<IndepVector>& grads, float lr) {
  RUNTIME_ERROR("The subclass must implement Push function.")
}

bool Table::Pull(const std::vector<int64_t>& indices,
                 std::vector<Tensor>* vars) {
  RUNTIME_ERROR("The subclass must implement Pull function.")
}

}  // namespace kraken
