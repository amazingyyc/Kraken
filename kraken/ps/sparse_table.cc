#include "ps/sparse_table.h"

namespace kraken {

SparseTable::SparseTable(Optim* optim, uint64_t id, const std::string& name,
                         int64_t dimension, ElementType etype, size_t s_count)
    : Table(optim, id, name),
      dimension_(dimension),
      etype_(etype),
      s_count_(s_count),
      lockers_(s_count),
      vars_(s_count) {
}

int64_t SparseTable::Dimension() const {
  return dimension_;
}

ElementType SparseTable::EType() const {
  return etype_;
}

bool SparseTable::Push(const std::vector<IndepVector>& grads, float lr) {
  for (const auto& grad : grads) {
    int64_t id = grad.indice;
    if (id < 0) {
      return false;
    }

    int64_t vidx = (size_t)id % s_count_;

    SpinLockerHandler handler(lockers_[vidx]);

    auto it = vars_[vidx].find(id);
    if (it == vars_[vidx].end()) {
      // Cannot find just return.
      return false;
    }

    if (optim_->Update(grad.val, lr, &(it->second)) == false) {
      return false;
    }
  }

  return true;
}

bool SparseTable::Pull(const std::vector<int64_t>& indices,
                       std::vector<Tensor>* vars) {
  // At here the id maybe not exist in this table. If not exist create a new
  // vector.
  vars->resize(indices.size());

  for (size_t i = 0; i < indices.size(); ++i) {
    int64_t id = indices[i];
    if (id < 0) {
      return false;
    }

    int64_t vidx = (size_t)id % s_count_;

    SpinLockerHandler handler(lockers_[vidx]);

    auto it = vars_[vidx].find(id);
    if (it == vars_[vidx].end()) {
      vars_[vidx].emplace(id, Tensor::create({dimension_}, etype_));

      // (TODO) initialize tensor.
    }

    // Must clone or will not thread-safe.
    (*vars)[i] = vars_[vidx][id].clone();
  }

  return true;
}

}  // namespace kraken
