#include "ps/sparse_table.h"

namespace kraken {

SparseTable::SparseTable(Optim* optim, uint64_t id, const std::string& name,
                         int64_t dimension, ElementType etype, size_t s_count)
    : Table(TableType::kSparse, optim, id, name),
      dimension_(dimension),
      etype_(etype),
      s_count_(s_count),
      lockers_(s_count),
      vals_(s_count),
      bags_(s_count) {
}

int64_t SparseTable::dimension() const {
  return dimension_;
}

ElementType SparseTable::etype() const {
  return etype_;
}

int32_t SparseTable::Push(const std::vector<int64_t>& indices,
                          const std::vector<Tensor>& grads, float lr) {
  if (indices.size() != grads.size()) {
    return ErrorCode::kPushSparseTableParameterError;
  }

  for (size_t i = 0; i < indices.size(); ++i) {
    int64_t id = indices[i];
    if (id < 0) {
      return ErrorCode::kSparseTableIdError;
    }

    int64_t vidx = (size_t)id % s_count_;

    SpinLockerHandler handler(lockers_[vidx]);

    auto it = vals_[vidx].find(id);
    if (it == vals_[vidx].end()) {
      return ErrorCode::kSparseTableIdNotExistError;
    }

    // Try to find Bag.
    auto bit = bags_[vidx].find(id);
    if (bit == bags_[vidx].end()) {
      return ErrorCode::kSparseTableIdNotExistError;
    }

    int32_t ecode = optim_->Update(grads[i], lr, &(it->second), &(bit->second));
    if (ecode != ErrorCode::kSuccess) {
      return ecode;
    }
  }

  return ErrorCode::kSuccess;
}

int32_t SparseTable::Pull(const std::vector<int64_t>& indices,
                          std::vector<Tensor>* vals) {
  // At here the id maybe not exist in this table. If not exist create a new
  // vector.
  vals->resize(indices.size());

  for (size_t i = 0; i < indices.size(); ++i) {
    int64_t id = indices[i];
    if (id < 0) {
      return ErrorCode::kSparseTableIdError;
    }

    int64_t vidx = (size_t)id % s_count_;

    SpinLockerHandler handler(lockers_[vidx]);

    auto it = vals_[vidx].find(id);
    if (it == vals_[vidx].end()) {
      Tensor t = Tensor::Create({dimension_}, etype_);

      // (TODO) other initialize method.
      t.Norm(0, 1.0);

      vals_[vidx].emplace(id, t);
      bags_[vidx].emplace(id, Bag());
    }

    // Must clone or will not thread-safe.
    (*vals)[i] = vals_[vidx][id].Clone();
  }

  return ErrorCode::kSuccess;
}

}  // namespace kraken
