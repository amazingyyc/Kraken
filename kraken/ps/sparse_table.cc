#include "ps/sparse_table.h"

namespace kraken {

SparseTable::SparseTable(Optim* optim, uint64_t id, const std::string& name,
                         int64_t dimension, ElementType etype)
    : Table(TableType::kSparse, optim, id, name),
      dimension_(dimension),
      etype_(etype) {
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

  Optim* l_optim = optim_;

  for (size_t i = 0; i < indices.size(); ++i) {
    int64_t sparse_id = indices[i];
    if (sparse_id < 0) {
      return ErrorCode::kSparseTableIdError;
    }

    int32_t ecode;
    bool exist =
        vals_.modify_if(sparse_id, [l_optim, &grads, i, lr, &ecode](Value& v) {
          ecode = l_optim->Update(grads[i], lr, &v.val, &v.bag);
        });

    if (exist == false) {
      return ErrorCode::kSparseTableIdNotExistError;
    }

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
    int64_t sparse_id = indices[i];
    if (sparse_id < 0) {
      return ErrorCode::kSparseTableIdError;
    }

    bool exist = vals_.if_contains(
        sparse_id, [vals, i](const Value& v) { (*vals)[i] = v.val.Clone(); });

    if (exist == false) {
      // can not find, insert a new one.
      // (TODO) other initialize method.
      Tensor t = Tensor::Create({dimension_}, etype_);
      t.Normal(0, 1.0);

      // for result.
      (*vals)[i] = t.Clone();

      Value v;
      v.val = t;
      v.bag = Bag();

      vals_.emplace(sparse_id, v);
    }
  }

  return ErrorCode::kSuccess;
}

}  // namespace kraken
