#include "ps/sparse_table.h"

#include "common/exception.h"

namespace kraken {

SparseTable::SparseTable(uint64_t id, const std::string& name,
                         int64_t dimension, ElementType element_type,
                         std::unique_ptr<Initializer>&& initializer)
    : Table(TableType::kSparse, id, name),
      dimension_(dimension),
      element_type_(element_type),
      initializer_(std::move(initializer)),
      vals_() {
}

int64_t SparseTable::dimension() const {
  return dimension_;
}

ElementType SparseTable::element_type() const {
  return element_type_;
}

Initializer* SparseTable::initializer() const {
  return initializer_.get();
}

ParallelSkipList<uint64_t, Value>* SparseTable::vals() {
  return &vals_;
}

int32_t SparseTable::Push(const std::vector<uint64_t>& indices,
                          const std::vector<Tensor>& grads, float lr) {
  // if (indices.size() != grads.size()) {
  //   return ErrorCode::kPushSparseTableParameterError;
  // }

  // Optim* l_optim = optim_;
  // for (size_t i = 0; i < indices.size(); ++i) {
  //   uint64_t sparse_id = indices[i];

  //   int32_t ecode;
  //   bool exist =
  //       vals_.update_fn(sparse_id, [l_optim, &grads, i, lr, &ecode](Value& v)
  //       {
  //         ecode = l_optim->Update(grads[i], lr, &v.val, &v.bag);
  //       });

  //   if (exist == false) {
  //     return ErrorCode::kSparseTableIdNotExistError;
  //   }

  //   if (ecode != ErrorCode::kSuccess) {
  //     return ecode;
  //   }
  // }

  return ErrorCode::kSuccess;
}

int32_t SparseTable::Pull(const std::vector<uint64_t>& indices,
                          std::vector<Tensor>* vals) {
  // // At here the id maybe not exist in this table. If not exist create a new
  // // vector.
  // vals->resize(indices.size());

  // for (size_t i = 0; i < indices.size(); ++i) {
  //   uint64_t sparse_id = indices[i];

  //   bool exist = vals_.find_fn(
  //       sparse_id, [vals, i](const Value& v) { (*vals)[i] = v.val.Clone();
  //       });

  //   if (exist == false) {
  //     // can not find, insert a new one.
  //     Tensor t = Tensor::Dense({dimension_}, element_type_);
  //     initializer_->Initialize(&t);

  //     // for result.
  //     (*vals)[i] = t.Clone();

  //     Value v;
  //     v.val = t;
  //     v.bag = Bag();

  //     vals_.insert(sparse_id, v);
  //   }
  // }

  return ErrorCode::kSuccess;
}

}  // namespace kraken
