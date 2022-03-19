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

int32_t SparseTable::Pull(const std::vector<uint64_t>& sparse_ids,
                          std::vector<Tensor>* vals) {
  // At here the id maybe not exist in this table. If not exist create a new
  // one.
  vals->resize(sparse_ids.size());

  std::unordered_map<size_t, std::vector<size_t>> slot_idx_map;
  slot_idx_map.reserve(vals_.slot_count());

  for (size_t i = 0; i < sparse_ids.size(); ++i) {
    slot_idx_map[vals_.HitSlot(sparse_ids[i])].emplace_back(i);
  }

  for (const auto& [slot, v] : slot_idx_map) {
    // Lock the slot.
    auto h = vals_.UniqueSkipListHandler(slot);

    for (auto i : v) {
      uint64_t sparse_id = sparse_ids[i];

      auto it = h.skip_list.Find(sparse_id);

      if (it.Valid()) {
        // Exist.
        (*vals)[i] = it.value().val.Clone();
      } else {
        // Not exist create a new embedding.
        Tensor t = Tensor::Dense({dimension_}, element_type_);
        initializer_->Initialize(&t);

        // for result.
        (*vals)[i] = t.Clone();

        Value v;
        v.val = t;

        h.skip_list.Insert(sparse_id, v);
      }
    }
  }

  return ErrorCode::kSuccess;
}

int32_t SparseTable::Push(Optim* optim, const std::vector<uint64_t>& sparse_ids,
                          const std::vector<Tensor>& grads, float lr) {
  assert(sparse_ids.size() == grads.size());

  std::unordered_map<size_t, std::vector<size_t>> slot_idx_map;
  slot_idx_map.reserve(vals_.slot_count());

  for (size_t i = 0; i < sparse_ids.size(); ++i) {
    slot_idx_map[vals_.HitSlot(sparse_ids[i])].emplace_back(i);
  }

  for (const auto& [slot, v] : slot_idx_map) {
    // Lock the slot.
    auto h = vals_.UniqueSkipListHandler(slot);

    for (auto i : v) {
      uint64_t sparse_id = sparse_ids[i];

      auto it = h.skip_list.Find(sparse_id);
      if (it.Valid() == false) {
        return ErrorCode::kSparseIdNotExistError;
      }

      int32_t error_code = optim->Update(grads[i], lr, &(it.value()));
      if (error_code != ErrorCode::kSuccess) {
        return error_code;
      }
    }
  }

  return ErrorCode::kSuccess;
}

}  // namespace kraken
