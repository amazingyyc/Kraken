#pragma once

#include <unordered_map>
#include <vector>

#include "common/element_type.h"
#include "common/error_code.h"
#include "common/spin_locker.h"
#include "common/tensor.h"
#include "parallel_hashmap/parallel_hashmap/phmap.h"
#include "ps/optim/optim.h"
#include "ps/table.h"

namespace kraken {

class SparseTable : public Table {
private:
  struct Value {
    Tensor val;
    Bag bag;
  };

  // For sparse table this must be a matrix. shape is [N, dimension].
  // We donnot assign the N, so it means the matrix's row canbe increase
  // automatically.
  int64_t dimension_;
  ElementType etype_;

  phmap::parallel_flat_hash_map<
      int64_t, Value, phmap::priv::hash_default_hash<int64_t>,
      phmap::priv::hash_default_eq<int64_t>,
      std::allocator<std::pair<const int64_t, Value>>, 4, std::shared_mutex>
      vals_;

public:
  SparseTable(Optim* optim, uint64_t id, const std::string& name,
              int64_t dimension, ElementType etype);

public:
  int64_t dimension() const;

  ElementType etype() const;

  int32_t Push(const std::vector<int64_t>& indices,
               const std::vector<Tensor>& grads, float lr) override;

  int32_t Pull(const std::vector<int64_t>& indices,
               std::vector<Tensor>* vals) override;
};

}  // namespace kraken
