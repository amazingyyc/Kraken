#pragma once

#include <unordered_map>
#include <vector>

#include "common/error_code.h"
#include "common/parallel_skip_list.h"
#include "common/spin_locker.h"
#include "libcuckoo/libcuckoo/cuckoohash_map.hh"
#include "ps/initializer/initializer.h"
#include "ps/optim/optim.h"
#include "ps/table.h"
#include "t/element_type.h"
#include "t/tensor.h"

namespace kraken {

class SparseTable : public Table {
private:
  // For sparse table this must be a matrix. shape is [N, dimension].
  // We donnot assign the N, so it means the matrix's row canbe increase
  // automatically.
  int64_t dimension_;
  ElementType element_type_;

  std::unique_ptr<Initializer> initializer_;

  ParallelSkipList<uint64_t, Value> vals_;

public:
  SparseTable(uint64_t id, const std::string& name, int64_t dimension,
              ElementType element_type,
              std::unique_ptr<Initializer>&& initializer);

public:
  int64_t dimension() const;

  ElementType element_type() const;

  Initializer* initializer() const;

  ParallelSkipList<uint64_t, Value>* mutable_vals();

  int32_t Pull(const std::vector<uint64_t>& sparse_ids,
               std::vector<Tensor>* vals) override;

  int32_t Push(Optim* optim, const std::vector<uint64_t>& sparse_ids,
               const std::vector<Tensor>& grads, float lr) override;
};

}  // namespace kraken
