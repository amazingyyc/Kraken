#pragma once

#include <unordered_map>
#include <vector>

#include "common/element_type.h"
#include "common/spin_locker.h"
#include "common/tensor.h"
#include "ps/optim.h"
#include "ps/table.h"

namespace kraken {

class SparseTable : public Table {
private:
  // For sparse table this must be a matrix. shape is [row, dimension].
  // We donnot assign the N, so it means the matrix's row canbe increase
  // automatically.
  int64_t dimension_;
  ElementType etype_;

  size_t s_count_;

  // Use segment locker.
  std::vector<SpinLocker> lockers_;
  std::vector<std::unordered_map<int64_t, Tensor>> vars_;

public:
  SparseTable(Optim* optim, uint64_t id, const std::string& name,
              int64_t dimension, ElementType etype, size_t s_count);

public:
  int64_t Dimension() const;

  ElementType EType() const;

  bool Push(const std::vector<IndepVector>& grads, float lr) override;

  bool Pull(const std::vector<int64_t>& indices,
            std::vector<Tensor>* vars) override;
};

}  // namespace kraken
