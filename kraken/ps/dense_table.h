#pragma once

#include <shared_mutex>

#include "ps/optim/optim.h"
#include "ps/table.h"
#include "t/element_type.h"
#include "t/shape.h"
#include "t/tensor.h"

namespace kraken {

namespace io {
class CheckpointExecutor;
class Checkpoint;
}  // namespace io

namespace watch {
class Watcher;
}

class DenseTable : public Table {
  friend class io::CheckpointExecutor;
  friend class io::Checkpoint;
  friend class watch::Watcher;

private:
  std::shared_mutex mu_;

  Value val_;

public:
  DenseTable(Optim* optim, uint64_t id, const std::string& name,
             const Tensor& val);

public:
  const Tensor& val() const;

  int32_t Push(const Tensor& grad, float lr) override;

  int32_t Pull(Tensor* val) override;

  int32_t PushPull(const Tensor& grad, float lr, Tensor* val) override;
};

}  // namespace kraken
