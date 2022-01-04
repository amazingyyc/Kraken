#pragma once

#include <shared_mutex>

#include "common/element_type.h"
#include "common/shape.h"
#include "common/tensor.h"
#include "ps/optim/optim.h"
#include "ps/table.h"

namespace kraken {

class DenseTable : public Table {
private:
  std::shared_mutex mu_;

  Tensor val_;
  Bag bag_;

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
