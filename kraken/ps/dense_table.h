#pragma once

#include <shared_mutex>

#include "common/element_type.h"
#include "common/shape.h"
#include "common/tensor.h"
#include "ps/table.h"

namespace kraken {

class DenseTable : public Table {
private:
  std::shared_mutex mu_;

  Tensor var_;

public:
  DenseTable(Optim* optim, uint64_t id, const std::string& name,
             const Shape& shape, ElementType etype);

public:
  const Tensor& Var() const;

  bool Push(const Tensor& grad, float lr) override;

  bool Pull(Tensor* var) override;
};

}  // namespace kraken
