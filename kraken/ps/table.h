#pragma once

#include <string>

#include "common/info.h"
#include "ps/optim/optim.h"
#include "t/tensor.h"

namespace kraken {

class Table {
protected:
  TableType type_;

  uint64_t id_;

  std::string name_;

  Table(TableType type, uint64_t id, const std::string& name);

public:
  virtual ~Table() = default;

  TableType type() const;

  uint64_t id() const;

  const std::string& name() const;

  virtual int32_t Pull(Tensor* val);

  virtual int32_t Push(Optim* optim, const Tensor& grad, float lr);

  virtual int32_t PushPull(const Tensor& grad, float lr, Tensor* val);

  virtual int32_t Pull(const std::vector<uint64_t>& indices,
                       std::vector<Tensor>* vals);

  virtual int32_t Push(const std::vector<uint64_t>& indices,
                       const std::vector<Tensor>& grads, float lr);
};

}  // namespace kraken
