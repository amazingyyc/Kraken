#pragma once

#include <string>

#include "ps/optim/optim.h"

namespace kraken {

/**
 * \brief The table type.
 */
enum class TableType : uint8_t {
  kDense = 0,
  kSparse = 1,
};

class Table {
protected:
  struct Value {
    Tensor val;
    Bag bag;
  };

protected:
  TableType type_;

  Optim* optim_;

  uint64_t id_;

  std::string name_;

  Table(TableType type, Optim* optim, uint64_t id, const std::string& name);

public:
  virtual ~Table() = default;

  TableType type() const;

  uint64_t id() const;

  const std::string name() const;

  virtual int32_t Push(const Tensor& grad, float lr);

  virtual int32_t Pull(Tensor* val);

  virtual int32_t PushPull(const Tensor& grad, float lr, Tensor* val);

  virtual int32_t Push(const std::vector<int64_t>& indices,
                       const std::vector<Tensor>& grads, float lr);

  virtual int32_t Pull(const std::vector<int64_t>& indices,
                       std::vector<Tensor>* vals);
};

}  // namespace kraken
