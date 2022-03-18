#pragma once

#include <shared_mutex>

#include "ps/optim/optim.h"
#include "ps/table.h"
#include "t/element_type.h"
#include "t/shape.h"
#include "t/tensor.h"

namespace kraken {

class DenseTable : public Table {
public:
  class UniqueHandler {
  private:
    std::shared_mutex& mu_;

  public:
    UniqueHandler(std::shared_mutex& mu) : mu_(mu) {
      mu_.lock();
    }

    UniqueHandler(const UniqueHandler&) = delete;
    UniqueHandler(const UniqueHandler&&) = delete;
    const UniqueHandler& operator=(const UniqueHandler&) = delete;
    const UniqueHandler& operator=(const UniqueHandler&&) = delete;

    ~UniqueHandler() {
      mu_.unlock();
    }
  };

  class SharedHandler {
  private:
    std::shared_mutex& mu_;

  public:
    SharedHandler(std::shared_mutex& mu) : mu_(mu) {
      mu_.lock_shared();
    }

    SharedHandler(const SharedHandler&) = delete;
    SharedHandler(const SharedHandler&&) = delete;
    const SharedHandler& operator=(const SharedHandler&) = delete;
    const SharedHandler& operator=(const SharedHandler&&) = delete;

    ~SharedHandler() {
      mu_.unlock_shared();
    }
  };

private:
  std::shared_mutex mu_;

  Value val_;

public:
  DenseTable(uint64_t id, const std::string& name, const Tensor& val);

  DenseTable(uint64_t id, const std::string& name, const Value& val);

public:
  UniqueHandler unique_handler();

  SharedHandler shared_handler();

  const Value& val() const;

  int32_t Pull(Tensor* val) override;

  int32_t Push(Optim* optim, const Tensor& grad, float lr) override;
};

}  // namespace kraken
