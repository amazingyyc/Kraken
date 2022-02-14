#pragma once

#include <string>

#include "io/check_point.h"
#include "ps/optim/optim.h"
#include "t/tensor.h"

namespace kraken {

/**
 * \brief The table type.
 */
enum class TableType : uint8_t {
  kDense = 0,
  kSparse = 1,
};

enum class StateType : uint32_t {
  kSteps = 0,
  kMomentumBuffer = 1,
  kStateSum = 2,
  kFirstMoment = 3,
  kSecondMoment = 4,
  kSecondMomentMax = 5,
  kSquareAverage = 6,
  kGAve = 7,
};

/**
 * \brief This is a simple struct that store some useful resource.
 * Like in SGD optim we may need store some temporary tensor.
 */
struct Bag {
  // tensor state.
  std::unordered_map<StateType, Tensor> state;

  // integer state.
  std::unordered_map<StateType, int64_t> state_i;
};

class Table {
  friend class io::CheckPoint;

public:
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
