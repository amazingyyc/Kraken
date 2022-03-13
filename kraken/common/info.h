#pragma once

#include <cinttypes>
#include <string>
#include <unordered_map>

#include "t/element_type.h"
#include "t/tensor.h"

namespace kraken {

// Cluster node type.
enum class NodeType : uint8_t {
  kScheduler = 0,
  kPs = 1,
  kWorker = 2,
};

struct NodeStatus {
  static constexpr uint32_t kInit = 1 << 0;
  static constexpr uint32_t kWork = 1 << 1;
  static constexpr uint32_t kProxy = 1 << 2;
  static constexpr uint32_t kTransfer = 1 << 3;
  static constexpr uint32_t kLeave = 1 << 4;
};

// Table type.
enum class TableType : uint8_t {
  kDense = 0,
  kSparse = 1,
};

// Optimizer type.
enum class OptimType : uint8_t {
  kAdagrad = 0,
  kAdam = 1,
  kRMSprop = 2,
  kSGD = 3,
};

// Initializer type.
enum class InitializerType : uint8_t {
  kConstant = 0,
  kUniform = 1,
  kNormal = 2,
  kXavierUniform = 3,
  kXavierNormal = 4,
};

// State type.
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

// This is a simple struct that store some useful resource.
// Like in SGD optim we may need store some temporary tensor.
// struct Bag {
//   // tensor state.
//   std::unordered_map<StateType, Tensor> state;

//   // integer state.
//   std::unordered_map<StateType, int64_t> state_i;

//   Bag Clone() const {
//     Bag n_bag;

//     for (auto& [k, v] : state) {
//       n_bag.state.emplace(k, v.Clone());
//     }

//     n_bag.state_i = state_i;

//     return n_bag;
//   }
// };

struct Value {
  Tensor val;

  std::unordered_map<StateType, Tensor> states;
  std::unordered_map<StateType, int64_t> states_i;

  Value Clone() const {
    Value n_value;
    n_value.val = val.Clone();

    for (const auto& [k, v] : states) {
      n_value.states.emplace(k, v.Clone());
    }

    n_value.states_i = states_i;

    return n_value;
  }
};

struct Node {
  // What kind of NodeType.
  NodeType type;

  // Every node has a unique id.
  uint64_t id;

  // The node address hostname:port.
  std::string addr;
};

struct TableMetaData {
  uint64_t id;
  std::string name;
  TableType table_type;

  ElementType element_type;

  // For dense.
  Shape shape;

  // For sparse.
  int64_t dimension;
  InitializerType init_type;
  std::unordered_map<std::string, std::string> init_conf;
};

struct ModelMetaData {
  std::string name;

  OptimType optim_type;
  std::unordered_map<std::string, std::string> optim_conf;

  std::unordered_map<uint64_t, TableMetaData> table_mdatas;
};

}  // namespace kraken
