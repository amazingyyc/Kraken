#pragma once

#include <cinttypes>
#include <string>
#include <unordered_map>

#include "t/element_type.h"
#include "t/tensor.h"

namespace kraken {

// Worker emitter type.
enum class EmitterType : uint8_t {
  kDefault = 0,
  kDCT = 1,  // ref: Training Recommender Systems at Scale:
             // Communication-Efficient Model and Data Parallelism
};

// Cluster node type.
enum class NodeType : uint8_t {
  kScheduler = 0,
  kPs = 1,
  kWorker = 2,
};

struct NodeStatus {
  static constexpr uint32_t kInit = 1 << 0;
  static constexpr uint32_t kWork = 1 << 1;
  // In proxy from other Ps.
  static constexpr uint32_t kProxy = 1 << 2;
  // In transfer data to other Ps.
  static constexpr uint32_t kTransfer = 1 << 3;
  // In save model.
  static constexpr uint32_t kSave = 1 << 4;
  // In load model.
  static constexpr uint32_t kLoad = 1 << 5;
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
