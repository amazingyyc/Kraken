#pragma once

#include <cinttypes>
#include <string>
#include <unordered_map>

#include "ps/initializer/initializer.h"
#include "ps/table.h"
#include "t/element_type.h"

namespace kraken {

struct TableInfo {
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

struct ModelInfo {
  uint64_t id;
  std::string name;

  OptimType optim_type;
  std::unordered_map<std::string, std::string> optim_conf;

  std::unordered_map<uint64_t, TableInfo> table_infos;
};

}  // namespace kraken
