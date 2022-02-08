#pragma once

#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "ps/initializer/initializer.h"
#include "ps/optim/optim.h"
#include "ps/table.h"
#include "t/element_type.h"
#include "t/shape.h"

namespace kraken {

namespace io {
class CheckPoint;
}

class ModelManager {
  friend class io::CheckPoint;

private:
  struct Table {
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

  struct Model {
    uint64_t id;
    std::string name;

    OptimType optim_type;
    std::unordered_map<std::string, std::string> optim_conf;

    std::unordered_map<std::string, uint64_t> table_id_map_;
    std::unordered_map<uint64_t, Table> tables_;
  };

private:
  std::shared_mutex mu_;

  std::unordered_map<std::string, uint64_t> model_id_map_;
  std::unordered_map<uint64_t, Model> models_;

public:
  int32_t ApplyModel(
      const std::string& name, OptimType optim_type,
      const std::unordered_map<std::string, std::string>& optim_conf,
      uint64_t* model_id);

  int32_t ApplyDenseTable(uint64_t model_id, const std::string& table_name,
                          const Shape& shape, ElementType element_type,
                          uint64_t* table_id);

  int32_t ApplySparseTable(
      uint64_t model_id, const std::string& table_name, int64_t dimension,
      ElementType element_type, InitializerType init_type,
      const std::unordered_map<std::string, std::string>& init_conf,
      uint64_t* table_id);
};

}  // namespace kraken
