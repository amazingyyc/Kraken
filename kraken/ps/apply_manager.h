#pragma once

#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "parallel_hashmap/parallel_hashmap/phmap.h"
#include "ps/table.h"

namespace kraken {

/**
 * \brief Apply manager will manager how many model has been applied.
 * and assign the unique model id/table id.
 */
class ApplyManager {
private:
  struct Table {
    std::string name;
    uint64_t id;
    TableType table_type;
  };

  struct Model {
    std::string name;
    uint64_t id;

    phmap::flat_hash_map<std::string, uint64_t> table_id_map_;
    phmap::flat_hash_map<uint64_t, Table> tables_;
  };

private:
  std::shared_mutex mu_;

  phmap::flat_hash_map<std::string, uint64_t> model_id_map_;
  phmap::flat_hash_map<uint64_t, Model> models_;

public:
  /**
   * \brief Apply a model id. thread-safe.
   *
   * \param name Model name.
   * \param model_id Store the id.
   * \return int32_t Error code.
   */
  int32_t ApplyModel(const std::string& name, uint64_t* model_id);

  /**
   * \brief Apply a table id. thread-safe.
   *
   * \param mdoel_id Model id.
   * \param name Table name.
   * \param type Table type.
   * \param table_id Store the id.
   * \return int32_t Error code.
   */
  int32_t ApplyTable(uint64_t model_id, const std::string& name, TableType type,
                     uint64_t* table_id);
};

}  // namespace kraken
