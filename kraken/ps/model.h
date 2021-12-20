#pragma once

#include <memory>
#include <shared_mutex>
#include <unordered_map>

#include "ps/optim.h"
#include "ps/table.h"

namespace kraken {

/**
 * \brief The model represent a DeepLearning model will contain the tarinable Dense/Sparse table.
 */
class Model {
private:
  const static size_t kSparseTableSCount;

  uint64_t id_;
  std::string name_;

  std::unique_ptr<Optim> optim_;

  std::shared_mutex mu_;

  uint64_t table_id_gen_;

  std::unordered_map<std::string, uint64_t> table_name_id_;
  std::unordered_map<uint64_t, std::unique_ptr<Table>> tables_;

public:
  Model(uint64_t id, const std::string& name, std::unique_ptr<Optim>&& optim);

  ~Model() = default;

  /**
   * \brief Register a dense table in this model.
   *
   * \param name The table name, must be unique.
   * \param shape The table shape.
   * \param etype The table's element type.
   * \param table_id Store the table id when success.
   * \return true Register success.
   * \return false Register fail, like the name has been register but has different shape or element type.
   */
  bool RegisterDenseTable(const std::string& name, const Shape& shape,
                          ElementType etype, uint64_t* table_id);

  /**
   * \brief Register a sparse table.
   *
   * \param name The table name.
   * \param dimension The SparseTable dimension.
   * \param etype The Table tensor's element type.
   * \param table_id Store the table id.
   * \return true Reigster success.
   * \return false Register fail like already register a same name SparseTable but has different dimension or element type.
   */
  bool RegisterSparseTable(const std::string& name, int64_t dimension,
                           ElementType etype, uint64_t* table_id);
};

}  // namespace kraken
