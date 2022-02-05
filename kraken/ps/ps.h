#pragma once

#include <memory>
#include <shared_mutex>
#include <unordered_map>

#include "ps/apply_manager.h"
#include "ps/model_manager.h"
#include "ps/model.h"
#include "ps/optim/optim.h"

namespace kraken {

namespace io {
class CheckPoint;
}

class Ps {
  friend class io::CheckPoint;

private:
  ApplyManager apply_mgr_;

  ModelManager model_manager_;

  std::shared_mutex mu_;
  std::unordered_map<uint64_t, std::unique_ptr<Model>> models_;

  size_t shard_num_;
  size_t shard_id_;

public:
  size_t shard_num() const;

  size_t shard_id() const;

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

  /**
   * \brief Register a model.
   *
   * \param name The model name.
   * \param id Model id.
   * \param optim_type Optim type.
   * \param optim_conf Optim config.
   * \return int32_t Error code.
   */
  int32_t RegisterModel(
      uint64_t id, const std::string& name, OptimType optim_type,
      const std::unordered_map<std::string, std::string>& optim_conf);

  /**
   * \brief Register a DenseTabel for special model.
   *
   * \param model_id Model id.
   * \param id Table id.
   * \param name Table name.
   * \param var Table value.
   * \return int32_t Error code.
   */
  int32_t RegisterDenseTable(uint64_t model_id, uint64_t id,
                             const std::string& name, const Tensor& var);

  /**
   * \brief Register a sparse table.
   *
   * \param model_id Which model will be register.
   * \param id Table id.
   * \param name Table name.
   * \param dimension Sparse tensor dimension.
   * \param etype Sparse tensor element type.
   * \return int32_t error code.
   */
  int32_t RegisterSparseTable(uint64_t model_id, uint64_t id,
                              const std::string& name, int64_t dimension,
                              ElementType etype);

  /**
   * \brief Register a sparse table.
   *
   * \param model_id Which model will be register.
   * \param id Table id.
   * \param name Table name.
   * \param dimension Sparse tensor dimension.
   * \param etype Sparse tensor element type.
   * \param init_type Initialize type.
   * \param init_conf Initialize config.
   * \return int32_t Error code.
   */
  int32_t RegisterSparseTableV2(
      uint64_t model_id, uint64_t id, const std::string& name,
      int64_t dimension, ElementType etype, InitializerType init_type,
      const std::unordered_map<std::string, std::string>& init_conf);

  /**
   * \brief Pull Dense tensor
   *
   * \param model_id model id.
   * \param table_id table id.
   * \param val Table value.
   * \return int32_t error code.
   */
  int32_t PullDenseTable(uint64_t model_id, uint64_t table_id, Tensor* val);

  /**
   * \brief Pull list of dense table.
   *
   * \param model_id Model id.
   * \param table_id Table_id list.
   * \param vals The result.
   * \return int32_t Error code.
   */
  int32_t CombinePullDenseTable(uint64_t model_id,
                                const std::vector<uint64_t>& table_ids,
                                std::vector<Tensor>* vals);

  /**
   * \brief Push and Pull dense table.
   *
   * \param model_id Model id.
   * \param table_id Table id.
   * \param grad Gradient.
   * \param lr Learning rate.
   * \param val Store value.
   * \return int32_t Error code.
   */
  int32_t PushPullDenseTable(uint64_t model_id, uint64_t table_id,
                             const Tensor& grad, float lr, Tensor* val);

  /**
   * \brief Push update dense table.
   *
   * \param model_id Model id.
   * \param table_id Table id.
   * \param grad The gradient.
   * \param lr Learning rate.
   * \return int32_t error code.
   */
  int32_t PushDenseTable(uint64_t model_id, uint64_t table_id,
                         const Tensor& grad, float lr);

  /**
   * \brief Pull sparse vector from server.
   *
   * \param model_id Model id.
   * \param table_id Table id.
   * \param indices Sparse indice.
   * \param vals The result.
   * \return int32_t Error code.
   */
  int32_t PullSparseTable(uint64_t model_id, uint64_t table_id,
                          const std::vector<int64_t>& indices,
                          std::vector<Tensor>* vals);

  /**
   * \brief Push gradient to server.
   *
   * \param model_id Model id.
   * \param table_id Table id.
   * \param indices Gradient index.
   * \param grads Gradient.
   * \param lr Learning rate.
   * \return int32_t Error code.
   */
  int32_t PushSparseTable(uint64_t model_id, uint64_t table_id,
                          const std::vector<int64_t>& indices,
                          const std::vector<Tensor>& grads, float lr);
};

}  // namespace kraken
