#pragma once

#include <memory>
#include <shared_mutex>
#include <unordered_map>

#include "io/checkpoint.h"
#include "io/checkpoint_executor.h"
#include "ps/info.h"
#include "ps/model.h"
#include "ps/optim/optim.h"

namespace kraken {

class Ps {
  friend class io::CheckpointExecutor;
  friend class io::Checkpoint;

private:
  size_t shard_num_;
  size_t shard_id_;

  // protect model_infos_/models_.
  std::shared_mutex mu_;

  // model_infos_ contains the whole model information.
  std::unordered_map<uint64_t, ModelInfo> model_infos_;

  // models_ include the whole SparseTable and part of DenseTable.
  std::unordered_map<uint64_t, std::unique_ptr<Model>> models_;

  // save Checkpoint.
  io::CheckpointExecutor checkpoint_executor_;

public:
  Ps(size_t shard_num, size_t shard_id, const std::string& save_dir,
     size_t max_save_count);

public:
  size_t shard_num() const;

  size_t shard_id() const;

  void Load(const std::string& load_dir);

  void Stop();

  /**
   * \brief Apply a model will return a unique model id.
   * Will insert modelinfo to model_infos_.
   *
   * \param name Model name.
   * \param optim_type Optim type.
   * \param optim_conf Optim config.
   * \param model_id Returned model id.
   * \return int32_t ErrorCode.
   */
  int32_t ApplyModel(
      const std::string& name, OptimType optim_type,
      const std::unordered_map<std::string, std::string>& optim_conf,
      uint64_t* model_id);

  /**
   * \brief Apply DenseTable
   *
   * \param model_id Model id.
   * \param name Table name.
   * \param shape Table shape.
   * \param element_type ElementType.
   * \param table_id Returned table id.
   * \return int32_t ErrorCode.
   */
  int32_t ApplyDenseTable(uint64_t model_id, const std::string& name,
                          const Shape& shape, ElementType element_type,
                          uint64_t* table_id);

  /**
   * \brief Apply SparseTable.
   *
   * \param model_id Model id.
   * \param name Table name.
   * \param dimension Dimension.
   * \param element_type ElementType.
   * \param init_type Initialzie type.
   * \param init_conf Initialize config.
   * \param table_id Table id.
   * \return int32_t ErrorCode.
   */
  int32_t ApplySparseTable(
      uint64_t model_id, const std::string& name, int64_t dimension,
      ElementType element_type, InitializerType init_type,
      const std::unordered_map<std::string, std::string>& init_conf,
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
   * \brief Register the DenseTable Info. every server will store all denstable info.
   *
   * \param model_id Model id.
   * \param id Table id.
   * \param name Table name.
   * \param shape Table shape.
   * \param element_type Element type.
   * \return int32_t ErrorCode.
   */
  int32_t RegisterDenseTableInfo(uint64_t model_id, uint64_t id,
                                 const std::string& name, const Shape& shape,
                                 ElementType element_type);

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
   * \param init_type Initialize type.
   * \param init_conf Initialize config.
   * \return int32_t Error code.
   */
  int32_t RegisterSparseTable(
      uint64_t model_id, uint64_t id, const std::string& name,
      int64_t dimension, ElementType element_type, InitializerType init_type,
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
   * \brief Pull Sparse vector from this server.
   *
   * \param model_id Model id.
   * \param indices A <TableId, Indices> map.
   * \param vals The result.
   * \return int32_t Error code.
   */
  int32_t CombinePullSparseTable(
      uint64_t model_id,
      const std::unordered_map<uint64_t, std::vector<int64_t>>& indices,
      std::unordered_map<uint64_t, std::vector<Tensor>>* vals);

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

  /**
   * \brief Save CheckPoint.
   *
   * \param model_id Model id.
   * \return int32_t ErrorCode.
   */
  int32_t SaveCheckPoint(uint64_t model_id);
};

}  // namespace kraken
