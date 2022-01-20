#pragma once

#include <memory>

#include "worker/emitter.h"

namespace kraken {

/**
 * \brief A worker represent a model.
 */
class Worker {
private:
  std::unique_ptr<Emitter> emitter_;

public:
  Worker();

  /**
   * \brief Initialize this work select Emitter and connect server.
   * 
   * \param addrs Server address.
   * \param emitter_type Emitter type.
   * \param life_span For DCTEmitter.
   * \param eta For DCTEmitter.
   */
  void Initialize(const std::string& addrs,
                  EmitterType emitter_type = EmitterType::kDefault,
                  uint64_t life_span = 1000, float eta = 0.75);

  /**
   * \brief Stop worker.
   */
  void Stop();

  /**
   * \brief Register a model to server.
   *
   * \param model_name The model name.
   * \param optim_type Which optimizer algorithm to be used.
   * \param optim_conf Optimizer config.
   * \return uint64_t Model id.
   */
  uint64_t RegisterModel(
      const std::string& model_name, OptimType optim_type,
      const std::unordered_map<std::string, std::string>& optim_conf);

  /**
   * \brief Update LearningRate, thread-safe.
   *
   * \param lr The learning rate.
   */
  void UpdateLR(float lr);

  /**
   * \brief Register dense table in server.
   *
   * \param name Table name.
   * \param val The table value.
   * \return uint64_t Error code.
   */
  uint64_t RegisterDenseTable(const std::string& name, const Tensor& val);

  /**
   * \brief Register a sparse table. thread-safe.
   *
   * A sparse table will register in all server.
   *
   * \param name Sparse table name.
   * \param dimension the table dimension
   * \param etype table element type.
   * \return uint64_t table id.
   */
  uint64_t RegisterSparseTable(const std::string& name, int64_t dimension,
                               ElementType etype);

  /**
   * \brief Register a sparse table. thread-safe.
   *
   * A sparse table will register in all server.
   *
   * \param name Sparse table name.
   * \param dimension the table dimension
   * \param etype table element type.
   * \param init_type Initialize type.
   * \param init_conf Initialize config.
   * \return uint64_t table id.
   */
  uint64_t RegisterSparseTableV2(
      const std::string& name, int64_t dimension, ElementType etype,
      InitializerType init_type,
      const std::unordered_map<std::string, std::string>& init_conf);

  /**
   * \brief Push gradient for special dense table.
   *
   * \param table_id Table id.
   * \param grad Gradient.
   */
  void PushDenseTable(uint64_t table_id, const Tensor& grad);

  /**
   * \brief Pull Dense table from server.
   *
   * \param table_id Table id.
   * \return Tensor The dense table.
   */
  Tensor PullDenseTable(uint64_t table_id);

  /**
   * \brief Pull a list of dense table from server.
   *
   * \param table_ids The list of dense table ids.
   * \return std::vector<Tensor> Dense table value.
   */
  std::vector<Tensor> PullListDenseTable(
      const std::vector<uint64_t>& table_ids);

  /**
   * \brief Push gradient and pull val from server.
   *
   * \param table_id Table id.
   * \param grad Gradient
   * \return Tensor The Dense value.
   */
  Tensor PushPullDenseTable(uint64_t table_id, const Tensor& grad);

  /**
   * \brief Push sparse table gradient to server.
   *
   * \param table_id Table id.
   * \param indices Gradient index.
   * \param grads Gradient.
   */
  void PushSparseTable(uint64_t table_id, const Tensor& indices,
                       const Tensor& grads);

  /**
   * \brief Pull Sparse vector from server.
   *
   * Suppose the indices's shape is:[d0, d1,..., dk] and the embedding dimension
   * is dim. So the val'shape will be:[d0, d1,..., dk, dim].
   *
   * \param table_id Sparse table id.
   * \param indices The index of embedding.
   * \return The sparse embedding.
   */
  Tensor PullSparseTable(uint64_t table_id, const Tensor& indices);
};

}  // namespace kraken
