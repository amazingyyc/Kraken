#pragma once

#include <atomic>
#include <cinttypes>
#include <memory>
#include <vector>

#include "common/consistent_hasher.h"
#include "protocol/pull_sparse_table_prot.h"
#include "protocol/push_sparse_table_prot.h"
#include "protocol/rpc_func_type.h"
#include "worker/client.h"

namespace kraken {

enum class EmitterType : uint8_t {
  kDefault = 0,
  kDCT = 1,  // ref: Training Recommender Systems at Scale:
             // Communication-Efficient Model and Data Parallelism
};

class Emitter {
protected:
  EmitterType type_;

  std::atomic_bool initialized_;

  std::vector<std::unique_ptr<Client>> clients_;

  // use concsistent hash to route.
  ConsistentHasher router_;

  std::string model_name_;
  uint64_t model_id_;

  /**
   * \brief Ok this maybe weird the worker will store the learning rate.
   * We will use learning rate in Optimizer and Embedding and I donot want store
   * LearninRate in Embedding but in Optimizer. So We will store and update
   * LearningRate in Optimizer (One worker has only one but has
   * multi-Embedding).
   */
  std::atomic<float> lr_;

public:
  Emitter();

protected:
  Emitter(EmitterType type);

  /**
   * \brief Get server id by model_id and table_id.
   *
   * \param model_id Model id.
   * \param table_id Tabel id.
   * \return size_t Server id.
   */
  size_t DenseTableRouter(uint64_t model_id, uint64_t table_id);

  /**
   * \brief Get server id for sparse table by model_id/table_id/sparse_id.
   *
   * \param model_id Model id.
   * \param table_id Tabel id.
   * \param sparse_id Sparse id.
   * \return size_t Tabel id.
   */
  size_t SparseTableRouter(uint64_t model_id, uint64_t table_id,
                           int64_t sparse_id);

  /**
   * \brief Request all server with same request.
   *
   * \tparam ReqType Request type.
   * \tparam RspType Response type.
   * \param req The request.
   * \param rsps Store response.
   */
  template <typename ReqType, typename RspType>
  int32_t ParallelCallAll(uint32_t type, const ReqType& req,
                          std::vector<RspType>* rsps) const {
    size_t server_num = clients_.size();

    rsps->clear();
    rsps->resize(server_num);

    // Store error code.
    std::vector<int32_t> ecodes(server_num);

    ThreadBarrier barrier(server_num);

    for (size_t i = 0; i < server_num; ++i) {
      auto callback = [i, &ecodes, rsps, &barrier](int32_t code, RspType& rsp) {
        ecodes[i] = code;
        (*rsps)[i] = std::move(rsp);

        barrier.Release();
      };

      clients_[i]->CallAsync<ReqType, RspType>(type, req, std::move(callback));
    }

    barrier.Wait();

    for (size_t i = 0; i < server_num; ++i) {
      if (ecodes[i] != ErrorCode::kSuccess) {
        return ecodes[i];
      }
    }

    return ErrorCode::kSuccess;
  }

  /**
   * \brief Prallel send to multi server.
   *
   * \tparam ReqType Request type.
   * \tparam RspType Response type.
   * \param server_indices The corresponding server id.
   * \param reqs Request.
   * \param rsps Response.
   * \return int32_t Error code.
   */
  template <typename ReqType, typename RspType>
  int32_t ParallelCall(uint32_t type, const std::vector<size_t>& server_indices,
                       const std::vector<ReqType>& reqs,
                       std::vector<RspType>* rsps) {
    ARGUMENT_CHECK(!reqs.empty(), "ParallelCall's reqs is empty.")

    size_t count = reqs.size();

    rsps->clear();
    rsps->resize(count);

    std::vector<int32_t> ecodes(count);

    ThreadBarrier barrier(count);
    for (size_t i = 0; i < count; ++i) {
      auto callback = [i, &ecodes, rsps, &barrier](int32_t code, RspType& rsp) {
        ecodes[i] = code;
        (*rsps)[i] = std::move(rsp);

        barrier.Release();
      };

      clients_[server_indices[i]]->CallAsync<ReqType, RspType>(
          type, reqs[i], std::move(callback));
    }

    barrier.Wait();

    for (size_t i = 0; i < count; ++i) {
      if (ecodes[i] != ErrorCode::kSuccess) {
        return ecodes[i];
      }
    }

    return ErrorCode::kSuccess;
  }

  /**
   * \brief Same like ParallelCall not the parameter include all server's request and mask.
   *
   * \tparam ReqType Request type.
   * \tparam RspType Response type.
   * \param type Which RPC func will be call.
   * \param mask The server mask, true means will request this server.
   * \param reqs Request.
   * \param rsps Response
   * \return int32_t Error code.
   */
  template <typename ReqType, typename RspType>
  int32_t ParallelCall(uint32_t type, const std::vector<bool>& mask,
                       const std::vector<ReqType>& reqs,
                       std::vector<RspType>* rsps) {
    rsps->clear();
    rsps->resize(reqs.size());

    uint32_t b_count = 0;
    for (size_t i = 0; i < mask.size(); ++i) {
      if (mask[i]) {
        b_count += 1;
      }
    }

    std::vector<int32_t> ecodes(mask.size());
    ThreadBarrier barrier(b_count);

    for (size_t i = 0; i < mask.size(); ++i) {
      if (mask[i] == false) {
        continue;
      }

      auto callback = [&ecodes, i, rsps, &barrier](int32_t code, RspType& rsp) {
        ecodes[i] = code;
        (*rsps)[i] = std::move(rsp);

        barrier.Release();
      };

      clients_[i]->CallAsync<ReqType, RspType>(type, reqs[i],
                                               std::move(callback));
    }

    barrier.Wait();

    for (size_t i = 0; i < mask.size(); ++i) {
      if (mask[i] == false) {
        continue;
      }

      if (ecodes[i] != ErrorCode::kSuccess) {
        return ecodes[i];
      }
    }

    return ErrorCode::kSuccess;
  }

  template <typename ReqType, typename RspType>
  void ParallelCallAsync(uint32_t type,
                         const std::vector<size_t>& server_indices,
                         const std::vector<ReqType>& reqs) {
    ARGUMENT_CHECK(!reqs.empty(), "ParallelCallAsync's reqs is empty.")

    size_t count = reqs.size();
    for (size_t i = 0; i < count; ++i) {
      auto callback = [](int32_t ecode, RspType& rsp) {
        // Donot handle the error code, just crash.
        RPC_CALL(ecode);
      };

      clients_[server_indices[i]]->CallAsync<ReqType, RspType>(
          type, reqs[i], std::move(callback));
    }
  }

  template <typename ReqType, typename RspType>
  void ParallelCallAsync(uint32_t type, const std::vector<bool>& mask,
                         const std::vector<ReqType>& reqs) {
    for (size_t i = 0; i < mask.size(); ++i) {
      if (mask[i] == false) {
        continue;
      }

      auto callback = [](int32_t ecode, RspType& rsp) {
        // Donot handle the error code, just crash.
        RPC_CALL(ecode);
      };

      clients_[i]->CallAsync<ReqType, RspType>(type, reqs[i],
                                               std::move(callback));
    }
  }

public:
  virtual ~Emitter() = default;

  /**
   * \brief Initialize Emitter.
   *
   * \param addrs_
   * \param model_name
   */
  void Initialize(const std::string& addrs);

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
  virtual uint64_t RegisterModel(
      const std::string& model_name, OptimType optim_type,
      const std::unordered_map<std::string, std::string>& optim_conf);

  /**
   * \brief Update LearningRate, thread-safe.
   *
   * \param lr The learning rate.
   */
  virtual void UpdateLR(float lr);

  /**
   * \brief Register dense table in server.
   *
   * \param name Table name.
   * \param val The table value.
   * \return uint64_t Error code.
   */
  virtual uint64_t RegisterDenseTable(const std::string& name,
                                      const Tensor& val);

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
  virtual uint64_t RegisterSparseTable(
      const std::string& name, int64_t dimension, ElementType etype,
      InitializerType init_type,
      const std::unordered_map<std::string, std::string>& init_conf);

  /**
   * \brief Pull Dense table from server.
   *
   * \param table_id Table id.
   * \return Tensor The dense table.
   */
  virtual Tensor PullDenseTable(uint64_t table_id);

  /**
   * \brief Pull a list of dense table from server.
   *
   * \param table_ids The list of dense table ids.
   * \return std::vector<Tensor> Dense table value.
   */
  virtual std::vector<Tensor> CombinePullDenseTable(
      const std::vector<uint64_t>& table_ids);

  /**
   * \brief Push gradient for special dense table.
   *
   * \param table_id Table id.
   * \param grad Gradient.
   */
  virtual void PushDenseTable(uint64_t table_id, const Tensor& grad);

  /**
   * \brief Push gradient and pull val from server.
   *
   * \param table_id Table id.
   * \param grad Gradient
   * \return Tensor The Dense value.
   */
  virtual Tensor PushPullDenseTable(uint64_t table_id, const Tensor& grad);

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
  virtual Tensor PullSparseTable(uint64_t table_id, const Tensor& indices);

  /**
   * \brief Like PullSparseTable but request a list.
   *
   * \param table_ids Sparse table ids.
   * \param indices The index of embedding.
   * \return std::vector<Tensor> Sparse embeddings.
   */
  virtual std::vector<Tensor> CombinePullSparseTable(
      const std::vector<uint64_t>& table_ids,
      const std::vector<Tensor>& indices);

  /**
   * \brief Push sparse table gradient to server.
   *
   * \param table_id Table id.
   * \param indices Gradient index.
   * \param grads Gradient.
   */
  virtual void PushSparseTable(uint64_t table_id, const Tensor& indices,
                               const Tensor& grads);

  /**
   * \brief Request server to save check point.
   */
  virtual void SaveCheckPoint();
};

}  // namespace kraken
