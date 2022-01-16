#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>

#include "protocol/pull_sparse_table_prot.h"
#include "protocol/push_sparse_table_prot.h"
#include "protocol/rpc_func_type.h"
#include "ps/optim/optim.h"
#include "worker/client.h"

namespace kraken {

/**
 * \brief A worker represent a model.
 */
class Worker {
private:
  /**
   * \brief Client every client connect with a server. Client 0 is the master server.
   */
  std::vector<std::unique_ptr<Client>> clients_;

  /**
   * \brief Whether initialized.
   */
  std::atomic_bool initialized_;

  /**
   * \brief Model name.
   */
  std::string model_name_;

  /**
   * \brief Model id.
   */
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
  Worker();

private:
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
                          std::vector<RspType>* rsps) {
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

  template <typename T>
  void PushSparseTableImpl(uint64_t table_id, const Tensor& indices,
                           const Tensor& grads) {
    // If the indices shape is [do, d1,...,dk] than grads shape must be [do,
    // d1,...,dk, dimension]
    std::vector<int64_t> dims = indices.shape().dims();
    int64_t dimension = grads.shape()[-1];
    dims.emplace_back(dimension);

    ARGUMENT_CHECK(Shape(dims) == grads.shape(),
                   "PushSparseTable indices and grads shape error.");

    int64_t row = indices.Size();
    int64_t col = dimension;
    Tensor mgrads = grads.Reshape({row, col});

    // <server_id, <sparse_id, indice>>
    std::unordered_map<size_t, std::unordered_map<int64_t, size_t>>
        server_req_indice_map;

    std::unordered_map<size_t, PushSparseTableRequest> server_reqs;

    T* idp = indices.Data<T>();
    for (int64_t i = 0; i < row; ++i) {
      int64_t sparse_id = (int64_t)idp[i];
      size_t server_id = SparseTableRouter(model_id_, table_id, sparse_id);

      auto it = server_req_indice_map[server_id].find(sparse_id);
      if (it == server_req_indice_map[server_id].end()) {
        // Get a new SparseId.
        server_req_indice_map[server_id][sparse_id] =
            server_reqs[server_id].indices.size();

        server_reqs[server_id].indices.emplace_back(sparse_id);

        // Here must clone. The grads is share memory with torch tensor.
        server_reqs[server_id].grads.emplace_back(grads.Vector(i).Clone());
      } else {
        // Already exist accumulate the gradient.
        size_t r_idx = it->second;
        server_reqs[server_id].grads[r_idx] += grads.Vector(i);
      }
    }

    std::vector<size_t> server_indices;
    server_indices.reserve(server_reqs.size());

    std::vector<PushSparseTableRequest> reqs;
    std::vector<PushSparseTableResponse> rsps;

    reqs.reserve(server_reqs.size());
    for (auto& item : server_reqs) {
      item.second.model_id = model_id_;
      item.second.table_id = table_id;
      item.second.lr = lr_.load();

      server_indices.emplace_back(item.first);
      reqs.emplace_back(std::move(item.second));
    }

    ParallelCallAsync<PushSparseTableRequest, PushSparseTableResponse>(
        RPCFuncType::kPushSparseTableType, server_indices, reqs);
  }

  template <typename T>
  Tensor PullSparseTableImpl(uint64_t table_id, const Tensor& indices) {
    int64_t row = indices.Size();

    // Sparse table store in all server. So we need select which server to
    // send.
    std::unordered_map<size_t, PullSparseTableRequest> server_reqs;

    // <server_id, <sparse id, request indices's index>> map
    std::unordered_map<size_t, std::unordered_map<int64_t, size_t>>
        server_indice_map;

    T* idp = indices.Data<T>();
    for (int64_t i = 0; i < row; ++i) {
      int64_t sparse_id = (int64_t)idp[i];
      size_t server_id = SparseTableRouter(model_id_, table_id, sparse_id);

      if (server_indice_map[server_id].find(sparse_id) ==
          server_indice_map[server_id].end()) {
        server_indice_map[server_id][sparse_id] =
            server_reqs[server_id].indices.size();

        server_reqs[server_id].indices.emplace_back(sparse_id);
      }
    }

    std::vector<size_t> server_indices;
    server_indices.reserve(server_reqs.size());

    std::vector<PullSparseTableRequest> reqs;
    std::vector<PullSparseTableResponse> rsps;

    reqs.reserve(server_reqs.size());
    for (auto& item : server_reqs) {
      item.second.model_id = model_id_;
      item.second.table_id = table_id;

      server_indices.emplace_back(item.first);
      reqs.emplace_back(std::move(item.second));
    }

    // Send to server.
    RPC_CALL(ParallelCall(RPCFuncType::kPullSparseTableType, server_indices,
                          reqs, &rsps));

    // For now we already get result from server. So let's merge it.
    std::unordered_map<size_t, size_t> server_rsp_map;
    for (size_t i = 0; i < server_indices.size(); ++i) {
      server_rsp_map[server_indices[i]] = i;
    }

    std::vector<Tensor> vecs;
    vecs.reserve(row);

    for (int64_t i = 0; i < row; ++i) {
      int64_t sparse_id = (int64_t)idp[i];
      size_t server_id = SparseTableRouter(model_id_, table_id, sparse_id);

      size_t rsp_idx = server_rsp_map[server_id];
      size_t vec_idx = server_indice_map[server_id][sparse_id];

      vecs.emplace_back(rsps.at(rsp_idx).vals.at(vec_idx));
    }

    // concat to matrix than reshape.
    Tensor val = indices.ConcatVector(vecs);

    std::vector<int64_t> dims = indices.shape().dims();
    int64_t col = val.Size() / indices.Size();

    dims.emplace_back(col);

    return val.Reshape(dims);
  }

public:
  /**
   * \brief Initialize worker.
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
