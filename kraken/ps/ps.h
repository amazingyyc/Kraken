#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <shared_mutex>

#include "common/async_task_queue.h"
#include "common/info.h"
#include "common/router.h"
#include "common/skip_list.h"
#include "ps/optim/optim.h"
#include "ps/proxy.h"
#include "ps/table.h"

namespace kraken {

class Ps {
private:
  enum class EventType : uint8_t {
    kProxyFinishTransfer = 0,
  };

  AsyncTaskQueue task_que_;

  // Current node address.
  std::string addr_;
  std::string s_addr_;

  // Protect status_/node_id_/events_/old_router_/router_/proxy_.
  std::shared_mutex mu_;

  // The node Status.
  uint32_t status_;
  uint64_t node_id_;

  Router old_router_;
  Router router_;

  std::unique_ptr<Proxy> proxy_;

  // Store the event.
  std::unordered_map<EventType, std::unordered_set<uint64_t>> events_;

  std::shared_mutex model_mu_;
  bool model_init_;
  std::string model_name_;
  std::unique_ptr<Optim> optim_;
  SkipList<uint64_t, std::unique_ptr<Table>> tables_;

public:
  Ps(const std::string& addr, const std::string& s_addr);

  ~Ps() = default;

private:
  // Clean tables_ the not belong to this node.
  void Clean();

  // Transfer data to new node.
  void Transfer(uint64_t target_id);

  // Call this function must make sure the status is kWorker | kProxy.
  void TryFetchDenseTableFromProxy(uint64_t table_id);

  void TryCombineFetchDenseTableFromProxy(
      const std::vector<uint64_t>& table_ids);

  void TryFetchSparseMetaDataFromProxy(uint64_t table_id);

  void TryFetchSparseValuesFromProxy(uint64_t table_id,
                                     const std::vector<uint64_t>& sparse_ids);

public:
  void Start();

  // Call by scheduler.
  int32_t Heartbeat(uint32_t* status);

  // Call by other Ps node.
  // Notify this PS other Ps has finish transfer data.
  int32_t NotifyFinishTransfer(uint64_t node_id);

  // Call by scheduler.
  // Notify this Node that there is a new Node joined.
  int32_t NotifyNodeJoin(uint64_t joined_id, const Router& old_router,
                         const Router& new_router);

  // Call by other Scheduler.
  int32_t CreateModel(
      std::string name, OptimType optim_type,
      const std::unordered_map<std::string, std::string>& optim_conf);

  // Call by other Scheduler.
  int32_t CreateDenseTable(uint64_t table_id, std::string name, const Tensor& val);

  // Call by other Scheduler.
  int32_t CreateSparseTable(
      uint64_t table_id, std::string name, int64_t dimension,
      ElementType element_type, InitializerType init_type,
      const std::unordered_map<std::string, std::string>& init_conf);

  // Call by other Ps node.
  // Another node transfer DenseTable to this node.
  int32_t TransferDenseTable(uint64_t from_node_id, uint64_t table_id,
                             const std::string& name, const Value& value);

  // Call by other Ps node.
  // Another node transfer SparseTableMetaData to this node.
  int32_t TransferSparseMetaData(
      uint64_t table_id, std::string name, int64_t dimension,
      ElementType element_type, InitializerType init_type,
      const std::unordered_map<std::string, std::string>& init_conf);

  // Call by other Ps node.
  // Another node transfer SparseTable Embedding to this node.
  int32_t TransferSparseValues(uint64_t table_id,
                               const std::vector<uint64_t>& sparse_ids,
                               const std::vector<Value>& values);

  // Call by other Ps node.
  // A node try to fetch DenseTable.
  int32_t TryFetchDenseTable(uint64_t table_id, std::string* name,
                             Value* value);

  // Call by other Ps node.
  int32_t TryCombineFetchDenseTable(const std::vector<uint64_t>& table_ids,
                                    std::vector<uint64_t>* exist_table_ids,
                                    std::vector<std::string>* names,
                                    std::vector<Value>* values);

  // Call by other Ps node.
  int32_t TryFetchSparseMetaData(
      uint64_t table_id, std::string* name, int64_t* dimension,
      ElementType* element_type, InitializerType* init_type,
      std::unordered_map<std::string, std::string>* init_conf);

  // Call by other Ps node.
  int32_t TryFetchSparseValues(uint64_t table_id,
                               const std::vector<uint64_t>& sparse_ids,
                               std::vector<uint64_t>* exist_sparse_ids,
                               std::vector<Value>* values);

  //////////////////////////////////////////////////////////////////////////////////
  // For Worker.

  // Call by Worker.
  int32_t PullDenseTable(uint64_t router_version, uint64_t table_id,
                         Tensor* val);

  // Call by Worker.
  int32_t CombinePullDenseTable(uint64_t router_version,
                                const std::vector<uint64_t>& table_ids,
                                std::vector<Tensor>* vals);

  // Call by Worker.
  int32_t PushDenseTable(uint64_t router_version, uint64_t table_id,
                         const Tensor& grad, float lr);

  // Call by Worker.
  int32_t PullSparseTable(uint64_t router_version, uint64_t table_id,
                          const std::vector<uint64_t>& sparse_ids,
                          std::vector<Tensor>* vals);
};

}  // namespace kraken
