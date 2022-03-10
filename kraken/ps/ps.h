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

  // CompressType.
  // CompressType compress_type_;

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

public:
  void Start();

  int32_t Heartbeat(uint32_t *status);

  // Notify this PS other Ps has finish transfer data.
  int32_t NotifyFinishTransfer(uint64_t node_id);

  // Notify this Node that there is a new Node joined.
  int32_t NotifyNodeJoin(uint64_t joined_id, const Router& old_router,
                         const Router& new_router);

  int32_t InitModel(
      std::string name, OptimType optim_type,
      const std::unordered_map<std::string, std::string>& optim_conf);

  int32_t CreateDenseTable(uint64_t id, std::string name, const Tensor& val);

  int32_t CreateSparseTable(
      uint64_t id, std::string name, int64_t dimension,
      ElementType element_type, InitializerType init_type,
      const std::unordered_map<std::string, std::string>& init_conf);
};

}  // namespace kraken
