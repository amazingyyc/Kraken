#include <thread>

#include "common/log.h"
#include "ps/dense_table.h"
#include "ps/ps.h"

namespace kraken {

void Ps::TryFetchDenseTableFromProxy(uint64_t table_id) {
  assert(proxy_ != nullptr);

  std::string name;
  Value value;

  auto error_code = proxy_->TryFetchDenseTable(table_id, &name, &value);

  if (error_code == ErrorCode::kSuccess) {
    std::unique_lock<std::shared_mutex> ll(model_mu_);

    std::unique_ptr<DenseTable> table(new DenseTable(table_id, name, value));
    tables_.Insert(table_id, std::move(table));
  } else {
    LOG_ERROR("TryFetchDenseTableFromProxy get error code:"
              << error_code << ", msg:" << ErrorCode::Msg(error_code));
  }
}

void Ps::TryCombineFetchDenseTableFromProxy(
    const std::vector<uint64_t>& table_ids) {
  assert(proxy_ != nullptr);

  std::vector<std::string> names;
  std::vector<Value> values;

  auto error_code =
      proxy_->TryCombineFetchDenseTable(table_ids, &names, &values);

  if (error_code == ErrorCode::kSuccess) {
    assert(table_ids.size() == names.size() &&
           table_ids.size() == values.size());

    std::unique_lock<std::shared_mutex> ll(model_mu_);

    for (size_t i = 0; i < table_ids.size(); ++i) {
      std::unique_ptr<DenseTable> table(
          new DenseTable(table_ids[i], names[i], values[i]));

      tables_.Insert(table_ids[i], std::move(table));
    }
  } else {
    LOG_ERROR("TryCombineFetchDenseTableFromProxy get error code:"
              << error_code << ", msg:" << ErrorCode::Msg(error_code));
  }
}

int32_t Ps::PullDenseTable(uint64_t router_version, uint64_t table_id,
                           Tensor* val) {
  std::shared_lock<std::shared_mutex> l(mu_);

  if (router_.Hit(utils::Hash(table_id)) != node_id_) {
    // We need check the router version it not equal to current router we need
    // tell the worker to update the router.
    if (router_version != router_.version()) {
      return ErrorCode::kRouterVersionError;
    }

    return ErrorCode::kRouteWrongNodeError;
  }

  if (status_ == (NodeStatus::kWork | NodeStatus::kProxy)) {
    std::shared_lock<std::shared_mutex> ll(model_mu_);

    auto it = tables_.Find(table_id);
    if (it.Valid() == false) {
      // We need release the lock maybe the proxy node will transfer it to this
      // node.
      ll.unlock();

      // Try fetch the DenseTable from proxy node.
      TryFetchDenseTableFromProxy(table_id);

      // lock again second time to check.
      ll.lock();

      // Try to find it.
      it = tables_.Find(table_id);
    }

    if (it.Valid() == false) {
      return ErrorCode::kTableNotExistError;
    }

    return it.value()->Pull(val);
  } else if (status_ == NodeStatus::kWork) {
    std::shared_lock<std::shared_mutex> ll(model_mu_);

    auto it = tables_.Find(table_id);
    if (it.Valid() == false) {
      return ErrorCode::kTableNotExistError;
    }

    return it.value()->Pull(val);
  } else {
    return ErrorCode::kNodeStatusError;
  }
}

int32_t Ps::CombinePullDenseTable(uint64_t router_version,
                                  const std::vector<uint64_t>& table_ids,
                                  std::vector<Tensor>* vals) {
  std::shared_lock<std::shared_mutex> l(mu_);

  for (auto table_id : table_ids) {
    if (router_.Hit(utils::Hash(table_id)) != node_id_) {
      if (router_version != router_.version()) {
        return ErrorCode::kRouterVersionError;
      }

      return ErrorCode::kRouteWrongNodeError;
    }
  }

  if (status_ == (NodeStatus::kWork | NodeStatus::kProxy)) {
    std::shared_lock<std::shared_mutex> ll(model_mu_);

    std::vector<uint64_t> not_exist_table_ids;
    not_exist_table_ids.reserve(table_ids.size());

    for (auto table_id : table_ids) {
      auto it = tables_.Find(table_id);
      if (it.Valid() == false) {
        not_exist_table_ids.emplace_back(table_id);
      }
    }

    if (not_exist_table_ids.empty() == false) {
      // Try to fetch from proxy node.
      ll.unlock();
      TryCombineFetchDenseTableFromProxy(not_exist_table_ids);
      ll.lock();
    }

    // Try pull again.
    vals->reserve(table_ids.size());
    for (size_t i = 0; i < table_ids.size(); ++i) {
      auto it = tables_.Find(table_ids[i]);

      if (it.Valid() == false) {
        return ErrorCode::kTableNotExistError;
      }

      Tensor val;
      auto error_code = it.value()->Pull(&val);
      if (error_code != ErrorCode::kSuccess) {
        return error_code;
      }

      vals->emplace_back(val);
    }

    return ErrorCode::kSuccess;
  } else if (status_ == NodeStatus::kWork) {
    std::shared_lock<std::shared_mutex> ll(model_mu_);

    // Try pull again.
    vals->reserve(table_ids.size());
    for (size_t i = 0; i < table_ids.size(); ++i) {
      auto it = tables_.Find(table_ids[i]);

      if (it.Valid() == false) {
        return ErrorCode::kTableNotExistError;
      }

      Tensor val;
      auto error_code = it.value()->Pull(&val);
      if (error_code != ErrorCode::kSuccess) {
        return error_code;
      }

      vals->emplace_back(val);
    }

    return ErrorCode::kSuccess;
  } else {
    return ErrorCode::kNodeStatusError;
  }
}

int32_t Ps::PushDenseTable(uint64_t router_version, uint64_t table_id,
                           const Tensor& grad, float lr) {
  std::shared_lock<std::shared_mutex> l(mu_);

  if (router_.Hit(utils::Hash(table_id)) != node_id_) {
    // We need check the router version it not equal to current router we need
    // tell the worker to update the router.
    if (router_version != router_.version()) {
      return ErrorCode::kRouterVersionError;
    }

    return ErrorCode::kRouteWrongNodeError;
  }

  if (status_ == (NodeStatus::kWork | NodeStatus::kProxy)) {
    std::shared_lock<std::shared_mutex> ll(model_mu_);

    auto it = tables_.Find(table_id);
    if (it.Valid() == false) {
      // We need release the lock maybe the proxy node will transfer it to this
      // node.
      ll.unlock();

      // Try fetch the DenseTable from proxy node.
      TryFetchDenseTableFromProxy(table_id);

      // lock again second time to check.
      ll.lock();

      // Try to find it.
      it = tables_.Find(table_id);
    }

    if (it.Valid() == false) {
      return ErrorCode::kTableNotExistError;
    }

    return it.value()->Push(optim_.get(), grad, lr);
  } else if (status_ == NodeStatus::kWork) {
    std::shared_lock<std::shared_mutex> ll(model_mu_);

    auto it = tables_.Find(table_id);
    if (it.Valid() == false) {
      return ErrorCode::kTableNotExistError;
    }

    return it.value()->Push(optim_.get(), grad, lr);
  } else {
    return ErrorCode::kNodeStatusError;
  }
}

}  // namespace kraken
