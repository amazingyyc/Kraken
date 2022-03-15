#include <thread>

#include "common/log.h"
#include "ps/dense_table.h"
#include "ps/ps.h"
#include "ps/sparse_table.h"

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

  std::vector<uint64_t> exist_ids;
  std::vector<std::string> names;
  std::vector<Value> values;

  auto error_code =
      proxy_->TryCombineFetchDenseTable(table_ids, &exist_ids, &names, &values);

  if (error_code == ErrorCode::kSuccess) {
    assert(exist_ids.size() == names.size() &&
           exist_ids.size() == values.size());

    std::unique_lock<std::shared_mutex> ll(model_mu_);

    for (size_t i = 0; i < exist_ids.size(); ++i) {
      std::unique_ptr<DenseTable> table(
          new DenseTable(exist_ids[i], names[i], values[i]));

      tables_.Insert(exist_ids[i], std::move(table));
    }
  } else {
    LOG_ERROR("TryCombineFetchDenseTableFromProxy get error code:"
              << error_code << ", msg:" << ErrorCode::Msg(error_code));
  }
}

void Ps::TryFetchSparseMetaDataFromProxy(uint64_t table_id) {
  assert(proxy_ != nullptr);

  std::string name;
  int64_t dimension;
  ElementType element_type;
  InitializerType init_type;
  std::unordered_map<std::string, std::string> init_conf;

  auto error_code = proxy_->TryFetchSparseMetaData(
      table_id, &name, &dimension, &element_type, &init_type, &init_conf);

  if (error_code == ErrorCode::kSuccess) {
    std::unique_lock<std::shared_mutex> ll(model_mu_);

    auto it = tables_.Find(table_id);
    if (it.Valid()) {
      // Already exist just skip it.
      return;
    }

    if (dimension <= 0) {
      LOG_ERROR(
          "TryFetchSparseTableFromProxy get wrong dimension:" << dimension);
      return;
    }

    std::unique_ptr<Initializer> initializer =
        Initializer::Create(init_type, init_conf);
    if (initializer == nullptr) {
      LOG_ERROR("TryFetchSparseTableFromProxy get unsupport initialize type:"
                << init_type);
      return;
    }

    std::unique_ptr<SparseTable> table(new SparseTable(
        table_id, name, dimension, element_type, std::move(initializer)));

    tables_.Insert(table_id, std::move(table));
  } else {
    LOG_ERROR("TryFetchSparseTableFromProxy get error code:"
              << error_code << ", msg:" << ErrorCode::Msg(error_code));
  }
}

void Ps::TryFetchSparseValuesFromProxy(
    uint64_t table_id, const std::vector<uint64_t>& sparse_ids) {
  assert(proxy_ != nullptr);

  std::vector<uint64_t> exist_sparse_ids;
  std::vector<Value> values;

  auto error_code = proxy_->TryFetchSparseValues(table_id, sparse_ids,
                                                 &exist_sparse_ids, &values);
  if (error_code == ErrorCode::kSuccess) {
    assert(exist_sparse_ids.size() == values.size());

    std::shared_lock<std::shared_mutex> ll(model_mu_);

    auto it = tables_.Find(table_id);
    if (it.Valid() == false || it.value()->type() != TableType::kSparse) {
      LOG_ERROR("TryFetchSparseValuesFromPorxy cannot find SparseTable:["
                << table_id << "]");
      return;
    }

    SparseTable* table = (SparseTable*)it.value().get();
    table->vals()->Insert(exist_sparse_ids, values);
  } else {
    LOG_ERROR("TryFetchSparseValuesFromPorxy get error code:"
              << error_code << ", msg:" << ErrorCode::Msg(error_code));
  }
}

int32_t Ps::PullDenseTable(uint64_t router_version, uint64_t table_id,
                           Tensor* val) {
  std::shared_lock<std::shared_mutex> l(mu_);

  if (!(status_ & NodeStatus::kWork)) {
    return ErrorCode::kNodeStatusError;
  }

  if (router_.Hit(utils::Hash(table_id)) != node_id_) {
    // We need check the router version it not equal to current router we need
    // tell the worker to update the router.
    if (router_version != router_.version()) {
      return ErrorCode::kRouterVersionError;
    }

    return ErrorCode::kRouteWrongNodeError;
  }

  if (status_ & NodeStatus::kProxy) {
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
  } else {
    std::shared_lock<std::shared_mutex> ll(model_mu_);

    auto it = tables_.Find(table_id);
    if (it.Valid() == false) {
      return ErrorCode::kTableNotExistError;
    }

    return it.value()->Pull(val);
  }
}

int32_t Ps::CombinePullDenseTable(uint64_t router_version,
                                  const std::vector<uint64_t>& table_ids,
                                  std::vector<Tensor>* vals) {
  std::shared_lock<std::shared_mutex> l(mu_);

  if (!(status_ & NodeStatus::kWork)) {
    return ErrorCode::kNodeStatusError;
  }

  for (auto table_id : table_ids) {
    if (router_.Hit(utils::Hash(table_id)) != node_id_) {
      if (router_version != router_.version()) {
        return ErrorCode::kRouterVersionError;
      }

      return ErrorCode::kRouteWrongNodeError;
    }
  }

  if (status_ & NodeStatus::kProxy) {
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
        std::cout << "DenseTableId not exit:" << table_ids[i] << "\n";
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
  }
}

int32_t Ps::PushDenseTable(uint64_t router_version, uint64_t table_id,
                           const Tensor& grad, float lr) {
  std::shared_lock<std::shared_mutex> l(mu_);

  if (!(status_ & NodeStatus::kWork)) {
    return ErrorCode::kNodeStatusError;
  }

  if (router_.Hit(utils::Hash(table_id)) != node_id_) {
    // We need check the router version it not equal to current router we need
    // tell the worker to update the router.
    if (router_version != router_.version()) {
      return ErrorCode::kRouterVersionError;
    }

    return ErrorCode::kRouteWrongNodeError;
  }

  if (status_ & NodeStatus::kProxy) {
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
  } else {
    std::shared_lock<std::shared_mutex> ll(model_mu_);

    auto it = tables_.Find(table_id);
    if (it.Valid() == false) {
      return ErrorCode::kTableNotExistError;
    }

    return it.value()->Push(optim_.get(), grad, lr);
  }
}

int32_t Ps::PullSparseTable(uint64_t router_version, uint64_t table_id,
                            const std::vector<uint64_t>& sparse_ids,
                            std::vector<Tensor>* vals) {
  std::shared_lock<std::shared_mutex> l(mu_);

  if (!(status_ & NodeStatus::kWork)) {
    return ErrorCode::kNodeStatusError;
  }

  // Check whether Route to wrong node.
  if (router_version != router_.version()) {
    return ErrorCode::kRouterVersionError;
  }

  if (status_ & NodeStatus::kProxy) {
    std::shared_lock<std::shared_mutex> ll(model_mu_);

    auto it = tables_.Find(table_id);
    if (it.Valid() == false) {
      ll.unlock();
      TryFetchSparseMetaDataFromProxy(table_id);
      ll.lock();

      // Find again.
      it = tables_.Find(table_id);
    }

    if (it.Valid() == false || it.value()->type() != TableType::kSparse) {
      return ErrorCode::kTableNotExistError;
    }

    SparseTable* table = (SparseTable*)it.value().get();

    std::vector<uint64_t> not_exist_ids;
    not_exist_ids.reserve(sparse_ids.size());

    for (auto sparse_id : sparse_ids) {
      if (table->vals()->Contains(sparse_id) == false) {
        not_exist_ids.emplace_back(sparse_id);
      }
    }

    if (not_exist_ids.empty() == false) {
      ll.unlock();
      TryFetchSparseValuesFromProxy(table_id, not_exist_ids);
      ll.lock();

      // At here we release the locker so the it maybe become invalid.
      // We need try to find it agagin.
      it = tables_.Find(table_id);
      if (it.Valid() == false || it.value()->type() != TableType::kSparse) {
        return ErrorCode::kTableNotExistError;
      }
    }

    // Try to pull again.
    return it.value()->Pull(sparse_ids, vals);
  } else {
    // Normal.
    std::shared_lock<std::shared_mutex> ll(model_mu_);

    auto it = tables_.Find(table_id);
    if (it.Valid() == false) {
      return ErrorCode::kTableNotExistError;
    }

    return it.value()->Pull(sparse_ids, vals);
  }
}

int32_t Ps::PushSparseTable(uint64_t router_version, uint64_t table_id,
                            const std::vector<uint64_t>& sparse_ids,
                            const std::vector<Tensor>& grads, float lr) {
  std::shared_lock<std::shared_mutex> l(mu_);

  if (!(status_ & NodeStatus::kWork)) {
    return ErrorCode::kNodeStatusError;
  }

  // Check whether Route to wrong node.
  if (router_version != router_.version()) {
    return ErrorCode::kRouterVersionError;
  }

  if (status_ & NodeStatus::kProxy) {
    std::shared_lock<std::shared_mutex> ll(model_mu_);

    auto it = tables_.Find(table_id);
    if (it.Valid() == false) {
      ll.unlock();
      TryFetchSparseMetaDataFromProxy(table_id);
      ll.lock();

      // Find again.
      it = tables_.Find(table_id);
    }

    if (it.Valid() == false || it.value()->type() != TableType::kSparse) {
      return ErrorCode::kTableNotExistError;
    }

    SparseTable* table = (SparseTable*)it.value().get();

    std::vector<uint64_t> not_exist_ids;
    not_exist_ids.reserve(sparse_ids.size());

    for (auto sparse_id : sparse_ids) {
      if (table->vals()->Contains(sparse_id) == false) {
        not_exist_ids.emplace_back(sparse_id);
      }
    }

    if (not_exist_ids.empty() == false) {
      ll.unlock();
      TryFetchSparseValuesFromProxy(table_id, not_exist_ids);
      ll.lock();

      // At here we release the locker so the it maybe become invalid.
      // We need try to find it agagin.
      it = tables_.Find(table_id);
      if (it.Valid() == false || it.value()->type() != TableType::kSparse) {
        return ErrorCode::kTableNotExistError;
      }
    }

    return it.value()->Push(optim_.get(), sparse_ids, grads, lr);
  } else {
    std::shared_lock<std::shared_mutex> ll(model_mu_);

    auto it = tables_.Find(table_id);
    if (it.Valid() == false) {
      return ErrorCode::kTableNotExistError;
    }

    return it.value()->Push(optim_.get(), sparse_ids, grads, lr);
  }
}

}  // namespace kraken
