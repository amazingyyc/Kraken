#pragma once

#include <cinttypes>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "rpc/indep_connecter.h"

namespace kraken {

class GroupConnecters {
private:
  CompressType compress_type_;

  std::unordered_map<uint64_t /*Ps node id*/, std::unique_ptr<IndepConnecter>>
      connecters_;

public:
  GroupConnecters(CompressType compress_type) : compress_type_(compress_type) {
  }

  ~GroupConnecters() {
    for (auto& [_, conn] : connecters_) {
      conn->Stop();
    }

    connecters_.clear();
  }

public:
  void Add(uint64_t node_id, const std::string& addr) {
    auto it = connecters_.find(node_id);
    if (it != connecters_.end()) {
      if (it->second->addr() == addr) {
        return;
      }

      it->second->Stop();
      connecters_.erase(it);
    }

    std::unique_ptr<IndepConnecter> conn(
        new IndepConnecter(addr, compress_type_));
    conn->Start();

    connecters_.emplace(node_id, std::move(conn));

    return;
  }

  void Remove(uint64_t node_id) {
    auto it = connecters_.find(node_id);
    if (it != connecters_.end()) {
      it->second->Stop();
      connecters_.erase(it);
    }
  }

  // The caller must make sure the node_id is Added.
  template <typename ReqType, typename ReplyType>
  int32_t Call(uint64_t node_id, uint32_t rpc_type, const ReqType& req,
               ReplyType* reply, int64_t timeout_ms = 5000) const {
    auto it = connecters_.find(node_id);

    return it->second->Call<ReqType, ReplyType>(rpc_type, req, reply,
                                                timeout_ms);
  }

  template <typename ReqType, typename ReplyType>
  int32_t Call(uint32_t rpc_type,
               const std::unordered_map<uint64_t, ReqType>& reqs,
               std::unordered_map<uint64_t, ReplyType>* replies,
               int64_t timeout_ms = 5000) const {
    size_t count = reqs.size();
    if (count == 0) {
      return ErrorCode::kSuccess;
    }

    std::vector<uint64_t> node_ids;
    node_ids.reserve(count);
    for (const auto& [k, _] : reqs) {
      node_ids.emplace_back(k);
    }

    std::vector<ReplyType> replies_v;
    replies_v.resize(count);

    std::vector<int32_t> error_codes;
    error_codes.resize(count);

    ThreadBarrier barrier(count);
    for (size_t i = 0; i < count; ++i) {
      auto callback = [&error_codes, &replies_v, i, &barrier](
                          int32_t r_error_code, ReplyType& r_reply) {
        error_codes[i] = r_error_code;
        replies_v[i] = std::move(r_reply);

        barrier.Release();
      };

      auto cit = connecters_.find(node_ids[i]);
      auto rit = reqs.find(node_ids[i]);

      cit->second->CallAsync<ReqType, ReplyType>(
          rpc_type, rit->second, std::move(callback), timeout_ms);
    }

    barrier.Wait();

    for (size_t i = 0; i < count; ++i) {
      if (error_codes[i] != ErrorCode::kSuccess) {
        return error_codes[i];
      }
    }

    replies->reserve(count);
    for (size_t i = 0; i < count; ++i) {
      replies->emplace(node_ids[i], std::move(replies_v[i]));
    }

    return ErrorCode::kSuccess;
  }

  template <typename ReqType, typename ReplyType>
  void CallAsync(uint32_t rpc_type,
                 const std::unordered_map<uint64_t, ReqType>& reqs,
                 const std::function<void(int32_t, ReplyType&)>& callback,
                 int64_t timeout_ms = 5000) {
    for (const auto& [node_id, req] : reqs) {
      auto l_callback = callback;

      auto it = connecters_.find(node_id);
      it->second->CallAsync(rpc_type, req, std::move(l_callback), timeout_ms);
    }
  }

  // The caller must make sure the node_id is Added.
  template <typename ReqType, typename ReplyType>
  void CallAsync(uint64_t node_id, uint32_t rpc_type, const ReqType& req,
                 std::function<void(int32_t, ReplyType&)>&& callback,
                 int64_t timeout_ms = 5000) const {
    auto it = connecters_.find(node_id);

    it->second->CallAsync<ReqType, ReplyType>(rpc_type, req,
                                              std::move(callback), timeout_ms);
  }
};

}  // namespace kraken
