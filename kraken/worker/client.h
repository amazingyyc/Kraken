#pragma once

#include <cinttypes>
#include <string>

#include "rpc/caller.h"

namespace kraken {

class Client {
private:
  uint32_t server_id_;

  std::string addr_;

  Caller caller_;

public:
  Client(uint32_t server_id, const std::string& addr);

public:
  uint32_t server_id() const;

  const std::string& addr() const;

  void Start();

  void Stop();

  template <typename ReqType, typename RspType>
  int32_t Call(uint32_t type, const ReqType& req, RspType* rsp) {
    return caller_.Call<ReqType, RspType>(type, req, rsp);
  }

  template <typename ReqType, typename RspType>
  void CallAsync(uint32_t type, const ReqType& req,
                 std::function<void(int32_t, RspType&)>&& callback) {
    caller_.CallAsync<ReqType, RspType>(type, req, std::move(callback));
  }
};

}  // namespace kraken
