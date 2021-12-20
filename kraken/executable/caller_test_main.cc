#include <iostream>
#include <thread>

#include "common/log.h"
#include "rpc/caller.h"

int main(int argc, char* argv[]) {
  // tcp://localhost:5557
  std::string addr = "tcp://localhost:5000";

  kraken::Caller caller(addr);
  caller.Start();

  // std::function<void(int32_t, const RspType&)>&& callback
  auto callback = [](int32_t error_code, const std::string& reply) {
    LOG_INFO("error_code:" << error_code << ", reply:" << reply);
  };

  std::string hello = "Hello world!";

  caller.CallAsync<std::string, std::string>(0, hello, std::move(callback));

  using namespace std::chrono_literals;
  std::this_thread::sleep_for(200000ms);

  return 0;
}
