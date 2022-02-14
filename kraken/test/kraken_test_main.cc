#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>

#include "ps/ps_server.h"

namespace kraken {
namespace test {

class Environment : public ::testing::Environment {
protected:
  static std::unique_ptr<std::thread> server_t0_;
  static std::unique_ptr<std::thread> server_t1_;

public:
  ~Environment() override {
  }

  void SetUp() override {
    std::cout << "All test start, will start 2 server: "
                 "127.0.0.1:50000,127.0.0.1:50001.\n";

    server_t0_.reset(new std::thread([]() {
      uint32_t port = 50000;
      uint32_t thread_nums = 2;
      uint32_t shard_num = 2;
      uint32_t shard_id = 0;

      kraken::PsServer ps_server(port, thread_nums, shard_num, shard_id, "", 0);
      ps_server.Start();
    }));

    server_t1_.reset(new std::thread([]() {
      uint32_t port = 50001;
      uint32_t thread_nums = 2;
      uint32_t shard_num = 2;
      uint32_t shard_id = 1;

      kraken::PsServer ps_server(port, thread_nums, shard_num, shard_id, "", 0);
      ps_server.Start();
    }));

    // Wait server started.
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }

  void TearDown() override {
    std::cout << "All test finish.\n";

    server_t0_->detach();
    server_t0_.reset();

    server_t1_->detach();
    server_t1_.reset();
  }
};

std::unique_ptr<std::thread> Environment::server_t0_ = nullptr;
std::unique_ptr<std::thread> Environment::server_t1_ = nullptr;

}  // namespace test
}  // namespace kraken

int main(int argc, char* argv[]) {
  testing::AddGlobalTestEnvironment(new kraken::test::Environment);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
