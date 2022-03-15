#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>

#include "ps/ps_server.h"
#include "scheduler/scheduler_server.h"

namespace kraken {
namespace test {

class Environment : public ::testing::Environment {
protected:
  static std::unique_ptr<std::thread> scheduler_t_;
  static std::unique_ptr<std::thread> server_t0_;
  static std::unique_ptr<std::thread> server_t1_;

public:
  ~Environment() override {
  }

  void SetUp() override {
    std::cout << "All test start, will start scheduler: 127.0.0.1:50000, 2 "
                 "server: 127.0.0.1:50001,127.0.0.1:50002.\n";
    scheduler_t_.reset(new std::thread([]() {
      kraken::SchedulerServer scheduler_server(50000);
      scheduler_server.Start();
    }));

    // Wait scheduler_ started.
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    server_t0_.reset(new std::thread([]() {
      uint32_t port = 50001;
      uint32_t thread_nums = 2;
      std::string addr = "127.0.0.1:50001";
      std::string s_addr = "127.0.0.1:50000";

      kraken::PsServer ps_server(port, thread_nums, addr, s_addr);
      ps_server.Start();
    }));

    // 2 node try join in at same time will cause one of it sleep about
    // 10s. So let's join one by one.
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    server_t1_.reset(new std::thread([]() {
      uint32_t port = 50002;
      uint32_t thread_nums = 2;
      std::string addr = "127.0.0.1:50002";
      std::string s_addr = "127.0.0.1:50000";

      kraken::PsServer ps_server(port, thread_nums, addr, s_addr);
      ps_server.Start();
    }));

    // Wait server started.
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  }

  void TearDown() override {
    std::cout << "All test finish.\n";

    scheduler_t_->detach();
    scheduler_t_.reset();

    server_t0_->detach();
    server_t0_.reset();

    server_t1_->detach();
    server_t1_.reset();
  }
};

std::unique_ptr<std::thread> Environment::scheduler_t_ = nullptr;
std::unique_ptr<std::thread> Environment::server_t0_ = nullptr;
std::unique_ptr<std::thread> Environment::server_t1_ = nullptr;

}  // namespace test
}  // namespace kraken

int main(int argc, char* argv[]) {
  testing::AddGlobalTestEnvironment(new kraken::test::Environment);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
