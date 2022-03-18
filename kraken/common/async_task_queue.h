#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

namespace kraken {

class AsyncTaskQueue {
private:
  using TASK = std::function<void()>;

  std::mutex mu_;
  std::condition_variable cond_;
  std::queue<TASK> que_;

  std::vector<std::thread> workers_;

  bool stop_;

public:
  AsyncTaskQueue(size_t thread_nums);

  ~AsyncTaskQueue() = default;

private:
  void Run();

public:
  void Stop();

  void Enque(TASK&& task);
};

}  // namespace kraken
