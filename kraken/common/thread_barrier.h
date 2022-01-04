#pragma once

#include <condition_variable>
#include <mutex>

namespace kraken {

class ThreadBarrier {
private:
  uint32_t counter_;

  std::mutex mu_;

  std::condition_variable cond_;

public:
  ThreadBarrier(uint32_t counter) : counter_(counter) {
  }

  void Release() {
    bool done = false;
    {
      std::unique_lock<std::mutex> lock(mu_);
      counter_--;

      done = (0 == counter_);
    }

    if (done) {
      cond_.notify_one();
    }
  }

  void Wait() {
    std::unique_lock<std::mutex> lock(mu_);

    while (counter_ > 0) {
      cond_.wait(lock);
    }
  }
};

}  // namespace kraken
