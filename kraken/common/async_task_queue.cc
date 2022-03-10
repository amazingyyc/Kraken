#include "common/async_task_queue.h"

namespace kraken {

AsyncTaskQueue::AsyncTaskQueue(size_t thread_nums) : stop_(false) {
  for (size_t i = 0; i < thread_nums; ++i) {
    std::thread t(&AsyncTaskQueue::Run, this);

    workers_.emplace_back(std::move(t));
  }
}

void AsyncTaskQueue::Run() {
  while (true) {
    std::unique_lock<std::mutex> lock(mu_);
    if (que_.empty() == false) {
      TASK task = std::move(que_.front());
      que_.pop();
      lock.unlock();

      task();
    } else if (stop_) {
      break;
    } else {
      cond_.wait(lock, [this]() -> bool {
        return this->stop_ || this->que_.empty() == false;
      });
    }
  }
}

void AsyncTaskQueue::Stop() {
  std::unique_lock<std::mutex> lock(mu_);
  stop_ = true;
  lock.unlock();

  cond_.notify_all();

  for (auto& t : workers_) {
    if (t.joinable()) {
      t.join();
    }
  }
}

void AsyncTaskQueue::Enque(std::function<void()>&& task) {
  std::unique_lock<std::mutex> lock(mu_);
  que_.push(std::move(task));
  lock.unlock();

  cond_.notify_one();
}

}  // namespace kraken
