#pragma once

#include <atomic>

namespace kraken {

class SpinLocker {
private:
  std::atomic_flag flag_;

public:
  SpinLocker();

  SpinLocker(const SpinLocker&) = delete;
  SpinLocker(SpinLocker&&) = delete;

  SpinLocker& operator=(const SpinLocker&) = delete;
  SpinLocker& operator=(SpinLocker&&) = delete;

  ~SpinLocker() = default;

public:
  void lock();

  void unlock();
};

class SpinLockerHandler {
private:
  SpinLocker& locker_;

public:
  SpinLockerHandler(SpinLocker& locker);

  ~SpinLockerHandler();
};

}  // namespace kraken
