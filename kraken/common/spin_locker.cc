#include "common/spin_locker.h"

namespace kraken {

SpinLocker::SpinLocker() {
  flag_.clear();
}

void SpinLocker::lock() {
  while (flag_.test_and_set(std::memory_order_acq_rel))
    ;
}

void SpinLocker::unlock() {
  flag_.clear(std::memory_order_release);
}

SpinLockerHandler::SpinLockerHandler(SpinLocker& locker) : locker_(locker) {
  locker_.lock();
}

SpinLockerHandler::~SpinLockerHandler() {
  locker_.unlock();
}

}  // namespace kraken
