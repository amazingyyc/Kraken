#include "common/spin_locker.h"

namespace kraken {

SpinLocker::SpinLocker() {
  flag_.clear();
}

void SpinLocker::Lock() {
  while (flag_.test_and_set(std::memory_order_acq_rel))
    ;
}

void SpinLocker::UnLock() {
  flag_.clear(std::memory_order_release);
}

SpinLockerHandler::SpinLockerHandler(SpinLocker& locker) : locker_(locker) {
  locker_.Lock();
}

SpinLockerHandler::~SpinLockerHandler() {
  locker_.UnLock();
}

}  // namespace kraken
