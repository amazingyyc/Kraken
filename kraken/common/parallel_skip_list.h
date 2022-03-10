#pragma once

#include <cassert>
#include <functional>
#include <shared_mutex>
#include <vector>

#include "common/skip_list.h"

namespace kraken {

template <typename Key, typename Value, typename KeyHash = std::hash<Key>,
          size_t SlotCount = 16>
class ParallelSkipList {
public:
  class UniqueHandler {
  private:
    std::shared_mutex& mu_;

  public:
    SkipList<Key, Value>& skip_list;

    UniqueHandler(std::shared_mutex& mu, SkipList<Key, Value>& sl)
        : mu_(mu), skip_list(sl) {
      mu_.lock();
    }

    UniqueHandler(const UniqueHandler&) = delete;
    UniqueHandler(const UniqueHandler&&) = delete;
    const UniqueHandler& operator=(const UniqueHandler&) = delete;
    const UniqueHandler& operator=(const UniqueHandler&&) = delete;

    ~UniqueHandler() {
      mu_.unlock();
    }
  };

  class SharedHandler {
  private:
    std::shared_mutex& mu_;

  public:
    SkipList<Key, Value>& skip_list;

    SharedHandler(std::shared_mutex& mu, SkipList<Key, Value>& sl)
        : mu_(mu), skip_list(sl) {
      mu_.lock_shared();
    }

    SharedHandler(const SharedHandler&) = delete;
    SharedHandler(const SharedHandler&&) = delete;
    const SharedHandler& operator=(const SharedHandler&) = delete;
    const SharedHandler& operator=(const SharedHandler&&) = delete;

    ~SharedHandler() {
      mu_.unlock_shared();
    }
  };

private:
  KeyHash hash_;

  std::vector<std::shared_mutex> lockers_;
  std::vector<SkipList<Key, Value>> skip_lists_;

public:
  ParallelSkipList() : hash_(), lockers_(SlotCount), skip_lists_(SlotCount) {
  }

public:
  size_t slot_count() const {
    return SlotCount;
  }

  UniqueHandler UniqueSkipListHandler(size_t slot) {
    assert(slot < SlotCount);

    return UniqueHandler(lockers_[slot], skip_lists_[slot]);
  }

  SharedHandler SharedSkipListHandler(size_t slot) {
    assert(slot < SlotCount);

    return SharedHandler(lockers_[slot], skip_lists_[slot]);
  }

  void Clear() {
    for (size_t slot = 0; slot < SlotCount; ++slot) {
      std::unique_lock<std::shared_mutex> _(lockers_[slot]);

      skip_lists_[slot].Clear();
    }
  }

  bool Insert(const Key& key, const Value& value) {
    size_t slot = hash_(key) % SlotCount;

    std::unique_lock<std::shared_mutex> _(lockers_[slot]);
    return skip_lists_[slot].Insert(key, value);
  }

  bool Insert(const Key& key, const Value&& value) {
    size_t slot = hash_(key) % SlotCount;

    std::unique_lock<std::shared_mutex> _(lockers_[slot]);
    return skip_lists_[slot].Insert(key, std::move(value));
  }

  bool Remove(const Key& key) {
    size_t slot = hash_(key) % SlotCount;

    std::unique_lock<std::shared_mutex> _(lockers_[slot]);
    return skip_lists_[slot].Remove(key);
  }

  bool Contains(const Key& key) {
    size_t slot = hash_(key) % SlotCount;

    std::shared_lock<std::shared_mutex> _(lockers_[slot]);
    return skip_lists_[slot].Contains(key);
  }
};

}  // namespace kraken
