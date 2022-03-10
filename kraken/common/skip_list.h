#pragma once

#include <cassert>
#include <cstdlib>
#include <random>

namespace kraken {

// ref: https://github.com/google/leveldb/blob/main/db/skiplist.h
template <typename Key /*must be comparable*/, typename Value>
class SkipList {
private:
  constexpr static size_t kMaxHeight = 32;
  constexpr static int kBranching = 3;

private:
  struct Node {
    Key key;
    Value value;

    Node* next[0];

    inline Node* Next(size_t i) {
      return next[i];
    }

    inline void SetNext(size_t i, Node* x) {
      next[i] = x;
    }
  };

public:
  // The Iterator will be invalid when modify the SkipList.
  struct SeekIterator {
  private:
    friend class SkipList;

    const SkipList* list_;
    Node* prev_[kMaxHeight];
    Node* node_;

  public:
    SeekIterator(const SkipList* list) : list_(list), node_(nullptr) {
    }

    bool Valid() const {
      return node_ != nullptr;
    }

    const Key& key() const {
      assert(Valid());
      return node_->key;
    }

    const Value& value() const {
      assert(Valid());
      return node_->value;
    }

    void SeekGreaterOrEqual(const Key& target) {
      node_ = list_->FindGreaterOrEqual(target, prev_);
    }

    void Seek(const Key& target) {
      node_ = list_->FindGreaterOrEqual(target, prev_);

      if (node_ == nullptr || node_->key != target) {
        node_ = nullptr;
      }
    }
  };

private:
  Node* const header_;

  size_t max_height_;

  std::default_random_engine e_;
  std::uniform_int_distribution<> dist_;

public:
  SkipList()
      : header_(NewNode(kMaxHeight)),
        max_height_(1),
        e_(time(0)),
        dist_(0, kBranching) {
    for (size_t i = 0; i < kMaxHeight; ++i) {
      header_->SetNext(i, nullptr);
    }
  }

  ~SkipList() {
    Clear();

    free(header_);
  }

  SkipList(const SkipList&) = delete;
  SkipList(const SkipList&&) = delete;
  SkipList& operator=(const SkipList&) = delete;
  SkipList& operator=(const SkipList&&) = delete;

private:
  Node* NewNode(size_t height) {
    assert(height > 0);

    Node* node = (Node*)malloc(sizeof(Node) + height * sizeof(Node*));

    return node;
  }

  size_t RandomHeight() {
    size_t height = 1;
    while (height < kMaxHeight && dist_(e_) == 0) {
      height++;
    }

    return height;
  }

  bool KeyIsAfterNode(const Key& key, Node* x) const {
    return x != nullptr && x->key < key;
  }

  Node* FindGreaterOrEqual(const Key& key, Node** prev) const {
    Node* x = header_;
    size_t level = max_height_ - 1;

    while (true) {
      Node* next = x->Next(level);
      if (KeyIsAfterNode(key, next)) {
        x = next;
      } else {
        if (prev != nullptr) {
          prev[level] = x;
        }

        if (level == 0) {
          return next;
        } else {
          level--;
        }
      }
    }
  }

  Node* FindLessThan(const Key& key) const {
    Node* x = header_;
    int level = max_height_ - 1;

    while (true) {
      assert(x == header_ || x->key < key < 0);

      Node* next = x->Next(level);
      if (next == nullptr || next->key >= key) {
        if (level == 0) {
          return x;
        } else {
          level--;
        }
      } else {
        x = next;
      }
    }

    return (x != header_) ? x : nullptr;
  }

  Node* FindLast(const Key& key) const {
    Node* x = header_;
    int level = max_height_ - 1;

    while (true) {
      Node* next = x->Next(level);

      if (next == nullptr) {
        if (level == 0) {
          break;
        } else {
          level--;
        }
      } else {
        x = next;
      }
    }

    return x != header_ ? x : nullptr;
  }

public:
  void Clear() {
    while (header_->Next(0) != nullptr) {
      Node* node = header_->Next(0);
      header_->SetNext(0, node->Next(0));

      free(node);
    }

    for (size_t i = 0; i < kMaxHeight; ++i) {
      header_->SetNext(i, nullptr);
    }

    max_height_ = 1;
  }

  bool Insert(const Key& key, const Value& value) {
    Node* prev[kMaxHeight];
    Node* x = FindGreaterOrEqual(key, &prev);

    if (x != nullptr && x->key == key) {
      return false;
    }

    size_t height = RandomHeight();
    if (height > max_height_) {
      for (size_t i = max_height_; i < height; ++i) {
        prev[i] = header_;
      }

      max_height_ = height;
    }

    x = NewNode(height);
    x->key = key;
    x->value = value;

    for (size_t i = 0; i < height; ++i) {
      x->SetNext(i, prev[i]->Next(i));
      prev[i]->SetNext(i, x);
    }

    return true;
  }

  bool Insert(const Key& key, Value&& value) {
    Node* prev[kMaxHeight];
    Node* x = FindGreaterOrEqual(key, prev);

    if (x != nullptr && x->key == key) {
      return false;
    }

    size_t height = RandomHeight();
    if (height > max_height_) {
      for (size_t i = max_height_; i < height; ++i) {
        prev[i] = header_;
      }

      max_height_ = height;
    }

    x = NewNode(height);
    x->key = key;
    x->value = std::move(value);

    for (size_t i = 0; i < height; ++i) {
      x->SetNext(i, prev[i]->Next(i));
      prev[i]->SetNext(i, x);
    }

    return true;
  }

  bool Remove(const Key& key) {
    Node* prev[kMaxHeight];
    Node* x = FindGreaterOrEqual(key, &prev);

    if (x == nullptr || x->key != key) {
      return false;
    }

    for (size_t i = 0; i < max_height_; ++i) {
      if (prev[i]->Next(i) == x) {
        prev[i]->SetNext(x->Next(i));
        x->SetNext(i, nullptr);
      }
    }

    free(x);

    return true;
  }

  bool Contains(const Key& key) const {
    Node* x = FindGreaterOrEqual(key, nullptr);

    if (x != nullptr && key == x->key) {
      return true;
    } else {
      return false;
    }
  }

  SeekIterator Find(const Key& key) const {
    SeekIterator it(this);
    it.Seek(key);

    return it;
  }

  SeekIterator FindGreaterOrEqual(const Key& key) const {
    SeekIterator it(this);
    it.SeekGreaterOrEqual(key);

    return it;
  }

  // Becareful when call this function the user must make sure the SkipList not
  // modified after get the SeekIterator(it).
  // And after call this function the SeekIterator will become invalid.
  bool Remove(const SeekIterator& it) {
    if (it.Valid() == false) {
      return false;
    }

    Node* x = it.node_;

    for (size_t i = 0; i < max_height_; ++i) {
      if (it.prev_[i]->Next(i) == x) {
        it.prev_[i]->SetNext(i, x->Next(i));
        x->SetNext(i, nullptr);
      }
    }

    free(x);

    return true;
  }
};

}  // namespace kraken
