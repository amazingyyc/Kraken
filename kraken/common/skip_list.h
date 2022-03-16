#pragma once

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <random>

namespace kraken {

// ref: https://github.com/google/leveldb/blob/main/db/skiplist.h
template <typename Key /*must be comparable*/, typename Value>
class SkipList {
private:
  constexpr static size_t kMaxHeight = 24;
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

  private:
    SeekIterator(const SkipList* list) : list_(list), node_(nullptr) {
    }

  public:
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

    Value& value() {
      assert(Valid());
      return node_->value;
    }

    // Becareful call Next the caller must make sure not modify the List (like
    // Insert/Remove). Or the Iterator will be undefined behavior.
    void Next() {
      assert(Valid());

      Node* next = node_->Next(0);

      for (size_t i = 0; i < list_->max_height_; ++i) {
        if (prev_[i]->Next(i) == node_) {
          prev_[i] = node_;
        }
      }

      node_ = next;
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

    DeleteNode(header_);
  }

  SkipList(const SkipList&) = delete;
  SkipList(const SkipList&&) = delete;
  SkipList& operator=(const SkipList&) = delete;
  SkipList& operator=(const SkipList&&) = delete;

private:
  inline Node* NewNode(size_t height) {
    assert(height > 0);

    void* ptr = malloc(sizeof(Node) + height * sizeof(Node*));
    Node* node = new (ptr) Node();

    return node;
  }

  inline void DeleteNode(Node* x) {
    x->~Node();
    free(x);
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
  // O(n)
  size_t Size() const {
    size_t size = 0;

    Node* x = header_->Next(0);
    while (x != nullptr) {
      size += 1;
      x = x->Next(0);
    }

    return size;
  }

  void Clear() {
    while (header_->Next(0) != nullptr) {
      Node* node = header_->Next(0);
      header_->SetNext(0, node->Next(0));

      DeleteNode(node);
    }

    for (size_t i = 0; i < kMaxHeight; ++i) {
      header_->SetNext(i, nullptr);
    }

    max_height_ = 1;
  }

  bool Insert(const Key& key, const Value& value) {
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

  bool Contains(const Key& key) const {
    Node* x = FindGreaterOrEqual(key, nullptr);

    if (x != nullptr && key == x->key) {
      return true;
    } else {
      return false;
    }
  }

  SeekIterator Begin() const {
    SeekIterator it(this);
    it.node_ = header_->Next(0);

    for (size_t i = 0; i < max_height_; ++i) {
      it.prev_[i] = header_;
    }

    return it;
  }

  SeekIterator Find(const Key& key) const {
    SeekIterator it(this);
    it.node_ = FindGreaterOrEqual(key, it.prev_);

    if (it.node_ == nullptr || it.node_->key != key) {
      it.node_ = nullptr;
    }

    return it;
  }

  SeekIterator FindGreaterOrEqual(const Key& key) const {
    SeekIterator it(this);
    it.node_ = FindGreaterOrEqual(key, it.prev_);

    return it;
  }

  // Remove a node and return the next Iterator.
  // Becareful when get the SeekIterator the user cannot modify the List
  // (Insert/Remove) or will let the list be Undefined behavior.
  SeekIterator Remove(const SeekIterator& it) {
    if (it.Valid() == false) {
      throw std::runtime_error("Invalid SeekIterator.");
    }

    SeekIterator n_it(this);
    n_it.node_ = it.node_->Next(0);

    for (size_t i = 0; i < max_height_; ++i) {
      n_it.prev_[i] = it.prev_[i];
    }

    Node* x = it.node_;
    for (size_t i = 0; i < max_height_; ++i) {
      if (it.prev_[i]->Next(i) == x) {
        it.prev_[i]->SetNext(i, x->Next(i));
        x->SetNext(i, nullptr);
      }
    }

    DeleteNode(x);

    return n_it;
  }
};

}  // namespace kraken
