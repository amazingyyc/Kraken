#include "common/consistent_hasher.h"

#include <limits>

#include "common/exception.h"

namespace kraken {

ConsistentHasher::ConsistentHasher(size_t bucket_count)
    : bucket_count_(bucket_count) {
  uint64_t max_v = std::numeric_limits<uint64_t>::max();

  ARGUMENT_CHECK(
      bucket_count_ > 0 && bucket_count_ < (max_v / 2),
      "ConsistentHasher need bucket_count > 0 and < uint64_t_max_value / 2.");

  buckets_.resize(bucket_count_);
  stride_ = max_v / bucket_count_;

  if (bucket_count_ != 1 && (max_v - bucket_count_ + 1) % bucket_count == 0) {
    stride_++;
  }

  for (size_t i = 0; i < bucket_count_; ++i) {
    buckets_[i] = stride_ * i;
  }
}

uint64_t ConsistentHasher::Hash(uint64_t v1, uint64_t v2) const {
  uint64_t seed = 0;
  seed ^= v1 + 0x9e3779b9;
  seed ^= v2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);

  return seed;
}

uint64_t ConsistentHasher::Hash(uint64_t v1, uint64_t v2, uint64_t v3) const {
  uint64_t seed = 0;
  seed ^= v1 + 0x9e3779b9;
  seed ^= v2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= v3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);

  return seed;
}

size_t ConsistentHasher::bucket_count() const {
  return bucket_count_;
}

const std::vector<uint64_t>& ConsistentHasher::buckets() const {
  return buckets_;
}

std::pair<uint64_t, uint64_t> ConsistentHasher::Boundary(
    size_t shard_id) const {
  ARGUMENT_CHECK(shard_id < buckets_.size(),
                 "shard_id:" << shard_id << " out of range.");

  uint64_t lower = buckets_[shard_id];
  uint64_t upper = (shard_id + 1 == buckets_.size())
                       ? std::numeric_limits<uint64_t>::max()
                       : buckets_[shard_id + 1] - 1;

  return std::make_pair(lower, upper);
}

size_t ConsistentHasher::Hit(uint64_t h) const {
  size_t s = 0;
  size_t e = bucket_count_;

  while (s + 1 < e) {
    size_t m = s + (e - s) / 2;

    if (h >= buckets_[m]) {
      s = m;
    } else {
      e = m;
    }
  }

  return s;
}

size_t ConsistentHasher::operator()(uint64_t v1, uint64_t v2) const {
  uint64_t h = Hash(v1, v2);

  return Hit(h);
}

size_t ConsistentHasher::operator()(uint64_t v1, uint64_t v2,
                                    uint64_t v3) const {
  uint64_t h = Hash(v1, v2, v3);

  return Hit(h);
}

}  // namespace kraken
