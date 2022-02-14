#pragma once

#include <cinttypes>
#include <cstddef>
#include <vector>

namespace kraken {

class ConsistentHasher {
private:
  // how many bucket for this hasher.
  size_t bucket_count_;
  uint64_t stride_;

  std::vector<uint64_t> buckets_;

public:
  ConsistentHasher(size_t bucket_count);

private:
  uint64_t Hash(uint64_t v1, uint64_t v2) const;

  uint64_t Hash(uint64_t v1, uint64_t v2, uint64_t v3) const;

public:
  size_t bucket_count() const;

  const std::vector<uint64_t>& buckets() const;

  // the boundary is: [lower, upper] both are closed interval.
  std::pair<uint64_t, uint64_t> Boundary(size_t shard_id) const;

  size_t Hit(uint64_t h) const;

  size_t operator()(uint64_t v1, uint64_t v2) const;

  size_t operator()(uint64_t v1, uint64_t v2, uint64_t v3) const;
};

}  // namespace kraken
