#pragma once

#include <atomic>
#include <cinttypes>
#include <memory>
#include <vector>

#include "worker/emitter.h"

namespace kraken {

// ref: Training Recommender Systems at Scale: Communication-Efficient Model and
// Data Parallelism
class DCTEmitter : public Emitter {
public:
  DCTEmitter();

  ~DCTEmitter() = default;

public:
  void PushDenseTable(uint64_t table_id, const Tensor& grad) override;

  Tensor PushPullDenseTable(uint64_t table_id, const Tensor& grad) override;
};

}  // namespace kraken
