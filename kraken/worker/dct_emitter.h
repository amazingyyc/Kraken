#pragma once

#include <atomic>
#include <cinttypes>
#include <memory>
#include <unordered_map>
#include <vector>

#include "parallel_hashmap/parallel_hashmap/phmap.h"
#include "t/tensor.h"
#include "worker/emitter.h"

namespace kraken {

// ref: Training Recommender Systems at Scale: Communication-Efficient Model and
// Data Parallelism
class DCTEmitter : public Emitter {
private:
  // Every DenseTable has a Bag to store the e_grad, step etc.
  class DenseBag {
  private:
    std::shared_mutex mu_;

    Tensor e_grad_;

    // topk value.
    float tau_;

    // current step.
    uint64_t step_;

  public:
    DenseBag(const Tensor& e_grad);

    Tensor MaybeToCoo(uint64_t life_span, float eta, const Tensor& grad);
  };

  uint64_t life_span_;
  float eta_;

  std::shared_mutex mu_;
  phmap::flat_hash_map<uint64_t, std::unique_ptr<DenseBag>> dense_bags_;

public:
  DCTEmitter(uint64_t life_span, float eta);

  ~DCTEmitter() = default;

public:
  uint64_t RegisterDenseTable(const std::string& name,
                              const Tensor& val) override;

  void PushDenseTable(uint64_t table_id, const Tensor& grad) override;

  Tensor PushPullDenseTable(uint64_t table_id, const Tensor& grad) override;
};

}  // namespace kraken