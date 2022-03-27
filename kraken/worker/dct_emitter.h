#pragma once

#include <atomic>
#include <cinttypes>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

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

  std::unordered_map<uint64_t /*DenseTable id*/, DenseBag> dense_bags_;

public:
  DCTEmitter(uint64_t life_span, float eta);

  ~DCTEmitter() = default;

public:
  uint64_t RegisterDenseTable(const std::string& name,
                              const Tensor& val) override;

  void PushDenseTable(uint64_t table_id, const Tensor& grad) override;
};

}  // namespace kraken
