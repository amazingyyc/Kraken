#pragma once

#include <atomic>
#include <cinttypes>
#include <memory>
#include <shared_mutex>
#include <vector>

#include "common/router.h"
#include "rpc/indep_connecter.h"
#include "rpc/protocol.h"

namespace kraken {

enum class EmitterType : uint8_t {
  kDefault = 0,
  kDCT = 1,  // ref: Training Recommender Systems at Scale:
             // Communication-Efficient Model and Data Parallelism
};

class Emitter {
protected:
  EmitterType type_;
  std::atomic<bool> initialized_;

  // Scheduler addr.
  std::string s_addr_;
  CompressType compress_type_;

  std::shared_mutex mu_;
  // Scheduler connecter.
  std::unique_ptr<IndepConnecter> s_connecter_;
  Router router_;

  std::string model_name_;
  std::atomic<float> lr_;

public:
  Emitter();

protected:
  Emitter(EmitterType type);

public:
  void Initialize(const std::string& s_addr, CompressType compress_type);

  void Stop();

  void UpdateLR(float lr);

  void InitModel(
      const std::string& model_name, OptimType optim_type,
      const std::unordered_map<std::string, std::string>& optim_conf);

  uint64_t RegisterDenseTable(const std::string& name, const Tensor& val);

  uint64_t RegisterSparseTable(
      const std::string& name, int64_t dimension, ElementType element_type,
      InitializerType init_type,
      const std::unordered_map<std::string, std::string>& init_conf);
};

}  // namespace kraken
