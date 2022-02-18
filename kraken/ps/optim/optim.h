#pragma once

#include <cinttypes>
#include <string>
#include <unordered_map>

#include "common/utils.h"
#include "t/tensor.h"

namespace kraken {

struct Bag;

enum class OptimType : uint8_t {
  kAdagrad = 0,
  kAdam = 1,
  kRMSprop = 2,
  kSGD = 3,
};

class Optim {
protected:
  OptimType optim_type_;

protected:
  Optim(OptimType optim_type);

public:
  virtual ~Optim() = default;

  virtual int32_t Update(const Tensor& grad, float lr, Tensor* val,
                         Bag* bag) const = 0;

public:
  static std::unique_ptr<Optim> Create(
      OptimType optim_type,
      const std::unordered_map<std::string, std::string>& optim_conf);
};

}  // namespace kraken
