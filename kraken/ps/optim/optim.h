#pragma once

#include <cinttypes>
#include <string>
#include <unordered_map>

#include "common/info.h"
#include "t/tensor.h"

namespace kraken {

class Optim {
protected:
  OptimType optim_type_;

protected:
  Optim(OptimType optim_type);

public:
  virtual ~Optim() = default;

  OptimType optim_type() const;

  virtual int32_t Update(const Tensor& grad, float lr, Value* value) const = 0;

public:
  static std::unique_ptr<Optim> Create(
      OptimType optim_type,
      const std::unordered_map<std::string, std::string>& optim_conf);
};

}  // namespace kraken
