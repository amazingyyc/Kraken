#pragma once

#include <cinttypes>
#include <string>
#include <unordered_map>

#include "common/info.h"
#include "common/utils.h"
#include "t/tensor.h"

namespace kraken {

struct Bag;

class Optim {
protected:
  OptimType optim_type_;

protected:
  Optim(OptimType optim_type);

public:
  virtual ~Optim() = default;

  OptimType optim_type() const;

  virtual int32_t Update(const Tensor& grad, float lr, Tensor* val,
                         Bag* bag) const = 0;

public:
  static std::unique_ptr<Optim> Create(
      OptimType optim_type,
      const std::unordered_map<std::string, std::string>& optim_conf);
};

}  // namespace kraken
