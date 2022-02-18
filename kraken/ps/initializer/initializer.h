#pragma once

#include <cinttypes>
#include <string>
#include <unordered_map>

#include "common/utils.h"
#include "t/tensor.h"

namespace kraken {

enum class InitializerType : uint8_t {
  kConstant = 0,
  kUniform = 1,
  kNormal = 2,
  kXavierUniform = 3,
  kXavierNormal = 4,
};

class Initializer {
protected:
  InitializerType type_;

protected:
  Initializer(InitializerType type);

public:
  virtual ~Initializer() = default;

  InitializerType type() const;

  virtual std::unordered_map<std::string, std::string> conf() const;

  virtual void Initialize(Tensor* val) const = 0;

public:
  static std::unique_ptr<Initializer> Create(
      InitializerType init_type,
      const std::unordered_map<std::string, std::string>& init_conf);
};

}  // namespace kraken
