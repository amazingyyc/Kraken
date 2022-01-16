#pragma once

#include <cstdint>

namespace kraken {

/**
 * \brief The tensor layout kStride means the Tensor is dense and the memory is continue.
 * kCoo means the tensor is coo sparse.
 */
enum class Layout : uint8_t {
  kStride = 0,
  kCoo = 1,
};

}  // namespace kraken
