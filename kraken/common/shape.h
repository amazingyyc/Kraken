#pragma once

#include <cstdint>
#include <vector>

namespace kraken {

class Shape {
private:
  std::vector<int64_t> dims_;
  std::vector<int64_t> strides_;

public:
  Shape() = default;

  explicit Shape(const Shape& other);
  explicit Shape(const std::vector<int64_t>& dims);
  explicit Shape(std::vector<int64_t>&& dims);

private:
  void update_strides();

public:
  const Shape& operator=(const Shape& other);
  const Shape& operator=(Shape&& other);

  bool operator==(const Shape& other) const;
  bool operator!=(const Shape& other) const;

  int64_t operator[](int64_t axis) const;

  const std::vector<int64_t> dims() const;

  int64_t ndims() const;

  int64_t size() const;

  int64_t dim(int64_t axis) const;

  int64_t stride(int64_t axis) const;
};

}  // namespace kraken
