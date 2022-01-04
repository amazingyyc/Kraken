#pragma once

#include <cstdint>
#include <string>
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
  void UpdateStrides();

public:
  const Shape& operator=(const Shape& other);
  const Shape& operator=(Shape&& other);

  bool operator==(const Shape& other) const;
  bool operator!=(const Shape& other) const;

  int64_t operator[](int64_t axis) const;

  const std::vector<int64_t>& dims() const;

  const std::vector<int64_t>& strides() const;

  int64_t NDims() const;

  int64_t Size() const;

  int64_t Dim(int64_t axis) const;

  int64_t Stride(int64_t axis) const;

  std::string Str() const;
};

}  // namespace kraken
