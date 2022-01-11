#pragma once

#include "snappy-sinksource.h"

namespace kraken {

class SnappySource : public ::snappy::Source {
private:
  char* ptr_;
  size_t offset_;
  size_t capacity_;

  void (*deleter_)(void*);

public:
  SnappySource(char* ptr, size_t capacity, void (*deleter)(void*));

  ~SnappySource() override;

  size_t Available() const override;

  const char* Peek(size_t* len) override;

  void Skip(size_t n) override;
};

// class SnappySink : public ::snappy::Sink {
// private:
//   char* ptr_;
//   size_t offset_;
//   size_t capacity_;

// public:
//   SnappySink();

//   ~SnappySink() override;

// private:
//   size_t Growth(size_t new_size) const;

// public:
//   void Append(const char* bytes, size_t n) override;

//   char* GetAppendBuffer(size_t length, char* scratch) override;

//   void AppendAndTakeOwnership(char* bytes, size_t n,
//                               void (*deleter)(void*, const char*, size_t),
//                               void* deleter_arg) override;

//   char* GetAppendBufferVariable(size_t min_size, size_t desired_size_hint,
//                                 char* scratch, size_t scratch_size,
//                                 size_t* allocated_size) override;
// };

}  // namespace kraken
