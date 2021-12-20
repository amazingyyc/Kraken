#pragma once

#include <cstdlib>

namespace kraken {

/**
 * \brief A mutable buffer not thread-safe.
 *
 * This buffer use to store binary data and can be increase length automaticlly.
 */
class MutableBuffer {
private:
  char* ptr_;

  size_t length_;
  size_t offset_;

public:
  MutableBuffer();

  explicit MutableBuffer(size_t expect);
  explicit MutableBuffer(MutableBuffer&&);

  const MutableBuffer& operator=(MutableBuffer&&);

  MutableBuffer(const MutableBuffer&) = delete;
  MutableBuffer& operator=(const MutableBuffer&) = delete;

  ~MutableBuffer();

private:
  size_t Growth(size_t new_size) const;

public:
  size_t Length() const;

  size_t Offset() const;

  void Append(const char* data, size_t data_size);

  /**
   * \brief Fetch the buffer outof this class, transfer the ownership to
   * outside and the outsider must responsible to release it.
   */
  void Transfer(char** ptr);

  /**
   * \brief Transfer buffer to outside, the outsider must release the buffer by
   * MutableBuffer::Malloc.
   *
   * \param ptr char** store memory pointer.
   * \param length the memory byte num.
   * \param offset how many bytes data the buffer stored.
   */
  void Transfer(char** ptr, size_t* length, size_t* offset);

public:
  static void* Malloc(size_t);
  static void Free(void*);

  /**
   * \brief A special free func for ZMQ.
   */
  static void ZMQFree(void*, void*);
};

}  // namespace kraken
