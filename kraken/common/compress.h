#pragma once

#include <snappy.h>

#include <cinttypes>

#include "common/deserialize.h"
#include "common/error_code.h"
#include "common/exception.h"
#include "common/mem_buffer.h"
#include "common/mem_reader.h"
#include "common/serialize.h"
#include "common/snappy.h"

namespace kraken {

struct Compress {
  template <typename Type>
  static bool NoUnCompressDeser(const char* body, size_t body_len, Type* v) {
    MemReader reader(body, body_len);
    Deserialize deserialize(&reader);
    if ((deserialize >> (*v)) == false) {
      return false;
    }

    return true;
  }

  template <typename Type>
  static bool SnappyUnCompressDeser(const char* body, size_t body_len,
                                    Type* v) {
    SnappySource source(body, body_len);
    SnappySink sink;

    if (snappy::Uncompress(&source, &sink) == false) {
      return false;
    }

    MemReader reader(sink.ptr(), sink.offset());
    Deserialize deserialize(&reader);
    if ((deserialize >> (*v)) == false) {
      return false;
    }

    return true;
  }

  template <typename ReplyType>
  static bool NoCompressSeria(const ReplyHeader& reply_header,
                              const ReplyType& reply, ZMQBuffer* z_buf) {
    MemBuffer buffer;
    Serialize serialize(&buffer);

    ARGUMENT_CHECK(serialize << reply_header, "Serialize reply header error!");
    if ((serialize << reply) == false) {
      // return ErrorCode::kSerializeReplyError;
      return false;
    }

    // Transfer to ZMQBuffer.
    buffer.TransferForZMQ(z_buf);

    return true;
  }

  template <typename ReplyType>
  static int32_t SnappyCompressSeria(const ReplyHeader& reply_header,
                                     const ReplyType& reply, ZMQBuffer* z_buf) {
    SnappySink sink;

    // step1 serialize header.
    {
      Serialize serialize(&sink);
      ARGUMENT_CHECK(serialize << reply_header,
                     "Serialize reply header error!");
    }

    // step2 serialize body to tmp buffer.
    MemBuffer body_buf;
    {
      Serialize serialize(&body_buf);
      if ((serialize << reply) == false) {
        // return ErrorCode::kSerializeReplyError;
        return false;
      }
    }

    // Step3 compress body data to sink.
    {
      SnappySource source(body_buf.ptr(), body_buf.offset());
      if (snappy::Compress(&source, &sink) <= 0) {
        // return ErrorCode::kSnappyCompressError;
        return false;
      }
    }

    sink.TransferForZMQ(z_buf);

    return true;
  }
};

}  // namespace kraken
