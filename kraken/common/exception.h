#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include "common/error_code.h"
#include "zmq.h"

namespace kraken {

#define ARGUMENT_CHECK(cond, msg) \
  if (!(cond)) { \
    std::ostringstream _oss; \
    _oss << "[" << std::this_thread::get_id() << "] "; \
    _oss << __FILE__ << ":"; \
    _oss << __LINE__ << ":"; \
    _oss << msg << "."; \
    throw std::invalid_argument(_oss.str()); \
  }

#define RUNTIME_ERROR(msg) \
  { \
    std::ostringstream _oss; \
    _oss << "[" << std::this_thread::get_id() << "] "; \
    _oss << __FILE__ << ":"; \
    _oss << __LINE__ << ":"; \
    _oss << msg << "."; \
    throw std::runtime_error(_oss.str()); \
  }

#define ZMQ_CALL(func) \
  { \
    auto _code = (func); \
    if (_code < 0) { \
      RUNTIME_ERROR("zmq errno:" << zmq_errno() \
                                 << ", msg:" << zmq_strerror(zmq_errno())); \
    } \
  }

#define RPC_CALL(func) \
  { \
    auto _code = (func); \
    if (_code != ErrorCode::kSuccess) { \
      RUNTIME_ERROR("RPC call error:" << _code); \
    } \
  }

}  // namespace kraken
