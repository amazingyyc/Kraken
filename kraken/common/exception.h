#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include "zmq.h"

namespace kraken {

#define ARGUMENT_CHECK(cond, msg) \
  if (!(cond)) { \
    std::ostringstream oss; \
    oss << "[" << std::this_thread::get_id() << "] "; \
    oss << __FILE__ << ":"; \
    oss << __LINE__ << ":"; \
    oss << msg << "."; \
    throw std::invalid_argument(oss.str()); \
  }

#define RUNTIME_ERROR(msg) \
  { \
    std::ostringstream oss; \
    oss << "[" << std::this_thread::get_id() << "] "; \
    oss << __FILE__ << ":"; \
    oss << __LINE__ << ":"; \
    oss << msg << "."; \
    throw std::runtime_error(oss.str()); \
  }

#define ZMQ_CALL(code) \
  if (code < 0) { \
    RUNTIME_ERROR("zmq errno:" << zmq_errno() \
                               << ", msg:" << zmq_strerror(zmq_errno())); \
  }

}  // namespace kraken
