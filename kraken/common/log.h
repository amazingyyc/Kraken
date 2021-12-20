#pragma once

#include <iostream>
#include <sstream>
#include <thread>

#include "common/utils.h"

namespace kraken {
namespace log {
extern uint32_t LOG_LEVEL;
}

#define LOG_DEBUG_LEVEL 0
#define LOG_INFO_LEVEL 1
#define LOG_WARNING_LEVEL 2
#define LOG_ERROR_LEVEL 3

#define REAL_PRINT_LOG(msg) \
  std::ostringstream oss; \
  oss << "[" << std::this_thread::get_id() << "] "; \
  oss << kraken::utils::current_timestamp() << " "; \
  oss << __FILE__ << ":"; \
  oss << __LINE__ << ":"; \
  oss << msg << "\n"; \
  std::cout << oss.str();

#define PRINT_LOG(msg, level) \
  if (level >= kraken::log::LOG_LEVEL) { \
    REAL_PRINT_LOG(msg) \
  }

#define LOG_DEBUG(msg) PRINT_LOG(msg, LOG_DEBUG_LEVEL)
#define LOG_INFO(msg) PRINT_LOG(msg, LOG_INFO_LEVEL)
#define LOG_WARNING(msg) PRINT_LOG(msg, LOG_WARNING_LEVEL)
#define LOG_ERROR(msg) PRINT_LOG(msg, LOG_ERROR_LEVEL)

}  // namespace kraken
