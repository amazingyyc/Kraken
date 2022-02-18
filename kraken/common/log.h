#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

#include "common/utils.h"

namespace kraken {
namespace log {
extern uint32_t LOG_LEVEL;

class Logger {
private:
  std::ostringstream& oss_;

public:
  Logger(std::ostringstream& oss) : oss_(oss) {
  }

  template <typename T>
  Logger& operator<<(const T& v) {
    oss_ << v;
    return *this;
  }
};

template <>
inline Logger& Logger::operator<<(
    const std::unordered_map<std::string, std::string>& map) {
  for (const auto& [k, v] : map) {
    oss_ << "[" << k << ", " << v << "], ";
  }

  return *this;
}

}  // namespace log

#define LOG_DEBUG_LEVEL 0
#define LOG_INFO_LEVEL 1
#define LOG_WARNING_LEVEL 2
#define LOG_ERROR_LEVEL 3

#define REAL_PRINT_LOG(msg) \
  std::ostringstream _oss; \
  kraken::log::Logger _logger(_oss); \
  _logger << "[" << std::this_thread::get_id() << "] "; \
  _logger << kraken::utils::CurrentTimestamp() << " "; \
  _logger << __FILE__ << ":"; \
  _logger << __LINE__ << ":"; \
  _logger << msg << "\n"; \
  std::cout << _oss.str();

#define PRINT_LOG(msg, level) \
  if (level >= kraken::log::LOG_LEVEL) { \
    REAL_PRINT_LOG(msg) \
  }

#define LOG_DEBUG(msg) PRINT_LOG(msg, LOG_DEBUG_LEVEL)
#define LOG_INFO(msg) PRINT_LOG(msg, LOG_INFO_LEVEL)
#define LOG_WARNING(msg) PRINT_LOG(msg, LOG_WARNING_LEVEL)
#define LOG_ERROR(msg) PRINT_LOG(msg, LOG_ERROR_LEVEL)

}  // namespace kraken
