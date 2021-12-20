#include "common/utils.h"

#include <chrono>
#include <ctime>

namespace kraken {
namespace utils {

std::string current_timestamp() {
  using std::chrono::system_clock;
  auto cur_time = std::chrono::system_clock::now();

  char buffer[80];
  auto transformed = cur_time.time_since_epoch().count() / 1000000;

  auto millis = transformed % 1000;

  std::time_t tt = system_clock::to_time_t(cur_time);

  auto timeinfo = localtime(&tt);
#pragma GCC diagnostic ignored "-Wrestrict"
  strftime(buffer, 80, "%F %H:%M:%S", timeinfo);
#pragma GCC diagnostic ignored "-Wformat-overflow="
  sprintf(buffer, "%s.%03d", buffer, (int)millis);

  return std::string(buffer);
}

}  // namespace utils
}  // namespace kraken
