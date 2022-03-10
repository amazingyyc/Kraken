#include "common/utils.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>

namespace kraken {
namespace utils {

std::string CurrentTimestamp() {
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

void Split(const std::string& str, const std::string& delim,
           std::vector<std::string>* tokens) {
  tokens->clear();

  if (str.empty()) {
    return;
  }

  std::string::size_type start = 0;
  auto pos = str.find_first_of(delim, start);

  while (pos != std::string::npos) {
    tokens->emplace_back(std::move(str.substr(start, pos - start)));

    start = pos + delim.size();
    pos = str.find_first_of(delim, start);
  }

  if (start < str.size()) {
    tokens->emplace_back(std::move(str.substr(start)));
  } else if (start == str.size()) {
    tokens->emplace_back(std::string());
  }
}

std::string ToLower(const std::string& v) {
  std::string lv;
  lv.resize(v.size());

  std::transform(v.begin(), v.end(), lv.begin(), tolower);

  return lv;
}

bool EndWith(const std::string& value, const std::string& ending) {
  if (ending.size() > value.size()) {
    return false;
  }

  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

uint64_t Hash(uint64_t v1, uint64_t v2) {
  uint64_t seed = 0;
  seed ^= v1 + 0x9e3779b9;
  seed ^= v2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);

  return seed;
}

uint64_t Hash(uint64_t v1, uint64_t v2, uint64_t v3) {
  uint64_t seed = 0;
  seed ^= v1 + 0x9e3779b9;
  seed ^= v2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= v3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);

  return seed;
}

}  // namespace utils
}  // namespace kraken
