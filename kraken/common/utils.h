#pragma once

#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace kraken {
namespace utils {

static thread_local std::default_random_engine e(time(0));
static thread_local std::uniform_real_distribution<double> u(0., 1.);

template <typename T>
T ThreadLocalRandom(T a, T b) {
  return a + (T)((b - a) * u(e));
}

std::string CurrentTimestamp();

void Split(const std::string& str, const std::string& delim,
           std::vector<std::string>* tokens);

std::string ToLower(const std::string& v);

bool EndWith(const std::string& value, const std::string& ending);

bool StartWith(const std::string& value, const std::string& beginning);

inline uint64_t Hash(uint64_t v) {
  v = v * 3935559000370003845 + 2691343689449507681;

  v ^= v >> 21;
  v ^= v << 37;
  v ^= v >> 4;

  v *= 4768777513237032717;

  v ^= v << 20;
  v ^= v >> 41;
  v ^= v << 5;

  return v;
}

inline uint64_t Hash(uint64_t v1, uint64_t v2) {
  return Hash(v1) ^ Hash(v2);
}

template <typename T>
bool ParseConf(const std::unordered_map<std::string, std::string>& conf,
               const std::string& key, T* v) {
  return false;
}

template <>
inline bool ParseConf<float>(
    const std::unordered_map<std::string, std::string>& conf,
    const std::string& key, float* v) {
  auto it = conf.find(key);
  if (it == conf.end()) {
    return false;
  }

  try {
    *v = std::stof(it->second);
  } catch (...) {
    return false;
  }

  return true;
}

template <>
inline bool ParseConf<bool>(
    const std::unordered_map<std::string, std::string>& conf,
    const std::string& key, bool* v) {
  auto it = conf.find(key);
  if (it == conf.end()) {
    return false;
  }

  std::string lv = utils::ToLower(it->second);

  if (lv == "true" || lv == "1") {
    *v = true;
  } else {
    *v = false;
  }

  return true;
}

}  // namespace utils
}  // namespace kraken
