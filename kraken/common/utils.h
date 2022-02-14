#pragma once

#include <random>
#include <string>
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

}  // namespace utils
}  // namespace kraken
