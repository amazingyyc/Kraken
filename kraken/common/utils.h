#pragma once

#include <string>
#include <vector>

namespace kraken {
namespace utils {

std::string CurrentTimestamp();

void Split(const std::string& str, const std::string& delim,
           std::vector<std::string>* tokens);

std::string ToLower(const std::string& v);

}  // namespace utils
}  // namespace kraken
