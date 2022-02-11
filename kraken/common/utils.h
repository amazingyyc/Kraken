#pragma once

#include <string>
#include <vector>

namespace kraken {
namespace utils {

std::string CurrentTimestamp();

void Split(const std::string& str, const std::string& delim,
           std::vector<std::string>* tokens);

std::string ToLower(const std::string& v);

bool EndWith(const std::string& value, const std::string& ending);

}  // namespace utils
}  // namespace kraken
