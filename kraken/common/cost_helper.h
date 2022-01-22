#pragma once

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

namespace kraken {

class CostHelper {
private:
  std::chrono::system_clock::time_point prev_time_;

public:
  void Start() {
    prev_time_ = std::chrono::system_clock::now();
  }

  void Stop(const std::string& label) {
    uint64_t duration =
        (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now() - prev_time_)
            .count();

    std::cout << label << ": " << duration << "us" << std::endl;
  }
};

}  // namespace kraken
