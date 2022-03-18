#pragma once

#include <chrono>

#include "common/deserialize.h"
#include "common/error_code.h"
#include "common/exception.h"
#include "common/mem_reader.h"
#include "common/serialize.h"
#include "common/snappy.h"
#include "snappy.h"

namespace kraken {

class Connecter {
protected:
  using CALLBACK = std::function<void(bool)>;

  using ZMQ_CALLBACK =
      std::function<void(const ReplyHeader&, const char*, size_t)>;

  struct TimerEvent {
    std::chrono::time_point<std::chrono::steady_clock> when;
    uint64_t timestamp;
  };

  struct TimerEventGrater {
    bool operator()(const TimerEvent& e1, const TimerEvent& e2) {
      return e1.when > e2.when;
    }
  };
};

}  // namespace kraken
