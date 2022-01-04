#include "worker/client.h"

namespace kraken {

Client::Client(uint32_t server_id, const std::string& addr)
    : server_id_(server_id), addr_(addr), caller_(addr) {
}

uint32_t Client::ServerId() const {
  return server_id_;
}

const std::string& Client::Addr() const {
  return addr_;
}

void Client::Start() {
  caller_.Start();
}

void Client::Stop() {
  caller_.Stop();
}

}  // namespace kraken
