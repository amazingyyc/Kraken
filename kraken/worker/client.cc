// #include "worker/client.h"

// namespace kraken {

// Client::Client(CompressType compress_type) : compress_type_(compress_type) {
// }

// Client::~Client() {
//   for (auto& [_, conn] : connecters_) {
//     conn->Stop();
//   }

//   connecters_.clear();
// }

// void Client::Add(uint64_t node_id, const std::string& addr) {
//   auto it = connecters_.find(node_id);
//   if (it != connecters_.end()) {
//     if (it->second->addr() == addr) {
//       return;
//     }

//     it->second->Stop();
//     connecters_.erase(it);
//   }

//   std::unique_ptr<IndepConnecter> conn(
//       new IndepConnecter(addr, compress_type_));
//   conn->Start();

//   connecters_.emplace(node_id, std::move(conn));

//   return;
// }

// void Client::Remove(uint64_t node_id) {
//   auto it = connecters_.find(node_id);
//   if (it != connecters_.end()) {
//     it->second->Stop();
//     connecters_.erase(it);
//   }
// }

// }  // namespace kraken
