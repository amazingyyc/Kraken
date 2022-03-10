#include "common/router.h"

#include <algorithm>
#include <functional>
#include <sstream>

#include "common/utils.h"

namespace kraken {

const std::size_t Router::kVirtualNodeNum = 3;
const std::string Router::kVirtualNodeSep = "#";

Router::Router() : version_(0) {
}

bool Router::operator==(const Router& other) const {
  if (version_ != other.version_ || nodes_.size() != other.nodes_.size() ||
      vnodes_.size() != other.vnodes_.size()) {
    return false;
  }

  for (const auto& [k, v] : nodes_) {
    auto it = other.nodes_.find(k);
    if (it == other.nodes_.end()) {
      return false;
    }

    if (v.id != it->second.id || v.name != it->second.name) {
      return false;
    }

    if (v.vnode_list.size() != it->second.vnode_list.size()) {
      return false;
    }

    for (size_t i = 0; i < v.vnode_list.size(); ++i) {
      if (v.vnode_list[i] != it->second.vnode_list[i]) {
        return false;
      }
    }
  }

  for (const auto& [k, v] : vnodes_) {
    auto it = other.vnodes_.find(k);
    if (it == other.vnodes_.end()) {
      return false;
    }

    if (v.hash_v != it->second.hash_v || v.node_id != it->second.node_id ||
        v.name != it->second.name) {
      return false;
    }
  }

  return true;
}

bool Router::operator!=(const Router& other) const {
  return !((*this) == other);
}

uint64_t Router::version() const {
  return version_;
}

const std::map<uint64_t, Router::Node>& Router::nodes() const {
  return nodes_;
}

const std::map<uint64_t, Router::VirtualNode>& Router::vnodes() const {
  return vnodes_;
}

bool Router::node(uint64_t id, Router::Node* node) const {
  auto it = nodes_.find(id);
  if (it == nodes_.end()) {
    return false;
  }

  *node = it->second;

  return true;
}

bool Router::Add(uint64_t id, const std::string& name) {
  // The new node id must bigger than all exist id.
  if (nodes_.empty() == false) {
    if (nodes_.rbegin()->first >= id) {
      return false;
    }
  }

  uint64_t interval = std::numeric_limits<uint64_t>::max();
  if (vnodes_.empty() == false) {
    interval = std::numeric_limits<uint64_t>::max() / vnodes_.size();
  }

  Node node;
  node.id = id;
  node.name = name;

  for (size_t i = 0; i < kVirtualNodeNum; ++i) {
    std::string name_v = name + kVirtualNodeSep + std::to_string(i);
    uint64_t hash_v = (uint64_t)std::hash<std::string>{}(name_v);

    while (vnodes_.find(hash_v) != vnodes_.end()) {
      // Not care about the overflow.
      hash_v += utils::ThreadLocalRandom<uint64_t>(1, interval);
    }

    VirtualNode vnode;
    vnode.hash_v = hash_v;
    vnode.node_id = id;
    vnode.name = name_v;

    vnodes_.emplace(hash_v, std::move(vnode));
    node.vnode_list.emplace_back(hash_v);
  }

  nodes_.emplace(id, std::move(node));

  version_++;

  return true;
}

bool Router::Remove(uint64_t id) {
  if (nodes_.find(id) == nodes_.end()) {
    return false;
  }

  for (auto hash_v : nodes_[id].vnode_list) {
    vnodes_.erase(hash_v);
  }

  nodes_.erase(id);

  version_++;

  return true;
}

bool Router::Empty() const {
  return nodes_.empty();
}

bool Router::Contains(uint64_t id) const {
  return nodes_.find(id) != nodes_.end();
}

std::vector<uint64_t> Router::VirtualHashs(uint64_t id) const {
  auto it = nodes_.find(id);
  if (it != nodes_.end()) {
    return it->second.vnode_list;
  }

  return {};
}

uint64_t Router::Hit(uint64_t hash) const {
  if (vnodes_.empty()) {
    return uint64_t(-1);
  }

  auto it = vnodes_.lower_bound(hash);
  if (it == vnodes_.end()) {
    return vnodes_.begin()->second.node_id;
  } else {
    return it->second.node_id;
  }
}

uint64_t Router::Hit(uint64_t v1, uint64_t v2) const {
  return Hit(utils::Hash(v1, v2));
}

std::string Router::Str() const {
  std::ostringstream oss;

  oss << "Version:" << version_ << ", Nodes:[";
  for (const auto& [_, v] : nodes_) {
    oss << "(id:" << v.id << ", name:" << v.name << ", vnode_list:";
    for (auto vn : v.vnode_list) {
      oss << vn << ", ";
    }
  }
  oss << "]";

  oss << ", Ring:[";
  for (const auto& [_, v] : vnodes_) {
    oss << v.node_id << ", ";
  }
  oss << "]";

  return oss.str();
}

}  // namespace kraken
