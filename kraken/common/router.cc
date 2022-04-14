#include "common/router.h"

#include <algorithm>
#include <functional>
#include <sstream>

#include "common/exception.h"
#include "common/utils.h"

namespace kraken {

const std::size_t Router::kVirtualNodeNum = 3;
const std::string Router::kVirtualNodeSep = "#";

Router::Router() : version_(0) {
}

bool Router::ContainVirtualNode(uint64_t hash_v) const {
  return BinaryFindVirtualNode(hash_v) != vnodes_.end();
}

std::vector<Router::Node>::const_iterator Router::BinaryFindNode(
    uint64_t id) const {
  auto it = std::lower_bound(
      nodes_.begin(), nodes_.end(), id,
      [](const Node& node, uint64_t id) -> bool { return node.id < id; });

  if (it != nodes_.end() && it->id == id) {
    return it;
  }

  return nodes_.end();
}

std::vector<Router::VirtualNode>::const_iterator Router::BinaryFindVirtualNode(
    uint64_t hash_v) const {
  auto it =
      std::lower_bound(vnodes_.begin(), vnodes_.end(), hash_v,
                       [](const VirtualNode& vnode, uint64_t hash_v) -> bool {
                         return vnode.hash_v < hash_v;
                       });

  if (it != vnodes_.end() && it->hash_v == hash_v) {
    return it;
  }

  return vnodes_.end();
}

bool Router::operator==(const Router& other) const {
  if (version_ != other.version_ || nodes_.size() != other.nodes_.size() ||
      vnodes_.size() != other.vnodes_.size()) {
    return false;
  }

  for (size_t i = 0; i < nodes_.size(); ++i) {
    if (nodes_[i].id != other.nodes_[i].id ||
        nodes_[i].name != other.nodes_[i].name ||
        nodes_[i].vnode_list.size() != other.nodes_[i].vnode_list.size()) {
      return false;
    }

    for (size_t j = 0; j < nodes_[i].vnode_list.size(); ++j) {
      if (nodes_[i].vnode_list[j] != other.nodes_[i].vnode_list[j]) {
        return false;
      }
    }
  }

  for (size_t i = 0; i < vnodes_.size(); ++i) {
    if (vnodes_[i].hash_v != other.vnodes_[i].hash_v ||
        vnodes_[i].node_id != other.vnodes_[i].node_id ||
        vnodes_[i].name != other.vnodes_[i].name) {
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

const std::vector<Router::Node>& Router::nodes() const {
  return nodes_;
}

const std::vector<Router::VirtualNode>& Router::vnodes() const {
  return vnodes_;
}

const Router::Node& Router::node(uint64_t id) const {
  auto it = BinaryFindNode(id);
  if (it == nodes_.end()) {
    RUNTIME_ERROR("Con't find node:" << id);
  }

  return *it;
}

const Router::VirtualNode& Router::virtual_node(uint64_t hash_v) const {
  auto it = BinaryFindVirtualNode(hash_v);
  if (it == vnodes_.end()) {
    RUNTIME_ERROR("Con't find virtual node:" << hash_v);
  }

  return *it;
}

bool Router::Add(uint64_t id, const std::string& name) {
  // The new node id must bigger than all exist id.
  if (nodes_.empty() == false) {
    if (nodes_.back().id >= id) {
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

    while (ContainVirtualNode(hash_v)) {
      // Not care about the overflow.
      hash_v += utils::ThreadLocalRandom<uint64_t>(1, interval);
    }

    VirtualNode vnode;
    vnode.hash_v = hash_v;
    vnode.node_id = id;
    vnode.name = name_v;

    vnodes_.emplace_back(std::move(vnode));
    node.vnode_list.emplace_back(hash_v);
  }

  // Always is ascending order.
  nodes_.emplace_back(std::move(node));

  // Sort ascending order.
  std::sort(vnodes_.begin(), vnodes_.end(),
            [](const VirtualNode& v1, const VirtualNode& v2) -> bool {
              return v1.hash_v < v2.hash_v;
            });

  version_++;

  return true;
}

bool Router::Remove(uint64_t id) {
  auto it = BinaryFindNode(id);
  if (it == nodes_.end()) {
    return false;
  }

  for (size_t i = 0; i < it->vnode_list.size(); ++i) {
    auto vit = BinaryFindVirtualNode(it->vnode_list[i]);
    if (vit == vnodes_.end()) {
      continue;
    }

    vnodes_.erase(vit);
  }

  nodes_.erase(it);

  version_++;

  return true;
}

bool Router::Empty() const {
  return nodes_.empty();
}

bool Router::Contains(uint64_t id) const {
  return BinaryFindNode(id) != nodes_.end();
}

std::vector<uint64_t> Router::VirtualHashs(uint64_t id) const {
  auto it = BinaryFindNode(id);
  if (it != nodes_.end()) {
    return it->vnode_list;
  }

  return {};
}

std::vector<Router::Range> Router::NodeHashRanges(uint64_t id) const {
  std::vector<Router::Range> ranges;
  ranges.reserve(kVirtualNodeNum);

  auto it = BinaryFindNode(id);
  if (it == nodes_.end()) {
    return ranges;
  }

  for (const auto& hash_v : it->vnode_list) {
    auto vit = BinaryFindVirtualNode(hash_v);
    if (vit == vnodes_.end()) {
      continue;
    }

    if (vit == vnodes_.begin()) {
      Range head_range;
      head_range.start = 0;
      head_range.end = vit->hash_v;

      Range zero_range;
      zero_range.start = 0;
      zero_range.end = 0;

      Range tail_range;
      tail_range.start = vnodes_.rbegin()->hash_v;
      tail_range.end = std::numeric_limits<uint64_t>::max();

      ranges.emplace_back(head_range);
      ranges.emplace_back(zero_range);
      ranges.emplace_back(tail_range);
    } else {
      Range range;
      range.start = (vit - 1)->hash_v;
      range.end = vit->hash_v;

      ranges.emplace_back(range);
    }
  }

  return ranges;
}

std::unordered_set<uint64_t> Router::IntersectNodes(const Range& range) const {
  std::unordered_set<uint64_t> intersect_ids;

  for (const auto& node : nodes_) {
    auto ranges = NodeHashRanges(node.id);

    for (const auto& other : ranges) {
      if (range.start < other.end && range.end > other.start) {
        intersect_ids.emplace(node.id);
      }
    }
  }

  return intersect_ids;
}

uint64_t Router::Hit(uint64_t hv) const {
  if (vnodes_.empty()) {
    return uint64_t(-1);
  }

  auto it = std::lower_bound(vnodes_.begin(), vnodes_.end(), hv,
                             [](const VirtualNode& vnode, uint64_t hv) -> bool {
                               return vnode.hash_v < hv;
                             });

  if (it == vnodes_.end()) {
    return vnodes_.begin()->node_id;
  } else {
    return it->node_id;
  }
}

std::string Router::Str() const {
  std::ostringstream oss;

  oss << "Version:" << version_ << ", Nodes:[";
  for (const auto& node : nodes_) {
    oss << "id:" << node.id << ", name:" << node.name << ", vnode_list:";
    for (auto vid : node.vnode_list) {
      oss << vid << ", ";
    }
  }
  oss << "]";

  oss << ", Ring:[";
  for (const auto& vnode : vnodes_) {
    oss << vnode.node_id << ", ";
  }
  oss << "]";

  return oss.str();
}

}  // namespace kraken
