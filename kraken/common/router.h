#pragma once

#include <cinttypes>
#include <cstddef>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace kraken {

class Deserialize;
class Serialize;

class Router {
  friend class Deserialize;
  friend class Serialize;

private:
  // How many virtual node for every real node.
  static const size_t kVirtualNodeNum;
  static const std::string kVirtualNodeSep;

public:
  // A range left open right close: (start, end].
  struct Range {
    uint64_t start;
    uint64_t end;
  };

  struct Node {
    uint64_t id;
    std::string name;

    // Virtual list (virtual node hash value).
    std::vector<uint64_t> vnode_list;
  };

  struct VirtualNode {
    uint64_t hash_v;

    // The real node id.
    uint64_t node_id;

    // Node name.
    std::string name;
  };

private:
  uint64_t version_;

  // Sort by id.
  std::vector<Node> nodes_;

  // sort by hash hash_v.
  std::vector<VirtualNode> vnodes_;

public:
  Router();

private:
  bool ContainVirtualNode(uint64_t hash_v) const;

  std::vector<Node>::const_iterator BinaryFindNode(uint64_t id) const;

  std::vector<VirtualNode>::const_iterator BinaryFindVirtualNode(
      uint64_t hash_v) const;

public:
  bool operator==(const Router& other) const;

  bool operator!=(const Router& other) const;

  uint64_t version() const;

  const std::vector<Node>& nodes() const;

  const std::vector<VirtualNode>& vnodes() const;

  const Node& node(uint64_t id) const;

  const VirtualNode& virtual_node(uint64_t hash_v) const;

  bool Add(uint64_t id, const std::string& name);

  bool Remove(uint64_t id);

  bool Empty() const;

  bool Contains(uint64_t id) const;

  std::vector<uint64_t> VirtualHashs(uint64_t id) const;

  std::vector<Range> NodeHashRanges(uint64_t id) const;

  std::unordered_set<uint64_t> IntersectNodes(const Range& range) const;

  uint64_t Hit(uint64_t hv) const;

  std::string Str() const;
};

}  // namespace kraken
