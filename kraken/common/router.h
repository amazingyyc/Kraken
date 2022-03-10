#pragma once

#include <cinttypes>
#include <cstddef>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace kraken {

class Deserialize;

class Router {
  friend class Deserialize;

private:
  // How many virtual node for every real node.
  static const size_t kVirtualNodeNum;
  static const std::string kVirtualNodeSep;

public:
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

  // <id, Node> map.
  std::map<uint64_t, Node> nodes_;

  // sort by hash hash_v.
  std::map<uint64_t, VirtualNode> vnodes_;

public:
  Router();

public:
  bool operator==(const Router& other) const;

  bool operator!=(const Router& other) const;

  uint64_t version() const;

  const std::map<uint64_t, Node>& nodes() const;

  const std::map<uint64_t, VirtualNode>& vnodes() const;

  bool node(uint64_t id, Router::Node* node) const;

  bool Add(uint64_t id, const std::string& name);

  bool Remove(uint64_t id);

  bool Empty() const;

  bool Contains(uint64_t id) const;

  std::vector<uint64_t> VirtualHashs(uint64_t id) const;

  uint64_t Hit(uint64_t hash) const;

  uint64_t Hit(uint64_t v1, uint64_t v2) const;

  std::string Str() const;
};

}  // namespace kraken
