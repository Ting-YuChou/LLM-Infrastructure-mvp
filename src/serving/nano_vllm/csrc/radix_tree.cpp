// The radix-tree prefix cache and block allocator of block_manager.py in C++.
//
// Same algorithm and the same block order as the Python BlockManager (so the
// two produce identical block tables): a block-granular radix tree whose
// nodes hold runs of whole blocks, sequences locking the path to their
// deepest cached node, commit() handing completed blocks to the tree, and
// eviction from the tails of the least recently used unlocked leaves.
//
// Token ids live here per sequence (add_sequence once, then extend_sequence
// with the new ones), so matching and committing never convert Python lists.
// Children are keyed by a 64-bit hash of their first block; a lookup checks
// the actual tokens, and a colliding block is simply not cached.
//
// Bound with nanobind (radix_cpp.py builds it).

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nb = nanobind;

using Tokens = std::vector<int32_t>;

namespace {

uint64_t hash_block(const int32_t* tokens, int n) {
  uint64_t h = 1469598103934665603ULL;  // FNV-1a with an extra mix
  for (int i = 0; i < n; ++i) {
    h ^= static_cast<uint32_t>(tokens[i]);
    h *= 1099511628211ULL;
    h ^= h >> 29;
  }
  return h;
}

struct Node {
  Tokens key;                 // blocks.size() * block_size token ids
  std::vector<int32_t> blocks;
  std::unordered_map<uint64_t, int> children;  // by hash of the first block
  int parent = -1;
  int lock_ref = 0;
  uint64_t last_access = 0;
  uint32_t generation = 0;    // bumped when the slot is freed (stale heap entries)
  bool alive = false;
};

struct HeapEntry {
  uint64_t last_access;
  uint64_t tiebreak;
  int node;
  uint32_t generation;
  bool operator>(const HeapEntry& other) const {
    return std::tie(last_access, tiebreak) > std::tie(other.last_access, other.tiebreak);
  }
};

}  // namespace

class RadixBlockManager {
 public:
  RadixBlockManager(int num_blocks, int block_size, bool enable_prefix_cache)
      : bs_(block_size), num_blocks_(num_blocks), enabled_(enable_prefix_cache) {
    if (num_blocks <= 0) throw std::invalid_argument("num_blocks must be positive");
    if (block_size <= 0) throw std::invalid_argument("block_size must be positive");
    // A node holds at least one block, so this many slots never reallocate.
    nodes_.reserve(static_cast<size_t>(num_blocks) + 2);
    nodes_.emplace_back();
    nodes_[0].alive = true;  // the root
    for (int b = 0; b < num_blocks; ++b) free_blocks_.push_back(b);
  }

  // --- sequences' tokens -------------------------------------------------

  void add_sequence(int64_t seq_id, const Tokens& tokens) { seqs_[seq_id] = tokens; }

  void extend_sequence(int64_t seq_id, const Tokens& tokens) {
    Tokens& stored = sequence(seq_id);
    stored.insert(stored.end(), tokens.begin(), tokens.end());
  }

  void remove_sequence(int64_t seq_id) { seqs_.erase(seq_id); }

  // --- allocation ----------------------------------------------------------

  // Prefix-cache hit blocks for the sequence, or -1 if it won't fit.
  int can_allocate(int64_t seq_id) {
    const Tokens& tokens = sequence(seq_id);
    int needed_blocks = num_blocks_for(tokens.size());
    Match m = match(tokens, std::max(0, needed_blocks - 1));
    if (blocks_needed(needed_blocks, m) > num_free()) return -1;
    return static_cast<int>(m.blocks.size());
  }

  // (block table, deepest locked node, shared blocks); changes nothing on failure.
  std::tuple<std::vector<int32_t>, int, int> allocate(int64_t seq_id, int num_cached_blocks) {
    const Tokens& tokens = sequence(seq_id);
    int needed_blocks = num_blocks_for(tokens.size());
    Match m = match(tokens, std::min(std::max(0, needed_blocks - 1), num_cached_blocks));
    if (static_cast<int>(m.blocks.size()) < num_cached_blocks)
      throw std::runtime_error("prefix cache changed since can_allocate()");
    if (blocks_needed(needed_blocks, m) > num_free())
      throw std::runtime_error("No free KV blocks available");
    int node = m.node;
    if (node != 0 && m.matched < static_cast<int>(nodes_[node].blocks.size()))
      node = split(node, m.matched);
    lock(node);  // before allocating: eviction must not reclaim the hits
    touch(node);
    int fresh = needed_blocks - static_cast<int>(m.blocks.size());
    reserve(fresh);
    std::vector<int32_t> table = std::move(m.blocks);
    int shared = static_cast<int>(table.size());
    for (int i = 0; i < fresh; ++i) {
      table.push_back(free_blocks_.front());
      free_blocks_.pop_front();
    }
    return {std::move(table), node, shared};
  }

  void deallocate(int node, const std::vector<int32_t>& private_blocks) {
    for (int32_t block : private_blocks) free_blocks_.push_back(block);
    if (node > 0) unlock(node);
  }

  std::vector<int32_t> take_blocks(int count) {
    reserve(count);
    std::vector<int32_t> out;
    out.reserve(count);
    for (int i = 0; i < count; ++i) {
      out.push_back(free_blocks_.front());
      free_blocks_.pop_front();
    }
    return out;
  }

  // Hand blocks [num_shared, end) to the tree. ``tail`` is the sequence's
  // block table from num_shared to end. Returns (new deepest node, new shared
  // count, swaps): swaps are (block index, tree block) where the tree already
  // held the content; the sequence's own copies are freed.
  std::tuple<int, int, std::vector<std::pair<int, int32_t>>> commit(
      int64_t seq_id, int node, int num_shared, int end, std::vector<int32_t> tail) {
    std::vector<std::pair<int, int32_t>> swaps;
    if (!enabled_ || node < 0 || num_shared >= end) return {node, num_shared, swaps};
    const Tokens& tokens = sequence(seq_id);
    if (static_cast<int64_t>(tokens.size()) < static_cast<int64_t>(end) * bs_)
      throw std::runtime_error("commit past the sequence's known tokens");
    int index = num_shared;
    while (index < end) {
      const int32_t* first = tokens.data() + static_cast<size_t>(index) * bs_;
      uint64_t h = hash_block(first, bs_);
      auto it = nodes_[node].children.find(h);
      if (it == nodes_[node].children.end()) {
        Tokens key(first, tokens.data() + static_cast<size_t>(end) * bs_);
        std::vector<int32_t> blocks(tail.begin() + (index - num_shared),
                                    tail.begin() + (end - num_shared));
        Node& current = nodes_[node];
        if (node != 0 && current.children.empty() && current.lock_ref == 1) {
          // Only this sequence uses the leaf: extend its edge in place.
          current.key.insert(current.key.end(), key.begin(), key.end());
          current.blocks.insert(current.blocks.end(), blocks.begin(), blocks.end());
        } else {
          int child = new_node(std::move(key), std::move(blocks), node);
          nodes_[node].children[h] = child;
          evictable_ += static_cast<int64_t>(nodes_[child].blocks.size());
          move_lock(node, child);
          node = child;
        }
        index = end;
        break;
      }
      int child = it->second;
      if (!std::equal(first, first + bs_, nodes_[child].key.begin())) break;  // hash collision
      int matched = match_edge(child, tokens, index, end);
      for (int off = 0; off < matched; ++off) {
        int32_t tree_block = nodes_[child].blocks[off];
        int32_t& mine = tail[index - num_shared + off];
        if (mine != tree_block) {  // same content, another copy
          free_blocks_.push_back(mine);
          swaps.emplace_back(index + off, tree_block);
          mine = tree_block;
        }
      }
      if (matched < static_cast<int>(nodes_[child].blocks.size())) child = split(child, matched);
      move_lock(node, child);
      node = child;
      index += matched;
    }
    touch(node);
    return {node, index, swaps};
  }

  int reset_prefix_cache() {
    int freed = static_cast<int>(evictable_);
    evict(freed);
    return freed;
  }

  // --- state ---------------------------------------------------------------

  int num_free_list() const { return static_cast<int>(free_blocks_.size()); }
  std::vector<int32_t> free_list() const { return {free_blocks_.begin(), free_blocks_.end()}; }
  int num_evictable() const { return static_cast<int>(evictable_); }
  int64_t evicted_blocks() const { return evicted_; }

  std::vector<int32_t> path_blocks(int node) const {
    std::vector<std::vector<int32_t>> parts;
    while (node > 0) {
      parts.push_back(nodes_[node].blocks);
      node = nodes_[node].parent;
    }
    std::vector<int32_t> out;
    for (auto it = parts.rbegin(); it != parts.rend(); ++it) out.insert(out.end(), it->begin(), it->end());
    return out;
  }

  // (node, parent, key, blocks, lock_ref) for every node but the root (tests).
  std::vector<std::tuple<int, int, Tokens, std::vector<int32_t>, int>> tree_dump() const {
    std::vector<std::tuple<int, int, Tokens, std::vector<int32_t>, int>> out;
    for (size_t i = 1; i < nodes_.size(); ++i) {
      const Node& n = nodes_[i];
      if (n.alive) out.emplace_back(static_cast<int>(i), n.parent, n.key, n.blocks, n.lock_ref);
    }
    return out;
  }

 private:
  struct Match {
    std::vector<int32_t> blocks;
    int node = 0;
    int matched = 0;
  };

  Tokens& sequence(int64_t seq_id) {
    auto it = seqs_.find(seq_id);
    if (it == seqs_.end()) throw std::runtime_error("unknown sequence");
    return it->second;
  }

  int num_blocks_for(size_t num_tokens) const {
    return static_cast<int>((num_tokens + bs_ - 1) / bs_);
  }

  int num_free() const { return static_cast<int>(free_blocks_.size() + evictable_); }

  Match match(const Tokens& tokens, int limit) const {
    Match m;
    if (!enabled_) return m;
    int node = 0, index = 0;
    while (index < limit) {
      const int32_t* first = tokens.data() + static_cast<size_t>(index) * bs_;
      auto it = nodes_[node].children.find(hash_block(first, bs_));
      if (it == nodes_[node].children.end()) break;
      int child = it->second;
      if (!std::equal(first, first + bs_, nodes_[child].key.begin())) break;
      int matched = match_edge(child, tokens, index, limit);
      const auto& blocks = nodes_[child].blocks;
      m.blocks.insert(m.blocks.end(), blocks.begin(), blocks.begin() + matched);
      node = child;
      index += matched;
      m.matched = matched;
      if (matched < static_cast<int>(blocks.size())) break;
    }
    m.node = node;
    if (node == 0) m.matched = 0;
    return m;
  }

  // Whole blocks of the node's edge equal to tokens from block ``index`` on
  // (at most limit - index); the first block is known to match.
  int match_edge(int node, const Tokens& tokens, int index, int limit) const {
    const Node& n = nodes_[node];
    int matched = 1;
    int edge_blocks = static_cast<int>(n.blocks.size());
    while (matched < edge_blocks && index + matched < limit) {
      const int32_t* mine = tokens.data() + static_cast<size_t>(index + matched) * bs_;
      if (!std::equal(mine, mine + bs_, n.key.begin() + static_cast<size_t>(matched) * bs_)) break;
      ++matched;
    }
    return matched;
  }

  // Hits in unlocked nodes stop being evictable once locked, so they cost
  // free capacity just like fresh blocks.
  int blocks_needed(int needed_blocks, const Match& m) const {
    int revived = 0, node = m.node, count = m.matched;
    while (node > 0) {
      if (nodes_[node].lock_ref == 0) revived += count;
      node = nodes_[node].parent;
      count = node > 0 ? static_cast<int>(nodes_[node].blocks.size()) : 0;
    }
    return needed_blocks - static_cast<int>(m.blocks.size()) + revived;
  }

  int new_node(Tokens key, std::vector<int32_t> blocks, int parent) {
    int id;
    if (!free_nodes_.empty()) {
      id = free_nodes_.back();
      free_nodes_.pop_back();
    } else {
      id = static_cast<int>(nodes_.size());
      nodes_.emplace_back();
    }
    Node& n = nodes_[id];
    n.key = std::move(key);
    n.blocks = std::move(blocks);
    n.children.clear();
    n.parent = parent;
    n.lock_ref = 0;
    n.last_access = 0;
    n.alive = true;
    return id;
  }

  void free_node(int id) {
    Node& n = nodes_[id];
    n.alive = false;
    ++n.generation;
    n.key.clear();
    n.blocks.clear();
    n.children.clear();
    n.parent = -1;
    free_nodes_.push_back(id);
  }

  // Cut the node's edge after ``matched`` blocks; return the new upper part.
  int split(int node, int matched) {
    int parent = nodes_[node].parent;
    uint64_t old_first = hash_block(nodes_[node].key.data(), bs_);
    Tokens upper_key(nodes_[node].key.begin(), nodes_[node].key.begin() + static_cast<size_t>(matched) * bs_);
    std::vector<int32_t> upper_blocks(nodes_[node].blocks.begin(), nodes_[node].blocks.begin() + matched);
    int upper = new_node(std::move(upper_key), std::move(upper_blocks), parent);
    Node& lower = nodes_[node];
    nodes_[upper].lock_ref = lower.lock_ref;
    nodes_[upper].last_access = lower.last_access;
    nodes_[parent].children[old_first] = upper;
    lower.key.erase(lower.key.begin(), lower.key.begin() + static_cast<size_t>(matched) * bs_);
    lower.blocks.erase(lower.blocks.begin(), lower.blocks.begin() + matched);
    lower.parent = upper;
    nodes_[upper].children[hash_block(lower.key.data(), bs_)] = node;
    return upper;
  }

  void lock(int node) {
    while (node > 0) {
      Node& n = nodes_[node];
      if (n.lock_ref == 0) evictable_ -= static_cast<int64_t>(n.blocks.size());
      ++n.lock_ref;
      node = n.parent;
    }
  }

  void unlock(int node) {
    while (node > 0) {
      Node& n = nodes_[node];
      --n.lock_ref;
      if (n.lock_ref == 0) {
        evictable_ += static_cast<int64_t>(n.blocks.size());
        if (n.children.empty()) push_leaf(node);
      }
      node = n.parent;
    }
  }

  void move_lock(int upper, int lower) {
    lock(lower);
    unlock(upper);
  }

  void touch(int node) {
    uint64_t now = ++clock_;
    while (node > 0) {
      nodes_[node].last_access = now;
      node = nodes_[node].parent;
    }
  }

  void push_leaf(int node) {
    heap_.push({nodes_[node].last_access, tiebreak_++, node, nodes_[node].generation});
  }

  void reserve(int count) {
    int missing = count - static_cast<int>(free_blocks_.size());
    if (missing > 0) {
      if (missing > evictable_) throw std::runtime_error("No free KV blocks available");
      evict(missing);
    }
  }

  // Free ``count`` blocks from the tails of the least recently used unlocked leaves.
  void evict(int count) {
    while (count > 0) {
      if (heap_.empty())
        throw std::runtime_error("prefix cache accounting is inconsistent: no evictable leaf");
      HeapEntry e = heap_.top();
      heap_.pop();
      Node& n = nodes_[e.node];
      if (!n.alive || n.generation != e.generation || !n.children.empty() || n.lock_ref ||
          n.blocks.empty() || n.last_access != e.last_access)
        continue;  // stale entry
      int take = std::min<int>(count, static_cast<int>(n.blocks.size()));
      uint64_t first = hash_block(n.key.data(), bs_);
      for (size_t i = n.blocks.size() - take; i < n.blocks.size(); ++i) free_blocks_.push_back(n.blocks[i]);
      n.blocks.resize(n.blocks.size() - take);
      n.key.resize(n.blocks.size() * static_cast<size_t>(bs_));
      evictable_ -= take;
      evicted_ += take;
      count -= take;
      if (!n.blocks.empty()) {
        push_leaf(e.node);
      } else {
        int parent = n.parent;
        nodes_[parent].children.erase(first);
        free_node(e.node);
        Node& p = nodes_[parent];
        if (parent > 0 && p.children.empty() && p.lock_ref == 0) push_leaf(parent);
      }
    }
  }

  int bs_;
  int num_blocks_;
  bool enabled_;
  std::vector<Node> nodes_;
  std::vector<int> free_nodes_;
  std::deque<int32_t> free_blocks_;
  int64_t evictable_ = 0;
  int64_t evicted_ = 0;
  uint64_t clock_ = 0;
  uint64_t tiebreak_ = 0;
  std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap_;
  std::unordered_map<int64_t, Tokens> seqs_;
};

NB_MODULE(nano_radix, m) {
  m.doc() = "nano-vLLM's radix-tree prefix cache and KV block allocator";
  nb::class_<RadixBlockManager>(m, "RadixBlockManager")
      .def(nb::init<int, int, bool>(), nb::arg("num_blocks"), nb::arg("block_size"),
           nb::arg("enable_prefix_cache") = true)
      .def("add_sequence", &RadixBlockManager::add_sequence)
      .def("extend_sequence", &RadixBlockManager::extend_sequence)
      .def("remove_sequence", &RadixBlockManager::remove_sequence)
      .def("can_allocate", &RadixBlockManager::can_allocate)
      .def("allocate", &RadixBlockManager::allocate)
      .def("deallocate", &RadixBlockManager::deallocate)
      .def("take_blocks", &RadixBlockManager::take_blocks)
      .def("commit", &RadixBlockManager::commit)
      .def("reset_prefix_cache", &RadixBlockManager::reset_prefix_cache)
      .def("num_free_list", &RadixBlockManager::num_free_list)
      .def("free_list", &RadixBlockManager::free_list)
      .def("num_evictable", &RadixBlockManager::num_evictable)
      .def("evicted_blocks", &RadixBlockManager::evicted_blocks)
      .def("path_blocks", &RadixBlockManager::path_blocks)
      .def("tree_dump", &RadixBlockManager::tree_dump);
}
