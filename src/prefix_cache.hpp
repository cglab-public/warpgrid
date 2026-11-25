#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ink {

static constexpr int kKvBlockSize = 64;

inline uint64_t hash_kv_block(uint64_t parent_hash, const int32_t* tokens, int count = kKvBlockSize) {
    // FNV-1a / splitmix64 mix over parent_hash and block tokens
    uint64_t h = parent_hash ^ 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < count; ++i) {
        uint64_t x = static_cast<uint64_t>(static_cast<uint32_t>(tokens[i])) + 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return h;
}

// Longest common prefix length aligned down to kKvBlockSize, strictly less than min(a.size(), b.size()).
inline int compute_lcp_aligned(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    // Keep at least 1 token remaining in both prompts so we always run at least 1 forward token
    if (i == n && i > 0) --i;
    return static_cast<int>((i / kKvBlockSize) * kKvBlockSize);
}

struct PrefixMatch {
    int hit_tokens = 0;                // Multiple of kKvBlockSize, strictly < prompt.size()
    std::vector<int32_t> phys_blocks;  // Length = hit_tokens / kKvBlockSize (pinned with ref_count++)
    int32_t snap_id = -1;              // Boundary snapshot ID for local SWA + sconv at pos = hit_tokens
};

// Cross-slot Paged Global KV Block Allocator + Radix Prefix Cache + Boundary Snapshot Pool.
//
// Invariants:
// - Every cached node owns exactly one physical global KV block (kKvBlockSize = 64 tokens).
// - Published blocks are strictly immutable while in the cache or while ref_count > 0.
// - A prefix match only returns a boundary that has a valid local-SWA + sconv snapshot (snap_id >= 0).
// - Eviction only reclaims unreferenced leaf blocks (ref_count == 0 && children.empty()) in LRU order.
class RadixPrefixCache {
    struct Node {
        uint64_t hash = 0;
        std::array<int32_t, kKvBlockSize> tokens{};
        int32_t phys_block = -1;
        int32_t snap_id = -1;
        int ref_count = 0;
        uint64_t last_touch = 0;
        Node* parent = nullptr;
        std::unordered_map<uint64_t, std::unique_ptr<Node>> children;
    };

    int total_blocks_ = 0;
    int total_snaps_ = 0;
    uint64_t clock_ = 0;

    std::unique_ptr<Node> root_;
    std::deque<int32_t> free_blocks_;
    std::deque<int32_t> free_snaps_;
    // Pin counts for snapshots acquired by admitted requests awaiting restore_boundary_snapshot
    std::vector<int> snap_pins_;
    // Map physical block ID -> owning radix node (nullptr if block is private/uncached)
    std::vector<Node*> block_owner_;
    // Track all allocated radix nodes (excluding root) for fast LRU victim selection
    std::vector<Node*> all_nodes_;

    static bool tokens_equal(const Node* n, const int32_t* toks) {
        for (int i = 0; i < kKvBlockSize; ++i) {
            if (n->tokens[i] != toks[i]) return false;
        }
        return true;
    }

    Node* find_child(Node* parent, uint64_t h, const int32_t* toks) const {
        auto it = parent->children.find(h);
        if (it != parent->children.end() && tokens_equal(it->second.get(), toks)) {
            return it->second.get();
        }
        // Linear fallback in the rare event of a hash collision
        for (const auto& kv : parent->children) {
            if (tokens_equal(kv.second.get(), toks)) return kv.second.get();
        }
        return nullptr;
    }

    void remove_from_all_nodes(Node* target) {
        for (size_t i = 0; i < all_nodes_.size(); ++i) {
            if (all_nodes_[i] == target) {
                all_nodes_[i] = all_nodes_.back();
                all_nodes_.pop_back();
                return;
            }
        }
    }

    bool evict_one_block() {
        Node* victim = nullptr;
        uint64_t best_ts = std::numeric_limits<uint64_t>::max();
        for (Node* n : all_nodes_) {
            if (n->ref_count == 0 && n->children.empty() &&
                !(n->snap_id >= 0 && snap_pins_[n->snap_id] > 0) &&
                n->last_touch < best_ts) {
                best_ts = n->last_touch;
                victim = n;
            }
        }
        if (!victim) return false;

        int32_t blk = victim->phys_block;
        int32_t sid = victim->snap_id;
        Node* parent = victim->parent;
        uint64_t h = victim->hash;

        remove_from_all_nodes(victim);
        if (blk >= 0 && blk < total_blocks_) {
            block_owner_[blk] = nullptr;
            free_blocks_.push_back(blk);
        }
        if (sid >= 0) {
            free_snaps_.push_back(sid);
        }
        if (parent) {
            // Erase by pointer match in case of hash collision
            auto it = parent->children.find(h);
            if (it != parent->children.end() && it->second.get() == victim) {
                parent->children.erase(it);
            } else {
                for (auto jt = parent->children.begin(); jt != parent->children.end(); ++jt) {
                    if (jt->second.get() == victim) {
                        parent->children.erase(jt);
                        break;
                    }
                }
            }
        }
        return true;
    }

    int32_t evict_one_snapshot() {
        Node* victim = nullptr;
        uint64_t best_ts = std::numeric_limits<uint64_t>::max();
        for (Node* n : all_nodes_) {
            if (n->snap_id >= 0 && snap_pins_[n->snap_id] == 0 && n->last_touch < best_ts) {
                best_ts = n->last_touch;
                victim = n;
            }
        }
        if (!victim) return -1;
        int32_t sid = victim->snap_id;
        victim->snap_id = -1;
        return sid;
    }

public:
    RadixPrefixCache() : root_(std::make_unique<Node>()) {}

    void init(int total_blocks, int total_snaps) {
        total_blocks_ = total_blocks;
        total_snaps_ = total_snaps;
        clock_ = 0;
        root_ = std::make_unique<Node>();
        free_blocks_.clear();
        for (int i = 0; i < total_blocks_; ++i) free_blocks_.push_back(i);
        free_snaps_.clear();
        for (int i = 0; i < total_snaps_; ++i) free_snaps_.push_back(i);
        snap_pins_.assign(total_snaps_, 0);
        block_owner_.assign(total_blocks_, nullptr);
        all_nodes_.clear();
    }

    int total_blocks() const { return total_blocks_; }
    int total_snapshots() const { return total_snaps_; }
    int free_block_count() const { return static_cast<int>(free_blocks_.size()); }
    int cached_block_count() const { return static_cast<int>(all_nodes_.size()); }

    // Peek the longest cached prefix (aligned to kKvBlockSize, strictly < prompt.size())
    // that has a live boundary snapshot. Does NOT modify ref_counts.
    int peek_prefix_hit(const std::vector<int32_t>& prompt) const {
        if (prompt.size() <= static_cast<size_t>(kKvBlockSize)) return 0;
        int max_blocks = static_cast<int>((prompt.size() - 1) / kKvBlockSize);
        const Node* cur = root_.get();
        uint64_t h = 0;
        int best_blocks = 0;
        for (int b = 0; b < max_blocks; ++b) {
            const int32_t* toks = prompt.data() + b * kKvBlockSize;
            h = hash_kv_block(h, toks);
            Node* nxt = find_child(const_cast<Node*>(cur), h, toks);
            if (!nxt) break;
            cur = nxt;
            if (cur->snap_id >= 0) best_blocks = b + 1;
        }
        return best_blocks * kKvBlockSize;
    }

    // Match and pin (ref_count++) the longest cached prefix with a valid boundary snapshot.
    PrefixMatch acquire_prefix(const std::vector<int32_t>& prompt) {
        PrefixMatch res;
        if (prompt.size() <= static_cast<size_t>(kKvBlockSize)) return res;
        int max_blocks = static_cast<int>((prompt.size() - 1) / kKvBlockSize);

        std::vector<Node*> path;
        path.reserve(max_blocks);
        Node* cur = root_.get();
        uint64_t h = 0;
        int best_len = 0;
        int32_t best_snap = -1;

        for (int b = 0; b < max_blocks; ++b) {
            const int32_t* toks = prompt.data() + b * kKvBlockSize;
            h = hash_kv_block(h, toks);
            Node* nxt = find_child(cur, h, toks);
            if (!nxt) break;
            cur = nxt;
            path.push_back(cur);
            if (cur->snap_id >= 0) {
                best_len = b + 1;
                best_snap = cur->snap_id;
            }
        }
        if (best_len == 0) return res;

        ++clock_;
        res.hit_tokens = best_len * kKvBlockSize;
        res.snap_id = best_snap;
        if (best_snap >= 0 && best_snap < total_snaps_) {
            snap_pins_[best_snap]++;
        }
        res.phys_blocks.reserve(best_len);
        for (int b = 0; b < best_len; ++b) {
            path[b]->ref_count++;
            path[b]->last_touch = clock_;
            res.phys_blocks.push_back(path[b]->phys_block);
        }
        return res;
    }

    // Release a snapshot pin obtained via acquire_prefix() once restore_boundary_snapshot() completes.
    void release_snapshot_pin(int32_t snap_id) {
        if (snap_id >= 0 && snap_id < total_snaps_ && snap_pins_[snap_id] > 0) {
            snap_pins_[snap_id]--;
        }
    }

    // Allocate one private writable physical block (evicting an LRU unreferenced cached leaf if needed).
    // Returns -1 if all physical blocks are currently pinned by active slots.
    int32_t alloc_block() {
        if (free_blocks_.empty()) {
            if (!evict_one_block()) return -1;
        }
        int32_t blk = free_blocks_.front();
        free_blocks_.pop_front();
        return blk;
    }

    // Allocate a boundary snapshot slot (evicting an LRU cached snapshot if needed).
    // Returns -1 if total_snaps_ == 0.
    int32_t alloc_snapshot() {
        if (total_snaps_ <= 0) return -1;
        if (!free_snaps_.empty()) {
            int32_t sid = free_snaps_.front();
            free_snaps_.pop_front();
            return sid;
        }
        return evict_one_snapshot();
    }

    // Publish full blocks [0 .. num_full_blocks) of `tokens` into the radix cache,
    // attaching `snap_id` to the final block `num_full_blocks - 1` (if snap_id >= 0).
    // Any block in `slot_blocks` that is already cached in the radix tree is deduplicated:
    // the slot's duplicate private block is freed and replaced by the canonical cached block.
    // Returns the snap_id actually attached to the final block (may reuse an existing one if
    // already present, freeing the passed-in snap_id back to free_snaps_).
    int32_t publish_prefix(const std::vector<int32_t>& tokens,
                           int num_full_blocks,
                           std::vector<int32_t>& slot_blocks,
                           int32_t snap_id) {
        if (num_full_blocks <= 0 ||
            static_cast<int>(tokens.size()) < num_full_blocks * kKvBlockSize ||
            static_cast<int>(slot_blocks.size()) < num_full_blocks) {
            if (snap_id >= 0) free_snaps_.push_back(snap_id);
            return -1;
        }

        ++clock_;
        Node* cur = root_.get();
        uint64_t h = 0;

        for (int b = 0; b < num_full_blocks; ++b) {
            const int32_t* toks = tokens.data() + b * kKvBlockSize;
            h = hash_kv_block(h, toks);
            int32_t sblk = slot_blocks[b];

            Node* existing = find_child(cur, h, toks);
            if (existing) {
                existing->last_touch = clock_;
                if (existing->phys_block != sblk) {
                    // Slot had a private duplicate of an already-cached block: deduplicate!
                    if (sblk >= 0 && sblk < total_blocks_) {
                        if (block_owner_[sblk] != nullptr) {
                            block_owner_[sblk]->ref_count--;
                        } else {
                            free_blocks_.push_back(sblk);
                        }
                    }
                    existing->ref_count++;
                    slot_blocks[b] = existing->phys_block;
                }
                cur = existing;
            } else {
                // Insert new radix node owning `sblk`
                auto node = std::make_unique<Node>();
                node->hash = h;
                for (int i = 0; i < kKvBlockSize; ++i) node->tokens[i] = toks[i];
                node->phys_block = sblk;
                node->snap_id = -1;
                node->ref_count = 1; // Still held by this active slot!
                node->last_touch = clock_;
                node->parent = cur;

                Node* raw = node.get();
                // Handle potential hash key collision in children map by choosing a unique key
                uint64_t map_key = h;
                while (cur->children.find(map_key) != cur->children.end()) ++map_key;
                raw->hash = map_key;
                cur->children.emplace(map_key, std::move(node));
                if (sblk >= 0 && sblk < total_blocks_) block_owner_[sblk] = raw;
                all_nodes_.push_back(raw);
                cur = raw;
            }
        }

        if (snap_id >= 0) {
            if (cur->snap_id >= 0 && cur->snap_id != snap_id) {
                // Node already has a valid snapshot; return the newly allocated one to free pool
                free_snaps_.push_back(snap_id);
            } else {
                cur->snap_id = snap_id;
            }
        }
        return cur->snap_id;
    }

    // Release all blocks held by a finished or cancelled slot.
    // Cached blocks stay in the radix tree (with ref_count--) for future reuse;
    // uncached private blocks return immediately to free_blocks_.
    void release_slot_blocks(std::vector<int32_t>& slot_blocks) {
        ++clock_;
        for (int32_t blk : slot_blocks) {
            if (blk < 0 || blk >= total_blocks_) continue;
            Node* owner = block_owner_[blk];
            if (owner) {
                if (owner->ref_count > 0) owner->ref_count--;
                owner->last_touch = clock_;
            } else {
                free_blocks_.push_back(blk);
            }
        }
        slot_blocks.clear();
    }
};

inline int match_inplace_slot_prefix(const std::vector<int32_t>& slot_toks,
                                     const std::vector<int32_t>& prompt_ids) {
    int slen = static_cast<int>(slot_toks.size());
    int plen = static_cast<int>(prompt_ids.size());
    if (slen <= 0 || plen <= slen) return 0;
    if (!std::equal(slot_toks.begin(), slot_toks.end(), prompt_ids.begin())) return 0;
    return slen;
}

} // namespace ink
