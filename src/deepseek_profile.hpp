#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ink {

struct DsStageProfile {
    uint64_t calls = 0;
    uint64_t tokens = 0;
    double emb_engram_ms = 0.0;
    double attn_router_ms = 0.0;
    double exp_pread_ms = 0.0;
    double moe_gpu_ms = 0.0;
    double head_ms = 0.0;
    double dspark_commit_ms = 0.0;
    double dspark_draft_ms = 0.0;
    uint64_t exp_hits = 0;
    uint64_t exp_misses = 0;
    uint64_t exp_unique_total = 0;

    void reset() {
        calls = 0;
        tokens = 0;
        emb_engram_ms = 0.0;
        attn_router_ms = 0.0;
        exp_pread_ms = 0.0;
        moe_gpu_ms = 0.0;
        head_ms = 0.0;
        dspark_commit_ms = 0.0;
        dspark_draft_ms = 0.0;
        exp_hits = 0;
        exp_misses = 0;
        exp_unique_total = 0;
    }

    double hit_rate_pct() const {
        uint64_t tot = exp_hits + exp_misses;
        return tot > 0 ? (100.0 * (double)exp_hits / (double)tot) : 100.0;
    }

    std::string format_summary() const {
        double div = calls > 0 ? (double)calls : 1.0;
        double tot_ms = emb_engram_ms + attn_router_ms + exp_pread_ms + moe_gpu_ms + head_ms + dspark_commit_ms + dspark_draft_ms;
        char buf[1024];
        snprintf(buf, sizeof(buf),
                 "[ds-profile] calls=%llu tokens=%llu (avg_T=%.2f) total=%.2fms/call\n"
                 "  emb_engram=%.2fms attn_router=%.2fms exp_pread=%.2fms moe_gpu=%.2fms head=%.2fms dspark_draft=%.2fms dspark_commit=%.2fms\n"
                 "  exp_cache: hits=%llu misses=%llu hit_rate=%.1f%% avg_uniq=%.1f/layer",
                 (unsigned long long)calls, (unsigned long long)tokens,
                 calls > 0 ? (double)tokens / div : 0.0,
                 tot_ms / div,
                 emb_engram_ms / div, attn_router_ms / div, exp_pread_ms / div,
                 moe_gpu_ms / div, head_ms / div, dspark_draft_ms / div, dspark_commit_ms / div,
                 (unsigned long long)exp_hits, (unsigned long long)exp_misses, hit_rate_pct(),
                 calls > 0 ? (double)exp_unique_total / (div * 40.0) : 0.0);
        return std::string(buf);
    }
};

struct ExpertUsageProfile {
    int num_layers = 0;
    int num_experts = 0;
    std::vector<uint64_t> counts;

    void init(int L, int E) {
        num_layers = L;
        num_experts = E;
        counts.assign((size_t)std::max(0, L) * (size_t)std::max(0, E), 0ull);
    }

    void record(int layer, int expert) {
        if (layer < 0 || layer >= num_layers || expert < 0 || expert >= num_experts) return;
        counts[(size_t)layer * (size_t)num_experts + (size_t)expert]++;
    }

    std::vector<std::pair<int, int>> top_experts(size_t k) const {
        struct Item {
            uint64_t cnt;
            int layer;
            int expert;
        };
        std::vector<Item> items;
        items.reserve(counts.size());
        for (int l = 0; l < num_layers; ++l) {
            for (int e = 0; e < num_experts; ++e) {
                uint64_t c = counts[(size_t)l * (size_t)num_experts + (size_t)e];
                if (c > 0) items.push_back({c, l, e});
            }
        }
        std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
            if (a.cnt != b.cnt) return a.cnt > b.cnt;
            if (a.layer != b.layer) return a.layer < b.layer;
            return a.expert < b.expert;
        });
        size_t n = std::min(k, items.size());
        std::vector<std::pair<int, int>> out;
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) out.push_back({items[i].layer, items[i].expert});
        return out;
    }

    bool save_tsv(const std::string& path) const {
        std::string tmp = path + ".tmp";
        std::ofstream f(tmp, std::ios::trunc);
        if (!f) return false;
        f << num_layers << "\t" << num_experts << "\n";
        for (int l = 0; l < num_layers; ++l) {
            for (int e = 0; e < num_experts; ++e) {
                uint64_t c = counts[(size_t)l * (size_t)num_experts + (size_t)e];
                if (c > 0) f << l << "\t" << e << "\t" << c << "\n";
            }
        }
        f.close();
        if (!f) return false;
        return std::rename(tmp.c_str(), path.c_str()) == 0;
    }

    bool load_tsv(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;
        int L = 0, E = 0;
        if (!(f >> L >> E) || L != num_layers || E != num_experts) return false;
        int l = 0, e = 0;
        uint64_t c = 0;
        while (f >> l >> e >> c) {
            if (l >= 0 && l < num_layers && e >= 0 && e < num_experts) {
                counts[(size_t)l * (size_t)num_experts + (size_t)e] += c;
            }
        }
        return true;
    }
};

struct AsyncPrewarmState {
    std::atomic<bool> stop_flag{false};
    std::atomic<int> infer_depth{0};
    std::mutex cache_mtx;
    std::thread worker;

    struct InferScope {
        AsyncPrewarmState& st;
        explicit InferScope(AsyncPrewarmState& s) : st(s) {
            st.infer_depth.fetch_add(1, std::memory_order_acq_rel);
        }
        ~InferScope() {
            st.infer_depth.fetch_sub(1, std::memory_order_acq_rel);
        }
        InferScope(const InferScope&) = delete;
        InferScope& operator=(const InferScope&) = delete;
    };

    ~AsyncPrewarmState() {
        stop_and_join();
    }

    bool should_stop() const {
        return stop_flag.load(std::memory_order_acquire);
    }

    bool should_yield() const {
        return infer_depth.load(std::memory_order_acquire) > 0;
    }

    template <typename Fn>
    void start(Fn&& fn) {
        stop_and_join();
        stop_flag.store(false, std::memory_order_release);
        worker = std::thread(std::forward<Fn>(fn));
    }

    void stop_and_join() {
        stop_flag.store(true, std::memory_order_release);
        if (worker.joinable()) worker.join();
    }
};

}
