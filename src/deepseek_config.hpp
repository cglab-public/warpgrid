#pragma once
#include "gguf.hpp"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ink {

struct DeepSeekConfig {
    std::string arch = "deepseek41";
    int vocab_size = 129280;
    int hidden_size = 4096;
    int num_hidden_layers = 40;
    int encoder_layers = 20;
    int num_attention_heads = 32;
    int num_key_value_heads = 1;
    int head_dim = 128;
    int n_rot = 64;
    int q_lora_rank = 1536;
    int o_group_count = 4;
    int o_lora_rank = 512;
    int sliding_window = 2048;
    int ffn_intermediate_size = 10944;
    int expert_intermediate_size = 1536;
    int n_routed_experts = 384;
    int num_experts_per_tok = 6;
    int n_shared_experts = 1;
    int hc_mult = 4;
    int hc_sinkhorn_iters = 20;
    float hc_eps = 1e-6f;
    float rms_norm_eps = 1e-6f;
    float rope_freq_base = 10000.0f;
    float compress_rope_freq_base = 10000.0f;
    float route_scale = 1.0f;
    bool expert_weights_norm = true;
    float swiglu_clamp_exp = 0.0f;
    float swiglu_clamp_shexp = 0.0f;
    int context_length = 1048576;
    int index_topk = 2048;
    int index_head_count = 64;
    int index_head_dim = 128;

    std::vector<int> compress_ratios;
    std::vector<int> kv_src;
    std::vector<int> topk_src;

    std::vector<int> ple_layers;
    int ple_ngram_size = 0;
    int ple_heads_per_ngram = 0;
    int ple_head_dim = 0;
    int ple_n_heads = 0;
    std::vector<uint64_t> ple_layer_multipliers;
    std::vector<uint32_t> ple_head_offsets;
    std::vector<uint32_t> ple_head_vocab_sizes;
    uint32_t engram_pad_id = 0;
    std::vector<uint32_t> engram_token_map;

    bool is_encoder_layer(int layer_id) const {
        return layer_id >= 0 && layer_id < encoder_layers;
    }

    int compress_ratio_for(int layer_id) const {
        if (layer_id >= 0 && (size_t)layer_id < compress_ratios.size()) {
            return compress_ratios[(size_t)layer_id];
        }
        return 0;
    }

    int engram_index(int layer_id) const {
        for (size_t i = 0; i < ple_layers.size(); ++i) {
            if (ple_layers[i] == layer_id) return (int)i;
        }
        return -1;
    }

    void engram_hash_token(int e_idx, const std::vector<int32_t>& hist, int pos, int32_t* out_rows) const {
        int n_gram = ple_ngram_size;
        int n_head = ple_heads_per_ngram;
        int n_cols = ple_n_heads;
        if (e_idx < 0 || n_gram < 2 || n_cols <= 0 || engram_token_map.empty()) return;

        const uint64_t* mult = ple_layer_multipliers.data() + (size_t)e_idx * (size_t)n_gram;
        const uint32_t* primes = ple_head_vocab_sizes.data() + (size_t)e_idx * (size_t)n_cols;
        const uint32_t* offs = ple_head_offsets.data() + (size_t)e_idx * (size_t)n_cols;
        int64_t pad = (engram_pad_id < engram_token_map.size()) ? (int64_t)engram_token_map[engram_pad_id] : 0;

        auto cid = [&](int32_t t) -> int64_t {
            if (t < 0 || (size_t)t >= engram_token_map.size()) return -1;
            return (int64_t)engram_token_map[(size_t)t];
        };

        std::vector<uint64_t> toks((size_t)n_gram, (uint64_t)pad);
        bool blocked = false;
        for (int s = 0; s < n_gram; ++s) {
            int p = pos - s;
            int64_t c = (p < 0 || (size_t)p >= hist.size()) ? -1 : cid(hist[(size_t)p]);
            if (c < 0) blocked = true;
            toks[(size_t)s] = blocked ? (uint64_t)pad : (uint64_t)c;
        }

        uint64_t rolling = toks[0] * mult[0];
        for (int o = 1; o < n_gram; ++o) {
            rolling ^= toks[(size_t)o] * mult[(size_t)o];
            for (int h = 0; h < n_head; ++h) {
                int col = (o - 1) * n_head + h;
                out_rows[col] = (int32_t)(rolling % primes[col] + offs[col]);
            }
        }
    }

    size_t kv_bytes_per_token() const {
        return (size_t)num_hidden_layers * (size_t)head_dim * sizeof(float);
    }

    static DeepSeekConfig from_gguf(const GgufFile& gf) {
        DeepSeekConfig c;
        c.arch = gf.get_str("general.architecture", "deepseek41");
        const std::string p = c.arch + ".";

        c.num_hidden_layers = (int)gf.get_i64(p + "block_count", c.num_hidden_layers);
        c.encoder_layers = (int)gf.get_i64(p + "encoder_layer_count",
            c.arch == "deepseek41" ? (c.num_hidden_layers / 2) : 0);
        c.hidden_size = (int)gf.get_i64(p + "embedding_length", c.hidden_size);
        int default_vocab = c.vocab_size;
        if (gf.has_kv("tokenizer.ggml.tokens") && !gf.kv("tokenizer.ggml.tokens").str_arr.empty()) {
            default_vocab = (int)gf.kv("tokenizer.ggml.tokens").str_arr.size();
        }
        c.vocab_size = (int)gf.get_i64(p + "vocab_size", default_vocab);
        c.num_attention_heads = (int)gf.get_i64(p + "attention.head_count", c.num_attention_heads);
        c.num_key_value_heads = (int)gf.get_i64(p + "attention.head_count_kv", 1);
        int default_hd = (c.num_attention_heads > 0) ? (c.hidden_size / c.num_attention_heads) : 128;
        c.head_dim = (int)gf.get_i64(p + "attention.key_length", default_hd);
        c.n_rot = (int)gf.get_i64(p + "rope.dimension_count", c.head_dim / 2);
        c.q_lora_rank = (int)gf.get_i64(p + "attention.q_lora_rank", c.q_lora_rank);
        c.o_group_count = (int)gf.get_i64(p + "attention.output_group_count", c.o_group_count);
        c.o_lora_rank = (int)gf.get_i64(p + "attention.output_lora_rank", c.o_lora_rank);
        c.sliding_window = (int)gf.get_i64(p + "attention.sliding_window", c.sliding_window);
        c.ffn_intermediate_size = (int)gf.get_i64(p + "feed_forward_length", c.ffn_intermediate_size);
        c.expert_intermediate_size = (int)gf.get_i64(p + "expert_feed_forward_length", c.expert_intermediate_size);
        c.n_routed_experts = (int)gf.get_i64(p + "expert_count", c.n_routed_experts);
        c.num_experts_per_tok = (int)gf.get_i64(p + "expert_used_count", c.num_experts_per_tok);
        c.n_shared_experts = (int)gf.get_i64(p + "expert_shared_count", c.n_shared_experts);
        c.hc_mult = (int)gf.get_i64(p + "hyper_connection.count", c.hc_mult);
        c.hc_sinkhorn_iters = (int)gf.get_i64(p + "hyper_connection.sinkhorn_iterations", c.hc_sinkhorn_iters);
        c.hc_eps = (float)gf.get_f64(p + "hyper_connection.epsilon", c.hc_eps);
        c.rms_norm_eps = (float)gf.get_f64(p + "attention.layer_norm_rms_epsilon", c.rms_norm_eps);
        c.rope_freq_base = (float)gf.get_f64(p + "rope.freq_base", c.rope_freq_base);
        c.compress_rope_freq_base = (float)gf.get_f64(p + "attention.compress_rope_freq_base", c.rope_freq_base);
        c.route_scale = (float)gf.get_f64(p + "expert_weights_scale", c.route_scale);
        c.expert_weights_norm = gf.has_kv(p + "expert_weights_norm")
            ? gf.kv(p + "expert_weights_norm").as_bool(true) : true;
        c.swiglu_clamp_exp = (float)gf.get_f64(p + "swiglu_clamp_exp", 0.0);
        c.swiglu_clamp_shexp = (float)gf.get_f64(p + "swiglu_clamp_shexp", c.swiglu_clamp_exp);
        c.context_length = (int)gf.get_i64(p + "context_length", c.context_length);
        c.index_topk = (int)gf.get_i64(p + "attention.indexer.top_k", c.index_topk);
        c.index_head_count = (int)gf.get_i64(p + "attention.indexer.head_count", c.index_head_count);
        c.index_head_dim = (int)gf.get_i64(p + "attention.indexer.key_length", c.index_head_dim);

        c.compress_ratios.assign((size_t)c.num_hidden_layers, 0);
        if (gf.has_kv(p + "attention.compress_ratios")) {
            const auto& v = gf.kv(p + "attention.compress_ratios");
            size_t n = std::max(v.i64_arr.size(), v.u64_arr.size());
            for (size_t i = 0; i < std::min(n, c.compress_ratios.size()); ++i) {
                c.compress_ratios[i] = !v.u64_arr.empty() ? (int)v.u64_arr[i] : (int)v.i64_arr[i];
            }
        }

        c.kv_src.assign((size_t)c.num_hidden_layers, -1);
        c.topk_src.assign((size_t)c.num_hidden_layers, -1);
        std::vector<int> kv_src_list, idx_src_list;
        if (gf.has_kv(p + "attention.kv_source_layers")) {
            const auto& v = gf.kv(p + "attention.kv_source_layers");
            for (uint64_t x : v.u64_arr) kv_src_list.push_back((int)x);
            for (int64_t x : v.i64_arr) kv_src_list.push_back((int)x);
        }
        if (gf.has_kv(p + "attention.indexer.source_layers")) {
            const auto& v = gf.kv(p + "attention.indexer.source_layers");
            for (uint64_t x : v.u64_arr) idx_src_list.push_back((int)x);
            for (int64_t x : v.i64_arr) idx_src_list.push_back((int)x);
        }
        int last_kv = -1, last_idx = -1;
        for (int il = 0; il < c.num_hidden_layers; ++il) {
            bool is_kv = std::find(kv_src_list.begin(), kv_src_list.end(), il) != kv_src_list.end();
            bool is_idx = std::find(idx_src_list.begin(), idx_src_list.end(), il) != idx_src_list.end();
            if (is_kv) last_kv = il;
            if (is_idx) last_idx = il;
            if (c.compress_ratios[(size_t)il] != 0) {
                c.kv_src[(size_t)il] = last_kv;
                c.topk_src[(size_t)il] = last_idx;
            }
        }

        if (gf.has_kv(p + "ple.layers")) {
            const auto& v = gf.kv(p + "ple.layers");
            for (uint64_t x : v.u64_arr) c.ple_layers.push_back((int)x);
            for (int64_t x : v.i64_arr) c.ple_layers.push_back((int)x);
            c.ple_ngram_size = (int)gf.get_i64(p + "ple.ngram_size", 0);
            c.ple_heads_per_ngram = (int)gf.get_i64(p + "ple.heads_per_ngram", 0);
            c.ple_head_dim = (int)gf.get_i64(p + "embedding_length_per_layer_input", 0);
            c.ple_n_heads = (c.ple_ngram_size >= 2) ? (c.ple_ngram_size - 1) * c.ple_heads_per_ngram : 0;
            if (gf.has_kv(p + "ple.layer_multipliers")) {
                const auto& mv = gf.kv(p + "ple.layer_multipliers");
                for (uint64_t x : mv.u64_arr) c.ple_layer_multipliers.push_back(x);
                for (int64_t x : mv.i64_arr) c.ple_layer_multipliers.push_back((uint64_t)x);
            }
            if (gf.has_kv(p + "ple.head_offsets")) {
                const auto& ov = gf.kv(p + "ple.head_offsets");
                for (uint64_t x : ov.u64_arr) c.ple_head_offsets.push_back((uint32_t)x);
                for (int64_t x : ov.i64_arr) c.ple_head_offsets.push_back((uint32_t)x);
            }
            if (gf.has_kv(p + "ple.head_vocab_sizes")) {
                const auto& pv = gf.kv(p + "ple.head_vocab_sizes");
                for (uint64_t x : pv.u64_arr) c.ple_head_vocab_sizes.push_back((uint32_t)x);
                for (int64_t x : pv.i64_arr) c.ple_head_vocab_sizes.push_back((uint32_t)x);
            }
            c.engram_pad_id = (uint32_t)gf.get_i64(p + "engram.pad_token_id", 0);
            if (gf.has_kv(p + "engram.token_map")) {
                const auto& tv = gf.kv(p + "engram.token_map");
                for (uint64_t x : tv.u64_arr) c.engram_token_map.push_back((uint32_t)x);
                for (int64_t x : tv.i64_arr) c.engram_token_map.push_back((uint32_t)x);
            }
        }
        return c;
    }
};

}
