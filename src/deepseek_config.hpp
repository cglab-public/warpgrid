#pragma once
#include "gguf.hpp"
#include <cstdint>
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
    int num_key_value_heads = 8;
    int head_dim = 128;
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
    float route_scale = 1.0f;
    int context_length = 1048576;
    int index_topk = 2048;
    int index_head_count = 64;
    std::vector<int> compress_ratios;
    std::vector<int> engram_layers = {1, 14};

    bool is_encoder_layer(int layer_id) const {
        return layer_id >= 0 && layer_id < encoder_layers;
    }

    int compress_ratio_for(int layer_id) const {
        if (layer_id >= 0 && (size_t)layer_id < compress_ratios.size()) {
            return compress_ratios[(size_t)layer_id];
        }
        return 1;
    }

    size_t kv_bytes_per_token() const {
        size_t per_layer = (size_t)2 * (size_t)num_key_value_heads * (size_t)head_dim * sizeof(float);
        return (size_t)num_hidden_layers * per_layer;
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
        c.num_key_value_heads = (int)gf.get_i64(p + "attention.head_count_kv", c.num_attention_heads);
        int default_hd = (c.num_attention_heads > 0) ? (c.hidden_size / c.num_attention_heads) : 128;
        c.head_dim = (int)gf.get_i64(p + "attention.key_length", default_hd);
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
        c.route_scale = (float)gf.get_f64(p + "expert_weights_scale", c.route_scale);
        c.context_length = (int)gf.get_i64(p + "context_length", c.context_length);
        c.index_topk = (int)gf.get_i64(p + "attention.indexer.top_k", c.index_topk);
        c.index_head_count = (int)gf.get_i64(p + "attention.indexer.head_count", c.index_head_count);

        if (gf.has_kv(p + "attention.compress_ratios")) {
            const auto& v = gf.kv(p + "attention.compress_ratios");
            c.compress_ratios.clear();
            for (int64_t x : v.i64_arr) c.compress_ratios.push_back((int)x);
            for (uint64_t x : v.u64_arr) c.compress_ratios.push_back((int)x);
        }
        return c;
    }
};

}
