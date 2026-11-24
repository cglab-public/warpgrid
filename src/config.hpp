#pragma once
#include "json.hpp"
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct InklingConfig {
    int vocab_size = 201024;
    int unpadded_vocab_size = 200058;
    int hidden_size = 2304;
    int num_hidden_layers = 24;
    int num_attention_heads = 24;
    int num_key_value_heads = 8;
    int head_dim = 128;
    int d_rel = 16;
    int rel_extent = 1024;
    int sliding_window_size = 1024;
    int sconv_kernel_size = 4;
    float rms_norm_eps = 1e-6f;
    float qk_l2_clip_threshold = 22.627416997969522f;
    float rel_l2_clip_threshold = 8.0f;
    float log_scaling_n_floor = 8192.0f;
    float log_scaling_alpha = 0.1f;
    float final_logit_softcapping = 30.0f;
    int n_layers_per_attn_res_block = 6;
    int n_routed_experts = 128;
    int num_experts_per_tok = 8;
    int intermediate_size = 1152;
    float mlp_gate_softcap = 4.0f;
    float mlp_up_softcap = 25.0f;
    float route_scale = 8.0f;

    // Loop config
    int loop_core_start = 6;
    int loop_core_end = 18;
    int loop_num_passes = 3;
    int loop_gate_rank = 64;

    int eos_token_id = 200006; // <|content_model_end_sampling|>
    int end_message_token_id = 200008;
    std::set<int> local_layer_ids;

    bool is_local(int layer_id) const { return local_layer_ids.count(layer_id) > 0; }
    bool is_loop_core(int layer_id) const { return layer_id >= loop_core_start && layer_id < loop_core_end; }
    int num_visits() const {
        return num_hidden_layers + (loop_num_passes - 1) * (loop_core_end - loop_core_start);
    }

    static InklingConfig load(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("Cannot open config.json: " + path);
        std::stringstream ss;
        ss << f.rdbuf();
        auto root = json::parse(ss.str());
        const auto& tc = root.contains("text_config") ? root["text_config"] : root;

        InklingConfig c;
        if (root.contains("eos_token_id")) c.eos_token_id = root["eos_token_id"].as_int(c.eos_token_id);
        c.vocab_size = tc["vocab_size"].as_int(c.vocab_size);
        c.unpadded_vocab_size = tc["unpadded_vocab_size"].as_int(c.unpadded_vocab_size);
        c.hidden_size = tc["hidden_size"].as_int(c.hidden_size);
        c.num_hidden_layers = tc["num_hidden_layers"].as_int(c.num_hidden_layers);
        c.num_attention_heads = tc["num_attention_heads"].as_int(c.num_attention_heads);
        c.num_key_value_heads = tc["num_key_value_heads"].as_int(c.num_key_value_heads);
        c.head_dim = tc["head_dim"].as_int(c.head_dim);
        c.d_rel = tc["d_rel"].as_int(c.d_rel);
        c.rel_extent = tc["rel_extent"].as_int(c.rel_extent);
        c.sliding_window_size = tc["sliding_window_size"].as_int(c.sliding_window_size);
        c.sconv_kernel_size = tc["sconv_kernel_size"].as_int(c.sconv_kernel_size);
        c.rms_norm_eps = (float)tc["rms_norm_eps"].as_double(c.rms_norm_eps);
        c.qk_l2_clip_threshold = (float)tc["qk_l2_clip_threshold"].as_double(c.qk_l2_clip_threshold);
        c.rel_l2_clip_threshold = (float)tc["rel_l2_clip_threshold"].as_double(c.rel_l2_clip_threshold);
        c.log_scaling_n_floor = (float)tc["log_scaling_n_floor"].as_double(c.log_scaling_n_floor);
        c.log_scaling_alpha = (float)tc["log_scaling_alpha"].as_double(c.log_scaling_alpha);
        c.final_logit_softcapping = (float)tc["final_logit_softcapping"].as_double(c.final_logit_softcapping);
        c.n_layers_per_attn_res_block = tc["n_layers_per_attn_res_block"].as_int(c.n_layers_per_attn_res_block);
        c.n_routed_experts = tc["n_routed_experts"].as_int(c.n_routed_experts);
        c.num_experts_per_tok = tc["num_experts_per_tok"].as_int(c.num_experts_per_tok);
        c.intermediate_size = tc["intermediate_size"].as_int(c.intermediate_size);
        c.mlp_gate_softcap = (float)tc["mlp_gate_softcap"].as_double(c.mlp_gate_softcap);
        c.mlp_up_softcap = (float)tc["mlp_up_softcap"].as_double(c.mlp_up_softcap);
        c.route_scale = (float)tc["route_scale"].as_double(c.route_scale);

        if (tc.contains("local_layer_ids") && tc["local_layer_ids"].is_array()) {
            for (const auto& x : tc["local_layer_ids"].arr) {
                c.local_layer_ids.insert(x.as_int());
            }
        }
        if (tc.contains("loop_config") && tc["loop_config"].is_object()) {
            const auto& lc = tc["loop_config"];
            c.loop_core_start = lc["core_start"].as_int(6);
            c.loop_core_end = lc["core_end"].as_int(18);
            c.loop_num_passes = lc["num_passes"].as_int(3);
            c.loop_gate_rank = lc["gate_rank"].as_int(64);
        }
        return c;
    }
};
