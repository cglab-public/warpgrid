#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../src/metal/deepseek_v41_metal.hpp"
#include "../src/gguf_iq_tables.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mach/mach.h>
#include <numeric>
#include <string>
#include <vector>

namespace {

static uint64_t process_rss_bytes() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        return (uint64_t)info.phys_footprint;
    }
    return 0;
}

static void wr_u32(std::vector<uint8_t>& b, uint32_t v) {
    size_t p = b.size();
    b.resize(p + 4);
    memcpy(b.data() + p, &v, 4);
}

static void wr_i32(std::vector<uint8_t>& b, int32_t v) {
    size_t p = b.size();
    b.resize(p + 4);
    memcpy(b.data() + p, &v, 4);
}

static void wr_u64(std::vector<uint8_t>& b, uint64_t v) {
    size_t p = b.size();
    b.resize(p + 8);
    memcpy(b.data() + p, &v, 8);
}

static void wr_f32(std::vector<uint8_t>& b, float v) {
    size_t p = b.size();
    b.resize(p + 4);
    memcpy(b.data() + p, &v, 4);
}

static void wr_str(std::vector<uint8_t>& b, const std::string& s) {
    wr_u64(b, (uint64_t)s.size());
    b.insert(b.end(), s.begin(), s.end());
}

static void kv_str(std::vector<uint8_t>& b, const std::string& k, const std::string& v) {
    wr_str(b, k);
    wr_u32(b, 8u);
    wr_str(b, v);
}

static void kv_u32(std::vector<uint8_t>& b, const std::string& k, uint32_t v) {
    wr_str(b, k);
    wr_u32(b, 4u);
    wr_u32(b, v);
}

static void kv_f32(std::vector<uint8_t>& b, const std::string& k, float v) {
    wr_str(b, k);
    wr_u32(b, 6u);
    wr_f32(b, v);
}

static void kv_i32_arr(std::vector<uint8_t>& b, const std::string& k, const std::vector<int32_t>& arr) {
    wr_str(b, k);
    wr_u32(b, 9u);
    wr_u32(b, 5u);
    wr_u64(b, (uint64_t)arr.size());
    for (int32_t v : arr) wr_i32(b, v);
}

static void kv_u64_arr(std::vector<uint8_t>& b, const std::string& k, const std::vector<uint64_t>& arr) {
    wr_str(b, k);
    wr_u32(b, 9u);
    wr_u32(b, 10u);
    wr_u64(b, (uint64_t)arr.size());
    for (uint64_t v : arr) wr_u64(b, v);
}

static void kv_f32_arr(std::vector<uint8_t>& b, const std::string& k, const std::vector<float>& arr) {
    wr_str(b, k);
    wr_u32(b, 9u);
    wr_u32(b, 6u);
    wr_u64(b, (uint64_t)arr.size());
    for (float v : arr) wr_f32(b, v);
}

struct SynthTensor {
    std::string name;
    std::vector<uint64_t> dims;
    ink::GgmlType type;
    size_t nbytes;
    uint64_t offset;
};

static std::string create_synthetic_deepseek41_gguf(const std::string& path, int n_layers, int n_experts) {
    const int D = 5120;
    const int H = 64;
    const int HD = 512;
    const int Q_LORA = 1280;
    const int O_GROUPS = 8;
    const int O_LORA = 1024;
    const int HC = 4;
    const int TOPK = 6;
    const int INTER = 2304;
    const int IDX_H = 32;
    const int IDX_D = 128;
    const int VOCAB = 1024;

    std::vector<SynthTensor> tensors;
    auto add_t = [&](const std::string& name, std::vector<uint64_t> dims, ink::GgmlType t) {
        size_t row_bytes = ink::ggml_row_nbytes(t, dims[0]);
        size_t nrows = 1;
        for (size_t i = 1; i < dims.size(); ++i) nrows *= (size_t)dims[i];
        tensors.push_back({name, dims, t, row_bytes * nrows, 0});
    };

    add_t("token_embd.weight", {(uint64_t)D, (uint64_t)VOCAB}, ink::GgmlType::Q6_K);
    add_t("output_norm.weight", {(uint64_t)D}, ink::GgmlType::F32);
    add_t("output.weight", {(uint64_t)D, (uint64_t)VOCAB}, ink::GgmlType::Q6_K);

    for (int l = 0; l < n_layers; ++l) {
        std::string p = "blk." + std::to_string(l) + ".";
        add_t(p + "attn_norm.weight", {(uint64_t)D}, ink::GgmlType::F32);
        add_t(p + "attn_sinks.weight", {(uint64_t)H}, ink::GgmlType::F32);
        add_t(p + "attn_q_a.weight", {(uint64_t)D, (uint64_t)Q_LORA}, ink::GgmlType::Q6_K);
        add_t(p + "attn_q_a_norm.weight", {(uint64_t)Q_LORA}, ink::GgmlType::F32);
        add_t(p + "attn_q_b.weight", {(uint64_t)Q_LORA, (uint64_t)(H * HD)}, ink::GgmlType::Q6_K);
        add_t(p + "attn_kv.weight", {(uint64_t)D, (uint64_t)HD}, ink::GgmlType::Q6_K);
        add_t(p + "attn_kv_a_norm.weight", {(uint64_t)HD}, ink::GgmlType::F32);
        add_t(p + "attn_output_a.weight", {(uint64_t)((H / O_GROUPS) * HD), (uint64_t)(O_GROUPS * O_LORA)}, ink::GgmlType::Q6_K);
        add_t(p + "attn_output_b.weight", {(uint64_t)(O_GROUPS * O_LORA), (uint64_t)D}, ink::GgmlType::Q6_K);

        add_t(p + "hc_attn_fn.weight", {(uint64_t)(HC * D), (uint64_t)((2 + HC) * HC)}, ink::GgmlType::F32);
        add_t(p + "hc_attn_base.weight", {(uint64_t)((2 + HC) * HC)}, ink::GgmlType::F32);
        add_t(p + "hc_attn_scale.weight", {3}, ink::GgmlType::F32);
        add_t(p + "hc_ffn_fn.weight", {(uint64_t)(HC * D), (uint64_t)((2 + HC) * HC)}, ink::GgmlType::F32);
        add_t(p + "hc_ffn_base.weight", {(uint64_t)((2 + HC) * HC)}, ink::GgmlType::F32);
        add_t(p + "hc_ffn_scale.weight", {3}, ink::GgmlType::F32);

        if (l >= 2) {
            if (l == 2) {
                add_t(p + "attn_compressor_kv.weight", {(uint64_t)D, (uint64_t)HD}, ink::GgmlType::Q5_K);
                add_t(p + "attn_compressor_gate.weight", {(uint64_t)D, (uint64_t)HD}, ink::GgmlType::Q5_K);
                add_t(p + "attn_compressor_norm.weight", {(uint64_t)HD}, ink::GgmlType::F32);
                add_t(p + "indexer.attn_k.weight", {(uint64_t)HD, (uint64_t)IDX_D}, ink::GgmlType::Q8_0);
                add_t(p + "indexer.k_norm.weight", {(uint64_t)IDX_D}, ink::GgmlType::F32);
                add_t(p + "indexer.attn_q_b.weight", {(uint64_t)Q_LORA, (uint64_t)(IDX_H * IDX_D)}, ink::GgmlType::Q8_0);
                add_t(p + "indexer.proj.weight", {(uint64_t)D, (uint64_t)IDX_H}, ink::GgmlType::BF16);
            }
        }

        add_t(p + "ffn_norm.weight", {(uint64_t)D}, ink::GgmlType::F32);
        add_t(p + "ffn_gate_inp.weight", {(uint64_t)D, (uint64_t)n_experts}, ink::GgmlType::F32);
        add_t(p + "exp_probs_b.bias", {(uint64_t)n_experts}, ink::GgmlType::F32);
        ink::GgmlType exp_t = (l == 0) ? ink::GgmlType::IQ2_XXS : ink::GgmlType::IQ1_S;
        add_t(p + "ffn_gate_exps.weight", {(uint64_t)D, (uint64_t)INTER, (uint64_t)n_experts}, exp_t);
        add_t(p + "ffn_up_exps.weight", {(uint64_t)D, (uint64_t)INTER, (uint64_t)n_experts}, exp_t);
        add_t(p + "ffn_down_exps.weight", {(uint64_t)INTER, (uint64_t)D, (uint64_t)n_experts}, exp_t);
        add_t(p + "ffn_gate_shexp.weight", {(uint64_t)D, (uint64_t)INTER}, ink::GgmlType::Q6_K);
        add_t(p + "ffn_up_shexp.weight", {(uint64_t)D, (uint64_t)INTER}, ink::GgmlType::Q6_K);
        add_t(p + "ffn_down_shexp.weight", {(uint64_t)INTER, (uint64_t)D}, ink::GgmlType::Q6_K);
    }

    std::vector<uint8_t> hdr;
    hdr.insert(hdr.end(), {'G', 'G', 'U', 'F'});
    wr_u32(hdr, 3u);
    wr_u64(hdr, (uint64_t)tensors.size());
    std::vector<int32_t> comp_ratios(n_layers, 0);
    for (int l = 2; l < n_layers; ++l) comp_ratios[l] = 2;
    std::vector<int32_t> kv_srcs;
    std::vector<int32_t> idx_srcs;
    if (n_layers > 2) { kv_srcs.push_back(2); idx_srcs.push_back(2); }

    uint64_t n_kv = 26u + (kv_srcs.empty() ? 0u : 2u);
    wr_u64(hdr, n_kv);
    kv_str(hdr, "general.architecture", "deepseek41");
    kv_u32(hdr, "deepseek41.block_count", (uint32_t)n_layers);
    kv_u32(hdr, "deepseek41.context_length", 32768u);
    kv_u32(hdr, "deepseek41.embedding_length", (uint32_t)D);
    kv_u32(hdr, "deepseek41.attention.head_count", (uint32_t)H);
    kv_u32(hdr, "deepseek41.attention.head_count_kv", 1u);
    kv_u32(hdr, "deepseek41.attention.key_length", (uint32_t)HD);
    kv_u32(hdr, "deepseek41.attention.value_length", (uint32_t)HD);
    kv_u32(hdr, "deepseek41.rope.dimension_count", 64u);
    kv_f32(hdr, "deepseek41.rope.freq_base", 10000.0f);
    kv_f32(hdr, "deepseek41.attention.compress_rope_freq_base", 160000.0f);
    kv_f32(hdr, "deepseek41.attention.layer_norm_rms_epsilon", 1e-6f);
    kv_u32(hdr, "deepseek41.attention.q_lora_rank", (uint32_t)Q_LORA);
    kv_u32(hdr, "deepseek41.attention.output_group_count", (uint32_t)O_GROUPS);
    kv_u32(hdr, "deepseek41.attention.output_lora_rank", (uint32_t)O_LORA);
    kv_u32(hdr, "deepseek41.attention.sliding_window", 128u);
    kv_u32(hdr, "deepseek41.attention.indexer.head_count", (uint32_t)IDX_H);
    kv_u32(hdr, "deepseek41.attention.indexer.key_length", (uint32_t)IDX_D);
    kv_u32(hdr, "deepseek41.attention.indexer.top_k", 512u);
    kv_u32(hdr, "deepseek41.expert_count", (uint32_t)n_experts);
    kv_u32(hdr, "deepseek41.expert_used_count", (uint32_t)TOPK);
    kv_u32(hdr, "deepseek41.expert_feed_forward_length", (uint32_t)INTER);
    kv_f32(hdr, "deepseek41.expert_weights_scale", 1.5f);
    kv_u32(hdr, "deepseek41.hyper_connection.count", (uint32_t)HC);
    kv_u32(hdr, "deepseek41.hyper_connection.sinkhorn_iterations", 20u);
    kv_i32_arr(hdr, "deepseek41.attention.compress_ratios", comp_ratios);
    if (!kv_srcs.empty()) {
        kv_i32_arr(hdr, "deepseek41.attention.kv_source_layers", kv_srcs);
        kv_i32_arr(hdr, "deepseek41.attention.indexer.source_layers", idx_srcs);
    }

    uint64_t cur_off = 0;
    for (auto& t : tensors) {
        cur_off = (cur_off + 31u) & ~31ull;
        t.offset = cur_off;
        cur_off += t.nbytes;
    }

    for (const auto& t : tensors) {
        wr_str(hdr, t.name);
        wr_u32(hdr, (uint32_t)t.dims.size());
        for (uint64_t d : t.dims) wr_u64(hdr, d);
        wr_u32(hdr, (uint32_t)t.type);
        wr_u64(hdr, t.offset);
    }

    while (hdr.size() % 32 != 0) hdr.push_back(0);
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    ofs.write(reinterpret_cast<const char*>(hdr.data()), (std::streamsize)hdr.size());

    std::vector<uint8_t> chunk(1 << 20);
    for (size_t i = 0; i < chunk.size(); ++i) chunk[i] = (uint8_t)((i * 37 + 13) & 0x3F);
    uint64_t rem = cur_off;
    while (rem > 0) {
        size_t w = (size_t)std::min<uint64_t>(rem, chunk.size());
        ofs.write(reinterpret_cast<const char*>(chunk.data()), (std::streamsize)w);
        rem -= w;
    }
    ofs.close();
    return path;
}

}

int main(int argc, char** argv) {
    @autoreleasepool {
        int n_layers = 4;
        int n_experts = 16;
        int samples = 7;
        int decode_steps = 16;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--layers" && i + 1 < argc) n_layers = std::atoi(argv[++i]);
            else if (a == "--experts" && i + 1 < argc) n_experts = std::atoi(argv[++i]);
            else if (a == "--samples" && i + 1 < argc) samples = std::atoi(argv[++i]);
            else if (a == "--steps" && i + 1 < argc) decode_steps = std::atoi(argv[++i]);
        }

        std::string tmp_path = "/tmp/warpgrid_synth_ds41.gguf";
        create_synthetic_deepseek41_gguf(tmp_path, n_layers, n_experts);

        auto t_load0 = std::chrono::steady_clock::now();
        ink::DeepSeekV41Model model;
        model.load(tmp_path, 4096, 64, 1);
        auto t_load1 = std::chrono::steady_clock::now();
        double load_ms = std::chrono::duration<double, std::milli>(t_load1 - t_load0).count();

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8};
        model.reset_cache(0);
        int32_t warm_tok = model.forward_and_greedy(prompt.data(), (int)prompt.size(), 0, 0);
        for (int s = 0; s < 4; ++s) {
            warm_tok = model.forward_and_greedy(&warm_tok, 1, (int)prompt.size() + s, 0);
        }

        std::vector<double> c1_ms;
        int32_t ref_tok = -1;
        bool greedy_ok = true;
        for (int rep = 0; rep < samples; ++rep) {
            model.reset_cache(0);
            int32_t tok = model.forward_and_greedy(prompt.data(), (int)prompt.size(), 0, 0);
            auto t0 = std::chrono::steady_clock::now();
            for (int s = 0; s < decode_steps; ++s) {
                tok = model.forward_and_greedy(&tok, 1, (int)prompt.size() + s, 0);
            }
            auto t1 = std::chrono::steady_clock::now();
            double ms_per_tok = std::chrono::duration<double, std::milli>(t1 - t0).count() / (double)decode_steps;
            c1_ms.push_back(ms_per_tok);
            if (rep == 0) ref_tok = tok;
            else if (tok != ref_tok) greedy_ok = false;
        }

        std::sort(c1_ms.begin(), c1_ms.end());
        double med_ms = c1_ms[c1_ms.size() / 2];
        double min_ms = c1_ms.front();
        double max_ms = c1_ms.back();
        double proj_40l_ms = med_ms * (40.0 / (double)n_layers);
        double proj_40l_tok_s = 1000.0 / proj_40l_ms;
        double rss_gib = (double)process_rss_bytes() / (1024.0 * 1024.0 * 1024.0);

        std::vector<float> ref_logits;
        model.reset_cache(0);
        model.forward_logits(prompt.data(), (int)prompt.size(), 0, ref_logits, 0);
        int32_t cpu_argmax = 0;
        float best_v = ref_logits.empty() ? 0.0f : ref_logits[0];
        bool all_finite = true;
        for (size_t i = 0; i < ref_logits.size(); ++i) {
            if (!std::isfinite(ref_logits[i])) all_finite = false;
            if (ref_logits[i] > best_v) { best_v = ref_logits[i]; cpu_argmax = (int32_t)i; }
        }
        model.reset_cache(0);
        int32_t gpu_argmax = model.forward_and_greedy(prompt.data(), (int)prompt.size(), 0, 0);
        bool argmax_parity = (cpu_argmax == gpu_argmax) && all_finite;

        printf("{\"layers\":%d,\"experts\":%d,\"samples\":%d,\"steps\":%d,"
               "\"load_ms\":%.2f,\"med_ms_per_tok\":%.3f,\"min_ms\":%.3f,\"max_ms\":%.3f,"
               "\"proj_40l_ms\":%.3f,\"proj_40l_tok_s\":%.2f,\"rss_gib\":%.3f,"
               "\"ref_tok\":%d,\"greedy_deterministic\":%s,\"argmax_parity\":%s}\n",
               n_layers, n_experts, samples, decode_steps,
               load_ms, med_ms, min_ms, max_ms,
               proj_40l_ms, proj_40l_tok_s, rss_gib,
               ref_tok, greedy_ok ? "true" : "false", argmax_parity ? "true" : "false");

        std::filesystem::remove(tmp_path);
        return (greedy_ok && argmax_parity) ? 0 : 1;
    }
}
