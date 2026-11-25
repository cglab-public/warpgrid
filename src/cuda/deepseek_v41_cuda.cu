#include "deepseek_v41_cuda.hpp"
#include "kernels_gguf.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ink {

namespace {

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(err));
    }
}

struct CudaWeightBuf {
    const uint8_t* ptr = nullptr;
    bool owned = false;
    GgmlType type = GgmlType::F32;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t groups = 1;
    uint32_t group_stride_bytes = 0;
    bool valid() const { return ptr != nullptr; }
};

struct CudaLayerWeights {
    float* attn_norm = nullptr;
    float* attn_sinks = nullptr;
    CudaWeightBuf wq_a;
    float* attn_q_a_norm = nullptr;
    CudaWeightBuf wq_b;
    CudaWeightBuf wkv;
    float* attn_kv_norm = nullptr;
    CudaWeightBuf wo_a;
    CudaWeightBuf wo_b;

    CudaWeightBuf hc_attn_fn;
    float* hc_attn_base = nullptr;
    float* hc_attn_scale = nullptr;
    CudaWeightBuf hc_ffn_fn;
    float* hc_ffn_base = nullptr;
    float* hc_ffn_scale = nullptr;

    CudaWeightBuf comp_wkv;
    CudaWeightBuf comp_wgate;
    float* comp_norm = nullptr;
    CudaWeightBuf idx_k;
    float* idx_k_norm = nullptr;
    CudaWeightBuf idx_q_b;
    CudaWeightBuf idx_proj;

    CudaWeightBuf engram_kv;
    float* engram_gate = nullptr;

    float* ffn_norm = nullptr;
    CudaWeightBuf router;
    float* router_bias = nullptr;
    CudaWeightBuf ffn_gate_exps;
    CudaWeightBuf ffn_up_exps;
    CudaWeightBuf ffn_down_exps;
    CudaWeightBuf ffn_gate_shexp;
    CudaWeightBuf ffn_up_shexp;
    CudaWeightBuf ffn_down_shexp;
};

struct CudaLayerKv {
    float* kv_cache = nullptr;
    float* comp_raw_kv = nullptr;
    float* comp_raw_score = nullptr;
    float* comp_kv = nullptr;
    float* comp_ik = nullptr;
    int32_t* topk_sel = nullptr;
};

}

struct DeepSeekV41Model::Impl {
    cudaStream_t stream = nullptr;
    std::vector<void*> owned_allocs;
    std::vector<void*> registered_shards;
    std::vector<const uint8_t*> shard_dev_ptrs;

    CudaWeightBuf lm_head;
    float* output_norm = nullptr;
    bool engram_in_main_gguf = false;
    std::vector<CudaLayerWeights> layers;
    std::vector<CudaLayerKv> kv_caches;
    std::vector<std::vector<int32_t>> slot_tok_history;

    float* b_streams_a = nullptr;
    float* b_streams_b = nullptr;
    float* b_flat_norm = nullptr;
    float* b_hc_raw = nullptr;
    float* b_hc_pre_a = nullptr;
    float* b_hc_pre_b = nullptr;
    float* b_hc_post = nullptr;
    float* b_hc_comb = nullptr;

    float* b_hidden = nullptr;
    float* b_norm = nullptr;
    float* b_qr = nullptr;
    float* b_qr_norm = nullptr;
    float* b_q = nullptr;
    float* b_kv = nullptr;
    float* b_kv_norm = nullptr;
    float* b_attn_out = nullptr;
    float* b_oa = nullptr;
    float* b_sub_out = nullptr;

    float* b_comp_tmp_kv = nullptr;
    float* b_comp_tmp_score = nullptr;
    float* b_idx_k_tmp = nullptr;
    float* b_idx_q = nullptr;
    float* b_idx_w = nullptr;
    float* b_idx_scores = nullptr;

    std::vector<float*> b_engram_rows;
    float* b_engram_kv = nullptr;

    float* b_router_logits = nullptr;
    int32_t* b_topk_ids = nullptr;
    float* b_topk_w = nullptr;
    float* b_ffn_gate = nullptr;
    float* b_ffn_up = nullptr;
    float* b_ffn_act = nullptr;
    float* b_logits = nullptr;
    int32_t* b_argmax = nullptr;
    float* b_dummy_bias = nullptr;

    std::vector<float> h_hidden;
    std::vector<float> h_engram_tmp;

    ~Impl() {
        if (stream) cudaStreamDestroy(stream);
        for (void* p : owned_allocs) cudaFree(p);
        for (void* p : registered_shards) cudaHostUnregister(p);
    }

    template <typename T>
    T* alloc_dev(size_t count) {
        size_t bytes = std::max<size_t>(16, count * sizeof(T));
        void* p = nullptr;
        cuda_check(cudaMalloc(&p, bytes), "cudaMalloc");
        cuda_check(cudaMemset(p, 0, bytes), "cudaMemset");
        owned_allocs.push_back(p);
        return static_cast<T*>(p);
    }

    void init_shard_mappings(const GgufFile& gf) {
        shard_dev_ptrs.assign(gf.num_shards(), nullptr);
        for (size_t s = 0; s < gf.num_shards(); ++s) {
            const void* ptr = gf.shard_mapped(s);
            size_t sz = gf.shard_size(s);
            if (!ptr || sz == 0) continue;
            cudaError_t reg = cudaHostRegister(const_cast<void*>(ptr), sz, cudaHostRegisterMapped | cudaHostRegisterReadOnly);
            if (reg == cudaSuccess) {
                void* dptr = nullptr;
                if (cudaHostGetDevicePointer(&dptr, const_cast<void*>(ptr), 0) == cudaSuccess) {
                    registered_shards.push_back(const_cast<void*>(ptr));
                    shard_dev_ptrs[s] = static_cast<const uint8_t*>(dptr);
                } else {
                    cudaHostUnregister(const_cast<void*>(ptr));
                }
            }
        }
    }

    float* load_f32_required(const GgufFile& gf, const std::string& name) {
        if (!gf.has_tensor(name)) throw std::runtime_error("Missing required DeepSeek-V4.1 tensor: " + name);
        const auto& ti = gf.tensor(name);
        size_t n = (size_t)ti.dims[0] * (size_t)ti.num_rows();
        std::vector<float> tmp(n);
        dequant_row_fp32(ti.type, gf.tensor_data(name), tmp.data(), n);
        float* d = alloc_dev<float>(n);
        cuda_check(cudaMemcpy(d, tmp.data(), n * sizeof(float), cudaMemcpyHostToDevice), "load_f32_required");
        return d;
    }

    CudaWeightBuf load_weight_required(const GgufFile& gf, const std::string& name, uint32_t force_groups = 0) {
        if (!gf.has_tensor(name)) throw std::runtime_error("Missing required DeepSeek-V4.1 tensor: " + name);
        const auto& ti = gf.tensor(name);
        CudaWeightBuf wb;
        wb.type = ti.type;
        wb.cols = ti.dims.size() >= 1 ? (uint32_t)ti.dims[0] : 0;
        wb.rows = ti.dims.size() >= 2 ? (uint32_t)ti.dims[1] : 1;
        wb.groups = ti.dims.size() >= 3 ? (uint32_t)ti.dims[2] : 1;
        if (force_groups > 1 && wb.groups == 1 && wb.rows % force_groups == 0) {
            wb.groups = force_groups;
            wb.rows /= force_groups;
        }
        if (wb.groups > 1 && wb.cols > 0) {
            wb.group_stride_bytes = (uint32_t)(ggml_row_nbytes(ti.type, wb.cols) * wb.rows);
        }
        if (ti.type == GgmlType::F16) {
            size_t total = (size_t)wb.cols * wb.rows * wb.groups;
            std::vector<float> tmp(total);
            dequant_row_fp32(ti.type, gf.tensor_data(name), tmp.data(), total);
            float* d = alloc_dev<float>(total);
            cuda_check(cudaMemcpy(d, tmp.data(), total * sizeof(float), cudaMemcpyHostToDevice), "load_weight_f16");
            wb.ptr = reinterpret_cast<const uint8_t*>(d);
            wb.owned = true;
            wb.type = GgmlType::F32;
            if (wb.groups > 1) wb.group_stride_bytes = wb.cols * wb.rows * (uint32_t)sizeof(float);
        } else if (ti.shard_index < shard_dev_ptrs.size() && shard_dev_ptrs[ti.shard_index] != nullptr) {
            wb.ptr = shard_dev_ptrs[ti.shard_index] + gf.shard_data_offset(ti.shard_index) + (size_t)ti.offset;
            wb.owned = false;
        } else {
            uint8_t* d = alloc_dev<uint8_t>(ti.nbytes);
            cuda_check(cudaMemcpy(d, gf.tensor_data(name), ti.nbytes, cudaMemcpyHostToDevice), "load_weight_copy");
            wb.ptr = d;
            wb.owned = true;
        }
        return wb;
    }

    void encode_hc_mixes(const float* streams, const CudaWeightBuf& fn, const float* scale, const float* base,
                         float* out_pre, float* out_post, float* out_comb,
                         uint32_t T, uint32_t D, uint32_t hc, uint32_t iters, float rms_eps, float hc_eps) {
        cuda_gguf::launch_rmsnorm_unweighted(streams, b_flat_norm, T, hc * D, rms_eps, stream);
        cuda_gguf::launch_gemv(fn.type, b_flat_norm, fn.ptr, b_hc_raw, T, fn.rows, fn.cols, stream);
        cuda_gguf::launch_hc_mixes(b_hc_raw, scale, base, out_pre, out_post, out_comb, T, hc, iters, hc_eps, stream);
    }
};

DeepSeekV41Model::DeepSeekV41Model() : impl_(std::make_unique<Impl>()) {}
DeepSeekV41Model::~DeepSeekV41Model() = default;

void DeepSeekV41Model::load_tokenizer(Tokenizer& tok) const {
    gguf.load_tokenizer(tok);
}

void DeepSeekV41Model::load(const std::string& model_path, int max_seq, int max_chunk, int n_slots) {
    cuda_check(cudaStreamCreate(&impl_->stream), "cudaStreamCreate");
    gguf.open(model_path, false);
    impl_->init_shard_mappings(gguf);
    cfg = DeepSeekConfig::from_gguf(gguf);

    if (!engram_path.empty()) {
        engram.open(engram_path, 4096);
        if (!engram.has_tensor("per_layer_token_embd.weight")) {
            throw std::runtime_error("Engram GGUF is missing per_layer_token_embd.weight: " + engram_path);
        }
    }
    impl_->engram_in_main_gguf = gguf.has_tensor("per_layer_token_embd.weight");
    if (!cfg.ple_layers.empty() && !engram.is_open() && !impl_->engram_in_main_gguf) {
        throw std::runtime_error("Model declares Engram layers (ple.layers) but per_layer_token_embd.weight is missing; pass --engram <path>");
    }

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t kv_per_slot = cfg.kv_bytes_per_token() * (size_t)max_seq;
    int eff_slots = std::max(1, n_slots);
    if (total_mem > 0 && kv_per_slot > 0) {
        size_t kv_budget = total_mem / 4;
        while (eff_slots > 1 && (size_t)eff_slots * kv_per_slot > kv_budget) --eff_slots;
    }
    max_slots = eff_slots;
    max_seq_len = max_seq;
    max_batch_tokens = std::max(1, max_chunk);
    max_blocks_per_seq = (max_seq_len + 63) / 64;
    num_kv_blocks = max_slots * max_blocks_per_seq;
    num_snapshots = 0;

    if (!gguf.has_tensor("token_embd.weight")) {
        throw std::runtime_error("Missing required DeepSeek-V4.1 tensor: token_embd.weight");
    }
    impl_->output_norm = impl_->load_f32_required(gguf, "output_norm.weight");
    impl_->lm_head = impl_->load_weight_required(gguf, gguf.has_tensor("output.weight") ? "output.weight" : "token_embd.weight");

    impl_->layers.resize((size_t)cfg.num_hidden_layers);
    impl_->kv_caches.resize((size_t)cfg.num_hidden_layers);
    impl_->slot_tok_history.assign((size_t)max_slots, {});

    size_t kv_layer_elems = (size_t)max_slots * (size_t)max_seq_len * (size_t)cfg.head_dim;
    uint32_t topk_cap = (uint32_t)std::max(1, cfg.index_topk);

    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
        std::string p = "blk." + std::to_string(l) + ".";
        auto& lw = impl_->layers[(size_t)l];

        lw.attn_norm = impl_->load_f32_required(gguf, p + "attn_norm.weight");
        lw.attn_sinks = impl_->load_f32_required(gguf, p + "attn_sinks.weight");
        lw.wq_a = impl_->load_weight_required(gguf, p + "attn_q_a.weight");
        lw.attn_q_a_norm = impl_->load_f32_required(gguf, p + "attn_q_a_norm.weight");
        lw.wq_b = impl_->load_weight_required(gguf, p + "attn_q_b.weight");
        lw.wkv = impl_->load_weight_required(gguf, p + "attn_kv.weight");
        lw.attn_kv_norm = impl_->load_f32_required(gguf, p + "attn_kv_a_norm.weight");
        lw.wo_a = impl_->load_weight_required(gguf, p + "attn_output_a.weight", (uint32_t)std::max(1, cfg.o_group_count));
        lw.wo_b = impl_->load_weight_required(gguf, p + "attn_output_b.weight");

        lw.hc_attn_fn = impl_->load_weight_required(gguf, p + "hc_attn_fn.weight");
        lw.hc_attn_base = impl_->load_f32_required(gguf, p + "hc_attn_base.weight");
        lw.hc_attn_scale = impl_->load_f32_required(gguf, p + "hc_attn_scale.weight");
        lw.hc_ffn_fn = impl_->load_weight_required(gguf, p + "hc_ffn_fn.weight");
        lw.hc_ffn_base = impl_->load_f32_required(gguf, p + "hc_ffn_base.weight");
        lw.hc_ffn_scale = impl_->load_f32_required(gguf, p + "hc_ffn_scale.weight");

        int ratio = cfg.compress_ratio_for(l);
        if (ratio > 0 && cfg.kv_src[(size_t)l] == l) {
            lw.comp_wkv = impl_->load_weight_required(gguf, p + "attn_compressor_kv.weight");
            lw.comp_norm = impl_->load_f32_required(gguf, p + "attn_compressor_norm.weight");
            if (ratio > 1) lw.comp_wgate = impl_->load_weight_required(gguf, p + "attn_compressor_gate.weight");
            lw.idx_k = impl_->load_weight_required(gguf, p + "indexer.attn_k.weight");
            lw.idx_k_norm = impl_->load_f32_required(gguf, p + "indexer.k_norm.weight");

            size_t max_comp = (size_t)(max_seq_len / ratio) + 1;
            impl_->kv_caches[(size_t)l].comp_raw_kv = impl_->alloc_dev<float>(kv_layer_elems);
            impl_->kv_caches[(size_t)l].comp_raw_score = impl_->alloc_dev<float>(kv_layer_elems);
            impl_->kv_caches[(size_t)l].comp_kv = impl_->alloc_dev<float>((size_t)max_slots * max_comp * (size_t)cfg.head_dim);
            impl_->kv_caches[(size_t)l].comp_ik = impl_->alloc_dev<float>((size_t)max_slots * max_comp * (size_t)cfg.index_head_dim);
        }
        if (ratio > 0 && cfg.topk_src[(size_t)l] == l) {
            lw.idx_q_b = impl_->load_weight_required(gguf, p + "indexer.attn_q_b.weight");
            lw.idx_proj = impl_->load_weight_required(gguf, p + "indexer.proj.weight");
            impl_->kv_caches[(size_t)l].topk_sel = impl_->alloc_dev<int32_t>((size_t)max_batch_tokens * topk_cap);
        }
        if (cfg.engram_index(l) >= 0) {
            lw.engram_kv = impl_->load_weight_required(gguf, p + "engram_kv.weight");
            lw.engram_gate = impl_->load_f32_required(gguf, p + "engram_gate.weight");
        }

        lw.ffn_norm = impl_->load_f32_required(gguf, p + "ffn_norm.weight");
        lw.router = impl_->load_weight_required(gguf, p + "ffn_gate_inp.weight");
        lw.router_bias = impl_->load_f32_required(gguf, p + "exp_probs_b.bias");
        lw.ffn_gate_exps = impl_->load_weight_required(gguf, p + "ffn_gate_exps.weight");
        lw.ffn_up_exps = impl_->load_weight_required(gguf, p + "ffn_up_exps.weight");
        lw.ffn_down_exps = impl_->load_weight_required(gguf, p + "ffn_down_exps.weight");
        lw.ffn_gate_shexp = impl_->load_weight_required(gguf, p + "ffn_gate_shexp.weight");
        lw.ffn_up_shexp = impl_->load_weight_required(gguf, p + "ffn_up_shexp.weight");
        lw.ffn_down_shexp = impl_->load_weight_required(gguf, p + "ffn_down_shexp.weight");

        impl_->kv_caches[(size_t)l].kv_cache = impl_->alloc_dev<float>(kv_layer_elems);
    }

    size_t Tmax = (size_t)max_batch_tokens;
    size_t D = (size_t)cfg.hidden_size;
    size_t hc = (size_t)std::max(1, cfg.hc_mult);
    size_t Qdim = (size_t)cfg.num_attention_heads * (size_t)cfg.head_dim;
    size_t Inter = (size_t)std::max(cfg.ffn_intermediate_size, cfg.expert_intermediate_size);
    for (const auto& lw : impl_->layers) {
        Inter = std::max<size_t>(Inter, lw.ffn_gate_shexp.rows);
        Inter = std::max<size_t>(Inter, lw.ffn_up_shexp.rows);
        Inter = std::max<size_t>(Inter, lw.ffn_gate_exps.rows);
        Inter = std::max<size_t>(Inter, lw.ffn_up_exps.rows);
    }

    impl_->b_streams_a = impl_->alloc_dev<float>(Tmax * hc * D);
    impl_->b_streams_b = impl_->alloc_dev<float>(Tmax * hc * D);
    impl_->b_flat_norm = impl_->alloc_dev<float>(Tmax * hc * D);
    impl_->b_hc_raw = impl_->alloc_dev<float>(Tmax * (2 + hc) * hc);
    impl_->b_hc_pre_a = impl_->alloc_dev<float>(Tmax * hc);
    impl_->b_hc_pre_b = impl_->alloc_dev<float>(Tmax * hc);
    impl_->b_hc_post = impl_->alloc_dev<float>(Tmax * hc);
    impl_->b_hc_comb = impl_->alloc_dev<float>(Tmax * hc * hc);

    impl_->b_hidden = impl_->alloc_dev<float>(Tmax * D);
    impl_->b_norm = impl_->alloc_dev<float>(Tmax * D);
    impl_->b_qr = impl_->alloc_dev<float>(Tmax * (size_t)std::max(1, cfg.q_lora_rank));
    impl_->b_qr_norm = impl_->alloc_dev<float>(Tmax * (size_t)std::max(1, cfg.q_lora_rank));
    impl_->b_q = impl_->alloc_dev<float>(Tmax * Qdim);
    impl_->b_kv = impl_->alloc_dev<float>(Tmax * (size_t)cfg.head_dim);
    impl_->b_kv_norm = impl_->alloc_dev<float>(Tmax * (size_t)cfg.head_dim);
    impl_->b_attn_out = impl_->alloc_dev<float>(Tmax * Qdim);
    impl_->b_oa = impl_->alloc_dev<float>(Tmax * (size_t)std::max(1, cfg.o_group_count * cfg.o_lora_rank));
    impl_->b_sub_out = impl_->alloc_dev<float>(Tmax * D);

    impl_->b_comp_tmp_kv = impl_->alloc_dev<float>(Tmax * (size_t)cfg.head_dim);
    impl_->b_comp_tmp_score = impl_->alloc_dev<float>(Tmax * (size_t)cfg.head_dim);
    impl_->b_idx_k_tmp = impl_->alloc_dev<float>(Tmax * (size_t)std::max(1, cfg.index_head_dim));
    impl_->b_idx_q = impl_->alloc_dev<float>(Tmax * (size_t)std::max(1, cfg.index_head_count * cfg.index_head_dim));
    impl_->b_idx_w = impl_->alloc_dev<float>(Tmax * (size_t)std::max(1, cfg.index_head_count));
    impl_->b_idx_scores = impl_->alloc_dev<float>(Tmax * (size_t)(max_seq_len + 1));

    size_t ple_dim = (size_t)std::max(1, cfg.ple_n_heads * cfg.ple_head_dim);
    size_t n_ple = std::max<size_t>(1, cfg.ple_layers.size());
    impl_->b_engram_rows.resize(n_ple);
    for (size_t i = 0; i < n_ple; ++i) impl_->b_engram_rows[i] = impl_->alloc_dev<float>(Tmax * ple_dim);
    impl_->b_engram_kv = impl_->alloc_dev<float>(Tmax * (hc + 1) * D);

    size_t top_k_sz = (size_t)std::max(1, cfg.num_experts_per_tok);
    impl_->b_router_logits = impl_->alloc_dev<float>(Tmax * (size_t)std::max(1, cfg.n_routed_experts));
    impl_->b_topk_ids = impl_->alloc_dev<int32_t>(Tmax * top_k_sz);
    impl_->b_topk_w = impl_->alloc_dev<float>(Tmax * top_k_sz);
    impl_->b_ffn_gate = impl_->alloc_dev<float>(Tmax * top_k_sz * Inter);
    impl_->b_ffn_up = impl_->alloc_dev<float>(Tmax * top_k_sz * Inter);
    impl_->b_ffn_act = impl_->alloc_dev<float>(Tmax * top_k_sz * Inter);
    impl_->b_logits = impl_->alloc_dev<float>((size_t)cfg.vocab_size);
    impl_->b_argmax = impl_->alloc_dev<int32_t>(1);
    impl_->b_dummy_bias = impl_->alloc_dev<float>((size_t)std::max(1, cfg.n_routed_experts));

    impl_->h_hidden.resize(Tmax * D);
    impl_->h_engram_tmp.resize(Tmax * ple_dim);

    fprintf(stderr, "[warpgrid] Loaded DeepSeek-V4.1 CUDA (%s): %d layers, D=%d, H=%d, hc=%d, E=%d (top-%d), slots=%d, max_seq=%d, engram=%s\n",
            cfg.arch.c_str(), cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads,
            cfg.hc_mult, cfg.n_routed_experts, cfg.num_experts_per_tok, max_slots, max_seq_len,
            (engram.is_open() || impl_->engram_in_main_gguf) ? "ON" : "OFF");
}

void DeepSeekV41Model::reset_cache(int slot_id) {
    if (slot_id < 0 || slot_id >= max_slots) return;
    if ((size_t)slot_id < impl_->slot_tok_history.size()) impl_->slot_tok_history[(size_t)slot_id].clear();
    size_t slot_elems = (size_t)max_seq_len * (size_t)cfg.head_dim;
    size_t slot_bytes = slot_elems * sizeof(float);
    for (auto& kv : impl_->kv_caches) {
        cudaMemsetAsync(kv.kv_cache + (size_t)slot_id * slot_elems, 0, slot_bytes, impl_->stream);
        if (kv.comp_raw_kv) {
            cudaMemsetAsync(kv.comp_raw_kv + (size_t)slot_id * slot_elems, 0, slot_bytes, impl_->stream);
            cudaMemsetAsync(kv.comp_raw_score + (size_t)slot_id * slot_elems, 0, slot_bytes, impl_->stream);
        }
    }
    cudaStreamSynchronize(impl_->stream);
}

void DeepSeekV41Model::forward_logits(const int32_t* h_tokens, int T, int start_pos, std::vector<float>& h_logits, int slot_id) {
    forward_and_greedy(h_tokens, T, start_pos, slot_id);
    h_logits.resize((size_t)cfg.vocab_size);
    cuda_check(cudaMemcpy(h_logits.data(), impl_->b_logits, (size_t)cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost), "copy_logits");
}

int32_t DeepSeekV41Model::forward_and_greedy(const int32_t* h_tokens, int T, int start_pos, int slot_id) {
    if (T <= 0 || T > max_batch_tokens) throw std::runtime_error("Invalid token batch size T");
    if (slot_id < 0 || slot_id >= max_slots) throw std::runtime_error("Invalid slot_id");
    if (start_pos < 0 || start_pos + T > max_seq_len) throw std::runtime_error("Sequence length exceeds max_seq_len");

    auto& hist = impl_->slot_tok_history[(size_t)slot_id];
    if ((int)hist.size() < start_pos + T) hist.resize((size_t)(start_pos + T), 0);
    for (int t = 0; t < T; ++t) hist[(size_t)(start_pos + t)] = h_tokens[t];

    size_t D = (size_t)cfg.hidden_size;
    uint32_t hc = (uint32_t)std::max(1, cfg.hc_mult);
    const auto& emb_ti = gguf.tensor("token_embd.weight");
    size_t row_stride = ggml_row_nbytes(emb_ti.type, emb_ti.dims[0]);
    const uint8_t* base = gguf.tensor_data("token_embd.weight");
    uint64_t nrows = emb_ti.num_rows();
    for (int t = 0; t < T; ++t) {
        uint64_t tid = (uint64_t)std::clamp(h_tokens[t], 0, (int32_t)(nrows > 0 ? nrows - 1 : 0));
        dequant_row_fp32(emb_ti.type, base + tid * row_stride, impl_->h_hidden.data() + (size_t)t * D, D);
    }
    cudaMemcpyAsync(impl_->b_hidden, impl_->h_hidden.data(), (size_t)T * D * sizeof(float), cudaMemcpyHostToDevice, impl_->stream);

    if ((engram.is_open() || impl_->engram_in_main_gguf) && cfg.ple_n_heads > 0 && cfg.ple_head_dim > 0) {
        std::vector<int32_t> row_ids((size_t)cfg.ple_n_heads, 0);
        size_t ple_dim = (size_t)cfg.ple_n_heads * (size_t)cfg.ple_head_dim;
        for (size_t e_idx = 0; e_idx < cfg.ple_layers.size(); ++e_idx) {
            for (int t = 0; t < T; ++t) {
                cfg.engram_hash_token((int)e_idx, hist, start_pos + t, row_ids.data());
                for (int h = 0; h < cfg.ple_n_heads; ++h) {
                    float* dst = impl_->h_engram_tmp.data() + ((size_t)t * (size_t)cfg.ple_n_heads + (size_t)h) * (size_t)cfg.ple_head_dim;
                    if (engram.is_open()) {
                        engram.lookup_row("per_layer_token_embd.weight", (uint64_t)row_ids[(size_t)h], dst, (size_t)cfg.ple_head_dim);
                    } else {
                        const auto& pti = gguf.tensor("per_layer_token_embd.weight");
                        size_t pstride = ggml_row_nbytes(pti.type, pti.dims[0]);
                        const uint8_t* pbase = gguf.tensor_data("per_layer_token_embd.weight");
                        dequant_row_fp32(pti.type, pbase + (size_t)row_ids[(size_t)h] * pstride, dst, (size_t)cfg.ple_head_dim);
                    }
                }
            }
            cudaMemcpyAsync(impl_->b_engram_rows[e_idx], impl_->h_engram_tmp.data(),
                            (size_t)T * ple_dim * sizeof(float), cudaMemcpyHostToDevice, impl_->stream);
        }
    }

    cuda_gguf::launch_mhc_expand(impl_->b_hidden, impl_->b_streams_a, (uint32_t)T, (uint32_t)D, hc, impl_->stream);

    float* pre_attn = impl_->b_hc_pre_a;
    float* pre_ffn = impl_->b_hc_pre_b;
    size_t slot_kv_elems = (size_t)max_seq_len * (size_t)cfg.head_dim;
    size_t slot_kv_off = (size_t)slot_id * slot_kv_elems;
    uint32_t n_nope = (uint32_t)std::max(0, cfg.head_dim - cfg.n_rot);
    uint32_t n_rot = (uint32_t)std::max(0, cfg.n_rot);
    uint32_t topk_cap = (uint32_t)std::max(1, cfg.index_topk);

    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
        const auto& lw = impl_->layers[(size_t)l];
        auto& kv = impl_->kv_caches[(size_t)l];

        int e_idx = cfg.engram_index(l);
        if (e_idx >= 0 && (engram.is_open() || impl_->engram_in_main_gguf) && cfg.ple_n_heads > 0 && cfg.ple_head_dim > 0) {
            cuda_gguf::launch_gemv(lw.engram_kv.type, impl_->b_engram_rows[(size_t)e_idx], lw.engram_kv.ptr,
                                   impl_->b_engram_kv, (uint32_t)T, lw.engram_kv.rows, lw.engram_kv.cols, impl_->stream);
            cuda_gguf::launch_engram_apply(impl_->b_streams_a, impl_->b_engram_kv, lw.engram_gate,
                                           (uint32_t)T, (uint32_t)D, hc, cfg.rms_norm_eps, impl_->stream);
        }

        impl_->encode_hc_mixes(impl_->b_streams_a, lw.hc_attn_fn, lw.hc_attn_scale, lw.hc_attn_base,
                               pre_attn, impl_->b_hc_post, impl_->b_hc_comb,
                               (uint32_t)T, (uint32_t)D, hc, (uint32_t)cfg.hc_sinkhorn_iters, cfg.rms_norm_eps, cfg.hc_eps);

        if (l > 0) {
            cuda_gguf::launch_hc_pre(impl_->b_streams_a, pre_ffn, impl_->b_hidden, (uint32_t)T, (uint32_t)D, hc, impl_->stream);
        } else {
            for (int t = 0; t < T; ++t) {
                cudaMemcpyAsync(impl_->b_hidden + (size_t)t * D,
                                impl_->b_streams_a + (size_t)t * hc * D,
                                D * sizeof(float), cudaMemcpyDeviceToDevice, impl_->stream);
            }
        }

        cuda_gguf::launch_rmsnorm(impl_->b_hidden, lw.attn_norm, impl_->b_norm, (uint32_t)T, (uint32_t)D, cfg.rms_norm_eps, impl_->stream);
        cuda_gguf::launch_gemv(lw.wq_a.type, impl_->b_norm, lw.wq_a.ptr, impl_->b_qr, (uint32_t)T, lw.wq_a.rows, lw.wq_a.cols, impl_->stream);
        cuda_gguf::launch_rmsnorm(impl_->b_qr, lw.attn_q_a_norm, impl_->b_qr_norm, (uint32_t)T, (uint32_t)cfg.q_lora_rank, cfg.rms_norm_eps, impl_->stream);
        cuda_gguf::launch_gemv(lw.wq_b.type, impl_->b_qr_norm, lw.wq_b.ptr, impl_->b_q, (uint32_t)T, lw.wq_b.rows, lw.wq_b.cols, impl_->stream);
        cuda_gguf::launch_gemv(lw.wkv.type, impl_->b_norm, lw.wkv.ptr, impl_->b_kv, (uint32_t)T, lw.wkv.rows, lw.wkv.cols, impl_->stream);
        cuda_gguf::launch_rmsnorm(impl_->b_kv, lw.attn_kv_norm, impl_->b_kv_norm, (uint32_t)T, (uint32_t)cfg.head_dim, cfg.rms_norm_eps, impl_->stream);

        int ratio = cfg.compress_ratio_for(l);
        float freq_base_l = (ratio != 0) ? cfg.compress_rope_freq_base : cfg.rope_freq_base;

        if (n_rot >= 2u) {
            cuda_gguf::launch_rope_partial(impl_->b_q, (uint32_t)T, (uint32_t)start_pos, 1u,
                                           (uint32_t)cfg.num_attention_heads, (uint32_t)cfg.head_dim, n_nope, n_rot, 0u, freq_base_l, impl_->stream);
            cuda_gguf::launch_rope_partial(impl_->b_kv_norm, (uint32_t)T, (uint32_t)start_pos, 1u,
                                           1u, (uint32_t)cfg.head_dim, n_nope, n_rot, 0u, freq_base_l, impl_->stream);
        }

        size_t kv_dst_off = slot_kv_off + (size_t)start_pos * (size_t)cfg.head_dim;
        size_t kv_copy_bytes = (size_t)T * (size_t)cfg.head_dim * sizeof(float);
        cudaMemcpyAsync(kv.kv_cache + kv_dst_off, impl_->b_kv_norm, kv_copy_bytes, cudaMemcpyDeviceToDevice, impl_->stream);

        if (ratio > 0 && cfg.kv_src[(size_t)l] == l) {
            cuda_gguf::launch_gemv(lw.comp_wkv.type, impl_->b_norm, lw.comp_wkv.ptr, impl_->b_comp_tmp_kv, (uint32_t)T, lw.comp_wkv.rows, lw.comp_wkv.cols, impl_->stream);
            if (ratio > 1) {
                cuda_gguf::launch_gemv(lw.comp_wgate.type, impl_->b_norm, lw.comp_wgate.ptr, impl_->b_comp_tmp_score, (uint32_t)T, lw.comp_wgate.rows, lw.comp_wgate.cols, impl_->stream);
            } else {
                cuda_gguf::launch_zero(impl_->b_comp_tmp_score, (uint32_t)((size_t)T * (size_t)cfg.head_dim), impl_->stream);
            }
            cudaMemcpyAsync(kv.comp_raw_kv + kv_dst_off, impl_->b_comp_tmp_kv, kv_copy_bytes, cudaMemcpyDeviceToDevice, impl_->stream);
            cudaMemcpyAsync(kv.comp_raw_score + kv_dst_off, impl_->b_comp_tmp_score, kv_copy_bytes, cudaMemcpyDeviceToDevice, impl_->stream);
        }

        uint32_t total_comp_blocks = (ratio > 0) ? (uint32_t)((start_pos + T) / ratio) : 0u;
        if (ratio > 0 && cfg.kv_src[(size_t)l] == l && total_comp_blocks > 0u) {
            uint32_t first_blk = (uint32_t)(start_pos / ratio);
            uint32_t n_new = total_comp_blocks - first_blk;
            if (n_new > 0u) {
                size_t max_comp = (size_t)(max_seq_len / ratio) + 1;
                size_t raw_off = slot_kv_off + (size_t)first_blk * (size_t)ratio * (size_t)cfg.head_dim;
                size_t comp_off = ((size_t)slot_id * max_comp + (size_t)first_blk) * (size_t)cfg.head_dim;
                size_t ik_off = ((size_t)slot_id * max_comp + (size_t)first_blk) * (size_t)cfg.index_head_dim;

                cuda_gguf::launch_comp_group_reduce(kv.comp_raw_kv + raw_off, kv.comp_raw_score + raw_off, lw.comp_norm,
                                                    kv.comp_kv + comp_off, n_new, (uint32_t)ratio, (uint32_t)cfg.head_dim, cfg.rms_norm_eps, impl_->stream);
                cuda_gguf::launch_gemv(lw.idx_k.type, kv.comp_kv + comp_off, lw.idx_k.ptr, impl_->b_idx_k_tmp,
                                       n_new, lw.idx_k.rows, lw.idx_k.cols, impl_->stream);
                cuda_gguf::launch_rmsnorm(impl_->b_idx_k_tmp, lw.idx_k_norm, kv.comp_ik + ik_off,
                                          n_new, (uint32_t)cfg.index_head_dim, cfg.rms_norm_eps, impl_->stream);
                if (n_rot >= 2u) {
                    uint32_t ik_nope = (uint32_t)std::max(0, cfg.index_head_dim - cfg.n_rot);
                    cuda_gguf::launch_rope_partial(kv.comp_ik + ik_off, n_new, first_blk * (uint32_t)ratio, (uint32_t)ratio,
                                                   1u, (uint32_t)cfg.index_head_dim, ik_nope, n_rot, 0u, freq_base_l, impl_->stream);
                    cuda_gguf::launch_rope_partial(kv.comp_kv + comp_off, n_new, first_blk * (uint32_t)ratio, (uint32_t)ratio,
                                                   1u, (uint32_t)cfg.head_dim, n_nope, n_rot, 0u, freq_base_l, impl_->stream);
                }
            }
        }

        if (ratio > 0 && cfg.topk_src[(size_t)l] == l && total_comp_blocks > 0u) {
            int src_kv_l = cfg.kv_src[(size_t)l];
            size_t max_comp = (size_t)(max_seq_len / ratio) + 1;
            size_t slot_ik_off = (size_t)slot_id * max_comp * (size_t)cfg.index_head_dim;

            cuda_gguf::launch_gemv(lw.idx_q_b.type, impl_->b_qr_norm, lw.idx_q_b.ptr, impl_->b_idx_q,
                                   (uint32_t)T, lw.idx_q_b.rows, lw.idx_q_b.cols, impl_->stream);
            if (n_rot >= 2u) {
                uint32_t iq_nope = (uint32_t)std::max(0, cfg.index_head_dim - cfg.n_rot);
                cuda_gguf::launch_rope_partial(impl_->b_idx_q, (uint32_t)T, (uint32_t)start_pos, 1u,
                                               (uint32_t)cfg.index_head_count, (uint32_t)cfg.index_head_dim, iq_nope, n_rot, 0u, freq_base_l, impl_->stream);
            }
            cuda_gguf::launch_gemv(lw.idx_proj.type, impl_->b_norm, lw.idx_proj.ptr, impl_->b_idx_w,
                                   (uint32_t)T, lw.idx_proj.rows, lw.idx_proj.cols, impl_->stream);
            cuda_gguf::launch_indexer_score_topk(
                impl_->b_idx_q, impl_->kv_caches[(size_t)src_kv_l].comp_ik + slot_ik_off, impl_->b_idx_w,
                impl_->b_idx_scores, kv.topk_sel, (uint32_t)T, (uint32_t)start_pos, total_comp_blocks,
                (uint32_t)(max_seq_len + 1), (uint32_t)cfg.index_head_count, (uint32_t)cfg.index_head_dim,
                (uint32_t)ratio, (uint32_t)cfg.sliding_window, topk_cap,
                1.0f / std::sqrt((float)(cfg.index_head_dim * cfg.index_head_count)), impl_->stream);
        }

        if (ratio > 0 && total_comp_blocks > 0u && cfg.kv_src[(size_t)l] >= 0 && cfg.topk_src[(size_t)l] >= 0) {
            int src_kv_l = cfg.kv_src[(size_t)l];
            int src_tk_l = cfg.topk_src[(size_t)l];
            size_t max_comp = (size_t)(max_seq_len / ratio) + 1;
            size_t slot_comp_off = (size_t)slot_id * max_comp * (size_t)cfg.head_dim;
            cuda_gguf::launch_mqa_sink_comp_attn(
                impl_->b_q, kv.kv_cache + slot_kv_off, impl_->kv_caches[(size_t)src_kv_l].comp_kv + slot_comp_off,
                impl_->kv_caches[(size_t)src_tk_l].topk_sel, lw.attn_sinks, impl_->b_attn_out,
                (uint32_t)T, (uint32_t)start_pos, (uint32_t)cfg.num_attention_heads, (uint32_t)cfg.head_dim,
                (uint32_t)cfg.sliding_window, topk_cap, 1.0f / std::sqrt((float)cfg.head_dim), impl_->stream);
        } else {
            cuda_gguf::launch_mqa_sink_attn(
                impl_->b_q, kv.kv_cache + slot_kv_off, lw.attn_sinks, impl_->b_attn_out,
                (uint32_t)T, (uint32_t)start_pos, (uint32_t)cfg.num_attention_heads, (uint32_t)cfg.head_dim,
                (uint32_t)cfg.sliding_window, 1.0f / std::sqrt((float)cfg.head_dim), impl_->stream);
        }

        if (n_rot >= 2u) {
            cuda_gguf::launch_rope_partial(impl_->b_attn_out, (uint32_t)T, (uint32_t)start_pos, 1u,
                                           (uint32_t)cfg.num_attention_heads, (uint32_t)cfg.head_dim, n_nope, n_rot, 1u, freq_base_l, impl_->stream);
        }

        cuda_gguf::launch_grouped_gemv(lw.wo_a.type, impl_->b_attn_out, lw.wo_a.ptr, impl_->b_oa,
                                       (uint32_t)T, lw.wo_a.groups, lw.wo_a.rows, lw.wo_a.cols, lw.wo_a.group_stride_bytes, impl_->stream);
        cuda_gguf::launch_gemv(lw.wo_b.type, impl_->b_oa, lw.wo_b.ptr, impl_->b_sub_out, (uint32_t)T, lw.wo_b.rows, lw.wo_b.cols, impl_->stream);

        cuda_gguf::launch_hc_post(impl_->b_sub_out, impl_->b_streams_a, impl_->b_hc_post, impl_->b_hc_comb,
                                  impl_->b_streams_b, (uint32_t)T, (uint32_t)D, hc, impl_->stream);
        std::swap(impl_->b_streams_a, impl_->b_streams_b);

        impl_->encode_hc_mixes(impl_->b_streams_a, lw.hc_ffn_fn, lw.hc_ffn_scale, lw.hc_ffn_base,
                               pre_ffn, impl_->b_hc_post, impl_->b_hc_comb,
                               (uint32_t)T, (uint32_t)D, hc, (uint32_t)cfg.hc_sinkhorn_iters, cfg.rms_norm_eps, cfg.hc_eps);
        cuda_gguf::launch_hc_pre(impl_->b_streams_a, pre_attn, impl_->b_hidden, (uint32_t)T, (uint32_t)D, hc, impl_->stream);

        cuda_gguf::launch_rmsnorm(impl_->b_hidden, lw.ffn_norm, impl_->b_norm, (uint32_t)T, (uint32_t)D, cfg.rms_norm_eps, impl_->stream);

        cuda_gguf::launch_gemv(lw.ffn_gate_shexp.type, impl_->b_norm, lw.ffn_gate_shexp.ptr, impl_->b_ffn_gate,
                               (uint32_t)T, lw.ffn_gate_shexp.rows, lw.ffn_gate_shexp.cols, impl_->stream);
        cuda_gguf::launch_gemv(lw.ffn_up_shexp.type, impl_->b_norm, lw.ffn_up_shexp.ptr, impl_->b_ffn_up,
                               (uint32_t)T, lw.ffn_up_shexp.rows, lw.ffn_up_shexp.cols, impl_->stream);
        cuda_gguf::launch_swiglu(impl_->b_ffn_gate, impl_->b_ffn_up, impl_->b_ffn_act,
                                 (uint32_t)((size_t)T * lw.ffn_gate_shexp.rows), cfg.swiglu_clamp_shexp, impl_->stream);
        cuda_gguf::launch_gemv(lw.ffn_down_shexp.type, impl_->b_ffn_act, lw.ffn_down_shexp.ptr, impl_->b_sub_out,
                               (uint32_t)T, lw.ffn_down_shexp.rows, lw.ffn_down_shexp.cols, impl_->stream);

        cuda_gguf::launch_gemv(lw.router.type, impl_->b_norm, lw.router.ptr, impl_->b_router_logits,
                               (uint32_t)T, lw.router.rows, lw.router.cols, impl_->stream);
        cuda_gguf::launch_router_sqrtsoftplus_topk(
            impl_->b_router_logits, lw.router_bias ? lw.router_bias : impl_->b_dummy_bias,
            impl_->b_topk_ids, impl_->b_topk_w, (uint32_t)T, (uint32_t)cfg.n_routed_experts,
            (uint32_t)cfg.num_experts_per_tok, lw.router_bias ? 1u : 0u, cfg.route_scale, impl_->stream);

        uint32_t top_k_u = (uint32_t)cfg.num_experts_per_tok;
        if (top_k_u > 0u && lw.ffn_gate_exps.groups > 0u) {
            if (lw.ffn_gate_exps.type == lw.ffn_up_exps.type &&
                lw.ffn_gate_exps.group_stride_bytes == lw.ffn_up_exps.group_stride_bytes) {
                cuda_gguf::launch_moe_gate_up_swiglu(
                    lw.ffn_gate_exps.type, impl_->b_norm, lw.ffn_gate_exps.ptr, lw.ffn_up_exps.ptr,
                    impl_->b_topk_ids, impl_->b_ffn_act, (uint32_t)T, lw.ffn_gate_exps.rows,
                    lw.ffn_gate_exps.cols, top_k_u, lw.ffn_gate_exps.group_stride_bytes, cfg.swiglu_clamp_exp, impl_->stream);
            } else {
                cuda_gguf::launch_moe_indexed(lw.ffn_gate_exps.type, impl_->b_norm, lw.ffn_gate_exps.ptr, impl_->b_topk_ids,
                                              impl_->b_ffn_gate, (uint32_t)T, lw.ffn_gate_exps.rows, lw.ffn_gate_exps.cols,
                                              top_k_u, lw.ffn_gate_exps.group_stride_bytes, impl_->stream);
                cuda_gguf::launch_moe_indexed(lw.ffn_up_exps.type, impl_->b_norm, lw.ffn_up_exps.ptr, impl_->b_topk_ids,
                                              impl_->b_ffn_up, (uint32_t)T, lw.ffn_up_exps.rows, lw.ffn_up_exps.cols,
                                              top_k_u, lw.ffn_up_exps.group_stride_bytes, impl_->stream);
                cuda_gguf::launch_swiglu(impl_->b_ffn_gate, impl_->b_ffn_up, impl_->b_ffn_act,
                                         (uint32_t)((size_t)T * top_k_u * lw.ffn_gate_exps.rows), cfg.swiglu_clamp_exp, impl_->stream);
            }
            cuda_gguf::launch_moe_down_accum(lw.ffn_down_exps.type, impl_->b_ffn_act, lw.ffn_down_exps.ptr,
                                             impl_->b_topk_ids, impl_->b_topk_w, impl_->b_sub_out,
                                             (uint32_t)T, lw.ffn_down_exps.rows, lw.ffn_down_exps.cols,
                                             top_k_u, lw.ffn_down_exps.group_stride_bytes, impl_->stream);
        }

        cuda_gguf::launch_hc_post(impl_->b_sub_out, impl_->b_streams_a, impl_->b_hc_post, impl_->b_hc_comb,
                                  impl_->b_streams_b, (uint32_t)T, (uint32_t)D, hc, impl_->stream);
        std::swap(impl_->b_streams_a, impl_->b_streams_b);
    }

    if (cfg.num_hidden_layers > 0) {
        cuda_gguf::launch_hc_pre(impl_->b_streams_a, pre_ffn, impl_->b_hidden, (uint32_t)T, (uint32_t)D, hc, impl_->stream);
    }
    cuda_gguf::launch_rmsnorm(impl_->b_hidden, impl_->output_norm, impl_->b_norm, (uint32_t)T, (uint32_t)D, cfg.rms_norm_eps, impl_->stream);
    cuda_gguf::launch_gemv(impl_->lm_head.type, impl_->b_norm + (size_t)(T - 1) * D, impl_->lm_head.ptr, impl_->b_logits,
                           1u, impl_->lm_head.rows, impl_->lm_head.cols, impl_->stream);
    cuda_gguf::launch_argmax(impl_->b_logits, impl_->b_argmax, (uint32_t)cfg.vocab_size, impl_->stream);

    int32_t best = 0;
    cuda_check(cudaMemcpyAsync(&best, impl_->b_argmax, sizeof(int32_t), cudaMemcpyDeviceToHost, impl_->stream), "copy_argmax");
    cuda_check(cudaStreamSynchronize(impl_->stream), "forward_sync");
    return best;
}

void DeepSeekV41Model::forward_batch_decode_greedy(
    const int32_t* h_tokens, const int32_t* h_slots, const int32_t* h_positions,
    int B, int32_t* h_out_tokens) {
    for (int b = 0; b < B; ++b) {
        h_out_tokens[b] = forward_and_greedy(&h_tokens[b], 1, h_positions[b], h_slots[b]);
    }
}

void DeepSeekV41Model::enable_imatrix_capture(bool enable) {
    if (enable) throw std::runtime_error("--dump-imatrix is not supported for DeepSeek GGUF models");
}

void DeepSeekV41Model::save_imatrix(const std::string& path) const {
    (void)path;
    throw std::runtime_error("--dump-imatrix is not supported for DeepSeek GGUF models");
}

}
