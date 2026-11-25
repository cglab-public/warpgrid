#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "deepseek_v41_metal.hpp"
#ifdef WARPGRID_METAL_PRECOMPILED
#include "kernels_gguf_metallib_embed.hpp"
#else
#include "kernels_gguf_source_embed.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ink {

namespace {

struct GgufGemvParams {
    uint32_t M;
    uint32_t N;
    uint32_t K;
};

struct DsRmsParams {
    uint32_t T;
    uint32_t D;
    float eps;
};

struct DsHcMixParams {
    uint32_t T;
    uint32_t hc_mult;
    uint32_t sinkhorn_iters;
    float eps;
};

struct DsHcApplyParams {
    uint32_t T;
    uint32_t D;
    uint32_t hc_mult;
};

struct DsEngramApplyParams {
    uint32_t T;
    uint32_t D;
    uint32_t hc_mult;
    float eps;
};

struct DsPartialRopeParams {
    uint32_t T;
    uint32_t start_pos;
    uint32_t n_heads;
    uint32_t head_dim;
    uint32_t n_nope;
    uint32_t n_rot;
    uint32_t inverse;
    float freq_base;
};

struct DsMqaSinkAttnParams {
    uint32_t T;
    uint32_t start_pos;
    uint32_t n_heads;
    uint32_t head_dim;
    uint32_t sliding_window;
    uint32_t n_comp_sel;
    float scale;
};

struct DsCompGroupParams {
    uint32_t n_blocks;
    uint32_t ratio;
    uint32_t head_dim;
    float eps;
};

struct DsIndexerParams {
    uint32_t T;
    uint32_t start_pos;
    uint32_t n_comp;
    uint32_t n_idx_head;
    uint32_t idx_dim;
    uint32_t ratio;
    uint32_t sliding_window;
    uint32_t top_k;
    float iw_scale;
};

struct DsRouterParams {
    uint32_t T;
    uint32_t E;
    uint32_t top_k;
    uint32_t has_bias;
    float route_scale;
};

struct DsStreamCollapseParams {
    uint32_t T;
    uint32_t D;
    uint32_t hc_mult;
};

struct WeightBuf {
    id<MTLBuffer> buf = nil;
    GgmlType type = GgmlType::F32;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t groups = 1;
    uint32_t group_stride_bytes = 0;
    bool valid() const { return buf != nil; }
};

struct Ds41LayerWeights {
    id<MTLBuffer> attn_norm = nil;
    id<MTLBuffer> attn_sinks = nil;
    WeightBuf wq_a;
    id<MTLBuffer> attn_q_a_norm = nil;
    WeightBuf wq_b;
    WeightBuf wkv;
    id<MTLBuffer> attn_kv_norm = nil;
    WeightBuf wo_a;
    WeightBuf wo_b;

    WeightBuf hc_attn_fn;
    id<MTLBuffer> hc_attn_base = nil;
    id<MTLBuffer> hc_attn_scale = nil;
    WeightBuf hc_ffn_fn;
    id<MTLBuffer> hc_ffn_base = nil;
    id<MTLBuffer> hc_ffn_scale = nil;

    WeightBuf comp_wkv;
    WeightBuf comp_wgate;
    id<MTLBuffer> comp_norm = nil;
    WeightBuf idx_k;
    id<MTLBuffer> idx_k_norm = nil;
    WeightBuf idx_q_b;
    WeightBuf idx_proj;

    WeightBuf engram_kv;
    id<MTLBuffer> engram_gate = nil;

    id<MTLBuffer> ffn_norm = nil;
    WeightBuf router;
    id<MTLBuffer> router_bias = nil;
    WeightBuf ffn_gate_exps;
    WeightBuf ffn_up_exps;
    WeightBuf ffn_down_exps;
    WeightBuf ffn_gate_shexp;
    WeightBuf ffn_up_shexp;
    WeightBuf ffn_down_shexp;
};

struct Ds41LayerKv {
    id<MTLBuffer> kv_cache = nil;
    id<MTLBuffer> comp_raw_kv = nil;
    id<MTLBuffer> comp_raw_score = nil;
    id<MTLBuffer> comp_kv = nil;
    id<MTLBuffer> comp_ik = nil;
    id<MTLBuffer> topk_sel = nil;
};

}

struct DeepSeekV41Model::Impl {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> lib = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> psos;

    WeightBuf lm_head;
    id<MTLBuffer> output_norm = nil;
    bool engram_in_main_gguf = false;
    std::vector<Ds41LayerWeights> layers;
    std::vector<Ds41LayerKv> kv_caches;
    std::vector<std::vector<int32_t>> slot_tok_history;

    id<MTLBuffer> b_streams_a = nil;
    id<MTLBuffer> b_streams_b = nil;
    id<MTLBuffer> b_flat_norm = nil;
    id<MTLBuffer> b_hc_raw = nil;
    id<MTLBuffer> b_hc_pre_a = nil;
    id<MTLBuffer> b_hc_pre_b = nil;
    id<MTLBuffer> b_hc_post = nil;
    id<MTLBuffer> b_hc_comb = nil;

    id<MTLBuffer> b_hidden = nil;
    id<MTLBuffer> b_norm = nil;
    id<MTLBuffer> b_qr = nil;
    id<MTLBuffer> b_qr_norm = nil;
    id<MTLBuffer> b_q = nil;
    id<MTLBuffer> b_kv = nil;
    id<MTLBuffer> b_kv_norm = nil;
    id<MTLBuffer> b_attn_out = nil;
    id<MTLBuffer> b_oa = nil;
    id<MTLBuffer> b_sub_out = nil;

    id<MTLBuffer> b_comp_tmp_kv = nil;
    id<MTLBuffer> b_comp_tmp_score = nil;
    id<MTLBuffer> b_comp_block = nil;
    id<MTLBuffer> b_idx_k_tmp = nil;
    id<MTLBuffer> b_idx_q = nil;
    id<MTLBuffer> b_idx_w = nil;

    id<MTLBuffer> b_engram_rows = nil;
    id<MTLBuffer> b_engram_kv = nil;

    id<MTLBuffer> b_router_logits = nil;
    id<MTLBuffer> b_topk_ids = nil;
    id<MTLBuffer> b_topk_w = nil;
    id<MTLBuffer> b_ffn_gate = nil;
    id<MTLBuffer> b_ffn_up = nil;
    id<MTLBuffer> b_ffn_act = nil;
    id<MTLBuffer> b_ffn_tmp = nil;
    id<MTLBuffer> b_logits = nil;
    id<MTLBuffer> b_dummy_bias = nil;

    void init_metal() {
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) throw std::runtime_error("No Apple Metal device available");
        queue = [dev newCommandQueue];
        NSError* err = nil;
#ifdef WARPGRID_METAL_PRECOMPILED
        dispatch_data_t data = dispatch_data_create(
            kGgufMetalLibBytes, kGgufMetalLibLen, dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        lib = [dev newLibraryWithData:data error:&err];
#else
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion3_1;
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeSafe;
            opts.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        }
        NSString* src = [[NSString alloc] initWithBytes:kGgufMetalSourceBytes
                                                 length:kGgufMetalSourceLen
                                               encoding:NSUTF8StringEncoding];
        lib = [dev newLibraryWithSource:src options:opts error:&err];
#endif
        if (!lib) {
            throw std::runtime_error(std::string("Failed to load GGUF Metal library: ") +
                                     (err ? err.localizedDescription.UTF8String : "unknown"));
        }
    }

    id<MTLComputePipelineState> pso(const std::string& name) {
        auto it = psos.find(name);
        if (it != psos.end()) return it->second;
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
        if (!fn) throw std::runtime_error("Missing GGUF Metal kernel: " + name);
        NSError* err = nil;
        id<MTLComputePipelineState> st = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!st) {
            throw std::runtime_error("Failed to create PSO " + name + ": " +
                                     (err ? err.localizedDescription.UTF8String : "unknown"));
        }
        psos[name] = st;
        return st;
    }

    id<MTLBuffer> alloc_buf(size_t bytes) {
        size_t sz = std::max<size_t>(16, bytes);
        id<MTLBuffer> b = [dev newBufferWithLength:sz options:MTLResourceStorageModeShared];
        if (!b) throw std::runtime_error("Metal buffer allocation failed (" + std::to_string(sz) + " bytes)");
        memset(b.contents, 0, sz);
        return b;
    }

    id<MTLBuffer> load_f32_required(const GgufFile& gf, const std::string& name) {
        if (!gf.has_tensor(name)) {
            throw std::runtime_error("Missing required DeepSeek-V4.1 tensor: " + name);
        }
        const auto& ti = gf.tensor(name);
        size_t n = (size_t)ti.dims[0] * (size_t)ti.num_rows();
        id<MTLBuffer> b = alloc_buf(n * sizeof(float));
        dequant_row_fp32(ti.type, gf.tensor_data(name), static_cast<float*>(b.contents), n);
        return b;
    }

    WeightBuf load_weight_required(const GgufFile& gf, const std::string& name, uint32_t force_groups = 0) {
        if (!gf.has_tensor(name)) {
            throw std::runtime_error("Missing required DeepSeek-V4.1 tensor: " + name);
        }
        const auto& ti = gf.tensor(name);
        WeightBuf wb;
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
        if (ti.type == GgmlType::F16 || ti.type == GgmlType::BF16 || ti.type == GgmlType::Q5_K) {
            size_t total_elems = (size_t)wb.cols * wb.rows * wb.groups;
            wb.buf = alloc_buf(total_elems * sizeof(float));
            dequant_row_fp32(ti.type, gf.tensor_data(name), static_cast<float*>(wb.buf.contents), total_elems);
            wb.type = GgmlType::F32;
            if (wb.groups > 1) wb.group_stride_bytes = wb.cols * wb.rows * (uint32_t)sizeof(float);
        } else {
            wb.buf = [dev newBufferWithBytes:gf.tensor_data(name)
                                      length:std::max<size_t>(16, ti.nbytes)
                                     options:MTLResourceStorageModeShared];
            if (!wb.buf) throw std::runtime_error("Metal buffer allocation failed for tensor: " + name);
        }
        return wb;
    }

    static const char* gemv_kernel_for(GgmlType t) {
        switch (t) {
            case GgmlType::F32: return "gguf_gemv_f32";
            case GgmlType::Q8_0: return "gguf_gemv_q8_0";
            case GgmlType::IQ4_NL: return "gguf_gemv_iq4_nl";
            case GgmlType::MXFP4: return "gguf_gemv_mxfp4";
            case GgmlType::Q4_K: return "gguf_gemv_q4_k";
            case GgmlType::Q6_K: return "gguf_gemv_q6_k";
            case GgmlType::Q2_K: return "gguf_gemv_q2_k";
            default:
                throw std::runtime_error("Unsupported Metal GEMV GGUF quant type: " + std::to_string((uint32_t)t));
        }
    }

    void encode_gemv(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, const WeightBuf& w, id<MTLBuffer> y,
                     uint32_t M, size_t x_offset = 0, size_t w_offset = 0, size_t y_offset = 0) {
        if (!w.valid()) throw std::runtime_error("Attempted Metal GEMV on missing weight buffer");
        GgufGemvParams p{M, w.rows, w.cols};
        [enc setComputePipelineState:pso(gemv_kernel_for(w.type))];
        [enc setBuffer:x offset:x_offset atIndex:0];
        [enc setBuffer:w.buf offset:w_offset atIndex:1];
        [enc setBuffer:y offset:y_offset atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((w.rows + 3u) / 4u, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

    void encode_rmsnorm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, id<MTLBuffer> w, id<MTLBuffer> y,
                        uint32_t T, uint32_t D, float eps) {
        DsRmsParams p{T, D, eps};
        [enc setComputePipelineState:pso("ds_rmsnorm_f32")];
        [enc setBuffer:x offset:0 atIndex:0];
        [enc setBuffer:w offset:0 atIndex:1];
        [enc setBuffer:y offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    void encode_rmsnorm_unweighted(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, id<MTLBuffer> y,
                                   uint32_t T, uint32_t D, float eps) {
        DsRmsParams p{T, D, eps};
        [enc setComputePipelineState:pso("ds_rmsnorm_unweighted_f32")];
        [enc setBuffer:x offset:0 atIndex:0];
        [enc setBuffer:y offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    void encode_swiglu(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> gate, id<MTLBuffer> up, id<MTLBuffer> out, uint32_t n) {
        [enc setComputePipelineState:pso("ds_swiglu_f32")];
        [enc setBuffer:gate offset:0 atIndex:0];
        [enc setBuffer:up offset:0 atIndex:1];
        [enc setBuffer:out offset:0 atIndex:2];
        [enc setBytes:&n length:sizeof(n) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((n + 255u) / 256u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    void encode_hc_mixes(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> streams,
                         const WeightBuf& fn, id<MTLBuffer> scale, id<MTLBuffer> base,
                         id<MTLBuffer> out_pre, id<MTLBuffer> out_post, id<MTLBuffer> out_comb,
                         uint32_t T, uint32_t D, uint32_t hc, uint32_t iters, float rms_eps, float hc_eps) {
        encode_rmsnorm_unweighted(enc, streams, b_flat_norm, T, hc * D, rms_eps);
        encode_gemv(enc, b_flat_norm, fn, b_hc_raw, T);
        DsHcMixParams mp{T, hc, iters, hc_eps};
        [enc setComputePipelineState:pso("ds_hc_mixes_f32")];
        [enc setBuffer:b_hc_raw offset:0 atIndex:0];
        [enc setBuffer:scale offset:0 atIndex:1];
        [enc setBuffer:base offset:0 atIndex:2];
        [enc setBuffer:out_pre offset:0 atIndex:3];
        [enc setBuffer:out_post offset:0 atIndex:4];
        [enc setBuffer:out_comb offset:0 atIndex:5];
        [enc setBytes:&mp length:sizeof(mp) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    void encode_hc_pre(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> streams, id<MTLBuffer> pre_w,
                       id<MTLBuffer> out_hidden, uint32_t T, uint32_t D, uint32_t hc) {
        DsHcApplyParams ap{T, D, hc};
        [enc setComputePipelineState:pso("ds_hc_pre_f32")];
        [enc setBuffer:streams offset:0 atIndex:0];
        [enc setBuffer:pre_w offset:0 atIndex:1];
        [enc setBuffer:out_hidden offset:0 atIndex:2];
        [enc setBytes:&ap length:sizeof(ap) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(D, T, 1) threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(64u, D), 1, 1)];
    }

    void encode_hc_post(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> sub, id<MTLBuffer> residual,
                        id<MTLBuffer> post_w, id<MTLBuffer> comb_w, id<MTLBuffer> out_streams,
                        uint32_t T, uint32_t D, uint32_t hc) {
        DsHcApplyParams ap{T, D, hc};
        [enc setComputePipelineState:pso("ds_hc_post_f32")];
        [enc setBuffer:sub offset:0 atIndex:0];
        [enc setBuffer:residual offset:0 atIndex:1];
        [enc setBuffer:post_w offset:0 atIndex:2];
        [enc setBuffer:comb_w offset:0 atIndex:3];
        [enc setBuffer:out_streams offset:0 atIndex:4];
        [enc setBytes:&ap length:sizeof(ap) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(D, T, 1) threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(64u, D), 1, 1)];
    }
};

DeepSeekV41Model::DeepSeekV41Model() : impl_(std::make_unique<Impl>()) {}
DeepSeekV41Model::~DeepSeekV41Model() = default;

void DeepSeekV41Model::load_tokenizer(Tokenizer& tok) const {
    gguf.load_tokenizer(tok);
}

void DeepSeekV41Model::load(const std::string& model_path, int max_seq, int max_chunk, int n_slots) {
    impl_->init_metal();
    gguf.open(model_path, false);
    cfg = DeepSeekConfig::from_gguf(gguf);

    if (!engram_path.empty()) {
        engram.open(engram_path, 4096);
        if (!engram.has_tensor("per_layer_token_embd.weight")) {
            throw std::runtime_error("Engram GGUF is missing per_layer_token_embd.weight: " + engram_path);
        }
    }
    impl_->engram_in_main_gguf = gguf.has_tensor("per_layer_token_embd.weight");

    uint64_t rec_ws = (uint64_t)impl_->dev.recommendedMaxWorkingSetSize;
    size_t kv_per_slot = cfg.kv_bytes_per_token() * (size_t)max_seq;
    int eff_slots = std::max(1, n_slots);
    if (rec_ws > 0 && kv_per_slot > 0) {
        size_t kv_budget = (size_t)(rec_ws / 4);
        while (eff_slots > 1 && (size_t)eff_slots * kv_per_slot > kv_budget) {
            --eff_slots;
        }
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
    {
        auto et = gguf.tensor("token_embd.weight").type;
        bool ok_emb = (et == GgmlType::F32 || et == GgmlType::F16 || et == GgmlType::BF16 ||
                       et == GgmlType::Q8_0 || et == GgmlType::IQ4_NL || et == GgmlType::MXFP4 ||
                       et == GgmlType::Q4_K || et == GgmlType::Q5_K || et == GgmlType::Q6_K || et == GgmlType::Q2_K);
        if (!ok_emb) {
            throw std::runtime_error("Unsupported token_embd.weight GGUF type: " + std::to_string((uint32_t)et));
        }
    }
    impl_->output_norm = impl_->load_f32_required(gguf, "output_norm.weight");
    impl_->lm_head = impl_->load_weight_required(gguf, gguf.has_tensor("output.weight") ? "output.weight" : "token_embd.weight");

    impl_->layers.resize((size_t)cfg.num_hidden_layers);
    impl_->kv_caches.resize((size_t)cfg.num_hidden_layers);
    impl_->slot_tok_history.assign((size_t)max_slots, {});

    size_t kv_layer_bytes = (size_t)max_slots * (size_t)max_seq_len * (size_t)cfg.head_dim * sizeof(float);
    uint32_t topk_cap = (uint32_t)std::min(cfg.index_topk, 256);

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
            if (ratio > 1) {
                lw.comp_wgate = impl_->load_weight_required(gguf, p + "attn_compressor_gate.weight");
            }
            lw.idx_k = impl_->load_weight_required(gguf, p + "indexer.attn_k.weight");
            lw.idx_k_norm = impl_->load_f32_required(gguf, p + "indexer.k_norm.weight");

            size_t max_comp = (size_t)(max_seq_len / ratio) + 1;
            impl_->kv_caches[(size_t)l].comp_raw_kv = impl_->alloc_buf(kv_layer_bytes);
            impl_->kv_caches[(size_t)l].comp_raw_score = impl_->alloc_buf(kv_layer_bytes);
            impl_->kv_caches[(size_t)l].comp_kv = impl_->alloc_buf((size_t)max_slots * max_comp * (size_t)cfg.head_dim * sizeof(float));
            impl_->kv_caches[(size_t)l].comp_ik = impl_->alloc_buf((size_t)max_slots * max_comp * (size_t)cfg.index_head_dim * sizeof(float));
        }
        if (ratio > 0 && cfg.topk_src[(size_t)l] == l) {
            lw.idx_q_b = impl_->load_weight_required(gguf, p + "indexer.attn_q_b.weight");
            lw.idx_proj = impl_->load_weight_required(gguf, p + "indexer.proj.weight");
            impl_->kv_caches[(size_t)l].topk_sel = impl_->alloc_buf((size_t)max_batch_tokens * topk_cap * sizeof(int32_t));
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

        impl_->kv_caches[(size_t)l].kv_cache = impl_->alloc_buf(kv_layer_bytes);
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

    impl_->b_streams_a = impl_->alloc_buf(Tmax * hc * D * sizeof(float));
    impl_->b_streams_b = impl_->alloc_buf(Tmax * hc * D * sizeof(float));
    impl_->b_flat_norm = impl_->alloc_buf(Tmax * hc * D * sizeof(float));
    impl_->b_hc_raw = impl_->alloc_buf(Tmax * (2 + hc) * hc * sizeof(float));
    impl_->b_hc_pre_a = impl_->alloc_buf(Tmax * hc * sizeof(float));
    impl_->b_hc_pre_b = impl_->alloc_buf(Tmax * hc * sizeof(float));
    impl_->b_hc_post = impl_->alloc_buf(Tmax * hc * sizeof(float));
    impl_->b_hc_comb = impl_->alloc_buf(Tmax * hc * hc * sizeof(float));

    impl_->b_hidden = impl_->alloc_buf(Tmax * D * sizeof(float));
    impl_->b_norm = impl_->alloc_buf(Tmax * D * sizeof(float));
    impl_->b_qr = impl_->alloc_buf(Tmax * (size_t)std::max(1, cfg.q_lora_rank) * sizeof(float));
    impl_->b_qr_norm = impl_->alloc_buf(Tmax * (size_t)std::max(1, cfg.q_lora_rank) * sizeof(float));
    impl_->b_q = impl_->alloc_buf(Tmax * Qdim * sizeof(float));
    impl_->b_kv = impl_->alloc_buf(Tmax * (size_t)cfg.head_dim * sizeof(float));
    impl_->b_kv_norm = impl_->alloc_buf(Tmax * (size_t)cfg.head_dim * sizeof(float));
    impl_->b_attn_out = impl_->alloc_buf(Tmax * Qdim * sizeof(float));
    impl_->b_oa = impl_->alloc_buf(Tmax * (size_t)std::max(1, cfg.o_group_count * cfg.o_lora_rank) * sizeof(float));
    impl_->b_sub_out = impl_->alloc_buf(Tmax * D * sizeof(float));

    impl_->b_comp_tmp_kv = impl_->alloc_buf(Tmax * (size_t)cfg.head_dim * sizeof(float));
    impl_->b_comp_tmp_score = impl_->alloc_buf(Tmax * (size_t)cfg.head_dim * sizeof(float));
    impl_->b_comp_block = impl_->alloc_buf(Tmax * (size_t)cfg.head_dim * sizeof(float));
    impl_->b_idx_k_tmp = impl_->alloc_buf(Tmax * (size_t)std::max(1, cfg.index_head_dim) * sizeof(float));
    impl_->b_idx_q = impl_->alloc_buf(Tmax * (size_t)std::max(1, cfg.index_head_count * cfg.index_head_dim) * sizeof(float));
    impl_->b_idx_w = impl_->alloc_buf(Tmax * (size_t)std::max(1, cfg.index_head_count) * sizeof(float));

    size_t ple_dim = (size_t)std::max(1, cfg.ple_n_heads * cfg.ple_head_dim);
    impl_->b_engram_rows = impl_->alloc_buf(Tmax * ple_dim * sizeof(float));
    impl_->b_engram_kv = impl_->alloc_buf(Tmax * (hc + 1) * D * sizeof(float));

    impl_->b_router_logits = impl_->alloc_buf(Tmax * (size_t)std::max(1, cfg.n_routed_experts) * sizeof(float));
    impl_->b_topk_ids = impl_->alloc_buf(Tmax * (size_t)std::max(1, cfg.num_experts_per_tok) * sizeof(int32_t));
    impl_->b_topk_w = impl_->alloc_buf(Tmax * (size_t)std::max(1, cfg.num_experts_per_tok) * sizeof(float));
    impl_->b_ffn_gate = impl_->alloc_buf(Tmax * Inter * sizeof(float));
    impl_->b_ffn_up = impl_->alloc_buf(Tmax * Inter * sizeof(float));
    impl_->b_ffn_act = impl_->alloc_buf(Tmax * Inter * sizeof(float));
    impl_->b_ffn_tmp = impl_->alloc_buf(Tmax * D * sizeof(float));
    impl_->b_logits = impl_->alloc_buf((size_t)cfg.vocab_size * sizeof(float));
    impl_->b_dummy_bias = impl_->alloc_buf((size_t)std::max(1, cfg.n_routed_experts) * sizeof(float));

    fprintf(stderr, "[warpgrid] Loaded DeepSeek-V4.1 (%s): %d layers, D=%d, H=%d, hc=%d, E=%d (top-%d), slots=%d, max_seq=%d, engram=%s\n",
            cfg.arch.c_str(), cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads,
            cfg.hc_mult, cfg.n_routed_experts, cfg.num_experts_per_tok, max_slots, max_seq_len,
            (engram.is_open() || impl_->engram_in_main_gguf) ? "ON" : "OFF");
}

void DeepSeekV41Model::reset_cache(int slot_id) {
    if (slot_id < 0 || slot_id >= max_slots) return;
    if ((size_t)slot_id < impl_->slot_tok_history.size()) {
        impl_->slot_tok_history[(size_t)slot_id].clear();
    }
    size_t slot_bytes = (size_t)max_seq_len * (size_t)cfg.head_dim * sizeof(float);
    for (size_t l = 0; l < impl_->kv_caches.size(); ++l) {
        auto& kv = impl_->kv_caches[l];
        memset(static_cast<uint8_t*>(kv.kv_cache.contents) + (size_t)slot_id * slot_bytes, 0, slot_bytes);
        if (kv.comp_raw_kv != nil) {
            memset(static_cast<uint8_t*>(kv.comp_raw_kv.contents) + (size_t)slot_id * slot_bytes, 0, slot_bytes);
            memset(static_cast<uint8_t*>(kv.comp_raw_score.contents) + (size_t)slot_id * slot_bytes, 0, slot_bytes);
        }
    }
}

void DeepSeekV41Model::forward_logits(const int32_t* h_tokens, int T, int start_pos, std::vector<float>& h_logits, int slot_id) {
    if (T <= 0 || T > max_batch_tokens) throw std::runtime_error("Invalid token batch size T");
    if (slot_id < 0 || slot_id >= max_slots) throw std::runtime_error("Invalid slot_id");
    if (start_pos < 0 || start_pos + T > max_seq_len) throw std::runtime_error("Sequence length exceeds max_seq_len");

    auto& hist = impl_->slot_tok_history[(size_t)slot_id];
    if ((int)hist.size() < start_pos + T) hist.resize((size_t)(start_pos + T), 0);
    for (int t = 0; t < T; ++t) hist[(size_t)(start_pos + t)] = h_tokens[t];

    size_t D = (size_t)cfg.hidden_size;
    uint32_t hc = (uint32_t)std::max(1, cfg.hc_mult);
    float* hidden_ptr = static_cast<float*>(impl_->b_hidden.contents);
    const auto& emb_ti = gguf.tensor("token_embd.weight");
    size_t row_stride = ggml_row_nbytes(emb_ti.type, emb_ti.dims[0]);
    const uint8_t* base = gguf.tensor_data("token_embd.weight");
    uint64_t nrows = emb_ti.num_rows();
    for (int t = 0; t < T; ++t) {
        uint64_t tid = (uint64_t)std::clamp(h_tokens[t], 0, (int32_t)(nrows > 0 ? nrows - 1 : 0));
        dequant_row_fp32(emb_ti.type, base + tid * row_stride, hidden_ptr + (size_t)t * D, D);
    }

    {
        id<MTLCommandBuffer> cb0 = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> e0 = [cb0 computeCommandEncoder];
        DsStreamCollapseParams scp{(uint32_t)T, (uint32_t)D, hc};
        [e0 setComputePipelineState:impl_->pso("ds_mhc_expand_f32")];
        [e0 setBuffer:impl_->b_hidden offset:0 atIndex:0];
        [e0 setBuffer:impl_->b_streams_a offset:0 atIndex:1];
        [e0 setBytes:&scp length:sizeof(scp) atIndex:2];
        [e0 dispatchThreads:MTLSizeMake((uint32_t)D, (uint32_t)T, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(64u, (uint32_t)D), 1, 1)];
        [e0 endEncoding];
        [cb0 commit];
        [cb0 waitUntilCompleted];
    }

    id<MTLBuffer> pre_prev = nil;
    id<MTLBuffer> pre_cur = impl_->b_hc_pre_a;
    id<MTLBuffer> pre_next = impl_->b_hc_pre_b;

    size_t slot_kv_bytes = (size_t)max_seq_len * (size_t)cfg.head_dim * sizeof(float);
    size_t slot_kv_off = (size_t)slot_id * slot_kv_bytes;
    uint32_t n_nope = (uint32_t)std::max(0, cfg.head_dim - cfg.n_rot);
    uint32_t n_rot = (uint32_t)std::max(0, cfg.n_rot);
    uint32_t topk_cap = (uint32_t)std::min(cfg.index_topk, 256);

    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
        const auto& lw = impl_->layers[(size_t)l];
        auto& kv = impl_->kv_caches[(size_t)l];

        int e_idx = cfg.engram_index(l);
        if (e_idx >= 0 && (engram.is_open() || impl_->engram_in_main_gguf) && cfg.ple_n_heads > 0 && cfg.ple_head_dim > 0) {
            float* erows_ptr = static_cast<float*>(impl_->b_engram_rows.contents);
            std::vector<int32_t> row_ids((size_t)cfg.ple_n_heads, 0);
            for (int t = 0; t < T; ++t) {
                cfg.engram_hash_token(e_idx, hist, start_pos + t, row_ids.data());
                for (int h = 0; h < cfg.ple_n_heads; ++h) {
                    float* dst = erows_ptr + ((size_t)t * (size_t)cfg.ple_n_heads + (size_t)h) * (size_t)cfg.ple_head_dim;
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
            id<MTLCommandBuffer> cbe = [impl_->queue commandBuffer];
            id<MTLComputeCommandEncoder> ence = [cbe computeCommandEncoder];
            impl_->encode_gemv(ence, impl_->b_engram_rows, lw.engram_kv, impl_->b_engram_kv, (uint32_t)T);
            DsEngramApplyParams eap{(uint32_t)T, (uint32_t)D, hc, cfg.rms_norm_eps};
            [ence setComputePipelineState:impl_->pso("ds_engram_apply_f32")];
            [ence setBuffer:impl_->b_streams_a offset:0 atIndex:0];
            [ence setBuffer:impl_->b_engram_kv offset:0 atIndex:1];
            [ence setBuffer:lw.engram_gate offset:0 atIndex:2];
            [ence setBytes:&eap length:sizeof(eap) atIndex:3];
            [ence dispatchThreadgroups:MTLSizeMake(hc, (uint32_t)T, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [ence endEncoding];
            [cbe commit];
            [cbe waitUntilCompleted];
        }

        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        impl_->encode_hc_mixes(enc, impl_->b_streams_a, lw.hc_attn_fn, lw.hc_attn_scale, lw.hc_attn_base,
                               pre_cur, impl_->b_hc_post, impl_->b_hc_comb,
                               (uint32_t)T, (uint32_t)D, hc, (uint32_t)cfg.hc_sinkhorn_iters, cfg.rms_norm_eps, cfg.hc_eps);

        if (pre_prev != nil) {
            impl_->encode_hc_pre(enc, impl_->b_streams_a, pre_prev, impl_->b_hidden, (uint32_t)T, (uint32_t)D, hc);
        } else {
            [enc endEncoding];
            id<MTLBlitCommandEncoder> bl0 = [cb blitCommandEncoder];
            for (int t = 0; t < T; ++t) {
                [bl0 copyFromBuffer:impl_->b_streams_a sourceOffset:(size_t)t * hc * D * sizeof(float)
                           toBuffer:impl_->b_hidden destinationOffset:(size_t)t * D * sizeof(float)
                               size:D * sizeof(float)];
            }
            [bl0 endEncoding];
            enc = [cb computeCommandEncoder];
        }

        impl_->encode_rmsnorm(enc, impl_->b_hidden, lw.attn_norm, impl_->b_norm, (uint32_t)T, (uint32_t)D, cfg.rms_norm_eps);

        impl_->encode_gemv(enc, impl_->b_norm, lw.wq_a, impl_->b_qr, (uint32_t)T);
        impl_->encode_rmsnorm(enc, impl_->b_qr, lw.attn_q_a_norm, impl_->b_qr_norm, (uint32_t)T, (uint32_t)cfg.q_lora_rank, cfg.rms_norm_eps);
        impl_->encode_gemv(enc, impl_->b_qr_norm, lw.wq_b, impl_->b_q, (uint32_t)T);

        impl_->encode_gemv(enc, impl_->b_norm, lw.wkv, impl_->b_kv, (uint32_t)T);
        impl_->encode_rmsnorm(enc, impl_->b_kv, lw.attn_kv_norm, impl_->b_kv_norm, (uint32_t)T, (uint32_t)cfg.head_dim, cfg.rms_norm_eps);

        int ratio = cfg.compress_ratio_for(l);
        float freq_base_l = (ratio != 0) ? cfg.compress_rope_freq_base : cfg.rope_freq_base;

        if (n_rot >= 2) {
            DsPartialRopeParams rq{(uint32_t)T, (uint32_t)start_pos, (uint32_t)cfg.num_attention_heads, (uint32_t)cfg.head_dim, n_nope, n_rot, 0u, freq_base_l};
            [enc setComputePipelineState:impl_->pso("ds_rope_partial_f32")];
            [enc setBuffer:impl_->b_q offset:0 atIndex:0];
            [enc setBytes:&rq length:sizeof(rq) atIndex:1];
            [enc dispatchThreads:MTLSizeMake(n_rot / 2u, (uint32_t)cfg.num_attention_heads, (uint32_t)T)
                threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(32u, n_rot / 2u), 1, 1)];

            DsPartialRopeParams rk{(uint32_t)T, (uint32_t)start_pos, 1u, (uint32_t)cfg.head_dim, n_nope, n_rot, 0u, freq_base_l};
            [enc setBuffer:impl_->b_kv_norm offset:0 atIndex:0];
            [enc setBytes:&rk length:sizeof(rk) atIndex:1];
            [enc dispatchThreads:MTLSizeMake(n_rot / 2u, 1u, (uint32_t)T)
                threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(32u, n_rot / 2u), 1, 1)];
        }

        if (ratio > 0 && cfg.kv_src[(size_t)l] == l) {
            impl_->encode_gemv(enc, impl_->b_norm, lw.comp_wkv, impl_->b_comp_tmp_kv, (uint32_t)T);
            if (ratio > 1) {
                impl_->encode_gemv(enc, impl_->b_norm, lw.comp_wgate, impl_->b_comp_tmp_score, (uint32_t)T);
            } else {
                uint32_t zn = (uint32_t)((size_t)T * (size_t)cfg.head_dim);
                [enc setComputePipelineState:impl_->pso("ds_zero_f32")];
                [enc setBuffer:impl_->b_comp_tmp_score offset:0 atIndex:0];
                [enc setBytes:&zn length:sizeof(zn) atIndex:1];
                [enc dispatchThreadgroups:MTLSizeMake((zn + 255u) / 256u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }

        [enc endEncoding];

        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        size_t kv_copy_bytes = (size_t)T * (size_t)cfg.head_dim * sizeof(float);
        size_t kv_dst_off = slot_kv_off + (size_t)start_pos * (size_t)cfg.head_dim * sizeof(float);
        [blit copyFromBuffer:impl_->b_kv_norm sourceOffset:0 toBuffer:kv.kv_cache destinationOffset:kv_dst_off size:kv_copy_bytes];
        if (ratio > 0 && cfg.kv_src[(size_t)l] == l) {
            [blit copyFromBuffer:impl_->b_comp_tmp_kv sourceOffset:0 toBuffer:kv.comp_raw_kv destinationOffset:kv_dst_off size:kv_copy_bytes];
            [blit copyFromBuffer:impl_->b_comp_tmp_score sourceOffset:0 toBuffer:kv.comp_raw_score destinationOffset:kv_dst_off size:kv_copy_bytes];
        }
        [blit endEncoding];

        enc = [cb computeCommandEncoder];

        uint32_t total_comp_blocks = (ratio > 0) ? (uint32_t)((start_pos + T) / ratio) : 0u;
        if (ratio > 0 && cfg.kv_src[(size_t)l] == l && total_comp_blocks > 0) {
            uint32_t first_blk = (uint32_t)(start_pos / ratio);
            uint32_t n_new = total_comp_blocks - first_blk;
            if (n_new > 0) {
                size_t max_comp = (size_t)(max_seq_len / ratio) + 1;
                size_t raw_off = slot_kv_off + (size_t)first_blk * (size_t)ratio * (size_t)cfg.head_dim * sizeof(float);
                size_t comp_off = (size_t)slot_id * max_comp * (size_t)cfg.head_dim * sizeof(float) +
                                  (size_t)first_blk * (size_t)cfg.head_dim * sizeof(float);
                size_t ik_off = (size_t)slot_id * max_comp * (size_t)cfg.index_head_dim * sizeof(float) +
                                (size_t)first_blk * (size_t)cfg.index_head_dim * sizeof(float);

                DsCompGroupParams cgp{n_new, (uint32_t)ratio, (uint32_t)cfg.head_dim, cfg.rms_norm_eps};
                [enc setComputePipelineState:impl_->pso("ds_comp_group_reduce_f32")];
                [enc setBuffer:kv.comp_raw_kv offset:raw_off atIndex:0];
                [enc setBuffer:kv.comp_raw_score offset:raw_off atIndex:1];
                [enc setBuffer:lw.comp_norm offset:0 atIndex:2];
                [enc setBuffer:kv.comp_kv offset:comp_off atIndex:3];
                [enc setBytes:&cgp length:sizeof(cgp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(n_new, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                impl_->encode_gemv(enc, kv.comp_kv, lw.idx_k, impl_->b_idx_k_tmp, n_new, comp_off, 0, 0);
                impl_->encode_rmsnorm(enc, impl_->b_idx_k_tmp, lw.idx_k_norm, kv.comp_ik, n_new, (uint32_t)cfg.index_head_dim, cfg.rms_norm_eps);

                if (n_rot >= 2) {
                    uint32_t ik_nope = (uint32_t)std::max(0, cfg.index_head_dim - cfg.n_rot);
                    DsPartialRopeParams rik{n_new, first_blk * (uint32_t)ratio, 1u, (uint32_t)cfg.index_head_dim, ik_nope, n_rot, 0u, freq_base_l};
                    [enc setComputePipelineState:impl_->pso("ds_rope_partial_f32")];
                    [enc setBuffer:kv.comp_ik offset:ik_off atIndex:0];
                    [enc setBytes:&rik length:sizeof(rik) atIndex:1];
                    [enc dispatchThreads:MTLSizeMake(n_rot / 2u, 1u, n_new)
                        threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(32u, n_rot / 2u), 1, 1)];

                    DsPartialRopeParams rc{n_new, first_blk * (uint32_t)ratio, 1u, (uint32_t)cfg.head_dim, n_nope, n_rot, 0u, freq_base_l};
                    [enc setBuffer:kv.comp_kv offset:comp_off atIndex:0];
                    [enc setBytes:&rc length:sizeof(rc) atIndex:1];
                    [enc dispatchThreads:MTLSizeMake(n_rot / 2u, 1u, n_new)
                        threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(32u, n_rot / 2u), 1, 1)];
                }
            }
        }

        if (ratio > 0 && cfg.topk_src[(size_t)l] == l && total_comp_blocks > 0) {
            int src_kv_l = cfg.kv_src[(size_t)l];
            size_t max_comp = (size_t)(max_seq_len / ratio) + 1;
            size_t slot_ik_off = (size_t)slot_id * max_comp * (size_t)cfg.index_head_dim * sizeof(float);

            impl_->encode_gemv(enc, impl_->b_qr_norm, lw.idx_q_b, impl_->b_idx_q, (uint32_t)T);
            if (n_rot >= 2) {
                uint32_t iq_nope = (uint32_t)std::max(0, cfg.index_head_dim - cfg.n_rot);
                DsPartialRopeParams riq{(uint32_t)T, (uint32_t)start_pos, (uint32_t)cfg.index_head_count, (uint32_t)cfg.index_head_dim, iq_nope, n_rot, 0u, freq_base_l};
                [enc setComputePipelineState:impl_->pso("ds_rope_partial_f32")];
                [enc setBuffer:impl_->b_idx_q offset:0 atIndex:0];
                [enc setBytes:&riq length:sizeof(riq) atIndex:1];
                [enc dispatchThreads:MTLSizeMake(n_rot / 2u, (uint32_t)cfg.index_head_count, (uint32_t)T)
                    threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(32u, n_rot / 2u), 1, 1)];
            }
            impl_->encode_gemv(enc, impl_->b_norm, lw.idx_proj, impl_->b_idx_w, (uint32_t)T);

            DsIndexerParams ip{
                (uint32_t)T, (uint32_t)start_pos, total_comp_blocks,
                (uint32_t)cfg.index_head_count, (uint32_t)cfg.index_head_dim,
                (uint32_t)ratio, (uint32_t)cfg.sliding_window, topk_cap,
                1.0f / std::sqrt((float)(cfg.index_head_dim * cfg.index_head_count))
            };
            [enc setComputePipelineState:impl_->pso("ds_indexer_score_topk_f32")];
            [enc setBuffer:impl_->b_idx_q offset:0 atIndex:0];
            [enc setBuffer:impl_->kv_caches[(size_t)src_kv_l].comp_ik offset:slot_ik_off atIndex:1];
            [enc setBuffer:impl_->b_idx_w offset:0 atIndex:2];
            [enc setBuffer:kv.topk_sel offset:0 atIndex:3];
            [enc setBytes:&ip length:sizeof(ip) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((uint32_t)T, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }

        if (ratio > 0 && total_comp_blocks > 0 && cfg.kv_src[(size_t)l] >= 0 && cfg.topk_src[(size_t)l] >= 0) {
            int src_kv_l = cfg.kv_src[(size_t)l];
            int src_tk_l = cfg.topk_src[(size_t)l];
            size_t max_comp = (size_t)(max_seq_len / ratio) + 1;
            size_t slot_comp_off = (size_t)slot_id * max_comp * (size_t)cfg.head_dim * sizeof(float);
            DsMqaSinkAttnParams ap{
                (uint32_t)T, (uint32_t)start_pos, (uint32_t)cfg.num_attention_heads, (uint32_t)cfg.head_dim,
                (uint32_t)cfg.sliding_window, topk_cap, 1.0f / std::sqrt((float)cfg.head_dim)
            };
            [enc setComputePipelineState:impl_->pso("ds_mqa_sink_comp_attn_f32")];
            [enc setBuffer:impl_->b_q offset:0 atIndex:0];
            [enc setBuffer:kv.kv_cache offset:slot_kv_off atIndex:1];
            [enc setBuffer:impl_->kv_caches[(size_t)src_kv_l].comp_kv offset:slot_comp_off atIndex:2];
            [enc setBuffer:impl_->kv_caches[(size_t)src_tk_l].topk_sel offset:0 atIndex:3];
            [enc setBuffer:lw.attn_sinks offset:0 atIndex:4];
            [enc setBuffer:impl_->b_attn_out offset:0 atIndex:5];
            [enc setBytes:&ap length:sizeof(ap) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((uint32_t)cfg.num_attention_heads, (uint32_t)T, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        } else {
            DsMqaSinkAttnParams ap{
                (uint32_t)T, (uint32_t)start_pos, (uint32_t)cfg.num_attention_heads, (uint32_t)cfg.head_dim,
                (uint32_t)cfg.sliding_window, 0u, 1.0f / std::sqrt((float)cfg.head_dim)
            };
            [enc setComputePipelineState:impl_->pso("ds_mqa_sink_attn_f32")];
            [enc setBuffer:impl_->b_q offset:0 atIndex:0];
            [enc setBuffer:kv.kv_cache offset:slot_kv_off atIndex:1];
            [enc setBuffer:lw.attn_sinks offset:0 atIndex:2];
            [enc setBuffer:impl_->b_attn_out offset:0 atIndex:3];
            [enc setBytes:&ap length:sizeof(ap) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((uint32_t)cfg.num_attention_heads, (uint32_t)T, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }

        if (n_rot >= 2) {
            DsPartialRopeParams rinv{(uint32_t)T, (uint32_t)start_pos, (uint32_t)cfg.num_attention_heads, (uint32_t)cfg.head_dim, n_nope, n_rot, 1u, freq_base_l};
            [enc setComputePipelineState:impl_->pso("ds_rope_partial_f32")];
            [enc setBuffer:impl_->b_attn_out offset:0 atIndex:0];
            [enc setBytes:&rinv length:sizeof(rinv) atIndex:1];
            [enc dispatchThreads:MTLSizeMake(n_rot / 2u, (uint32_t)cfg.num_attention_heads, (uint32_t)T)
                threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(32u, n_rot / 2u), 1, 1)];
        }

        uint32_t o_groups = lw.wo_a.groups;
        for (int t = 0; t < T; ++t) {
            for (uint32_t g = 0; g < o_groups; ++g) {
                size_t x_off = ((size_t)t * o_groups + g) * lw.wo_a.cols * sizeof(float);
                size_t w_off = (size_t)g * lw.wo_a.group_stride_bytes;
                size_t y_off = ((size_t)t * o_groups + g) * lw.wo_a.rows * sizeof(float);
                impl_->encode_gemv(enc, impl_->b_attn_out, lw.wo_a, impl_->b_oa, 1u, x_off, w_off, y_off);
            }
        }
        impl_->encode_gemv(enc, impl_->b_oa, lw.wo_b, impl_->b_sub_out, (uint32_t)T);

        impl_->encode_hc_post(enc, impl_->b_sub_out, impl_->b_streams_a, impl_->b_hc_post, impl_->b_hc_comb,
                              impl_->b_streams_b, (uint32_t)T, (uint32_t)D, hc);
        std::swap(impl_->b_streams_a, impl_->b_streams_b);

        impl_->encode_hc_mixes(enc, impl_->b_streams_a, lw.hc_ffn_fn, lw.hc_ffn_scale, lw.hc_ffn_base,
                               pre_next, impl_->b_hc_post, impl_->b_hc_comb,
                               (uint32_t)T, (uint32_t)D, hc, (uint32_t)cfg.hc_sinkhorn_iters, cfg.rms_norm_eps, cfg.hc_eps);
        impl_->encode_hc_pre(enc, impl_->b_streams_a, pre_cur, impl_->b_hidden, (uint32_t)T, (uint32_t)D, hc);
        pre_prev = pre_next;
        std::swap(pre_cur, pre_next);

        impl_->encode_rmsnorm(enc, impl_->b_hidden, lw.ffn_norm, impl_->b_norm, (uint32_t)T, (uint32_t)D, cfg.rms_norm_eps);

        impl_->encode_gemv(enc, impl_->b_norm, lw.ffn_gate_shexp, impl_->b_ffn_gate, (uint32_t)T);
        impl_->encode_gemv(enc, impl_->b_norm, lw.ffn_up_shexp, impl_->b_ffn_up, (uint32_t)T);
        impl_->encode_swiglu(enc, impl_->b_ffn_gate, impl_->b_ffn_up, impl_->b_ffn_act, (uint32_t)((size_t)T * lw.ffn_gate_shexp.rows));
        impl_->encode_gemv(enc, impl_->b_ffn_act, lw.ffn_down_shexp, impl_->b_sub_out, (uint32_t)T);

        impl_->encode_gemv(enc, impl_->b_norm, lw.router, impl_->b_router_logits, (uint32_t)T);
        DsRouterParams rp{
            (uint32_t)T, (uint32_t)cfg.n_routed_experts, (uint32_t)cfg.num_experts_per_tok,
            lw.router_bias != nil ? 1u : 0u, cfg.route_scale
        };
        [enc setComputePipelineState:impl_->pso("ds_router_sqrtsoftplus_topk_f32")];
        [enc setBuffer:impl_->b_router_logits offset:0 atIndex:0];
        [enc setBuffer:(lw.router_bias != nil ? lw.router_bias : impl_->b_dummy_bias) offset:0 atIndex:1];
        [enc setBuffer:impl_->b_topk_ids offset:0 atIndex:2];
        [enc setBuffer:impl_->b_topk_w offset:0 atIndex:3];
        [enc setBytes:&rp length:sizeof(rp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((uint32_t)T, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        const int32_t* h_ids = static_cast<const int32_t*>(impl_->b_topk_ids.contents);
        cb = [impl_->queue commandBuffer];
        enc = [cb computeCommandEncoder];
        for (int t = 0; t < T; ++t) {
            size_t x_off = (size_t)t * D * sizeof(float);
            for (int r = 0; r < cfg.num_experts_per_tok; ++r) {
                int e = h_ids[t * cfg.num_experts_per_tok + r];
                if (e < 0 || e >= (int)lw.ffn_gate_exps.groups) continue;
                size_t g_off = (size_t)e * lw.ffn_gate_exps.group_stride_bytes;
                size_t u_off = (size_t)e * lw.ffn_up_exps.group_stride_bytes;
                size_t d_off = (size_t)e * lw.ffn_down_exps.group_stride_bytes;

                impl_->encode_gemv(enc, impl_->b_norm, lw.ffn_gate_exps, impl_->b_ffn_gate, 1u, x_off, g_off, 0);
                impl_->encode_gemv(enc, impl_->b_norm, lw.ffn_up_exps, impl_->b_ffn_up, 1u, x_off, u_off, 0);
                impl_->encode_swiglu(enc, impl_->b_ffn_gate, impl_->b_ffn_up, impl_->b_ffn_act, lw.ffn_gate_exps.rows);
                impl_->encode_gemv(enc, impl_->b_ffn_act, lw.ffn_down_exps, impl_->b_ffn_tmp, 1u, 0, d_off, 0);

                struct { uint32_t n; uint32_t top_k; uint32_t r; uint32_t D; } sap{
                    (uint32_t)D, (uint32_t)cfg.num_experts_per_tok, (uint32_t)r, (uint32_t)D
                };
                [enc setComputePipelineState:impl_->pso("ds_moe_scaled_add_f32")];
                [enc setBuffer:impl_->b_sub_out offset:x_off atIndex:0];
                [enc setBuffer:impl_->b_ffn_tmp offset:0 atIndex:1];
                [enc setBuffer:impl_->b_topk_w offset:(size_t)t * (size_t)cfg.num_experts_per_tok * sizeof(float) atIndex:2];
                [enc setBytes:&sap length:sizeof(sap) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(((uint32_t)D + 255u) / 256u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }

        impl_->encode_hc_post(enc, impl_->b_sub_out, impl_->b_streams_a, impl_->b_hc_post, impl_->b_hc_comb,
                              impl_->b_streams_b, (uint32_t)T, (uint32_t)D, hc);
        std::swap(impl_->b_streams_a, impl_->b_streams_b);

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (pre_prev != nil) {
        impl_->encode_hc_pre(enc, impl_->b_streams_a, pre_prev, impl_->b_hidden, (uint32_t)T, (uint32_t)D, hc);
    }
    impl_->encode_rmsnorm(enc, impl_->b_hidden, impl_->output_norm, impl_->b_norm, (uint32_t)T, (uint32_t)D, cfg.rms_norm_eps);
    size_t last_off = (size_t)(T - 1) * D * sizeof(float);
    impl_->encode_gemv(enc, impl_->b_norm, impl_->lm_head, impl_->b_logits, 1u, last_off, 0, 0);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    h_logits.resize((size_t)cfg.vocab_size);
    memcpy(h_logits.data(), impl_->b_logits.contents, (size_t)cfg.vocab_size * sizeof(float));
}

int32_t DeepSeekV41Model::forward_and_greedy(const int32_t* h_tokens, int T, int start_pos, int slot_id) {
    std::vector<float> logits;
    forward_logits(h_tokens, T, start_pos, logits, slot_id);
    int32_t best = 0;
    float bv = logits.empty() ? 0.0f : logits[0];
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > bv) {
            bv = logits[i];
            best = (int32_t)i;
        }
    }
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
    if (enable) {
        throw std::runtime_error("--dump-imatrix is not supported for DeepSeek GGUF models");
    }
}

void DeepSeekV41Model::save_imatrix(const std::string& path) const {
    (void)path;
    throw std::runtime_error("--dump-imatrix is not supported for DeepSeek GGUF models");
}

}
