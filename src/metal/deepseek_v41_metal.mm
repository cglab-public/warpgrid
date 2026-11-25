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

struct GgufMoeParams {
    uint32_t T;
    uint32_t N;
    uint32_t K;
    uint32_t top_k;
    uint32_t expert_stride_bytes;
};

struct DsRmsParams {
    uint32_t T;
    uint32_t D;
    float eps;
};

struct DsMhcParams {
    uint32_t T;
    uint32_t D;
    uint32_t hc_mult;
    uint32_t sinkhorn_iters;
    float eps;
};

struct DsRopeParams {
    uint32_t T;
    uint32_t start_pos;
    uint32_t n_heads;
    uint32_t head_dim;
    float freq_base;
};

struct DsRouterParams {
    uint32_t T;
    uint32_t E;
    uint32_t top_k;
    uint32_t has_bias;
    float route_scale;
};

struct DsAttnParams {
    uint32_t T;
    uint32_t start_pos;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t max_seq;
    uint32_t compress_ratio;
    uint32_t local_window;
    float scale;
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
    uint32_t experts = 1;
    uint32_t expert_stride_bytes = 0;
    bool valid() const { return buf != nil; }
};

struct DsLayerWeights {
    id<MTLBuffer> attn_norm = nil;
    id<MTLBuffer> ffn_norm = nil;
    WeightBuf wq;
    WeightBuf wk;
    WeightBuf wv;
    WeightBuf wo;
    WeightBuf router;
    id<MTLBuffer> router_bias = nil;
    WeightBuf ffn_gate_exps;
    WeightBuf ffn_up_exps;
    WeightBuf ffn_down_exps;
    WeightBuf ffn_gate_shexp;
    WeightBuf ffn_up_shexp;
    WeightBuf ffn_down_shexp;
    WeightBuf ffn_gate_dense;
    WeightBuf ffn_up_dense;
    WeightBuf ffn_down_dense;
    id<MTLBuffer> hc_attn_mat = nil;
    id<MTLBuffer> hc_attn_post = nil;
    id<MTLBuffer> hc_ffn_mat = nil;
    id<MTLBuffer> hc_ffn_post = nil;
};

struct DsLayerKv {
    id<MTLBuffer> k_cache = nil;
    id<MTLBuffer> v_cache = nil;
};

}

struct DeepSeekV41Model::Impl {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> lib = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> psos;

    WeightBuf lm_head;
    id<MTLBuffer> output_norm = nil;
    std::vector<DsLayerWeights> layers;
    std::vector<DsLayerKv> kv_caches;
    std::vector<std::vector<int32_t>> slot_tok_history;
    bool has_mhc = false;

    id<MTLBuffer> b_streams_a = nil;
    id<MTLBuffer> b_streams_b = nil;
    id<MTLBuffer> b_hidden = nil;
    id<MTLBuffer> b_norm = nil;
    id<MTLBuffer> b_q = nil;
    id<MTLBuffer> b_k = nil;
    id<MTLBuffer> b_v = nil;
    id<MTLBuffer> b_attn_out = nil;
    id<MTLBuffer> b_sub_out = nil;
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

    id<MTLBuffer> load_f32_1d(const GgufFile& gf, const std::string& name, size_t expected_elems, float fill_if_missing = 1.0f) {
        if (!gf.has_tensor(name)) {
            id<MTLBuffer> b = alloc_buf(expected_elems * sizeof(float));
            float* p = static_cast<float*>(b.contents);
            for (size_t i = 0; i < expected_elems; ++i) p[i] = fill_if_missing;
            return b;
        }
        const auto& ti = gf.tensor(name);
        size_t n = ti.dims.empty() ? expected_elems : (size_t)ti.dims[0] * (size_t)ti.num_rows();
        id<MTLBuffer> b = alloc_buf(n * sizeof(float));
        dequant_row_fp32(ti.type, gf.tensor_data(name), static_cast<float*>(b.contents), n);
        return b;
    }

    WeightBuf load_weight(const GgufFile& gf, const std::string& name) {
        WeightBuf wb;
        if (!gf.has_tensor(name)) return wb;
        const auto& ti = gf.tensor(name);
        wb.type = ti.type;
        wb.cols = ti.dims.size() >= 1 ? (uint32_t)ti.dims[0] : 0;
        wb.rows = ti.dims.size() >= 2 ? (uint32_t)ti.dims[1] : 1;
        wb.experts = ti.dims.size() >= 3 ? (uint32_t)ti.dims[2] : 1;
        if (wb.experts > 1 && wb.cols > 0) {
            wb.expert_stride_bytes = (uint32_t)(ggml_row_nbytes(ti.type, wb.cols) * wb.rows);
        }
        if (ti.type == GgmlType::F16 || ti.type == GgmlType::BF16 || ti.type == GgmlType::Q5_K) {
            size_t total_elems = (size_t)wb.cols * wb.rows * wb.experts;
            wb.buf = alloc_buf(total_elems * sizeof(float));
            dequant_row_fp32(ti.type, gf.tensor_data(name), static_cast<float*>(wb.buf.contents), total_elems);
            wb.type = GgmlType::F32;
            if (wb.experts > 1) wb.expert_stride_bytes = wb.cols * wb.rows * (uint32_t)sizeof(float);
        } else {
            wb.buf = [dev newBufferWithBytes:gf.tensor_data(name)
                                      length:std::max<size_t>(16, ti.nbytes)
                                     options:MTLResourceStorageModeShared];
            if (!wb.buf) {
                throw std::runtime_error("Metal buffer allocation failed for tensor: " + name);
            }
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

    void encode_gemv(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, const WeightBuf& w, id<MTLBuffer> y, uint32_t M, size_t w_offset = 0) {
        if (!w.valid()) throw std::runtime_error("Attempted Metal GEMV on missing weight buffer");
        GgufGemvParams p{M, w.rows, w.cols};
        [enc setComputePipelineState:pso(gemv_kernel_for(w.type))];
        [enc setBuffer:x offset:0 atIndex:0];
        [enc setBuffer:w.buf offset:w_offset atIndex:1];
        [enc setBuffer:y offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((w.rows + 3u) / 4u, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

    void encode_rmsnorm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, id<MTLBuffer> w, id<MTLBuffer> y, uint32_t T, uint32_t D, float eps) {
        DsRmsParams p{T, D, eps};
        [enc setComputePipelineState:pso("ds_rmsnorm_f32")];
        [enc setBuffer:x offset:0 atIndex:0];
        [enc setBuffer:w offset:0 atIndex:1];
        [enc setBuffer:y offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    void encode_add(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> dst, id<MTLBuffer> src, uint32_t n) {
        [enc setComputePipelineState:pso("ds_add_f32")];
        [enc setBuffer:dst offset:0 atIndex:0];
        [enc setBuffer:src offset:0 atIndex:1];
        [enc setBytes:&n length:sizeof(n) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((n + 255u) / 256u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    void encode_swiglu(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> gate, id<MTLBuffer> up, id<MTLBuffer> out, uint32_t n) {
        [enc setComputePipelineState:pso("ds_swiglu_f32")];
        [enc setBuffer:gate offset:0 atIndex:0];
        [enc setBuffer:up offset:0 atIndex:1];
        [enc setBuffer:out offset:0 atIndex:2];
        [enc setBytes:&n length:sizeof(n) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((n + 255u) / 256u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
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
    }

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
        throw std::runtime_error("Missing required GGUF tensor: token_embd.weight");
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
    impl_->output_norm = impl_->load_f32_1d(gguf, "output_norm.weight", (size_t)cfg.hidden_size, 1.0f);
    if (gguf.has_tensor("output.weight")) {
        impl_->lm_head = impl_->load_weight(gguf, "output.weight");
    } else {
        impl_->lm_head = impl_->load_weight(gguf, "token_embd.weight");
    }

    impl_->layers.resize((size_t)cfg.num_hidden_layers);
    impl_->kv_caches.resize((size_t)cfg.num_hidden_layers);
    impl_->slot_tok_history.assign((size_t)max_slots, {});
    impl_->has_mhc = false;
    size_t kv_layer_bytes = (size_t)max_slots * (size_t)max_seq_len * (size_t)cfg.num_key_value_heads * (size_t)cfg.head_dim * sizeof(float);

    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
        std::string p = "blk." + std::to_string(l) + ".";
        auto& lw = impl_->layers[(size_t)l];
        lw.attn_norm = impl_->load_f32_1d(gguf, p + "attn_norm.weight", (size_t)cfg.hidden_size, 1.0f);
        lw.ffn_norm = impl_->load_f32_1d(gguf, p + "ffn_norm.weight", (size_t)cfg.hidden_size, 1.0f);
        lw.wq = impl_->load_weight(gguf, p + "attn_q.weight");
        lw.wk = impl_->load_weight(gguf, p + "attn_k.weight");
        lw.wv = impl_->load_weight(gguf, p + "attn_v.weight");
        lw.wo = impl_->load_weight(gguf, p + "attn_output.weight");
        if (!lw.wq.valid() || !lw.wk.valid() || !lw.wv.valid() || !lw.wo.valid()) {
            throw std::runtime_error("Missing required attention projection weights in layer " + std::to_string(l));
        }
        lw.router = impl_->load_weight(gguf, p + "ffn_gate_inp.weight");
        if (gguf.has_tensor(p + "exp_probs_b.bias")) {
            lw.router_bias = impl_->load_f32_1d(gguf, p + "exp_probs_b.bias", (size_t)cfg.n_routed_experts, 0.0f);
        }
        lw.ffn_gate_exps = impl_->load_weight(gguf, p + "ffn_gate_exps.weight");
        lw.ffn_up_exps = impl_->load_weight(gguf, p + "ffn_up_exps.weight");
        lw.ffn_down_exps = impl_->load_weight(gguf, p + "ffn_down_exps.weight");
        lw.ffn_gate_shexp = impl_->load_weight(gguf, p + "ffn_gate_shexp.weight");
        lw.ffn_up_shexp = impl_->load_weight(gguf, p + "ffn_up_shexp.weight");
        lw.ffn_down_shexp = impl_->load_weight(gguf, p + "ffn_down_shexp.weight");
        lw.ffn_gate_dense = impl_->load_weight(gguf, p + "ffn_gate.weight");
        lw.ffn_up_dense = impl_->load_weight(gguf, p + "ffn_up.weight");
        lw.ffn_down_dense = impl_->load_weight(gguf, p + "ffn_down.weight");
        bool has_dense = lw.ffn_gate_dense.valid() && lw.ffn_up_dense.valid() && lw.ffn_down_dense.valid();
        bool has_moe = lw.router.valid() && lw.ffn_gate_exps.valid() && lw.ffn_up_exps.valid() && lw.ffn_down_exps.valid();
        bool has_shexp = lw.ffn_gate_shexp.valid() && lw.ffn_up_shexp.valid() && lw.ffn_down_shexp.valid();
        if (!has_dense && !has_moe && !has_shexp) {
            throw std::runtime_error("Missing required FFN weights in layer " + std::to_string(l));
        }

        size_t hc_m = (size_t)std::max(1, cfg.hc_mult);
        if (gguf.has_tensor(p + "hc_attn_mat.weight")) {
            lw.hc_attn_mat = impl_->load_f32_1d(gguf, p + "hc_attn_mat.weight", hc_m * hc_m, 0.0f);
            lw.hc_attn_post = impl_->load_f32_1d(gguf, p + "hc_attn_post.weight", hc_m, 1.0f / (float)hc_m);
            impl_->has_mhc = true;
        }
        if (gguf.has_tensor(p + "hc_ffn_mat.weight")) {
            lw.hc_ffn_mat = impl_->load_f32_1d(gguf, p + "hc_ffn_mat.weight", hc_m * hc_m, 0.0f);
            lw.hc_ffn_post = impl_->load_f32_1d(gguf, p + "hc_ffn_post.weight", hc_m, 1.0f / (float)hc_m);
            impl_->has_mhc = true;
        }

        impl_->kv_caches[(size_t)l].k_cache = impl_->alloc_buf(kv_layer_bytes);
        impl_->kv_caches[(size_t)l].v_cache = impl_->alloc_buf(kv_layer_bytes);
    }

    size_t Tmax = (size_t)max_batch_tokens;
    size_t D = (size_t)cfg.hidden_size;
    size_t Mhc = (size_t)std::max(1, cfg.hc_mult);
    size_t Qdim = (size_t)cfg.num_attention_heads * (size_t)cfg.head_dim;
    size_t KVdim = (size_t)cfg.num_key_value_heads * (size_t)cfg.head_dim;
    size_t Inter = (size_t)std::max(cfg.ffn_intermediate_size, cfg.expert_intermediate_size);
    for (const auto& lw : impl_->layers) {
        Inter = std::max<size_t>(Inter, lw.ffn_gate_dense.rows);
        Inter = std::max<size_t>(Inter, lw.ffn_up_dense.rows);
        Inter = std::max<size_t>(Inter, lw.ffn_gate_shexp.rows);
        Inter = std::max<size_t>(Inter, lw.ffn_up_shexp.rows);
        Inter = std::max<size_t>(Inter, lw.ffn_gate_exps.rows);
        Inter = std::max<size_t>(Inter, lw.ffn_up_exps.rows);
    }

    if (impl_->has_mhc) {
        impl_->b_streams_a = impl_->alloc_buf(Tmax * Mhc * D * sizeof(float));
        impl_->b_streams_b = impl_->alloc_buf(Tmax * Mhc * D * sizeof(float));
    }

    impl_->b_hidden = impl_->alloc_buf(Tmax * D * sizeof(float));
    impl_->b_norm = impl_->alloc_buf(Tmax * D * sizeof(float));
    impl_->b_q = impl_->alloc_buf(Tmax * Qdim * sizeof(float));
    impl_->b_k = impl_->alloc_buf(Tmax * KVdim * sizeof(float));
    impl_->b_v = impl_->alloc_buf(Tmax * KVdim * sizeof(float));
    impl_->b_attn_out = impl_->alloc_buf(Tmax * Qdim * sizeof(float));
    impl_->b_sub_out = impl_->alloc_buf(Tmax * D * sizeof(float));
    impl_->b_router_logits = impl_->alloc_buf(Tmax * (size_t)std::max(1, cfg.n_routed_experts) * sizeof(float));
    impl_->b_topk_ids = impl_->alloc_buf(Tmax * (size_t)std::max(1, cfg.num_experts_per_tok) * sizeof(int32_t));
    impl_->b_topk_w = impl_->alloc_buf(Tmax * (size_t)std::max(1, cfg.num_experts_per_tok) * sizeof(float));
    impl_->b_ffn_gate = impl_->alloc_buf(Tmax * Inter * sizeof(float));
    impl_->b_ffn_up = impl_->alloc_buf(Tmax * Inter * sizeof(float));
    impl_->b_ffn_act = impl_->alloc_buf(Tmax * Inter * sizeof(float));
    impl_->b_ffn_tmp = impl_->alloc_buf(Tmax * D * sizeof(float));
    impl_->b_logits = impl_->alloc_buf((size_t)cfg.vocab_size * sizeof(float));
    impl_->b_dummy_bias = impl_->alloc_buf((size_t)std::max(1, cfg.n_routed_experts) * sizeof(float));

    fprintf(stderr, "[warpgrid] Loaded DeepSeek (%s): %d layers (%d enc / %d dec), D=%d, E=%d (top-%d), slots=%d, max_seq=%d, engram=%s\n",
            cfg.arch.c_str(), cfg.num_hidden_layers, cfg.encoder_layers, cfg.num_hidden_layers - cfg.encoder_layers,
            cfg.hidden_size, cfg.n_routed_experts, cfg.num_experts_per_tok, max_slots, max_seq_len,
            engram.is_open() ? "ON" : "OFF");
}

void DeepSeekV41Model::reset_cache(int slot_id) {
    if (slot_id < 0 || slot_id >= max_slots) return;
    if ((size_t)slot_id < impl_->slot_tok_history.size()) {
        impl_->slot_tok_history[(size_t)slot_id].clear();
    }
    size_t slot_bytes = (size_t)max_seq_len * (size_t)cfg.num_key_value_heads * (size_t)cfg.head_dim * sizeof(float);
    for (auto& kv : impl_->kv_caches) {
        memset(static_cast<uint8_t*>(kv.k_cache.contents) + (size_t)slot_id * slot_bytes, 0, slot_bytes);
        memset(static_cast<uint8_t*>(kv.v_cache.contents) + (size_t)slot_id * slot_bytes, 0, slot_bytes);
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
    float* hidden_ptr = static_cast<float*>(impl_->b_hidden.contents);
    const auto& emb_ti = gguf.tensor("token_embd.weight");
    size_t row_stride = ggml_row_nbytes(emb_ti.type, emb_ti.dims[0]);
    const uint8_t* base = gguf.tensor_data("token_embd.weight");
    uint64_t nrows = emb_ti.num_rows();
    for (int t = 0; t < T; ++t) {
        uint64_t tid = (uint64_t)std::clamp(h_tokens[t], 0, (int32_t)(nrows > 0 ? nrows - 1 : 0));
        dequant_row_fp32(emb_ti.type, base + tid * row_stride, hidden_ptr + (size_t)t * D, D);
    }

    if (impl_->has_mhc) {
        id<MTLCommandBuffer> cb0 = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> e0 = [cb0 computeCommandEncoder];
        DsStreamCollapseParams scp{(uint32_t)T, (uint32_t)D, (uint32_t)std::max(1, cfg.hc_mult)};
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

    size_t slot_kv_elems = (size_t)max_seq_len * (size_t)cfg.num_key_value_heads * (size_t)cfg.head_dim;
    size_t kv_tok_elems = (size_t)cfg.num_key_value_heads * (size_t)cfg.head_dim;
    size_t slot_kv_offset_bytes = (size_t)slot_id * slot_kv_elems * sizeof(float);

    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
        const auto& lw = impl_->layers[(size_t)l];
        auto& kv = impl_->kv_caches[(size_t)l];

        if (engram.is_open() &&
            std::find(cfg.engram_layers.begin(), cfg.engram_layers.end(), l) != cfg.engram_layers.end()) {
            std::string ename = "blk." + std::to_string(l) + ".engram.weight";
            if (engram.has_tensor(ename)) {
                const auto& eti = engram.tensor_info(ename);
                uint64_t erows = eti.num_rows();
                std::vector<float> erow(D, 0.0f);
                for (int t = 0; t < T; ++t) {
                    int abs_pos = start_pos + t;
                    int ng_len = std::min(3, abs_pos + 1);
                    uint64_t ridx = engram_hash_row(hist.data() + (abs_pos - ng_len + 1), ng_len, (uint64_t)(l + 1), erows);
                    engram.lookup_row(ename, ridx, erow.data(), D);
                    for (size_t d = 0; d < D; ++d) hidden_ptr[(size_t)t * D + d] += erow[d];
                    if (impl_->has_mhc) {
                        float* sp = static_cast<float*>(impl_->b_streams_a.contents);
                        size_t Mhc = (size_t)std::max(1, cfg.hc_mult);
                        for (size_t r = 0; r < Mhc; ++r) {
                            for (size_t d = 0; d < D; ++d) {
                                sp[((size_t)t * Mhc + r) * D + d] += erow[d];
                            }
                        }
                    }
                }
            }
        }

        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        impl_->encode_rmsnorm(enc, impl_->b_hidden, lw.attn_norm, impl_->b_norm, (uint32_t)T, (uint32_t)D, cfg.rms_norm_eps);
        impl_->encode_gemv(enc, impl_->b_norm, lw.wq, impl_->b_q, (uint32_t)T);
        impl_->encode_gemv(enc, impl_->b_norm, lw.wk, impl_->b_k, (uint32_t)T);
        impl_->encode_gemv(enc, impl_->b_norm, lw.wv, impl_->b_v, (uint32_t)T);

        DsRopeParams rq{(uint32_t)T, (uint32_t)start_pos, (uint32_t)cfg.num_attention_heads, (uint32_t)cfg.head_dim, cfg.rope_freq_base};
        [enc setComputePipelineState:impl_->pso("ds_rope_f32")];
        [enc setBuffer:impl_->b_q offset:0 atIndex:0];
        [enc setBytes:&rq length:sizeof(rq) atIndex:1];
        [enc dispatchThreads:MTLSizeMake((uint32_t)(cfg.head_dim / 2), (uint32_t)cfg.num_attention_heads, (uint32_t)T)
            threadsPerThreadgroup:MTLSizeMake(std::min(32, std::max(1, cfg.head_dim / 2)), 1, 1)];

        DsRopeParams rk{(uint32_t)T, (uint32_t)start_pos, (uint32_t)cfg.num_key_value_heads, (uint32_t)cfg.head_dim, cfg.rope_freq_base};
        [enc setBuffer:impl_->b_k offset:0 atIndex:0];
        [enc setBytes:&rk length:sizeof(rk) atIndex:1];
        [enc dispatchThreads:MTLSizeMake((uint32_t)(cfg.head_dim / 2), (uint32_t)cfg.num_key_value_heads, (uint32_t)T)
            threadsPerThreadgroup:MTLSizeMake(std::min(32, std::max(1, cfg.head_dim / 2)), 1, 1)];

        [enc endEncoding];

        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        size_t kv_copy_bytes = (size_t)T * kv_tok_elems * sizeof(float);
        size_t kv_dst_off = slot_kv_offset_bytes + (size_t)start_pos * kv_tok_elems * sizeof(float);
        [blit copyFromBuffer:impl_->b_k sourceOffset:0 toBuffer:kv.k_cache destinationOffset:kv_dst_off size:kv_copy_bytes];
        [blit copyFromBuffer:impl_->b_v sourceOffset:0 toBuffer:kv.v_cache destinationOffset:kv_dst_off size:kv_copy_bytes];
        [blit endEncoding];

        enc = [cb computeCommandEncoder];
        DsAttnParams ap{
            (uint32_t)T, (uint32_t)start_pos,
            (uint32_t)cfg.num_attention_heads, (uint32_t)cfg.num_key_value_heads,
            (uint32_t)cfg.head_dim, (uint32_t)max_seq_len,
            (uint32_t)std::max(1, cfg.compress_ratio_for(l)),
            (uint32_t)std::max(1, cfg.index_topk),
            1.0f / std::sqrt((float)cfg.head_dim)
        };
        [enc setComputePipelineState:impl_->pso("ds_csa_hca_attn_f32")];
        [enc setBuffer:impl_->b_q offset:0 atIndex:0];
        [enc setBuffer:kv.k_cache offset:slot_kv_offset_bytes atIndex:1];
        [enc setBuffer:kv.v_cache offset:slot_kv_offset_bytes atIndex:2];
        [enc setBuffer:impl_->b_attn_out offset:0 atIndex:3];
        [enc setBytes:&ap length:sizeof(ap) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((uint32_t)cfg.num_attention_heads, (uint32_t)T, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        impl_->encode_gemv(enc, impl_->b_attn_out, lw.wo, impl_->b_sub_out, (uint32_t)T);
        if (lw.hc_attn_mat != nil && impl_->has_mhc) {
            DsMhcParams mp{(uint32_t)T, (uint32_t)D, (uint32_t)std::max(1, cfg.hc_mult), (uint32_t)cfg.hc_sinkhorn_iters, cfg.hc_eps};
            [enc setComputePipelineState:impl_->pso("ds_mhc_sinkhorn_mix")];
            [enc setBuffer:impl_->b_streams_a offset:0 atIndex:0];
            [enc setBuffer:impl_->b_sub_out offset:0 atIndex:1];
            [enc setBuffer:lw.hc_attn_mat offset:0 atIndex:2];
            [enc setBuffer:lw.hc_attn_post offset:0 atIndex:3];
            [enc setBuffer:impl_->b_streams_b offset:0 atIndex:4];
            [enc setBytes:&mp length:sizeof(mp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(((uint32_t)D + 63u) / 64u, (uint32_t)T, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            std::swap(impl_->b_streams_a, impl_->b_streams_b);
            DsStreamCollapseParams scp{(uint32_t)T, (uint32_t)D, (uint32_t)std::max(1, cfg.hc_mult)};
            [enc setComputePipelineState:impl_->pso("ds_mhc_collapse_f32")];
            [enc setBuffer:impl_->b_streams_a offset:0 atIndex:0];
            [enc setBuffer:impl_->b_hidden offset:0 atIndex:1];
            [enc setBytes:&scp length:sizeof(scp) atIndex:2];
            [enc dispatchThreads:MTLSizeMake((uint32_t)D, (uint32_t)T, 1)
                threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(64u, (uint32_t)D), 1, 1)];
        } else {
            impl_->encode_add(enc, impl_->b_hidden, impl_->b_sub_out, (uint32_t)((size_t)T * D));
            if (impl_->has_mhc) {
                DsStreamCollapseParams scp{(uint32_t)T, (uint32_t)D, (uint32_t)std::max(1, cfg.hc_mult)};
                [enc setComputePipelineState:impl_->pso("ds_mhc_expand_f32")];
                [enc setBuffer:impl_->b_hidden offset:0 atIndex:0];
                [enc setBuffer:impl_->b_streams_a offset:0 atIndex:1];
                [enc setBytes:&scp length:sizeof(scp) atIndex:2];
                [enc dispatchThreads:MTLSizeMake((uint32_t)D, (uint32_t)T, 1)
                    threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(64u, (uint32_t)D), 1, 1)];
            }
        }

        impl_->encode_rmsnorm(enc, impl_->b_hidden, lw.ffn_norm, impl_->b_norm, (uint32_t)T, (uint32_t)D, cfg.rms_norm_eps);

        if (lw.ffn_gate_dense.valid() && lw.ffn_up_dense.valid() && lw.ffn_down_dense.valid()) {
            impl_->encode_gemv(enc, impl_->b_norm, lw.ffn_gate_dense, impl_->b_ffn_gate, (uint32_t)T);
            impl_->encode_gemv(enc, impl_->b_norm, lw.ffn_up_dense, impl_->b_ffn_up, (uint32_t)T);
            impl_->encode_swiglu(enc, impl_->b_ffn_gate, impl_->b_ffn_up, impl_->b_ffn_act, (uint32_t)((size_t)T * lw.ffn_gate_dense.rows));
            impl_->encode_gemv(enc, impl_->b_ffn_act, lw.ffn_down_dense, impl_->b_sub_out, (uint32_t)T);
        } else {
            uint32_t sub_n = (uint32_t)((size_t)T * D);
            [enc setComputePipelineState:impl_->pso("ds_zero_f32")];
            [enc setBuffer:impl_->b_sub_out offset:0 atIndex:0];
            [enc setBytes:&sub_n length:sizeof(sub_n) atIndex:1];
            [enc dispatchThreadgroups:MTLSizeMake((sub_n + 255u) / 256u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            if (lw.ffn_gate_shexp.valid() && lw.ffn_up_shexp.valid() && lw.ffn_down_shexp.valid()) {
                impl_->encode_gemv(enc, impl_->b_norm, lw.ffn_gate_shexp, impl_->b_ffn_gate, (uint32_t)T);
                impl_->encode_gemv(enc, impl_->b_norm, lw.ffn_up_shexp, impl_->b_ffn_up, (uint32_t)T);
                impl_->encode_swiglu(enc, impl_->b_ffn_gate, impl_->b_ffn_up, impl_->b_ffn_act, (uint32_t)((size_t)T * lw.ffn_gate_shexp.rows));
                impl_->encode_gemv(enc, impl_->b_ffn_act, lw.ffn_down_shexp, impl_->b_sub_out, (uint32_t)T);
            }
            if (lw.router.valid() && lw.ffn_gate_exps.valid() && lw.ffn_up_exps.valid() && lw.ffn_down_exps.valid()) {
                impl_->encode_gemv(enc, impl_->b_norm, lw.router, impl_->b_router_logits, (uint32_t)T);
                DsRouterParams rp{
                    (uint32_t)T, (uint32_t)cfg.n_routed_experts, (uint32_t)cfg.num_experts_per_tok,
                    lw.router_bias != nil ? 1u : 0u, cfg.route_scale
                };
                [enc setComputePipelineState:impl_->pso("ds_router_topk_f32")];
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
                        if (e < 0 || e >= (int)lw.ffn_gate_exps.experts) continue;
                        size_t g_off = (size_t)e * lw.ffn_gate_exps.expert_stride_bytes;
                        size_t u_off = (size_t)e * lw.ffn_up_exps.expert_stride_bytes;
                        size_t d_off = (size_t)e * lw.ffn_down_exps.expert_stride_bytes;

                        GgufGemvParams gp{1u, lw.ffn_gate_exps.rows, lw.ffn_gate_exps.cols};
                        [enc setComputePipelineState:impl_->pso(Impl::gemv_kernel_for(lw.ffn_gate_exps.type))];
                        [enc setBuffer:impl_->b_norm offset:x_off atIndex:0];
                        [enc setBuffer:lw.ffn_gate_exps.buf offset:g_off atIndex:1];
                        [enc setBuffer:impl_->b_ffn_gate offset:0 atIndex:2];
                        [enc setBytes:&gp length:sizeof(gp) atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake((lw.ffn_gate_exps.rows + 3u) / 4u, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                        GgufGemvParams up{1u, lw.ffn_up_exps.rows, lw.ffn_up_exps.cols};
                        [enc setComputePipelineState:impl_->pso(Impl::gemv_kernel_for(lw.ffn_up_exps.type))];
                        [enc setBuffer:impl_->b_norm offset:x_off atIndex:0];
                        [enc setBuffer:lw.ffn_up_exps.buf offset:u_off atIndex:1];
                        [enc setBuffer:impl_->b_ffn_up offset:0 atIndex:2];
                        [enc setBytes:&up length:sizeof(up) atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake((lw.ffn_up_exps.rows + 3u) / 4u, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                        impl_->encode_swiglu(enc, impl_->b_ffn_gate, impl_->b_ffn_up, impl_->b_ffn_act, lw.ffn_gate_exps.rows);

                        GgufGemvParams dp{1u, lw.ffn_down_exps.rows, lw.ffn_down_exps.cols};
                        [enc setComputePipelineState:impl_->pso(Impl::gemv_kernel_for(lw.ffn_down_exps.type))];
                        [enc setBuffer:impl_->b_ffn_act offset:0 atIndex:0];
                        [enc setBuffer:lw.ffn_down_exps.buf offset:d_off atIndex:1];
                        [enc setBuffer:impl_->b_ffn_tmp offset:0 atIndex:2];
                        [enc setBytes:&dp length:sizeof(dp) atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake((lw.ffn_down_exps.rows + 3u) / 4u, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

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
            }
        }

        if (lw.hc_ffn_mat != nil && impl_->has_mhc) {
            DsMhcParams mp{(uint32_t)T, (uint32_t)D, (uint32_t)std::max(1, cfg.hc_mult), (uint32_t)cfg.hc_sinkhorn_iters, cfg.hc_eps};
            [enc setComputePipelineState:impl_->pso("ds_mhc_sinkhorn_mix")];
            [enc setBuffer:impl_->b_streams_a offset:0 atIndex:0];
            [enc setBuffer:impl_->b_sub_out offset:0 atIndex:1];
            [enc setBuffer:lw.hc_ffn_mat offset:0 atIndex:2];
            [enc setBuffer:lw.hc_ffn_post offset:0 atIndex:3];
            [enc setBuffer:impl_->b_streams_b offset:0 atIndex:4];
            [enc setBytes:&mp length:sizeof(mp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(((uint32_t)D + 63u) / 64u, (uint32_t)T, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            std::swap(impl_->b_streams_a, impl_->b_streams_b);
            DsStreamCollapseParams scp{(uint32_t)T, (uint32_t)D, (uint32_t)std::max(1, cfg.hc_mult)};
            [enc setComputePipelineState:impl_->pso("ds_mhc_collapse_f32")];
            [enc setBuffer:impl_->b_streams_a offset:0 atIndex:0];
            [enc setBuffer:impl_->b_hidden offset:0 atIndex:1];
            [enc setBytes:&scp length:sizeof(scp) atIndex:2];
            [enc dispatchThreads:MTLSizeMake((uint32_t)D, (uint32_t)T, 1)
                threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(64u, (uint32_t)D), 1, 1)];
        } else {
            impl_->encode_add(enc, impl_->b_hidden, impl_->b_sub_out, (uint32_t)((size_t)T * D));
            if (impl_->has_mhc) {
                DsStreamCollapseParams scp{(uint32_t)T, (uint32_t)D, (uint32_t)std::max(1, cfg.hc_mult)};
                [enc setComputePipelineState:impl_->pso("ds_mhc_expand_f32")];
                [enc setBuffer:impl_->b_hidden offset:0 atIndex:0];
                [enc setBuffer:impl_->b_streams_a offset:0 atIndex:1];
                [enc setBytes:&scp length:sizeof(scp) atIndex:2];
                [enc dispatchThreads:MTLSizeMake((uint32_t)D, (uint32_t)T, 1)
                    threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(64u, (uint32_t)D), 1, 1)];
            }
        }

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    impl_->encode_rmsnorm(enc, impl_->b_hidden, impl_->output_norm, impl_->b_norm, (uint32_t)T, (uint32_t)D, cfg.rms_norm_eps);
    size_t last_off = (size_t)(T - 1) * D * sizeof(float);
    if (impl_->lm_head.valid()) {
        GgufGemvParams lp{1u, impl_->lm_head.rows, impl_->lm_head.cols};
        [enc setComputePipelineState:impl_->pso(Impl::gemv_kernel_for(impl_->lm_head.type))];
        [enc setBuffer:impl_->b_norm offset:last_off atIndex:0];
        [enc setBuffer:impl_->lm_head.buf offset:0 atIndex:1];
        [enc setBuffer:impl_->b_logits offset:0 atIndex:2];
        [enc setBytes:&lp length:sizeof(lp) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((impl_->lm_head.rows + 3u) / 4u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
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
