#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "model_metal.hpp"
#ifdef WARPGRID_METAL_PRECOMPILED
#include "kernels_metallib_embed.hpp"
#else
#include "kernels_source_embed.hpp"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ink {

namespace {

struct RmsParams { uint32_t T; uint32_t D; float eps; };
struct SplitQrgsParams { uint32_t T; uint32_t q_dim; uint32_t r_dim; uint32_t n_heads; };
struct SplitKvParams { uint32_t T; uint32_t kv_dim; };
struct AttnResParams {
    uint32_t num_stack; uint32_t stack_stride; uint32_t has_addend;
    uint32_t has_norm_w; uint32_t T; uint32_t D; float eps;
};
struct CtrlParams { uint32_t T; uint32_t D; uint32_t rank; float eps; };
struct SconvStaticParams { uint32_t T; uint32_t C; uint32_t has_snap; };
struct SconvDynamicParams { uint32_t T; uint32_t num_kv_heads; uint32_t head_dim; uint32_t has_snap; };
struct AttnPrepQRParams {
    uint32_t start_pos; uint32_t T; uint32_t is_local; uint32_t n_heads;
    uint32_t head_dim; uint32_t d_rel; float qk_clip; float rel_clip;
    float log_n_floor; float log_alpha;
};
struct AttnPrepKVParams {
    uint32_t start_pos; uint32_t T; uint32_t cache_cap; uint32_t is_local;
    uint32_t n_kv_heads; uint32_t head_dim; float qk_clip; float eps;
};
struct RelMatmulParams { uint32_t M; uint32_t d_rel; uint32_t rel_extent; };
struct AttnParams {
    uint32_t start_pos; uint32_t T; uint32_t cache_cap; uint32_t is_local;
    uint32_t window_size; uint32_t n_heads; uint32_t n_kv_heads;
    uint32_t head_dim; uint32_t rel_extent;
};
struct RouterParams { uint32_t T; uint32_t D; uint32_t E; uint32_t K; float route_scale; };
struct MoeParams { uint32_t T; uint32_t D; uint32_t inter; uint32_t K; float gate_cap; float up_cap; };
struct GemvParams { uint32_t M; uint32_t N; uint32_t K; uint32_t out_fp32; };
struct QuantParams { uint32_t N; uint32_t K; };
struct ArgmaxParams { uint32_t vocab_size; float softcap; };

struct MetalLayerWeights {
    bool is_local = true;
    bool is_loop_core = false;
    bool is_mxfp8 = false;

    id<MTLBuffer> attn_norm = nil;
    id<MTLBuffer> mlp_norm = nil;
    id<MTLBuffer> w_qrgs = nil;
    id<MTLBuffer> w_kv = nil;
    id<MTLBuffer> wo = nil;
    id<MTLBuffer> w_qrgs_fp8 = nil;
    id<MTLBuffer> w_qrgs_sc = nil;
    id<MTLBuffer> w_kv_fp8 = nil;
    id<MTLBuffer> w_kv_sc = nil;
    id<MTLBuffer> wo_fp8 = nil;
    id<MTLBuffer> wo_sc = nil;

    id<MTLBuffer> q_clip_scale = nil;
    id<MTLBuffer> rel_clip_scale = nil;
    id<MTLBuffer> rel_proj = nil;

    id<MTLBuffer> k_sconv_w = nil;
    id<MTLBuffer> k_sconv_proj = nil;
    id<MTLBuffer> v_sconv_w = nil;
    id<MTLBuffer> v_sconv_proj = nil;
    id<MTLBuffer> attn_sconv_w = nil;
    id<MTLBuffer> mlp_sconv_w = nil;

    id<MTLBuffer> ctrl_down[4] = {nil, nil, nil, nil};
    id<MTLBuffer> ctrl_gain[4] = {nil, nil, nil, nil};
    id<MTLBuffer> ctrl_forget[4] = {nil, nil, nil, nil};

    id<MTLBuffer> gate_weight = nil;
    id<MTLBuffer> gate_bias = nil;

    id<MTLBuffer> w13_weight = nil;
    id<MTLBuffer> w13_scale = nil;
    id<MTLBuffer> w13_amax = nil;
    id<MTLBuffer> w13_denom = nil;
    id<MTLBuffer> w2_weight = nil;
    id<MTLBuffer> w2_scale = nil;
    id<MTLBuffer> w2_amax = nil;
    id<MTLBuffer> w2_denom = nil;
};

struct MetalVisitCache {
    bool is_local = true;
    int cap = 0;
    id<MTLBuffer> k_cache = nil;
    id<MTLBuffer> v_cache = nil;
    id<MTLBuffer> k_cache_fp8 = nil;
    id<MTLBuffer> v_cache_fp8 = nil;
    id<MTLBuffer> k_scale_fp8 = nil;
    id<MTLBuffer> v_scale_fp8 = nil;
    id<MTLBuffer> k_conv_state = nil;
    id<MTLBuffer> v_conv_state = nil;
    id<MTLBuffer> attn_conv_state = nil;
    id<MTLBuffer> mlp_conv_state = nil;
    id<MTLBuffer> local_k_prefill_snap = nil;
    id<MTLBuffer> local_v_prefill_snap = nil;
    id<MTLBuffer> k_conv_prefill_snap = nil;
    id<MTLBuffer> v_conv_prefill_snap = nil;
    id<MTLBuffer> attn_conv_prefill_snap = nil;
    id<MTLBuffer> mlp_conv_prefill_snap = nil;
};

}

struct Inkling2Model::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;

    id<MTLComputePipelineState> pso_embed_norm = nil;
    id<MTLComputePipelineState> pso_rmsnorm = nil;
    id<MTLComputePipelineState> pso_add_bf16 = nil;
    id<MTLComputePipelineState> pso_copy_bf16 = nil;
    id<MTLComputePipelineState> pso_split_qrgs = nil;
    id<MTLComputePipelineState> pso_split_kv = nil;
    id<MTLComputePipelineState> pso_attn_res = nil;
    id<MTLComputePipelineState> pso_pair_rms_norm = nil;
    id<MTLComputePipelineState> pso_fused_ctrl = nil;
    id<MTLComputePipelineState> pso_ctrl_gf_small_t = nil;
    id<MTLComputePipelineState> pso_fp32_rmsnorm = nil;
    id<MTLComputePipelineState> pso_sconv_static = nil;
    id<MTLComputePipelineState> pso_sconv_dynamic = nil;
    id<MTLComputePipelineState> pso_attn_prep_qr = nil;
    id<MTLComputePipelineState> pso_attn_prep_kv = nil;
    id<MTLComputePipelineState> pso_attn_prep_kv_fp8 = nil;
    id<MTLComputePipelineState> pso_rel_matmul = nil;
    id<MTLComputePipelineState> pso_attention = nil;
    id<MTLComputePipelineState> pso_attention_fp8 = nil;
    id<MTLComputePipelineState> pso_router_logits = nil;
    id<MTLComputePipelineState> pso_router_topk = nil;
    id<MTLComputePipelineState> pso_moe_nvfp4_w13 = nil;
    id<MTLComputePipelineState> pso_moe_nvfp4_w2 = nil;
    id<MTLComputePipelineState> pso_moe_mxfp8_w13 = nil;
    id<MTLComputePipelineState> pso_moe_mxfp8_w2 = nil;
    id<MTLComputePipelineState> pso_bf16_gemv = nil;
    id<MTLComputePipelineState> pso_bf16_gemv_1sg = nil;
    id<MTLComputePipelineState> pso_quant_fp8 = nil;
    id<MTLComputePipelineState> pso_fp8_gemv = nil;
    id<MTLComputePipelineState> pso_bf16_gemm_tiled = nil;
    id<MTLComputePipelineState> pso_softcap_argmax = nil;

    id<MTLBuffer> embed_w = nil;
    id<MTLBuffer> embed_norm_w = nil;
    id<MTLBuffer> final_norm_w = nil;
    id<MTLBuffer> unembed_w = nil;
    id<MTLBuffer> unembed_fp8 = nil;
    id<MTLBuffer> unembed_sc = nil;

    std::vector<id<MTLBuffer>> attn_queries;
    std::vector<id<MTLBuffer>> mlp_queries;
    id<MTLBuffer> output_query = nil;
    id<MTLBuffer> message_queries = nil;

    std::vector<MetalLayerWeights> layers;
    std::vector<std::pair<int, int>> visits;
    std::vector<MetalVisitCache> caches;

    id<MTLBuffer> b_dummy = nil;
    id<MTLBuffer> b_tokens = nil;
    id<MTLBuffer> b_stack = nil;
    id<MTLBuffer> b_prelude_part = nil;
    id<MTLBuffer> b_bank = nil;
    id<MTLBuffer> b_prelude_dest = nil;
    id<MTLBuffer> b_messages = nil;
    id<MTLBuffer> b_fp32_mix = nil;
    id<MTLBuffer> b_attn_in = nil;
    id<MTLBuffer> b_mlp_in = nil;
    id<MTLBuffer> b_pair = nil;
    id<MTLBuffer> b_ctrl_h = nil;
    id<MTLBuffer> b_ctrl_tmp = nil;
    id<MTLBuffer> b_qrgs = nil;
    id<MTLBuffer> b_kv_out = nil;
    id<MTLBuffer> b_Q = nil;
    id<MTLBuffer> b_K = nil;
    id<MTLBuffer> b_V = nil;
    id<MTLBuffer> b_K_conv = nil;
    id<MTLBuffer> b_V_conv = nil;
    id<MTLBuffer> b_dyn_w = nil;
    id<MTLBuffer> b_R = nil;
    id<MTLBuffer> b_rel_logits = nil;
    id<MTLBuffer> b_gate = nil;
    id<MTLBuffer> b_sinks = nil;
    id<MTLBuffer> b_attn_raw = nil;
    id<MTLBuffer> b_attn_proj = nil;
    id<MTLBuffer> b_attn_out = nil;
    id<MTLBuffer> b_mlp_raw = nil;
    id<MTLBuffer> b_mlp_out = nil;
    id<MTLBuffer> b_topk_ids = nil;
    id<MTLBuffer> b_topk_weights = nil;
    id<MTLBuffer> b_router_logits = nil;
    id<MTLBuffer> b_moe_act = nil;
    id<MTLBuffer> b_logits = nil;
    id<MTLBuffer> b_out_token = nil;

    int dspark_slots = 0;
    double gpu_ws_gib = 0.0;
    uint32_t gpu_max_tg = 256;
    bool gpu_is_apple9 = false;
    bool gpu_low_mem = false;
    double prof_attn_res_ctrl_ms = 0.0;
    double prof_dense_proj_ms = 0.0;
    double prof_attn_sconv_ms = 0.0;
    double prof_router_ms = 0.0;
    double prof_moe_nvfp4_ms = 0.0;
    double prof_moe_mxfp8_ms = 0.0;
    double prof_lm_head_ms = 0.0;
    int prof_steps = 0;

    id<MTLBuffer> alloc_shared(size_t bytes) {
        size_t sz = std::max<size_t>(16, bytes);
        id<MTLBuffer> b = [device newBufferWithLength:sz options:MTLResourceStorageModeShared];
        if (!b) {
            throw std::runtime_error("Metal newBufferWithLength failed (" + std::to_string(sz) + " bytes)");
        }
        return b;
    }

    id<MTLBuffer> load_tensor(const SafetensorsFile& st, const std::string& name) {
        const auto& info = st.info(name);
        id<MTLBuffer> b = alloc_shared(info.nbytes);
        st.read_into(name, b.contents, info.nbytes);
        return b;
    }

    id<MTLComputePipelineState> make_pso(const char* name) {
        id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (!fn) throw std::runtime_error(std::string("Missing Metal kernel: ") + name);
        NSError* err = nil;
        id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!p) {
            throw std::runtime_error(std::string("PSO error for ") + name + ": " +
                                     (err ? err.localizedDescription.UTF8String : "unknown"));
        }
        return p;
    }

    void commit_and_wait(id<MTLCommandBuffer> cb) {
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted || cb.error != nil) {
            throw std::runtime_error(std::string("Metal command buffer failed: ") +
                                     (cb.error ? cb.error.localizedDescription.UTF8String : "unknown"));
        }
    }

    void encode_linear_bf16(id<MTLComputeCommandEncoder> enc,
                            id<MTLBuffer> X, size_t x_off,
                            id<MTLBuffer> W, size_t w_off,
                            id<MTLBuffer> Y_bf, size_t y_bf_off,
                            id<MTLBuffer> Y_f32, size_t y_f32_off,
                            uint32_t M, uint32_t N, uint32_t K, bool out_fp32)
    {
        if (!out_fp32 && M >= 16u && (K & 31u) == 0u) {
            GemvParams gp{M, N, K, 0u};
            [enc setComputePipelineState:pso_bf16_gemm_tiled];
            [enc setBuffer:X offset:x_off atIndex:0];
            [enc setBuffer:W offset:w_off atIndex:1];
            [enc setBuffer:Y_bf offset:y_bf_off atIndex:2];
            [enc setBytes:&gp length:sizeof(gp) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((N + 31u) / 32u, (M + 31u) / 32u, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            return;
        }
        GemvParams gp{M, N, K, out_fp32 ? 1u : 0u};
        if (N <= 128u && (K & 3u) == 0u) {
            [enc setComputePipelineState:pso_bf16_gemv_1sg];
            [enc setBuffer:X offset:x_off atIndex:0];
            [enc setBuffer:W offset:w_off atIndex:1];
            [enc setBuffer:(Y_bf ? Y_bf : b_dummy) offset:y_bf_off atIndex:2];
            [enc setBuffer:(Y_f32 ? Y_f32 : b_dummy) offset:y_f32_off atIndex:3];
            [enc setBytes:&gp length:sizeof(gp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(N, M, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            return;
        }
        [enc setComputePipelineState:pso_bf16_gemv];
        [enc setBuffer:X offset:x_off atIndex:0];
        [enc setBuffer:W offset:w_off atIndex:1];
        [enc setBuffer:(Y_bf ? Y_bf : b_dummy) offset:y_bf_off atIndex:2];
        [enc setBuffer:(Y_f32 ? Y_f32 : b_dummy) offset:y_f32_off atIndex:3];
        [enc setBytes:&gp length:sizeof(gp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((N + 3u) / 4u, M, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

    void encode_linear_fp8(id<MTLComputeCommandEncoder> enc,
                           id<MTLBuffer> X, size_t x_off,
                           id<MTLBuffer> W8, size_t w8_off,
                           id<MTLBuffer> Wsc, size_t wsc_off,
                           id<MTLBuffer> Y_bf, size_t y_bf_off,
                           id<MTLBuffer> Y_f32, size_t y_f32_off,
                           uint32_t M, uint32_t N, uint32_t K, bool out_fp32)
    {
        GemvParams gp{M, N, K, out_fp32 ? 1u : 0u};
        [enc setComputePipelineState:pso_fp8_gemv];
        [enc setBuffer:X offset:x_off atIndex:0];
        [enc setBuffer:W8 offset:w8_off atIndex:1];
        [enc setBuffer:Wsc offset:wsc_off atIndex:2];
        [enc setBuffer:(Y_bf ? Y_bf : b_dummy) offset:y_bf_off atIndex:3];
        [enc setBuffer:(Y_f32 ? Y_f32 : b_dummy) offset:y_f32_off atIndex:4];
        [enc setBytes:&gp length:sizeof(gp) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((N + 3u) / 4u, M, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

    void quantize_rowwise_fp8(id<MTLComputeCommandEncoder> enc,
                              id<MTLBuffer> W_bf16, id<MTLBuffer> W_fp8, id<MTLBuffer> W_sc,
                              uint32_t N, uint32_t K)
    {
        QuantParams qp{N, K};
        [enc setComputePipelineState:pso_quant_fp8];
        [enc setBuffer:W_bf16 offset:0 atIndex:0];
        [enc setBuffer:W_fp8 offset:0 atIndex:1];
        [enc setBuffer:W_sc offset:0 atIndex:2];
        [enc setBytes:&qp length:sizeof(qp) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    void encode_attn_res(id<MTLComputeCommandEncoder> enc,
                         id<MTLBuffer> stack, uint32_t num_stack, uint32_t stack_stride,
                         id<MTLBuffer> last, size_t last_off,
                         id<MTLBuffer> addend, size_t add_off,
                         id<MTLBuffer> queries, size_t q_off, uint32_t num_q,
                         id<MTLBuffer> norm_w, size_t nw_off,
                         id<MTLBuffer> out_bf, size_t obf_off,
                         id<MTLBuffer> out_f32, size_t of32_off,
                         uint32_t T, uint32_t D, float eps)
    {
        AttnResParams ap{num_stack, stack_stride, addend ? 1u : 0u, norm_w ? 1u : 0u, T, D, eps};
        [enc setComputePipelineState:pso_attn_res];
        [enc setBuffer:stack offset:0 atIndex:0];
        [enc setBuffer:last offset:last_off atIndex:1];
        [enc setBuffer:(addend ? addend : b_dummy) offset:add_off atIndex:2];
        [enc setBuffer:queries offset:q_off atIndex:3];
        [enc setBuffer:(norm_w ? norm_w : b_dummy) offset:nw_off atIndex:4];
        [enc setBuffer:(out_bf ? out_bf : b_dummy) offset:obf_off atIndex:5];
        [enc setBuffer:(out_f32 ? out_f32 : b_dummy) offset:of32_off atIndex:6];
        [enc setBytes:&ap length:sizeof(ap) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(num_q, T, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    void encode_controller(id<MTLComputeCommandEncoder> enc,
                           const MetalLayerWeights& lw, int s,
                           id<MTLBuffer> dest, size_t dest_off,
                           id<MTLBuffer> msg, size_t msg_off,
                           id<MTLBuffer> norm_w, size_t nw_off,
                           id<MTLBuffer> out, size_t out_off,
                           uint32_t T, uint32_t D, uint32_t rank, float eps)
    {
        RmsParams rp{T, D, 1e-6f};
        [enc setComputePipelineState:pso_pair_rms_norm];
        [enc setBuffer:dest offset:dest_off atIndex:0];
        [enc setBuffer:msg offset:msg_off atIndex:1];
        [enc setBuffer:b_pair offset:0 atIndex:2];
        [enc setBytes:&rp length:sizeof(rp) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(T, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        encode_linear_bf16(enc, b_pair, 0, lw.ctrl_down[s], 0, b_ctrl_h, 0, nil, 0, T, rank, 2u * D, false);

        CtrlParams cp{T, D, rank, eps};
        if (T <= 8u && rank == 64u) {
            [enc setComputePipelineState:pso_ctrl_gf_small_t];
            [enc setBuffer:b_ctrl_h offset:0 atIndex:0];
            [enc setBuffer:lw.ctrl_gain[s] offset:0 atIndex:1];
            [enc setBuffer:lw.ctrl_forget[s] offset:0 atIndex:2];
            [enc setBuffer:dest offset:dest_off atIndex:3];
            [enc setBuffer:msg offset:msg_off atIndex:4];
            [enc setBuffer:b_ctrl_tmp offset:0 atIndex:5];
            [enc setBytes:&cp length:sizeof(cp) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((D + 3u) / 4u, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            RmsParams nrp{T, D, eps};
            [enc setComputePipelineState:pso_fp32_rmsnorm];
            [enc setBuffer:b_ctrl_tmp offset:0 atIndex:0];
            [enc setBuffer:norm_w offset:nw_off atIndex:1];
            [enc setBuffer:out offset:out_off atIndex:2];
            [enc setBytes:&nrp length:sizeof(nrp) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(T, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [enc setComputePipelineState:pso_fused_ctrl];
            [enc setBuffer:b_ctrl_h offset:0 atIndex:0];
            [enc setBuffer:lw.ctrl_gain[s] offset:0 atIndex:1];
            [enc setBuffer:lw.ctrl_forget[s] offset:0 atIndex:2];
            [enc setBuffer:dest offset:dest_off atIndex:3];
            [enc setBuffer:msg offset:msg_off atIndex:4];
            [enc setBuffer:norm_w offset:nw_off atIndex:5];
            [enc setBuffer:out offset:out_off atIndex:6];
            [enc setBytes:&cp length:sizeof(cp) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(T, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    }

    void forward_internal(const InklingConfig& cfg, int max_batch_tokens,
                          const int32_t* h_tokens, int T, int start_pos, int slot_id, bool do_argmax,
                          bool profile_stages)
    {
        if (T <= 0 || T > max_batch_tokens) {
            throw std::runtime_error("Invalid T for Metal forward_internal");
        }
        memcpy(b_tokens.contents, h_tokens, (size_t)T * sizeof(int32_t));

        uint32_t D = (uint32_t)cfg.hidden_size;
        uint32_t stack_stride = (uint32_t)max_batch_tokens * D;
        size_t stack_stride_bytes = (size_t)stack_stride * 2;
        uint32_t q_dim = (uint32_t)(cfg.num_attention_heads * cfg.head_dim);
        uint32_t kv_dim = (uint32_t)(cfg.num_key_value_heads * cfg.head_dim);
        uint32_t r_dim = (uint32_t)(cfg.num_attention_heads * cfg.d_rel);
        uint32_t qrgs_dim = 2u * q_dim + r_dim + (uint32_t)cfg.num_attention_heads;

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        auto flush_stage = [&](double& acc) {
            if (!profile_stages) return;
            [enc endEncoding];
            commit_and_wait(cb);
            acc += (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            cb = [queue commandBuffer];
            enc = [cb computeCommandEncoder];
        };

        RmsParams erp{(uint32_t)T, D, cfg.rms_norm_eps};
        [enc setComputePipelineState:pso_embed_norm];
        [enc setBuffer:b_tokens offset:0 atIndex:0];
        [enc setBuffer:embed_w offset:0 atIndex:1];
        [enc setBuffer:embed_norm_w offset:0 atIndex:2];
        [enc setBuffer:b_stack offset:0 atIndex:3];
        [enc setBytes:&erp length:sizeof(erp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((uint32_t)T, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        auto blit_copy = [&](id<MTLBuffer> src, size_t src_off, id<MTLBuffer> dst, size_t dst_off, size_t nbytes) {
            uint32_t n_elem = (uint32_t)(nbytes / 2);
            [enc setComputePipelineState:pso_copy_bf16];
            [enc setBuffer:src offset:src_off atIndex:0];
            [enc setBuffer:dst offset:dst_off atIndex:1];
            [enc setBytes:&n_elem length:sizeof(n_elem) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(n_elem, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        };

        int num_blocks = 0;
        int prelude_blocks = 0;
        bool have_prelude_dest = false;

        for (size_t v = 0; v < visits.size(); ++v) {
            int i = visits[v].first;
            int pass_idx = visits[v].second;
            bool looped = (pass_idx > 0);
            const auto& lw = layers[i];
            auto& vc = caches[v];

            if (i == cfg.loop_core_start && !looped) {
                prelude_blocks = num_blocks;
                blit_copy(b_stack, (size_t)num_blocks * stack_stride_bytes, b_prelude_part, 0, (size_t)T * D * 2);
                have_prelude_dest = false;
            } else if (i == cfg.loop_core_start && looped) {
                blit_copy(b_stack, (size_t)(prelude_blocks + 1) * stack_stride_bytes, b_bank, 0, (size_t)T * D * 2);
                blit_copy(b_stack, (size_t)(prelude_blocks + 2) * stack_stride_bytes, b_bank, stack_stride_bytes, (size_t)T * D * 2);
                num_blocks = prelude_blocks;
                blit_copy(b_prelude_part, 0, b_stack, (size_t)num_blocks * stack_stride_bytes, (size_t)T * D * 2);
                if (!have_prelude_dest) {
                    encode_attn_res(enc, b_stack, (uint32_t)num_blocks, stack_stride,
                                    b_prelude_part, 0, nil, 0,
                                    attn_queries[i - 1], 0, 1,
                                    nil, 0, nil, 0, b_prelude_dest, 0,
                                    (uint32_t)T, D, cfg.rms_norm_eps);
                    have_prelude_dest = true;
                }
            }

            size_t partial_off = (size_t)num_blocks * stack_stride_bytes;
            size_t q_in_off = 0, k_in_off = 0, v_in_off = 0;

            if (looped) {
                size_t mq_off = (size_t)(i - cfg.loop_core_start) * 4 * D * sizeof(float);
                encode_attn_res(enc, b_bank, 1, stack_stride,
                                b_bank, stack_stride_bytes, nil, 0,
                                message_queries, mq_off, 4,
                                nil, 0, nil, 0, b_messages, 0,
                                (uint32_t)T, D, cfg.rms_norm_eps);

                id<MTLBuffer> dest_q_buf = b_prelude_dest;
                size_t dest_q_off = 0, dest_k_off = 0, dest_v_off = 0;
                if (i != cfg.loop_core_start) {
                    uint32_t nq = lw.is_local ? 1u : 3u;
                    encode_attn_res(enc, b_stack, (uint32_t)num_blocks, stack_stride,
                                    b_stack, partial_off, nil, 0,
                                    attn_queries[i - 1], 0, nq,
                                    nil, 0, nil, 0, b_fp32_mix, 0,
                                    (uint32_t)T, D, cfg.rms_norm_eps);
                    dest_q_buf = b_fp32_mix;
                    dest_q_off = 0;
                    dest_k_off = lw.is_local ? 0 : (size_t)T * D * sizeof(float);
                    dest_v_off = lw.is_local ? 0 : (size_t)2 * T * D * sizeof(float);
                }
                size_t nw_k_off = lw.is_local ? 0 : (size_t)D * 2;
                size_t nw_v_off = lw.is_local ? 0 : (size_t)2 * D * 2;
                size_t msg_stride = (size_t)T * D * sizeof(float);
                size_t in_stride = (size_t)T * D * 2;

                encode_controller(enc, lw, 0, dest_q_buf, dest_q_off, b_messages, 0 * msg_stride,
                                  lw.attn_norm, 0, b_attn_in, 0 * in_stride, (uint32_t)T, D, (uint32_t)cfg.loop_gate_rank, cfg.rms_norm_eps);
                encode_controller(enc, lw, 1, dest_q_buf, dest_k_off, b_messages, 1 * msg_stride,
                                  lw.attn_norm, nw_k_off, b_attn_in, 1 * in_stride, (uint32_t)T, D, (uint32_t)cfg.loop_gate_rank, cfg.rms_norm_eps);
                encode_controller(enc, lw, 2, dest_q_buf, dest_v_off, b_messages, 2 * msg_stride,
                                  lw.attn_norm, nw_v_off, b_attn_in, 2 * in_stride, (uint32_t)T, D, (uint32_t)cfg.loop_gate_rank, cfg.rms_norm_eps);
                q_in_off = 0;
                k_in_off = 1 * in_stride;
                v_in_off = 2 * in_stride;
            } else if (i == 0) {
                RmsParams rp{(uint32_t)T, D, cfg.rms_norm_eps};
                [enc setComputePipelineState:pso_rmsnorm];
                [enc setBuffer:b_stack offset:partial_off atIndex:0];
                [enc setBuffer:lw.attn_norm offset:0 atIndex:1];
                [enc setBuffer:b_attn_in offset:0 atIndex:2];
                [enc setBytes:&rp length:sizeof(rp) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((uint32_t)T, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                q_in_off = k_in_off = v_in_off = 0;
            } else {
                uint32_t nq = lw.is_local ? 1u : 3u;
                encode_attn_res(enc, b_stack, (uint32_t)num_blocks, stack_stride,
                                b_stack, partial_off, nil, 0,
                                attn_queries[i - 1], 0, nq,
                                lw.attn_norm, 0, b_attn_in, 0, nil, 0,
                                (uint32_t)T, D, cfg.rms_norm_eps);
                q_in_off = 0;
                k_in_off = lw.is_local ? 0 : (size_t)T * D * 2;
                v_in_off = lw.is_local ? 0 : (size_t)2 * T * D * 2;
            }
            flush_stage(prof_attn_res_ctrl_ms);

            if (T <= 8) {
                encode_linear_fp8(enc, b_attn_in, q_in_off, lw.w_qrgs_fp8, 0, lw.w_qrgs_sc, 0,
                                  b_qrgs, 0, nil, 0, (uint32_t)T, qrgs_dim, D, false);
            } else {
                encode_linear_bf16(enc, b_attn_in, q_in_off, lw.w_qrgs, 0, b_qrgs, 0, nil, 0, (uint32_t)T, qrgs_dim, D, false);
            }
            {
                SplitQrgsParams sqp{(uint32_t)T, q_dim, r_dim, (uint32_t)cfg.num_attention_heads};
                [enc setComputePipelineState:pso_split_qrgs];
                [enc setBuffer:b_qrgs offset:0 atIndex:0];
                [enc setBuffer:b_Q offset:0 atIndex:1];
                [enc setBuffer:b_R offset:0 atIndex:2];
                [enc setBuffer:b_gate offset:0 atIndex:3];
                [enc setBuffer:b_sinks offset:0 atIndex:4];
                [enc setBytes:&sqp length:sizeof(sqp) atIndex:5];
                [enc dispatchThreads:MTLSizeMake(qrgs_dim, (uint32_t)T, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            if (k_in_off == v_in_off) {
                if (T <= 8) {
                    encode_linear_fp8(enc, b_attn_in, k_in_off, lw.w_kv_fp8, 0, lw.w_kv_sc, 0,
                                      b_kv_out, 0, nil, 0, (uint32_t)T, 2u * kv_dim, D, false);
                } else {
                    encode_linear_bf16(enc, b_attn_in, k_in_off, lw.w_kv, 0, b_kv_out, 0, nil, 0, (uint32_t)T, 2u * kv_dim, D, false);
                }
                SplitKvParams skp{(uint32_t)T, kv_dim};
                [enc setComputePipelineState:pso_split_kv];
                [enc setBuffer:b_kv_out offset:0 atIndex:0];
                [enc setBuffer:b_K offset:0 atIndex:1];
                [enc setBuffer:b_V offset:0 atIndex:2];
                [enc setBytes:&skp length:sizeof(skp) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(2u * kv_dim, (uint32_t)T, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                if (T <= 8) {
                    encode_linear_fp8(enc, b_attn_in, k_in_off, lw.w_kv_fp8, 0, lw.w_kv_sc, 0,
                                      b_K, 0, nil, 0, (uint32_t)T, kv_dim, D, false);
                    encode_linear_fp8(enc, b_attn_in, v_in_off, lw.w_kv_fp8, (size_t)kv_dim * D, lw.w_kv_sc, (size_t)kv_dim * sizeof(float),
                                      b_V, 0, nil, 0, (uint32_t)T, kv_dim, D, false);
                } else {
                    encode_linear_bf16(enc, b_attn_in, k_in_off, lw.w_kv, 0, b_K, 0, nil, 0, (uint32_t)T, kv_dim, D, false);
                    encode_linear_bf16(enc, b_attn_in, v_in_off, lw.w_kv, (size_t)kv_dim * D * 2, b_V, 0, nil, 0, (uint32_t)T, kv_dim, D, false);
                }
            }
            flush_stage(prof_dense_proj_ms);

            size_t kv_slot_off = (size_t)slot_id * (size_t)vc.cap * kv_dim * (vc.is_local ? 2 : 1);
            size_t sc_slot_off = (size_t)slot_id * (size_t)vc.cap * (size_t)cfg.num_key_value_heads * sizeof(float);
            size_t k_cs_off = (size_t)slot_id * 3 * kv_dim * 2;
            size_t d_cs_off = (size_t)slot_id * 3 * D * 2;

            if (lw.is_local) {
                SconvStaticParams sp{(uint32_t)T, kv_dim, 0u};
                [enc setComputePipelineState:pso_sconv_static];
                [enc setBuffer:b_K offset:0 atIndex:0];
                [enc setBuffer:lw.k_sconv_w offset:0 atIndex:1];
                [enc setBuffer:vc.k_conv_state offset:k_cs_off atIndex:2];
                [enc setBuffer:b_K_conv offset:0 atIndex:3];
                [enc setBuffer:b_dummy offset:0 atIndex:4];
                [enc setBytes:&sp length:sizeof(sp) atIndex:5];
                [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc setBuffer:b_V offset:0 atIndex:0];
                [enc setBuffer:lw.v_sconv_w offset:0 atIndex:1];
                [enc setBuffer:vc.v_conv_state offset:k_cs_off atIndex:2];
                [enc setBuffer:b_V_conv offset:0 atIndex:3];
                [enc setBuffer:b_dummy offset:0 atIndex:4];
                [enc setBytes:&sp length:sizeof(sp) atIndex:5];
                [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                uint32_t dyn_dim = (uint32_t)cfg.num_key_value_heads * 4u;
                SconvDynamicParams dp{(uint32_t)T, (uint32_t)cfg.num_key_value_heads, (uint32_t)cfg.head_dim, 0u};

                encode_linear_bf16(enc, b_attn_in, k_in_off, lw.k_sconv_proj, 0, b_dyn_w, 0, nil, 0, (uint32_t)T, dyn_dim, D, false);
                [enc setComputePipelineState:pso_sconv_dynamic];
                [enc setBuffer:b_K offset:0 atIndex:0];
                [enc setBuffer:lw.k_sconv_w offset:0 atIndex:1];
                [enc setBuffer:b_dyn_w offset:0 atIndex:2];
                [enc setBuffer:vc.k_conv_state offset:k_cs_off atIndex:3];
                [enc setBuffer:b_K_conv offset:0 atIndex:4];
                [enc setBuffer:b_dummy offset:0 atIndex:5];
                [enc setBytes:&dp length:sizeof(dp) atIndex:6];
                [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                encode_linear_bf16(enc, b_attn_in, v_in_off, lw.v_sconv_proj, 0, b_dyn_w, 0, nil, 0, (uint32_t)T, dyn_dim, D, false);
                [enc setComputePipelineState:pso_sconv_dynamic];
                [enc setBuffer:b_V offset:0 atIndex:0];
                [enc setBuffer:lw.v_sconv_w offset:0 atIndex:1];
                [enc setBuffer:b_dyn_w offset:0 atIndex:2];
                [enc setBuffer:vc.v_conv_state offset:k_cs_off atIndex:3];
                [enc setBuffer:b_V_conv offset:0 atIndex:4];
                [enc setBuffer:b_dummy offset:0 atIndex:5];
                [enc setBytes:&dp length:sizeof(dp) atIndex:6];
                [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            {
                AttnPrepQRParams qrp{
                    (uint32_t)start_pos, (uint32_t)T, lw.is_local ? 1u : 0u,
                    (uint32_t)cfg.num_attention_heads, (uint32_t)cfg.head_dim, (uint32_t)cfg.d_rel,
                    cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold,
                    cfg.log_scaling_n_floor, cfg.log_scaling_alpha
                };
                [enc setComputePipelineState:pso_attn_prep_qr];
                [enc setBuffer:b_Q offset:0 atIndex:0];
                [enc setBuffer:b_R offset:0 atIndex:1];
                [enc setBuffer:lw.q_clip_scale offset:0 atIndex:2];
                [enc setBuffer:lw.rel_clip_scale offset:0 atIndex:3];
                [enc setBytes:&qrp length:sizeof(qrp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((uint32_t)cfg.num_attention_heads, (uint32_t)T, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                AttnPrepKVParams kvp{
                    (uint32_t)start_pos, (uint32_t)T, (uint32_t)vc.cap, lw.is_local ? 1u : 0u,
                    (uint32_t)cfg.num_key_value_heads, (uint32_t)cfg.head_dim,
                    cfg.qk_l2_clip_threshold, cfg.rms_norm_eps
                };
                if (vc.is_local) {
                    [enc setComputePipelineState:pso_attn_prep_kv];
                    [enc setBuffer:b_K_conv offset:0 atIndex:0];
                    [enc setBuffer:b_V_conv offset:0 atIndex:1];
                    [enc setBuffer:vc.k_cache offset:kv_slot_off atIndex:2];
                    [enc setBuffer:vc.v_cache offset:kv_slot_off atIndex:3];
                    [enc setBytes:&kvp length:sizeof(kvp) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake((uint32_t)cfg.num_key_value_heads, (uint32_t)T, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                } else {
                    [enc setComputePipelineState:pso_attn_prep_kv_fp8];
                    [enc setBuffer:b_K_conv offset:0 atIndex:0];
                    [enc setBuffer:b_V_conv offset:0 atIndex:1];
                    [enc setBuffer:vc.k_cache_fp8 offset:kv_slot_off atIndex:2];
                    [enc setBuffer:vc.v_cache_fp8 offset:kv_slot_off atIndex:3];
                    [enc setBuffer:vc.k_scale_fp8 offset:sc_slot_off atIndex:4];
                    [enc setBuffer:vc.v_scale_fp8 offset:sc_slot_off atIndex:5];
                    [enc setBytes:&kvp length:sizeof(kvp) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake((uint32_t)cfg.num_key_value_heads, (uint32_t)T, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                }
            }

            {
                RelMatmulParams rmp{(uint32_t)(T * cfg.num_attention_heads), (uint32_t)cfg.d_rel, (uint32_t)cfg.rel_extent};
                [enc setComputePipelineState:pso_rel_matmul];
                [enc setBuffer:b_R offset:0 atIndex:0];
                [enc setBuffer:lw.rel_proj offset:0 atIndex:1];
                [enc setBuffer:b_rel_logits offset:0 atIndex:2];
                [enc setBytes:&rmp length:sizeof(rmp) atIndex:3];
                [enc dispatchThreads:MTLSizeMake((uint32_t)cfg.rel_extent, rmp.M, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
            }

            {
                AttnParams ap{
                    (uint32_t)start_pos, (uint32_t)T, (uint32_t)vc.cap, lw.is_local ? 1u : 0u,
                    (uint32_t)cfg.sliding_window_size, (uint32_t)cfg.num_attention_heads,
                    (uint32_t)cfg.num_key_value_heads, (uint32_t)cfg.head_dim, (uint32_t)cfg.rel_extent
                };
                if (vc.is_local) {
                    [enc setComputePipelineState:pso_attention];
                    [enc setBuffer:b_Q offset:0 atIndex:0];
                    [enc setBuffer:vc.k_cache offset:kv_slot_off atIndex:1];
                    [enc setBuffer:vc.v_cache offset:kv_slot_off atIndex:2];
                    [enc setBuffer:b_rel_logits offset:0 atIndex:3];
                    [enc setBuffer:b_gate offset:0 atIndex:4];
                    [enc setBuffer:b_sinks offset:0 atIndex:5];
                    [enc setBuffer:b_attn_raw offset:0 atIndex:6];
                    [enc setBytes:&ap length:sizeof(ap) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake((uint32_t)cfg.num_attention_heads, (uint32_t)T, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                } else {
                    [enc setComputePipelineState:pso_attention_fp8];
                    [enc setBuffer:b_Q offset:0 atIndex:0];
                    [enc setBuffer:vc.k_cache_fp8 offset:kv_slot_off atIndex:1];
                    [enc setBuffer:vc.v_cache_fp8 offset:kv_slot_off atIndex:2];
                    [enc setBuffer:vc.k_scale_fp8 offset:sc_slot_off atIndex:3];
                    [enc setBuffer:vc.v_scale_fp8 offset:sc_slot_off atIndex:4];
                    [enc setBuffer:b_rel_logits offset:0 atIndex:5];
                    [enc setBuffer:b_gate offset:0 atIndex:6];
                    [enc setBuffer:b_sinks offset:0 atIndex:7];
                    [enc setBuffer:b_attn_raw offset:0 atIndex:8];
                    [enc setBytes:&ap length:sizeof(ap) atIndex:9];
                    [enc dispatchThreadgroups:MTLSizeMake((uint32_t)cfg.num_attention_heads, (uint32_t)T, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                }
            }
            flush_stage(prof_attn_sconv_ms);

            if (T <= 8) {
                encode_linear_fp8(enc, b_attn_raw, 0, lw.wo_fp8, 0, lw.wo_sc, 0, b_attn_proj, 0, nil, 0, (uint32_t)T, D, q_dim, false);
            } else {
                encode_linear_bf16(enc, b_attn_raw, 0, lw.wo, 0, b_attn_proj, 0, nil, 0, (uint32_t)T, D, q_dim, false);
            }
            flush_stage(prof_dense_proj_ms);

            {
                SconvStaticParams sp{(uint32_t)T, D, 0u};
                [enc setComputePipelineState:pso_sconv_static];
                [enc setBuffer:b_attn_proj offset:0 atIndex:0];
                [enc setBuffer:lw.attn_sconv_w offset:0 atIndex:1];
                [enc setBuffer:vc.attn_conv_state offset:d_cs_off atIndex:2];
                [enc setBuffer:b_attn_out offset:0 atIndex:3];
                [enc setBuffer:b_dummy offset:0 atIndex:4];
                [enc setBytes:&sp length:sizeof(sp) atIndex:5];
                [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            flush_stage(prof_attn_sconv_ms);

            bool first = (i % cfg.n_layers_per_attn_res_block == 0);
            uint32_t mlp_n_stack = (uint32_t)(num_blocks + (first ? 1 : 0));
            id<MTLBuffer> mlp_last = first ? b_attn_out : b_stack;
            size_t mlp_last_off = first ? 0 : partial_off;
            id<MTLBuffer> mlp_add = first ? nil : b_attn_out;

            if (looped) {
                encode_attn_res(enc, b_stack, mlp_n_stack, stack_stride,
                                mlp_last, mlp_last_off, mlp_add, 0,
                                mlp_queries[i], 0, 1,
                                nil, 0, nil, 0, b_fp32_mix, 0,
                                (uint32_t)T, D, cfg.rms_norm_eps);
                encode_controller(enc, lw, 3, b_fp32_mix, 0, b_messages, (size_t)3 * T * D * sizeof(float),
                                  lw.mlp_norm, 0, b_mlp_in, 0, (uint32_t)T, D, (uint32_t)cfg.loop_gate_rank, cfg.rms_norm_eps);
            } else {
                encode_attn_res(enc, b_stack, mlp_n_stack, stack_stride,
                                mlp_last, mlp_last_off, mlp_add, 0,
                                mlp_queries[i], 0, 1,
                                lw.mlp_norm, 0, b_mlp_in, 0, nil, 0,
                                (uint32_t)T, D, cfg.rms_norm_eps);
            }
            flush_stage(prof_attn_res_ctrl_ms);

            {
                RouterParams rp{(uint32_t)T, D, (uint32_t)cfg.n_routed_experts, (uint32_t)cfg.num_experts_per_tok, cfg.route_scale};
                [enc setComputePipelineState:pso_router_logits];
                [enc setBuffer:b_mlp_in offset:0 atIndex:0];
                [enc setBuffer:lw.gate_weight offset:0 atIndex:1];
                [enc setBuffer:b_router_logits offset:0 atIndex:2];
                [enc setBytes:&rp length:sizeof(rp) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(rp.E, (uint32_t)T, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                [enc setComputePipelineState:pso_router_topk];
                [enc setBuffer:b_router_logits offset:0 atIndex:0];
                [enc setBuffer:lw.gate_bias offset:0 atIndex:1];
                [enc setBuffer:b_topk_ids offset:0 atIndex:2];
                [enc setBuffer:b_topk_weights offset:0 atIndex:3];
                [enc setBytes:&rp length:sizeof(rp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((uint32_t)T, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            flush_stage(prof_router_ms);

            {
                MoeParams mp{(uint32_t)T, D, (uint32_t)cfg.intermediate_size, (uint32_t)cfg.num_experts_per_tok,
                             cfg.mlp_gate_softcap, cfg.mlp_up_softcap};
                if (lw.is_mxfp8) {
                    [enc setComputePipelineState:pso_moe_mxfp8_w13];
                    [enc setBuffer:b_mlp_in offset:0 atIndex:0];
                    [enc setBuffer:b_topk_ids offset:0 atIndex:1];
                    [enc setBuffer:b_topk_weights offset:0 atIndex:2];
                    [enc setBuffer:lw.w13_weight offset:0 atIndex:3];
                    [enc setBuffer:lw.w13_scale offset:0 atIndex:4];
                    [enc setBuffer:b_moe_act offset:0 atIndex:5];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake((mp.inter + 3u) / 4u, mp.T * mp.K, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:pso_moe_mxfp8_w2];
                    [enc setBuffer:b_moe_act offset:0 atIndex:0];
                    [enc setBuffer:b_topk_ids offset:0 atIndex:1];
                    [enc setBuffer:lw.w2_weight offset:0 atIndex:2];
                    [enc setBuffer:lw.w2_scale offset:0 atIndex:3];
                    [enc setBuffer:b_mlp_raw offset:0 atIndex:4];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake((D + 3u) / 4u, mp.T, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    flush_stage(prof_moe_mxfp8_ms);
                } else {
                    [enc setComputePipelineState:pso_moe_nvfp4_w13];
                    [enc setBuffer:b_mlp_in offset:0 atIndex:0];
                    [enc setBuffer:b_topk_ids offset:0 atIndex:1];
                    [enc setBuffer:b_topk_weights offset:0 atIndex:2];
                    [enc setBuffer:lw.w13_weight offset:0 atIndex:3];
                    [enc setBuffer:lw.w13_scale offset:0 atIndex:4];
                    [enc setBuffer:lw.w13_amax offset:0 atIndex:5];
                    [enc setBuffer:lw.w13_denom offset:0 atIndex:6];
                    [enc setBuffer:b_moe_act offset:0 atIndex:7];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:8];
                    [enc dispatchThreadgroups:MTLSizeMake((mp.inter + 3u) / 4u, mp.T * mp.K, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:pso_moe_nvfp4_w2];
                    [enc setBuffer:b_moe_act offset:0 atIndex:0];
                    [enc setBuffer:b_topk_ids offset:0 atIndex:1];
                    [enc setBuffer:lw.w2_weight offset:0 atIndex:2];
                    [enc setBuffer:lw.w2_scale offset:0 atIndex:3];
                    [enc setBuffer:lw.w2_amax offset:0 atIndex:4];
                    [enc setBuffer:lw.w2_denom offset:0 atIndex:5];
                    [enc setBuffer:b_mlp_raw offset:0 atIndex:6];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake((D + 3u) / 4u, mp.T, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    flush_stage(prof_moe_nvfp4_ms);
                }
            }

            {
                SconvStaticParams sp{(uint32_t)T, D, 0u};
                [enc setComputePipelineState:pso_sconv_static];
                [enc setBuffer:b_mlp_raw offset:0 atIndex:0];
                [enc setBuffer:lw.mlp_sconv_w offset:0 atIndex:1];
                [enc setBuffer:vc.mlp_conv_state offset:d_cs_off atIndex:2];
                [enc setBuffer:b_mlp_out offset:0 atIndex:3];
                [enc setBuffer:b_dummy offset:0 atIndex:4];
                [enc setBytes:&sp length:sizeof(sp) atIndex:5];
                [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            uint32_t td_n = (uint32_t)T * D;
            [enc setComputePipelineState:pso_add_bf16];
            if (first) {
                num_blocks += 1;
                size_t new_off = (size_t)num_blocks * stack_stride_bytes;
                [enc setBuffer:b_attn_out offset:0 atIndex:0];
                [enc setBuffer:b_mlp_out offset:0 atIndex:1];
                [enc setBuffer:b_stack offset:new_off atIndex:2];
                [enc setBytes:&td_n length:sizeof(td_n) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(td_n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                [enc setBuffer:b_stack offset:partial_off atIndex:0];
                [enc setBuffer:b_attn_out offset:0 atIndex:1];
                [enc setBuffer:b_attn_proj offset:0 atIndex:2];
                [enc setBytes:&td_n length:sizeof(td_n) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(td_n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc setBuffer:b_attn_proj offset:0 atIndex:0];
                [enc setBuffer:b_mlp_out offset:0 atIndex:1];
                [enc setBuffer:b_stack offset:partial_off atIndex:2];
                [enc setBytes:&td_n length:sizeof(td_n) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(td_n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }

        size_t final_off = (size_t)num_blocks * stack_stride_bytes;
        encode_attn_res(enc, b_stack, (uint32_t)num_blocks, stack_stride,
                        b_stack, final_off, nil, 0,
                        output_query, 0, 1,
                        final_norm_w, 0, b_mlp_in, 0, nil, 0,
                        (uint32_t)T, D, cfg.rms_norm_eps);
        flush_stage(prof_attn_res_ctrl_ms);

        size_t last_tok_off = (size_t)(T - 1) * D * 2;
        encode_linear_fp8(enc, b_mlp_in, last_tok_off, unembed_fp8, 0, unembed_sc, 0, nil, 0, b_logits, 0,
                          1u, (uint32_t)cfg.unpadded_vocab_size, D, true);

        if (do_argmax) {
            ArgmaxParams amp{(uint32_t)cfg.unpadded_vocab_size, cfg.final_logit_softcapping};
            [enc setComputePipelineState:pso_softcap_argmax];
            [enc setBuffer:b_logits offset:0 atIndex:0];
            [enc setBuffer:b_out_token offset:0 atIndex:1];
            [enc setBytes:&amp length:sizeof(amp) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        [enc endEncoding];
        commit_and_wait(cb);
        if (profile_stages) {
            prof_lm_head_ms += (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            prof_steps += 1;
        }
    }
};

Inkling2Model::Inkling2Model() : impl_(std::make_unique<Impl>()) {}
Inkling2Model::~Inkling2Model() = default;

void Inkling2Model::load(const std::string& model_dir, int max_seq, int max_chunk, int n_slots) {
    @autoreleasepool {
        auto t0 = std::chrono::steady_clock::now();
        max_seq_len = max_seq;
        max_batch_tokens = max_chunk;
        max_slots = std::max(1, std::min(8, n_slots));

        cfg = InklingConfig::load(model_dir + "/config.json");
        st.open(model_dir + "/model-00001-of-00001.safetensors");

        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) throw std::runtime_error("No Apple Metal device available");
        impl_->gpu_ws_gib = (double)impl_->device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
        impl_->gpu_max_tg = (uint32_t)impl_->device.maxThreadsPerThreadgroup.width;
        impl_->gpu_is_apple9 = [impl_->device supportsFamily:MTLGPUFamilyApple9];
        impl_->gpu_low_mem = (impl_->gpu_ws_gib <= 24.0);
        impl_->queue = [impl_->device newCommandQueue];

        NSError* err = nil;
#ifdef WARPGRID_METAL_PRECOMPILED
        dispatch_data_t data = dispatch_data_create(
            kMetalLibBytes, kMetalLibLen, dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        impl_->library = [impl_->device newLibraryWithData:data error:&err];
#else
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion3_1;
        opts.fastMathEnabled = NO;
        NSString* src = [[NSString alloc] initWithBytes:kMetalSourceBytes
                                                 length:kMetalSourceLen
                                               encoding:NSUTF8StringEncoding];
        impl_->library = [impl_->device newLibraryWithSource:src options:opts error:&err];
#endif
        if (!impl_->library) {
            throw std::runtime_error(std::string("Failed to load Metal shader library: ") +
                                     (err ? err.localizedDescription.UTF8String : "unknown"));
        }

        impl_->pso_embed_norm = impl_->make_pso("embed_norm_kernel");
        impl_->pso_rmsnorm = impl_->make_pso("rmsnorm_kernel");
        impl_->pso_add_bf16 = impl_->make_pso("add_bf16_kernel");
        impl_->pso_copy_bf16 = impl_->make_pso("copy_bf16_kernel");
        impl_->pso_split_qrgs = impl_->make_pso("split_qrgs_kernel");
        impl_->pso_split_kv = impl_->make_pso("split_kv_kernel");
        impl_->pso_attn_res = impl_->make_pso("attn_res_kernel");
        impl_->pso_pair_rms_norm = impl_->make_pso("pair_rms_norm_kernel");
        impl_->pso_fused_ctrl = impl_->make_pso("fused_ctrl_blend_rmsnorm_kernel");
        impl_->pso_ctrl_gf_small_t = impl_->make_pso("ctrl_gf_blend_small_t_kernel");
        impl_->pso_fp32_rmsnorm = impl_->make_pso("fp32_rmsnorm_to_bf16_kernel");
        impl_->pso_sconv_static = impl_->make_pso("sconv_static_kernel");
        impl_->pso_sconv_dynamic = impl_->make_pso("sconv_dynamic_kernel");
        impl_->pso_attn_prep_qr = impl_->make_pso("attn_prep_q_r_kernel");
        impl_->pso_attn_prep_kv = impl_->make_pso("attn_prep_kv_kernel");
        impl_->pso_attn_prep_kv_fp8 = impl_->make_pso("attn_prep_kv_fp8_kernel");
        impl_->pso_rel_matmul = impl_->make_pso("rel_logits_matmul_kernel");
        impl_->pso_attention = impl_->make_pso("attention_kernel");
        impl_->pso_attention_fp8 = impl_->make_pso("attention_fp8_kernel");
        impl_->pso_router_logits = impl_->make_pso("moe_router_logits_kernel");
        impl_->pso_router_topk = impl_->make_pso("moe_router_topk_kernel");
        impl_->pso_moe_nvfp4_w13 = impl_->make_pso("moe_nvfp4_w13_kernel");
        impl_->pso_moe_nvfp4_w2 = impl_->make_pso("moe_nvfp4_w2_kernel");
        impl_->pso_moe_mxfp8_w13 = impl_->make_pso("moe_mxfp8_w13_kernel");
        impl_->pso_moe_mxfp8_w2 = impl_->make_pso("moe_mxfp8_w2_kernel");
        impl_->pso_bf16_gemv = impl_->make_pso("bf16_gemv_kernel");
        impl_->pso_bf16_gemv_1sg = impl_->make_pso("bf16_gemv_1sg_kernel");
        impl_->pso_quant_fp8 = impl_->make_pso("quantize_bf16_to_fp8_rowwise_kernel");
        impl_->pso_fp8_gemv = impl_->make_pso("fp8_gemv_kernel");
        impl_->pso_bf16_gemm_tiled = impl_->make_pso("bf16_gemm_tiled_kernel");
        impl_->pso_softcap_argmax = impl_->make_pso("softcap_argmax_kernel");

        impl_->b_dummy = impl_->alloc_shared(64);
        impl_->embed_w = impl_->load_tensor(st, "model.llm.embed.weight");
        impl_->embed_norm_w = impl_->load_tensor(st, "model.llm.embed_norm.weight");
        impl_->final_norm_w = impl_->load_tensor(st, "model.llm.norm.weight");
        impl_->unembed_w = impl_->load_tensor(st, "model.llm.unembed.weight");
        impl_->unembed_fp8 = impl_->alloc_shared((size_t)cfg.unpadded_vocab_size * cfg.hidden_size);
        impl_->unembed_sc = impl_->alloc_shared((size_t)cfg.unpadded_vocab_size * sizeof(float));

        float inv_sqrt_d = 1.0f / std::sqrt((float)cfg.hidden_size);
        auto load_scaled_query = [&](const std::string& name) -> id<MTLBuffer> {
            const auto& info = st.info(name);
            id<MTLBuffer> b = impl_->alloc_shared(info.nbytes);
            st.read_into(name, b.contents, info.nbytes);
            float* fp = static_cast<float*>(b.contents);
            size_t n = info.nbytes / sizeof(float);
            for (size_t k = 0; k < n; ++k) fp[k] *= inv_sqrt_d;
            return b;
        };

        impl_->attn_queries.resize(cfg.num_hidden_layers - 1);
        for (int i = 0; i < cfg.num_hidden_layers - 1; ++i) {
            impl_->attn_queries[i] = load_scaled_query("model.llm.attn_res.attn_queries." + std::to_string(i));
        }
        impl_->mlp_queries.resize(cfg.num_hidden_layers);
        for (int i = 0; i < cfg.num_hidden_layers; ++i) {
            impl_->mlp_queries[i] = load_scaled_query("model.llm.attn_res.mlp_queries." + std::to_string(i));
        }
        impl_->output_query = load_scaled_query("model.llm.attn_res.output_query");

        {
            int n_core = cfg.loop_core_end - cfg.loop_core_start;
            int D = cfg.hidden_size;
            std::vector<uint8_t> q_raw, kv_raw;
            st.load_to_host("model.llm.middle_cycle_cross_pass_queries", q_raw);
            st.load_to_host("model.llm.middle_cycle_cross_pass_kv_queries", kv_raw);
            const float* q_f = reinterpret_cast<const float*>(q_raw.data());
            const float* kv_f = reinterpret_cast<const float*>(kv_raw.data());
            impl_->message_queries = impl_->alloc_shared((size_t)n_core * 4 * D * sizeof(float));
            float* dst_all = static_cast<float*>(impl_->message_queries.contents);
            for (int c = 0; c < n_core; ++c) {
                const float* q_attn = q_f + (c * 2 + 0) * D;
                const float* q_mlp = q_f + (c * 2 + 1) * D;
                const float* q_k = kv_f + (c * 2 + 0) * D;
                const float* q_v = kv_f + (c * 2 + 1) * D;
                float* dst = dst_all + (size_t)c * 4 * D;
                for (int d = 0; d < D; ++d) {
                    dst[0 * D + d] = q_attn[d] * inv_sqrt_d;
                    dst[1 * D + d] = q_k[d] * inv_sqrt_d;
                    dst[2 * D + d] = q_v[d] * inv_sqrt_d;
                    dst[3 * D + d] = q_mlp[d] * inv_sqrt_d;
                }
            }
        }

        int D = cfg.hidden_size;
        int q_dim = cfg.num_attention_heads * cfg.head_dim;
        int kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        int r_dim = cfg.num_attention_heads * cfg.d_rel;
        int qrgs_dim = 2 * q_dim + r_dim + cfg.num_attention_heads;

        impl_->layers.resize(cfg.num_hidden_layers);
        for (int i = 0; i < cfg.num_hidden_layers; ++i) {
            auto& lw = impl_->layers[i];
            std::string p = "model.llm.layers." + std::to_string(i) + ".";
            lw.is_local = cfg.is_local(i);
            lw.is_loop_core = cfg.is_loop_core(i);

            if (lw.is_local) {
                lw.attn_norm = impl_->load_tensor(st, p + "attn_norm.weight");
            } else {
                lw.attn_norm = impl_->alloc_shared((size_t)3 * D * 2);
                uint8_t* base = static_cast<uint8_t*>(lw.attn_norm.contents);
                const char* nnames[3] = {"q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight"};
                for (int j = 0; j < 3; ++j) {
                    st.read_into(p + nnames[j], base + (size_t)j * D * 2, (size_t)D * 2);
                }
            }
            lw.mlp_norm = impl_->load_tensor(st, p + "mlp_norm.weight");

            lw.w_qrgs = impl_->alloc_shared((size_t)qrgs_dim * D * 2);
            uint8_t* qbase = static_cast<uint8_t*>(lw.w_qrgs.contents);
            st.read_into(p + "attn.wq_du.weight", qbase, (size_t)q_dim * D * 2);
            st.read_into(p + "attn.wr_du.weight", qbase + (size_t)q_dim * D * 2, (size_t)r_dim * D * 2);
            st.read_into(p + "attn.gate_attn.weight", qbase + (size_t)(q_dim + r_dim) * D * 2, (size_t)q_dim * D * 2);
            st.read_into(p + "attn.attn_sink_proj.weight", qbase + (size_t)(2 * q_dim + r_dim) * D * 2, (size_t)cfg.num_attention_heads * D * 2);

            lw.w_kv = impl_->alloc_shared((size_t)2 * kv_dim * D * 2);
            uint8_t* kvbase = static_cast<uint8_t*>(lw.w_kv.contents);
            st.read_into(p + "attn.wk_dv.weight", kvbase, (size_t)kv_dim * D * 2);
            st.read_into(p + "attn.wv_dv.weight", kvbase + (size_t)kv_dim * D * 2, (size_t)kv_dim * D * 2);

            lw.wo = impl_->load_tensor(st, p + "attn.wo_ud.weight");
            lw.w_qrgs_fp8 = impl_->alloc_shared((size_t)qrgs_dim * D);
            lw.w_qrgs_sc = impl_->alloc_shared((size_t)qrgs_dim * sizeof(float));
            lw.w_kv_fp8 = impl_->alloc_shared((size_t)2 * kv_dim * D);
            lw.w_kv_sc = impl_->alloc_shared((size_t)2 * kv_dim * sizeof(float));
            lw.wo_fp8 = impl_->alloc_shared((size_t)D * q_dim);
            lw.wo_sc = impl_->alloc_shared((size_t)D * sizeof(float));
            lw.q_clip_scale = impl_->load_tensor(st, p + "attn.q_l2_clip.scale");
            lw.rel_clip_scale = impl_->load_tensor(st, p + "attn.rel_l2_clip.scale");
            lw.rel_proj = impl_->load_tensor(st, p + "attn.rel_logits_proj.proj");

            if (lw.is_local) {
                lw.k_sconv_w = impl_->load_tensor(st, p + "attn.k_sconv.weight");
                lw.v_sconv_w = impl_->load_tensor(st, p + "attn.v_sconv.weight");
            } else {
                lw.k_sconv_w = impl_->load_tensor(st, p + "attn.k_sconv.static_w");
                lw.k_sconv_proj = impl_->load_tensor(st, p + "attn.k_sconv.weight_proj");
                lw.v_sconv_w = impl_->load_tensor(st, p + "attn.v_sconv.static_w");
                lw.v_sconv_proj = impl_->load_tensor(st, p + "attn.v_sconv.weight_proj");
            }
            lw.attn_sconv_w = impl_->load_tensor(st, p + "attn_sconv.weight");
            lw.mlp_sconv_w = impl_->load_tensor(st, p + "mlp_sconv.weight");

            if (lw.is_loop_core) {
                const char* snames[4] = {"attn", "k", "v", "mlp"};
                for (int s = 0; s < 4; ++s) {
                    std::string cp = p + "controllers." + snames[s] + ".";
                    lw.ctrl_down[s] = impl_->load_tensor(st, cp + "down_weight");
                    lw.ctrl_gain[s] = impl_->load_tensor(st, cp + "gain_weight");
                    lw.ctrl_forget[s] = impl_->load_tensor(st, cp + "forget_weight");
                }
            }

            lw.gate_weight = impl_->load_tensor(st, p + "mlp.gate.weight");
            lw.gate_bias = impl_->load_tensor(st, p + "mlp.gate.bias");

            lw.is_mxfp8 = (st.info(p + "mlp.experts.w13_weight").dtype == "F8_E4M3");
            lw.w13_weight = impl_->load_tensor(st, p + "mlp.experts.w13_weight");
            lw.w13_scale = impl_->load_tensor(st, p + "mlp.experts.w13_weight_scale");
            lw.w2_weight = impl_->load_tensor(st, p + "mlp.experts.w2_weight");
            lw.w2_scale = impl_->load_tensor(st, p + "mlp.experts.w2_weight_scale");
            if (!lw.is_mxfp8) {
                lw.w13_amax = impl_->load_tensor(st, p + "mlp.experts.w13_weight_amax");
                lw.w13_denom = impl_->load_tensor(st, p + "mlp.experts.w13_weight_outer_scale_denominator");
                lw.w2_amax = impl_->load_tensor(st, p + "mlp.experts.w2_weight_amax");
                lw.w2_denom = impl_->load_tensor(st, p + "mlp.experts.w2_weight_outer_scale_denominator");
            }
        }

        impl_->visits.clear();
        for (int i = 0; i < cfg.loop_core_start; ++i) impl_->visits.push_back({i, 0});
        for (int p = 0; p < cfg.loop_num_passes; ++p)
            for (int i = cfg.loop_core_start; i < cfg.loop_core_end; ++i) impl_->visits.push_back({i, p});
        for (int i = cfg.loop_core_end; i < cfg.num_hidden_layers; ++i) impl_->visits.push_back({i, 0});

        impl_->caches.resize(impl_->visits.size());
        size_t kv_bytes_total = 0;
        for (size_t v = 0; v < impl_->visits.size(); ++v) {
            int lid = impl_->visits[v].first;
            auto& vc = impl_->caches[v];
            vc.is_local = cfg.is_local(lid);
            vc.cap = vc.is_local ? (cfg.sliding_window_size + max_batch_tokens) : max_seq_len;
            if (vc.is_local) {
                size_t kv_sz = (size_t)max_slots * (size_t)vc.cap * kv_dim * 2;
                vc.k_cache = impl_->alloc_shared(kv_sz);
                vc.v_cache = impl_->alloc_shared(kv_sz);
                kv_bytes_total += 2 * kv_sz;
            } else {
                size_t kv_fp8_sz = (size_t)max_slots * (size_t)vc.cap * kv_dim;
                size_t sc_sz = (size_t)max_slots * (size_t)vc.cap * (size_t)cfg.num_key_value_heads * sizeof(float);
                vc.k_cache_fp8 = impl_->alloc_shared(kv_fp8_sz);
                vc.v_cache_fp8 = impl_->alloc_shared(kv_fp8_sz);
                vc.k_scale_fp8 = impl_->alloc_shared(sc_sz);
                vc.v_scale_fp8 = impl_->alloc_shared(sc_sz);
                kv_bytes_total += 2 * kv_fp8_sz + 2 * sc_sz;
            }

            vc.k_conv_state = impl_->alloc_shared((size_t)max_slots * 3 * kv_dim * 2);
            vc.v_conv_state = impl_->alloc_shared((size_t)max_slots * 3 * kv_dim * 2);
            vc.attn_conv_state = impl_->alloc_shared((size_t)max_slots * 3 * D * 2);
            vc.mlp_conv_state = impl_->alloc_shared((size_t)max_slots * 3 * D * 2);

            if (vc.is_local) {
                size_t kv_sz = (size_t)max_slots * (size_t)vc.cap * kv_dim * 2;
                vc.local_k_prefill_snap = impl_->alloc_shared(kv_sz);
                vc.local_v_prefill_snap = impl_->alloc_shared(kv_sz);
                kv_bytes_total += 2 * kv_sz;
            }
            vc.k_conv_prefill_snap = impl_->alloc_shared((size_t)max_slots * 3 * kv_dim * 2);
            vc.v_conv_prefill_snap = impl_->alloc_shared((size_t)max_slots * 3 * kv_dim * 2);
            vc.attn_conv_prefill_snap = impl_->alloc_shared((size_t)max_slots * 3 * D * 2);
            vc.mlp_conv_prefill_snap = impl_->alloc_shared((size_t)max_slots * 3 * D * 2);
        }

        int T = max_batch_tokens;
        impl_->b_tokens = impl_->alloc_shared((size_t)T * sizeof(int32_t));
        impl_->b_stack = impl_->alloc_shared((size_t)6 * T * D * 2);
        impl_->b_prelude_part = impl_->alloc_shared((size_t)T * D * 2);
        impl_->b_bank = impl_->alloc_shared((size_t)2 * T * D * 2);
        impl_->b_prelude_dest = impl_->alloc_shared((size_t)T * D * sizeof(float));
        impl_->b_messages = impl_->alloc_shared((size_t)4 * T * D * sizeof(float));
        impl_->b_fp32_mix = impl_->alloc_shared((size_t)3 * T * D * sizeof(float));
        impl_->b_attn_in = impl_->alloc_shared((size_t)3 * T * D * 2);
        impl_->b_mlp_in = impl_->alloc_shared((size_t)T * D * 2);
        impl_->b_pair = impl_->alloc_shared((size_t)T * 2 * D * 2);
        impl_->b_ctrl_h = impl_->alloc_shared((size_t)T * cfg.loop_gate_rank * 2);
        impl_->b_ctrl_tmp = impl_->alloc_shared((size_t)8 * D * sizeof(float));
        impl_->b_qrgs = impl_->alloc_shared((size_t)T * qrgs_dim * 2);
        impl_->b_kv_out = impl_->alloc_shared((size_t)T * 2 * kv_dim * 2);
        impl_->b_Q = impl_->alloc_shared((size_t)T * q_dim * 2);
        impl_->b_K = impl_->alloc_shared((size_t)T * kv_dim * 2);
        impl_->b_V = impl_->alloc_shared((size_t)T * kv_dim * 2);
        impl_->b_K_conv = impl_->alloc_shared((size_t)T * kv_dim * 2);
        impl_->b_V_conv = impl_->alloc_shared((size_t)T * kv_dim * 2);
        impl_->b_dyn_w = impl_->alloc_shared((size_t)T * cfg.num_key_value_heads * 4 * 2);
        impl_->b_R = impl_->alloc_shared((size_t)T * r_dim * 2);
        impl_->b_rel_logits = impl_->alloc_shared((size_t)T * cfg.num_attention_heads * cfg.rel_extent * 2);
        impl_->b_gate = impl_->alloc_shared((size_t)T * q_dim * 2);
        impl_->b_sinks = impl_->alloc_shared((size_t)T * cfg.num_attention_heads * 2);
        impl_->b_attn_raw = impl_->alloc_shared((size_t)T * q_dim * 2);
        impl_->b_attn_proj = impl_->alloc_shared((size_t)T * D * 2);
        impl_->b_attn_out = impl_->alloc_shared((size_t)T * D * 2);
        impl_->b_mlp_raw = impl_->alloc_shared((size_t)T * D * 2);
        impl_->b_mlp_out = impl_->alloc_shared((size_t)T * D * 2);
        impl_->b_topk_ids = impl_->alloc_shared((size_t)T * cfg.num_experts_per_tok * sizeof(int32_t));
        impl_->b_topk_weights = impl_->alloc_shared((size_t)T * cfg.num_experts_per_tok * sizeof(float));
        impl_->b_router_logits = impl_->alloc_shared((size_t)T * cfg.n_routed_experts * sizeof(float));
        impl_->b_moe_act = impl_->alloc_shared((size_t)T * cfg.num_experts_per_tok * cfg.intermediate_size * sizeof(float));
        impl_->b_logits = impl_->alloc_shared((size_t)cfg.vocab_size * sizeof(float));
        impl_->b_out_token = impl_->alloc_shared(16 * sizeof(int32_t));

        {
            id<MTLCommandBuffer> qcb = [impl_->queue commandBuffer];
            id<MTLComputeCommandEncoder> qenc = [qcb computeCommandEncoder];
            impl_->quantize_rowwise_fp8(qenc, impl_->unembed_w, impl_->unembed_fp8, impl_->unembed_sc,
                                        (uint32_t)cfg.unpadded_vocab_size, (uint32_t)D);
            for (int i = 0; i < cfg.num_hidden_layers; ++i) {
                auto& lw = impl_->layers[i];
                impl_->quantize_rowwise_fp8(qenc, lw.w_qrgs, lw.w_qrgs_fp8, lw.w_qrgs_sc, (uint32_t)qrgs_dim, (uint32_t)D);
                impl_->quantize_rowwise_fp8(qenc, lw.w_kv, lw.w_kv_fp8, lw.w_kv_sc, (uint32_t)(2 * kv_dim), (uint32_t)D);
                impl_->quantize_rowwise_fp8(qenc, lw.wo, lw.wo_fp8, lw.wo_sc, (uint32_t)D, (uint32_t)q_dim);
            }
            [qenc endEncoding];
            impl_->commit_and_wait(qcb);
        }

        for (int s = 0; s < max_slots; ++s) reset_cache(s);

        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        fprintf(stderr, "[warpgrid-metal] Loaded Inkling-2-Mini on %s in %.2f s (KV: %.2f GiB, %d slots x max_seq=%d)\n",
                impl_->device.name.UTF8String, sec,
                kv_bytes_total / (1024.0 * 1024.0 * 1024.0), max_slots, max_seq_len);
    }
}

void Inkling2Model::reset_cache(int slot_id) {
    int kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    int D = cfg.hidden_size;
    for (auto& vc : impl_->caches) {
        memset(static_cast<uint8_t*>(vc.k_conv_state.contents) + (size_t)slot_id * 3 * kv_dim * 2, 0, (size_t)3 * kv_dim * 2);
        memset(static_cast<uint8_t*>(vc.v_conv_state.contents) + (size_t)slot_id * 3 * kv_dim * 2, 0, (size_t)3 * kv_dim * 2);
        memset(static_cast<uint8_t*>(vc.attn_conv_state.contents) + (size_t)slot_id * 3 * D * 2, 0, (size_t)3 * D * 2);
        memset(static_cast<uint8_t*>(vc.mlp_conv_state.contents) + (size_t)slot_id * 3 * D * 2, 0, (size_t)3 * D * 2);
    }
}

void Inkling2Model::save_prefill_snapshot(int slot_id) {
    int kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    int D = cfg.hidden_size;
    for (auto& vc : impl_->caches) {
        if (vc.is_local) {
            size_t one_kv = (size_t)vc.cap * kv_dim * 2;
            size_t off = (size_t)slot_id * one_kv;
            memcpy(static_cast<uint8_t*>(vc.local_k_prefill_snap.contents) + off,
                   static_cast<uint8_t*>(vc.k_cache.contents) + off, one_kv);
            memcpy(static_cast<uint8_t*>(vc.local_v_prefill_snap.contents) + off,
                   static_cast<uint8_t*>(vc.v_cache.contents) + off, one_kv);
        }
        size_t k_sz = (size_t)3 * kv_dim * 2;
        size_t d_sz = (size_t)3 * D * 2;
        memcpy(static_cast<uint8_t*>(vc.k_conv_prefill_snap.contents) + (size_t)slot_id * k_sz,
               static_cast<uint8_t*>(vc.k_conv_state.contents) + (size_t)slot_id * k_sz, k_sz);
        memcpy(static_cast<uint8_t*>(vc.v_conv_prefill_snap.contents) + (size_t)slot_id * k_sz,
               static_cast<uint8_t*>(vc.v_conv_state.contents) + (size_t)slot_id * k_sz, k_sz);
        memcpy(static_cast<uint8_t*>(vc.attn_conv_prefill_snap.contents) + (size_t)slot_id * d_sz,
               static_cast<uint8_t*>(vc.attn_conv_state.contents) + (size_t)slot_id * d_sz, d_sz);
        memcpy(static_cast<uint8_t*>(vc.mlp_conv_prefill_snap.contents) + (size_t)slot_id * d_sz,
               static_cast<uint8_t*>(vc.mlp_conv_state.contents) + (size_t)slot_id * d_sz, d_sz);
    }
}

void Inkling2Model::restore_prefill_snapshot(int slot_id) {
    int kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    int D = cfg.hidden_size;
    for (auto& vc : impl_->caches) {
        if (vc.is_local) {
            size_t one_kv = (size_t)vc.cap * kv_dim * 2;
            size_t off = (size_t)slot_id * one_kv;
            memcpy(static_cast<uint8_t*>(vc.k_cache.contents) + off,
                   static_cast<uint8_t*>(vc.local_k_prefill_snap.contents) + off, one_kv);
            memcpy(static_cast<uint8_t*>(vc.v_cache.contents) + off,
                   static_cast<uint8_t*>(vc.local_v_prefill_snap.contents) + off, one_kv);
        }
        size_t k_sz = (size_t)3 * kv_dim * 2;
        size_t d_sz = (size_t)3 * D * 2;
        memcpy(static_cast<uint8_t*>(vc.k_conv_state.contents) + (size_t)slot_id * k_sz,
               static_cast<uint8_t*>(vc.k_conv_prefill_snap.contents) + (size_t)slot_id * k_sz, k_sz);
        memcpy(static_cast<uint8_t*>(vc.v_conv_state.contents) + (size_t)slot_id * k_sz,
               static_cast<uint8_t*>(vc.v_conv_prefill_snap.contents) + (size_t)slot_id * k_sz, k_sz);
        memcpy(static_cast<uint8_t*>(vc.attn_conv_state.contents) + (size_t)slot_id * d_sz,
               static_cast<uint8_t*>(vc.attn_conv_prefill_snap.contents) + (size_t)slot_id * d_sz, d_sz);
        memcpy(static_cast<uint8_t*>(vc.mlp_conv_state.contents) + (size_t)slot_id * d_sz,
               static_cast<uint8_t*>(vc.mlp_conv_prefill_snap.contents) + (size_t)slot_id * d_sz, d_sz);
    }
}

int32_t Inkling2Model::forward_and_greedy(const int32_t* h_tokens, int T, int start_pos, int slot_id) {
    @autoreleasepool {
        impl_->forward_internal(cfg, max_batch_tokens, h_tokens, T, start_pos, slot_id, true, profile_stages);
        return static_cast<const int32_t*>(impl_->b_out_token.contents)[0];
    }
}

void Inkling2Model::forward_batch_decode_greedy(
    const int32_t* h_tokens, const int32_t* h_slots, const int32_t* h_positions,
    int B, int32_t* h_out_tokens)
{
    for (int b = 0; b < B; ++b) {
        h_out_tokens[b] = forward_and_greedy(&h_tokens[b], 1, h_positions[b], h_slots[b]);
    }
}

void Inkling2Model::forward_verify_greedy(const int32_t* h_tokens, int T, int start_pos, int32_t* h_out_tokens) {
    for (int t = 0; t < T; ++t) {
        h_out_tokens[t] = forward_and_greedy(&h_tokens[t], 1, start_pos + t, 0);
    }
}

void Inkling2Model::forward_batch_verify_greedy(
    const int32_t* h_tokens, const int32_t* h_slots, const int32_t* h_positions,
    int B, int L, int32_t* h_out_tokens)
{
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < L; ++t) {
            h_out_tokens[b * L + t] = forward_and_greedy(&h_tokens[b * L + t], 1, h_positions[b] + t, h_slots[b]);
        }
    }
}

void Inkling2Model::rollback_conv_states(int, int) {}
void Inkling2Model::commit_dspark_context(int, int, int, int) {}

int Inkling2Model::max_dspark_slots() const {
    return impl_->dspark_slots;
}

void Inkling2Model::draft_tokens(int32_t, int, int, int32_t*, int) {
    throw std::runtime_error("Metal DSpark drafter not implemented yet");
}

void Inkling2Model::forward_logits(const int32_t* h_tokens, int T, int start_pos, std::vector<float>& h_logits, int slot_id) {
    @autoreleasepool {
        impl_->forward_internal(cfg, max_batch_tokens, h_tokens, T, start_pos, slot_id, true, profile_stages);
        h_logits.resize(cfg.unpadded_vocab_size);
        memcpy(h_logits.data(), impl_->b_logits.contents, (size_t)cfg.unpadded_vocab_size * sizeof(float));
    }
}

void Inkling2Model::reset_profile() {
    impl_->prof_attn_res_ctrl_ms = 0.0;
    impl_->prof_dense_proj_ms = 0.0;
    impl_->prof_attn_sconv_ms = 0.0;
    impl_->prof_router_ms = 0.0;
    impl_->prof_moe_nvfp4_ms = 0.0;
    impl_->prof_moe_mxfp8_ms = 0.0;
    impl_->prof_lm_head_ms = 0.0;
    impl_->prof_steps = 0;
}

void Inkling2Model::print_profile() const {
    if (impl_->prof_steps == 0) return;
    double n = (double)impl_->prof_steps;
    double tot = (impl_->prof_attn_res_ctrl_ms + impl_->prof_dense_proj_ms + impl_->prof_attn_sconv_ms +
                  impl_->prof_router_ms + impl_->prof_moe_nvfp4_ms + impl_->prof_moe_mxfp8_ms + impl_->prof_lm_head_ms) / n;
    fprintf(stderr, "\n=== METAL STAGE PROFILE (avg over %d steps, gpu_total=%.2f ms/tok) ===\n", impl_->prof_steps, tot);
    auto row = [&](const char* name, double ms) {
        fprintf(stderr, "  %-22s: %6.2f ms (%5.1f%%)\n", name, ms / n, 100.0 * (ms / n) / tot);
    };
    row("AttnRes + LoopCtrl", impl_->prof_attn_res_ctrl_ms);
    row("Dense FP8 Projs",    impl_->prof_dense_proj_ms);
    row("Attention + SConv",  impl_->prof_attn_sconv_ms);
    row("MoE Router",         impl_->prof_router_ms);
    row("MoE NVFP4 (44x)",    impl_->prof_moe_nvfp4_ms);
    row("MoE MXFP8 (4x)",     impl_->prof_moe_mxfp8_ms);
    row("LM Head + Sample",   impl_->prof_lm_head_ms);
}

}
