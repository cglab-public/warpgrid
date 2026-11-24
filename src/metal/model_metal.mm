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
#include <fstream>
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
struct AttnSplitKParams {
    uint32_t start_pos; uint32_t T; uint32_t cache_cap; uint32_t is_local;
    uint32_t window_size; uint32_t n_heads; uint32_t n_kv_heads;
    uint32_t head_dim; uint32_t rel_extent; uint32_t num_splits;
};
struct RouterParams { uint32_t T; uint32_t D; uint32_t E; uint32_t K; float route_scale; };
struct MoeParams { uint32_t T; uint32_t D; uint32_t inter; uint32_t K; float gate_cap; float up_cap; };
struct MoeGroupParams { uint32_t total_tk; uint32_t E; };
struct GemvParams { uint32_t M; uint32_t N; uint32_t K; uint32_t out_fp32; };
struct QuantParams { uint32_t N; uint32_t K; };
struct ArgmaxParams { uint32_t vocab_size; float softcap; };
struct BatchArgParams { uint32_t M; uint32_t vocab_size; };
struct DflareMixParams { uint32_t aux_stride; uint32_t T; uint32_t D; float eps; };
struct QkRopeParams { uint32_t start_pos; uint32_t T; uint32_t n_heads; uint32_t head_dim; float rope_theta; float eps; };
struct DraftWriteKvParams { uint32_t start_pos; uint32_t T; uint32_t cap; uint32_t is_local; uint32_t kv_dim; };
struct DraftAttnParams {
    uint32_t anchor_pos; uint32_t B; uint32_t cap; uint32_t is_local;
    uint32_t window_size; uint32_t n_heads; uint32_t n_kv_heads; uint32_t head_dim;
};
struct MarkovInParams { uint32_t rank; uint32_t D; };
struct MarkovArgParams { uint32_t V; uint32_t has_map; };
struct GatherParams { uint32_t M; uint32_t D; };
struct FoldParams { uint32_t V; uint32_t D; uint32_t rank; };
struct Df2ConcatParams { uint32_t T; uint32_t D; uint32_t stack_stride; };
struct Df2ConvParams { uint32_t T; uint32_t D; uint32_t group_size; uint32_t half_idx; };
struct Df2TopKParams { uint32_t M; uint32_t V; };

static float bf16_bits_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

struct MetalDraftLayerWeights {
    bool is_local = true;
    id<MTLBuffer> dflare_norm = nil;
    id<MTLBuffer> input_ln = nil;
    id<MTLBuffer> post_attn_ln = nil;
    id<MTLBuffer> k_proj_ctx = nil;
    id<MTLBuffer> v_proj_ctx = nil;
    id<MTLBuffer> q_norm = nil;
    id<MTLBuffer> k_norm = nil;
    id<MTLBuffer> k_norm_ctx = nil;
    id<MTLBuffer> q_fp8 = nil, q_sc = nil;
    id<MTLBuffer> k_fp8 = nil, k_sc = nil;
    id<MTLBuffer> v_fp8 = nil, v_sc = nil;
    id<MTLBuffer> o_fp8 = nil, o_sc = nil;
    id<MTLBuffer> gate_fp8 = nil, gate_sc = nil;
    id<MTLBuffer> up_fp8 = nil, up_sc = nil;
    id<MTLBuffer> down_fp8 = nil, down_sc = nil;
};

struct MetalDFlash2LayerWeights {
    id<MTLBuffer> input_ln = nil;
    id<MTLBuffer> post_ln = nil;
    id<MTLBuffer> q_norm = nil;
    id<MTLBuffer> k_norm = nil;
    id<MTLBuffer> attn_conv_base = nil;
    id<MTLBuffer> attn_conv_proj = nil;
    id<MTLBuffer> mlp_conv_base = nil;
    id<MTLBuffer> mlp_conv_proj = nil;
    id<MTLBuffer> k_proj_bf16 = nil;
    id<MTLBuffer> v_proj_bf16 = nil;
    id<MTLBuffer> q_fp8 = nil, q_sc = nil;
    id<MTLBuffer> k_fp8 = nil, k_sc = nil;
    id<MTLBuffer> v_fp8 = nil, v_sc = nil;
    id<MTLBuffer> o_fp8 = nil, o_sc = nil;
    id<MTLBuffer> g_fp8 = nil, g_sc = nil;
    id<MTLBuffer> u_fp8 = nil, u_sc = nil;
    id<MTLBuffer> d_fp8 = nil, d_sc = nil;
};

struct MetalLayerWeights {
    bool is_local = true;
    bool is_loop_core = false;
    bool is_mxfp8 = false;
    bool is_q2 = false;

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
    id<MTLBuffer> w13_min = nil;
    id<MTLBuffer> w13_amax = nil;
    id<MTLBuffer> w13_denom = nil;
    id<MTLBuffer> w2_weight = nil;
    id<MTLBuffer> w2_scale = nil;
    id<MTLBuffer> w2_min = nil;
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
    id<MTLBuffer> k_conv_snap = nil;
    id<MTLBuffer> v_conv_snap = nil;
    id<MTLBuffer> attn_conv_snap = nil;
    id<MTLBuffer> mlp_conv_snap = nil;
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
    id<MTLComputePipelineState> pso_attention_fp8_splitk = nil;
    id<MTLComputePipelineState> pso_attention_fp8_reduce = nil;
    id<MTLComputePipelineState> pso_attention_fp8_prefill4 = nil;
    id<MTLComputePipelineState> pso_router_logits = nil;
    id<MTLComputePipelineState> pso_router_topk = nil;
    id<MTLComputePipelineState> pso_moe_group = nil;
    id<MTLComputePipelineState> pso_moe_nvfp4_w13 = nil;
    id<MTLComputePipelineState> pso_moe_nvfp4_w2 = nil;
    id<MTLComputePipelineState> pso_moe_nvfp4_w13_grp = nil;
    id<MTLComputePipelineState> pso_moe_nvfp4_w2_grp = nil;
    id<MTLComputePipelineState> pso_moe_nvfp4_w13_dedup = nil;
    id<MTLComputePipelineState> pso_moe_nvfp4_w2_dedup = nil;
    id<MTLComputePipelineState> pso_moe_reduce_k = nil;
    id<MTLComputePipelineState> pso_moe_mxfp8_w13 = nil;
    id<MTLComputePipelineState> pso_moe_mxfp8_w2 = nil;
    id<MTLComputePipelineState> pso_moe_mxfp8_w13_grp = nil;
    id<MTLComputePipelineState> pso_moe_mxfp8_w2_grp = nil;
    id<MTLComputePipelineState> pso_moe_mxfp8_w13_dedup = nil;
    id<MTLComputePipelineState> pso_moe_mxfp8_w2_dedup = nil;
    id<MTLComputePipelineState> pso_moe_q2_w13 = nil;
    id<MTLComputePipelineState> pso_moe_q2_w2 = nil;
    id<MTLComputePipelineState> pso_moe_q2_w13_dedup = nil;
    id<MTLComputePipelineState> pso_moe_q2_w2_dedup = nil;
    id<MTLComputePipelineState> pso_bf16_gemv = nil;
    id<MTLComputePipelineState> pso_bf16_gemv_1sg = nil;
    id<MTLComputePipelineState> pso_quant_fp8 = nil;
    id<MTLComputePipelineState> pso_fp8_gemv = nil;
    id<MTLComputePipelineState> pso_fp8_small_gemm = nil;
    id<MTLComputePipelineState> pso_bf16_gemm_tiled = nil;
    id<MTLComputePipelineState> pso_softcap_argmax = nil;
    id<MTLComputePipelineState> pso_batched_argmax = nil;
    id<MTLComputePipelineState> pso_dflare_mix = nil;
    id<MTLComputePipelineState> pso_qk_rope = nil;
    id<MTLComputePipelineState> pso_draft_write_kv = nil;
    id<MTLComputePipelineState> pso_draft_attn = nil;
    id<MTLComputePipelineState> pso_add_rmsnorm = nil;
    id<MTLComputePipelineState> pso_silu_mul = nil;
    id<MTLComputePipelineState> pso_embed_draft = nil;
    id<MTLComputePipelineState> pso_assemble_markov = nil;
    id<MTLComputePipelineState> pso_markov_gate = nil;
    id<MTLComputePipelineState> pso_markov_argmax = nil;
    id<MTLComputePipelineState> pso_gather_bf16 = nil;
    id<MTLComputePipelineState> pso_fold_unembed = nil;
    id<MTLComputePipelineState> pso_zero_bf16 = nil;
    id<MTLComputePipelineState> pso_df2_concat = nil;
    id<MTLComputePipelineState> pso_df2_conv = nil;
    id<MTLComputePipelineState> pso_df2_attn = nil;
    id<MTLComputePipelineState> pso_df2_topk16 = nil;

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
    id<MTLBuffer> b_attn_split_ml = nil;
    id<MTLBuffer> b_attn_split_acc = nil;
    id<MTLBuffer> b_attn_proj = nil;
    id<MTLBuffer> b_attn_out = nil;
    id<MTLBuffer> b_mlp_raw = nil;
    id<MTLBuffer> b_mlp_out = nil;
    id<MTLBuffer> b_topk_ids = nil;
    id<MTLBuffer> b_topk_weights = nil;
    id<MTLBuffer> b_router_logits = nil;
    id<MTLBuffer> b_moe_act = nil;
    id<MTLBuffer> b_moe_offsets = nil;
    id<MTLBuffer> b_moe_sorted_tk = nil;
    id<MTLBuffer> b_moe_w2_out = nil;
    id<MTLBuffer> b_logits = nil;
    id<MTLBuffer> b_out_token = nil;
    id<MTLBuffer> b_aux_states = nil;
    id<MTLBuffer> b_last_aux = nil;

    bool ds_loaded = false;
    int dspark_slots = 0;
    int ds_num_layers = 3;
    int ds_hidden = 2304;
    int ds_heads = 24;
    int ds_kv_heads = 8;
    int ds_head_dim = 128;
    int ds_inter = 9216;
    int ds_block = 8;
    int ds_rank = 256;
    int ds_window = 1024;
    float ds_eps = 1e-6f;
    float ds_rope_theta = 1000000.0f;
    int ds_vocab_size = 200058;
    int ds_kv_cap[3] = {0, 0, 0};

    id<MTLBuffer> ds_fuse_alpha = nil;
    id<MTLBuffer> ds_mask_embed = nil;
    id<MTLBuffer> ds_final_norm = nil;
    std::vector<MetalDraftLayerWeights> ds_layers;
    id<MTLBuffer> ds_markov_dflare_norm = nil;
    id<MTLBuffer> ds_markov_w1 = nil;
    id<MTLBuffer> ds_rnn_proj_w = nil;
    id<MTLBuffer> ds_rnn_proj_b = nil;
    id<MTLBuffer> ds_pruned_unembed_fp8 = nil;
    id<MTLBuffer> ds_pruned_unembed_sc = nil;
    id<MTLBuffer> ds_draft_to_target = nil;
    id<MTLBuffer> ds_fold_w_fp8 = nil;
    id<MTLBuffer> ds_fold_w_sc = nil;
    id<MTLBuffer> ds_fold_b = nil;
    id<MTLBuffer> ds_k_cache[3] = {nil, nil, nil};
    id<MTLBuffer> ds_v_cache[3] = {nil, nil, nil};
    id<MTLBuffer> ds_ctx_mix = nil;
    id<MTLBuffer> ds_ctx_k = nil;
    id<MTLBuffer> ds_ctx_v = nil;
    id<MTLBuffer> ds_x = nil;
    id<MTLBuffer> ds_res = nil;
    id<MTLBuffer> ds_q = nil;
    id<MTLBuffer> ds_k = nil;
    id<MTLBuffer> ds_v = nil;
    id<MTLBuffer> ds_attn_out = nil;
    id<MTLBuffer> ds_proj_out = nil;
    id<MTLBuffer> ds_mlp_g = nil;
    id<MTLBuffer> ds_mlp_u = nil;
    id<MTLBuffer> ds_f_rnn = nil;
    id<MTLBuffer> ds_base_logits = nil;
    id<MTLBuffer> ds_rnn_in = nil;
    id<MTLBuffer> ds_rnn_raw = nil;
    id<MTLBuffer> ds_rnn_state = nil;
    id<MTLBuffer> ds_rnn_latent = nil;
    id<MTLBuffer> ds_markov_bias = nil;
    id<MTLBuffer> ds_draft_tok = nil;
    id<MTLBuffer> ds_anchor_tok = nil;

    bool df2_loaded = false;
    int df2_num_layers = 5;
    int df2_hidden = 2304;
    int df2_inter = 6912;
    int df2_heads = 32;
    int df2_kv_heads = 8;
    int df2_head_dim = 128;
    int df2_window = 2048;
    int df2_block = 8;
    int df2_mask_id = 200056;
    int df2_group_size = 16;
    int df2_sel_rank = 256;
    int df2_vocab_unpadded = 200058;
    float df2_rope_theta = 10000.0f;
    float df2_eps = 1e-6f;
    int df2_kv_cap = 2072;
    id<MTLBuffer> df2_fc = nil;
    id<MTLBuffer> df2_hidden_norm = nil;
    id<MTLBuffer> df2_final_norm = nil;
    id<MTLBuffer> df2_sel_hproj = nil;
    id<MTLBuffer> df2_sel_pred_cb = nil;
    id<MTLBuffer> df2_sel_succ_cb = nil;
    std::vector<MetalDFlash2LayerWeights> df2_layers;
    std::vector<id<MTLBuffer>> df2_k_cache;
    std::vector<id<MTLBuffer>> df2_v_cache;
    id<MTLBuffer> df2_ctx_cat = nil;
    id<MTLBuffer> df2_ctx_fc = nil;
    id<MTLBuffer> df2_ctx_norm = nil;
    id<MTLBuffer> df2_ctx_k = nil;
    id<MTLBuffer> df2_ctx_v = nil;
    id<MTLBuffer> df2_tok_ids = nil;
    id<MTLBuffer> df2_res = nil;
    id<MTLBuffer> df2_x = nil;
    id<MTLBuffer> df2_xc = nil;
    id<MTLBuffer> df2_dyn = nil;
    id<MTLBuffer> df2_q = nil;
    id<MTLBuffer> df2_k = nil;
    id<MTLBuffer> df2_v = nil;
    id<MTLBuffer> df2_attn_out = nil;
    id<MTLBuffer> df2_proj_out = nil;
    id<MTLBuffer> df2_mlp_g = nil;
    id<MTLBuffer> df2_mlp_u = nil;
    id<MTLBuffer> df2_h_sel = nil;
    id<MTLBuffer> df2_top_v = nil;
    id<MTLBuffer> df2_top_i = nil;

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
        if (M >= 2u && M <= 8u) {
            [enc setComputePipelineState:pso_fp8_small_gemm];
            [enc setBuffer:X offset:x_off atIndex:0];
            [enc setBuffer:W8 offset:w8_off atIndex:1];
            [enc setBuffer:Wsc offset:wsc_off atIndex:2];
            [enc setBuffer:(Y_bf ? Y_bf : b_dummy) offset:y_bf_off atIndex:3];
            [enc setBuffer:(Y_f32 ? Y_f32 : b_dummy) offset:y_f32_off atIndex:4];
            [enc setBytes:&gp length:sizeof(gp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((N + 7u) / 8u, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            return;
        }
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
                          bool profile_stages, bool save_conv_snaps, int aux_tok_offset = 0,
                          const int32_t* batch_slots = nullptr, const int32_t* batch_positions = nullptr)
    {
        bool batched_seq = (batch_slots != nullptr && batch_positions != nullptr);
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
            if (ds_loaded || df2_loaded) {
                int aux_slot = (v == 5) ? 0 : (v == 23 ? 1 : (v == 41 ? 2 : -1));
                if (aux_slot >= 0) {
                    blit_copy(b_stack, partial_off, b_aux_states,
                              (size_t)aux_slot * stack_stride_bytes + (size_t)aux_tok_offset * D * 2,
                              (size_t)T * D * 2);
                }
            }
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

            uint32_t snap_flag = save_conv_snaps ? 1u : 0u;
            int n_seq = batched_seq ? T : 1;
            int t_seq = batched_seq ? 1 : T;
            uint32_t dyn_dim = (uint32_t)cfg.num_key_value_heads * 4u;

            if (!lw.is_local) {
                encode_linear_bf16(enc, b_attn_in, k_in_off, lw.k_sconv_proj, 0, b_dyn_w, 0, nil, 0, (uint32_t)T, dyn_dim, D, false);
            }
            for (int b = 0; b < n_seq; ++b) {
                int s_id = batched_seq ? batch_slots[b] : slot_id;
                size_t k_cs_off = (size_t)s_id * 3 * kv_dim * 2;
                size_t k_snap_off = (size_t)s_id * 8 * 3 * kv_dim * 2;
                size_t tok_kv_off = (size_t)b * kv_dim * 2;
                size_t tok_dyn_off = (size_t)b * dyn_dim * 2;
                if (lw.is_local) {
                    SconvStaticParams sp{(uint32_t)t_seq, kv_dim, snap_flag};
                    [enc setComputePipelineState:pso_sconv_static];
                    [enc setBuffer:b_K offset:tok_kv_off atIndex:0];
                    [enc setBuffer:lw.k_sconv_w offset:0 atIndex:1];
                    [enc setBuffer:vc.k_conv_state offset:k_cs_off atIndex:2];
                    [enc setBuffer:b_K_conv offset:tok_kv_off atIndex:3];
                    [enc setBuffer:(save_conv_snaps ? vc.k_conv_snap : b_dummy) offset:(save_conv_snaps ? k_snap_off : 0) atIndex:4];
                    [enc setBytes:&sp length:sizeof(sp) atIndex:5];
                    [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    [enc setBuffer:b_V offset:tok_kv_off atIndex:0];
                    [enc setBuffer:lw.v_sconv_w offset:0 atIndex:1];
                    [enc setBuffer:vc.v_conv_state offset:k_cs_off atIndex:2];
                    [enc setBuffer:b_V_conv offset:tok_kv_off atIndex:3];
                    [enc setBuffer:(save_conv_snaps ? vc.v_conv_snap : b_dummy) offset:(save_conv_snaps ? k_snap_off : 0) atIndex:4];
                    [enc setBytes:&sp length:sizeof(sp) atIndex:5];
                    [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                } else {
                    SconvDynamicParams dp{(uint32_t)t_seq, (uint32_t)cfg.num_key_value_heads, (uint32_t)cfg.head_dim, snap_flag};
                    [enc setComputePipelineState:pso_sconv_dynamic];
                    [enc setBuffer:b_K offset:tok_kv_off atIndex:0];
                    [enc setBuffer:lw.k_sconv_w offset:0 atIndex:1];
                    [enc setBuffer:b_dyn_w offset:tok_dyn_off atIndex:2];
                    [enc setBuffer:vc.k_conv_state offset:k_cs_off atIndex:3];
                    [enc setBuffer:b_K_conv offset:tok_kv_off atIndex:4];
                    [enc setBuffer:(save_conv_snaps ? vc.k_conv_snap : b_dummy) offset:(save_conv_snaps ? k_snap_off : 0) atIndex:5];
                    [enc setBytes:&dp length:sizeof(dp) atIndex:6];
                    [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
            }
            if (!lw.is_local) {
                encode_linear_bf16(enc, b_attn_in, v_in_off, lw.v_sconv_proj, 0, b_dyn_w, 0, nil, 0, (uint32_t)T, dyn_dim, D, false);
                for (int b = 0; b < n_seq; ++b) {
                    int s_id = batched_seq ? batch_slots[b] : slot_id;
                    size_t k_cs_off = (size_t)s_id * 3 * kv_dim * 2;
                    size_t k_snap_off = (size_t)s_id * 8 * 3 * kv_dim * 2;
                    size_t tok_kv_off = (size_t)b * kv_dim * 2;
                    size_t tok_dyn_off = (size_t)b * dyn_dim * 2;
                    SconvDynamicParams dp{(uint32_t)t_seq, (uint32_t)cfg.num_key_value_heads, (uint32_t)cfg.head_dim, snap_flag};
                    [enc setComputePipelineState:pso_sconv_dynamic];
                    [enc setBuffer:b_V offset:tok_kv_off atIndex:0];
                    [enc setBuffer:lw.v_sconv_w offset:0 atIndex:1];
                    [enc setBuffer:b_dyn_w offset:tok_dyn_off atIndex:2];
                    [enc setBuffer:vc.v_conv_state offset:k_cs_off atIndex:3];
                    [enc setBuffer:b_V_conv offset:tok_kv_off atIndex:4];
                    [enc setBuffer:(save_conv_snaps ? vc.v_conv_snap : b_dummy) offset:(save_conv_snaps ? k_snap_off : 0) atIndex:5];
                    [enc setBytes:&dp length:sizeof(dp) atIndex:6];
                    [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
            }

            for (int b = 0; b < n_seq; ++b) {
                int s_id = batched_seq ? batch_slots[b] : slot_id;
                int pos_b = batched_seq ? batch_positions[b] : start_pos;
                size_t kv_slot_off = (size_t)s_id * (size_t)vc.cap * kv_dim * (vc.is_local ? 2 : 1);
                size_t sc_slot_off = (size_t)s_id * (size_t)vc.cap * (size_t)cfg.num_key_value_heads * sizeof(float);
                size_t tok_q_off = (size_t)b * q_dim * 2;
                size_t tok_r_off = (size_t)b * r_dim * 2;
                size_t tok_kv_off = (size_t)b * kv_dim * 2;

                AttnPrepQRParams qrp{
                    (uint32_t)pos_b, (uint32_t)t_seq, lw.is_local ? 1u : 0u,
                    (uint32_t)cfg.num_attention_heads, (uint32_t)cfg.head_dim, (uint32_t)cfg.d_rel,
                    cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold,
                    cfg.log_scaling_n_floor, cfg.log_scaling_alpha
                };
                [enc setComputePipelineState:pso_attn_prep_qr];
                [enc setBuffer:b_Q offset:tok_q_off atIndex:0];
                [enc setBuffer:b_R offset:tok_r_off atIndex:1];
                [enc setBuffer:lw.q_clip_scale offset:0 atIndex:2];
                [enc setBuffer:lw.rel_clip_scale offset:0 atIndex:3];
                [enc setBytes:&qrp length:sizeof(qrp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((uint32_t)cfg.num_attention_heads, (uint32_t)t_seq, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                AttnPrepKVParams kvp{
                    (uint32_t)pos_b, (uint32_t)t_seq, (uint32_t)vc.cap, lw.is_local ? 1u : 0u,
                    (uint32_t)cfg.num_key_value_heads, (uint32_t)cfg.head_dim,
                    cfg.qk_l2_clip_threshold, cfg.rms_norm_eps
                };
                if (vc.is_local) {
                    [enc setComputePipelineState:pso_attn_prep_kv];
                    [enc setBuffer:b_K_conv offset:tok_kv_off atIndex:0];
                    [enc setBuffer:b_V_conv offset:tok_kv_off atIndex:1];
                    [enc setBuffer:vc.k_cache offset:kv_slot_off atIndex:2];
                    [enc setBuffer:vc.v_cache offset:kv_slot_off atIndex:3];
                    [enc setBytes:&kvp length:sizeof(kvp) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake((uint32_t)cfg.num_key_value_heads, (uint32_t)t_seq, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                } else {
                    [enc setComputePipelineState:pso_attn_prep_kv_fp8];
                    [enc setBuffer:b_K_conv offset:tok_kv_off atIndex:0];
                    [enc setBuffer:b_V_conv offset:tok_kv_off atIndex:1];
                    [enc setBuffer:vc.k_cache_fp8 offset:kv_slot_off atIndex:2];
                    [enc setBuffer:vc.v_cache_fp8 offset:kv_slot_off atIndex:3];
                    [enc setBuffer:vc.k_scale_fp8 offset:sc_slot_off atIndex:4];
                    [enc setBuffer:vc.v_scale_fp8 offset:sc_slot_off atIndex:5];
                    [enc setBytes:&kvp length:sizeof(kvp) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake((uint32_t)cfg.num_key_value_heads, (uint32_t)t_seq, 1)
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

            for (int b = 0; b < n_seq; ++b) {
                int s_id = batched_seq ? batch_slots[b] : slot_id;
                int pos_b = batched_seq ? batch_positions[b] : start_pos;
                size_t kv_slot_off = (size_t)s_id * (size_t)vc.cap * kv_dim * (vc.is_local ? 2 : 1);
                size_t sc_slot_off = (size_t)s_id * (size_t)vc.cap * (size_t)cfg.num_key_value_heads * sizeof(float);
                size_t tok_q_off = (size_t)b * q_dim * 2;
                size_t tok_rel_off = (size_t)b * (size_t)cfg.num_attention_heads * (size_t)cfg.rel_extent * 2;
                size_t tok_sink_off = (size_t)b * (size_t)cfg.num_attention_heads * 2;

                if (vc.is_local) {
                    AttnParams ap{
                        (uint32_t)pos_b, (uint32_t)t_seq, (uint32_t)vc.cap, 1u,
                        (uint32_t)cfg.sliding_window_size, (uint32_t)cfg.num_attention_heads,
                        (uint32_t)cfg.num_key_value_heads, (uint32_t)cfg.head_dim, (uint32_t)cfg.rel_extent
                    };
                    [enc setComputePipelineState:pso_attention];
                    [enc setBuffer:b_Q offset:tok_q_off atIndex:0];
                    [enc setBuffer:vc.k_cache offset:kv_slot_off atIndex:1];
                    [enc setBuffer:vc.v_cache offset:kv_slot_off atIndex:2];
                    [enc setBuffer:b_rel_logits offset:tok_rel_off atIndex:3];
                    [enc setBuffer:b_gate offset:tok_q_off atIndex:4];
                    [enc setBuffer:b_sinks offset:tok_sink_off atIndex:5];
                    [enc setBuffer:b_attn_raw offset:tok_q_off atIndex:6];
                    [enc setBytes:&ap length:sizeof(ap) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake((uint32_t)cfg.num_attention_heads, (uint32_t)t_seq, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                } else if (t_seq <= 8 && (pos_b + t_seq) >= 512) {
                    uint32_t k_end = (uint32_t)(pos_b + t_seq);
                    uint32_t num_splits = std::min(64u, std::max(4u, k_end / 512u));
                    AttnSplitKParams skp{
                        (uint32_t)pos_b, (uint32_t)t_seq, (uint32_t)vc.cap, 0u,
                        (uint32_t)cfg.sliding_window_size, (uint32_t)cfg.num_attention_heads,
                        (uint32_t)cfg.num_key_value_heads, (uint32_t)cfg.head_dim,
                        (uint32_t)cfg.rel_extent, num_splits
                    };
                    [enc setComputePipelineState:pso_attention_fp8_splitk];
                    [enc setBuffer:b_Q offset:tok_q_off atIndex:0];
                    [enc setBuffer:vc.k_cache_fp8 offset:kv_slot_off atIndex:1];
                    [enc setBuffer:vc.v_cache_fp8 offset:kv_slot_off atIndex:2];
                    [enc setBuffer:vc.k_scale_fp8 offset:sc_slot_off atIndex:3];
                    [enc setBuffer:vc.v_scale_fp8 offset:sc_slot_off atIndex:4];
                    [enc setBuffer:b_rel_logits offset:tok_rel_off atIndex:5];
                    [enc setBuffer:b_attn_split_ml offset:0 atIndex:6];
                    [enc setBuffer:b_attn_split_acc offset:0 atIndex:7];
                    [enc setBytes:&skp length:sizeof(skp) atIndex:8];
                    [enc dispatchThreadgroups:MTLSizeMake((uint32_t)cfg.num_attention_heads, (uint32_t)t_seq, num_splits)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    [enc setComputePipelineState:pso_attention_fp8_reduce];
                    [enc setBuffer:b_attn_split_ml offset:0 atIndex:0];
                    [enc setBuffer:b_attn_split_acc offset:0 atIndex:1];
                    [enc setBuffer:b_gate offset:tok_q_off atIndex:2];
                    [enc setBuffer:b_sinks offset:tok_sink_off atIndex:3];
                    [enc setBuffer:b_attn_raw offset:tok_q_off atIndex:4];
                    [enc setBytes:&skp length:sizeof(skp) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake((uint32_t)cfg.num_attention_heads, (uint32_t)t_seq, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                } else if (t_seq >= 16) {
                    AttnParams ap{
                        (uint32_t)pos_b, (uint32_t)t_seq, (uint32_t)vc.cap, 0u,
                        (uint32_t)cfg.sliding_window_size, (uint32_t)cfg.num_attention_heads,
                        (uint32_t)cfg.num_key_value_heads, (uint32_t)cfg.head_dim, (uint32_t)cfg.rel_extent
                    };
                    [enc setComputePipelineState:pso_attention_fp8_prefill4];
                    [enc setBuffer:b_Q offset:tok_q_off atIndex:0];
                    [enc setBuffer:vc.k_cache_fp8 offset:kv_slot_off atIndex:1];
                    [enc setBuffer:vc.v_cache_fp8 offset:kv_slot_off atIndex:2];
                    [enc setBuffer:vc.k_scale_fp8 offset:sc_slot_off atIndex:3];
                    [enc setBuffer:vc.v_scale_fp8 offset:sc_slot_off atIndex:4];
                    [enc setBuffer:b_rel_logits offset:tok_rel_off atIndex:5];
                    [enc setBuffer:b_gate offset:tok_q_off atIndex:6];
                    [enc setBuffer:b_sinks offset:tok_sink_off atIndex:7];
                    [enc setBuffer:b_attn_raw offset:tok_q_off atIndex:8];
                    [enc setBytes:&ap length:sizeof(ap) atIndex:9];
                    [enc dispatchThreadgroups:MTLSizeMake((uint32_t)cfg.num_attention_heads, (uint32_t)((t_seq + 15) / 16), 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                } else {
                    AttnParams ap{
                        (uint32_t)pos_b, (uint32_t)t_seq, (uint32_t)vc.cap, 0u,
                        (uint32_t)cfg.sliding_window_size, (uint32_t)cfg.num_attention_heads,
                        (uint32_t)cfg.num_key_value_heads, (uint32_t)cfg.head_dim, (uint32_t)cfg.rel_extent
                    };
                    [enc setComputePipelineState:pso_attention_fp8];
                    [enc setBuffer:b_Q offset:tok_q_off atIndex:0];
                    [enc setBuffer:vc.k_cache_fp8 offset:kv_slot_off atIndex:1];
                    [enc setBuffer:vc.v_cache_fp8 offset:kv_slot_off atIndex:2];
                    [enc setBuffer:vc.k_scale_fp8 offset:sc_slot_off atIndex:3];
                    [enc setBuffer:vc.v_scale_fp8 offset:sc_slot_off atIndex:4];
                    [enc setBuffer:b_rel_logits offset:tok_rel_off atIndex:5];
                    [enc setBuffer:b_gate offset:tok_q_off atIndex:6];
                    [enc setBuffer:b_sinks offset:tok_sink_off atIndex:7];
                    [enc setBuffer:b_attn_raw offset:tok_q_off atIndex:8];
                    [enc setBytes:&ap length:sizeof(ap) atIndex:9];
                    [enc dispatchThreadgroups:MTLSizeMake((uint32_t)cfg.num_attention_heads, (uint32_t)t_seq, 1)
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

            for (int b = 0; b < n_seq; ++b) {
                int s_id = batched_seq ? batch_slots[b] : slot_id;
                size_t d_cs_off = (size_t)s_id * 3 * D * 2;
                size_t d_snap_off = (size_t)s_id * 8 * 3 * D * 2;
                size_t tok_d_off = (size_t)b * D * 2;
                SconvStaticParams sp{(uint32_t)t_seq, D, snap_flag};
                [enc setComputePipelineState:pso_sconv_static];
                [enc setBuffer:b_attn_proj offset:tok_d_off atIndex:0];
                [enc setBuffer:lw.attn_sconv_w offset:0 atIndex:1];
                [enc setBuffer:vc.attn_conv_state offset:d_cs_off atIndex:2];
                [enc setBuffer:b_attn_out offset:tok_d_off atIndex:3];
                [enc setBuffer:(save_conv_snaps ? vc.attn_conv_snap : b_dummy) offset:(save_conv_snaps ? d_snap_off : 0) atIndex:4];
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
                if (lw.is_q2 && T >= 2) {
                    MoeGroupParams gp{(uint32_t)(T * cfg.num_experts_per_tok), (uint32_t)cfg.n_routed_experts};
                    [enc setComputePipelineState:pso_moe_group];
                    [enc setBuffer:b_topk_ids offset:0 atIndex:0];
                    [enc setBuffer:b_moe_offsets offset:0 atIndex:1];
                    [enc setBuffer:b_moe_sorted_tk offset:0 atIndex:2];
                    [enc setBytes:&gp length:sizeof(gp) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:pso_moe_q2_w13_dedup];
                    [enc setBuffer:b_mlp_in offset:0 atIndex:0];
                    [enc setBuffer:b_topk_weights offset:0 atIndex:1];
                    [enc setBuffer:b_moe_offsets offset:0 atIndex:2];
                    [enc setBuffer:b_moe_sorted_tk offset:0 atIndex:3];
                    [enc setBuffer:lw.w13_weight offset:0 atIndex:4];
                    [enc setBuffer:lw.w13_scale offset:0 atIndex:5];
                    [enc setBuffer:lw.w13_min offset:0 atIndex:6];
                    [enc setBuffer:b_moe_act offset:0 atIndex:7];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:8];
                    [enc dispatchThreadgroups:MTLSizeMake((mp.inter + 3u) / 4u, (uint32_t)cfg.n_routed_experts, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:pso_moe_q2_w2_dedup];
                    [enc setBuffer:b_moe_act offset:0 atIndex:0];
                    [enc setBuffer:b_moe_offsets offset:0 atIndex:1];
                    [enc setBuffer:b_moe_sorted_tk offset:0 atIndex:2];
                    [enc setBuffer:lw.w2_weight offset:0 atIndex:3];
                    [enc setBuffer:lw.w2_scale offset:0 atIndex:4];
                    [enc setBuffer:lw.w2_min offset:0 atIndex:5];
                    [enc setBuffer:b_moe_w2_out offset:0 atIndex:6];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake((D + 3u) / 4u, (uint32_t)cfg.n_routed_experts, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:pso_moe_reduce_k];
                    [enc setBuffer:b_moe_w2_out offset:0 atIndex:0];
                    [enc setBuffer:b_mlp_raw offset:0 atIndex:1];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:2];
                    [enc dispatchThreads:MTLSizeMake(D, mp.T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    flush_stage(prof_moe_nvfp4_ms);
                } else if (lw.is_q2) {
                    [enc setComputePipelineState:pso_moe_q2_w13];
                    [enc setBuffer:b_mlp_in offset:0 atIndex:0];
                    [enc setBuffer:b_topk_ids offset:0 atIndex:1];
                    [enc setBuffer:b_topk_weights offset:0 atIndex:2];
                    [enc setBuffer:lw.w13_weight offset:0 atIndex:3];
                    [enc setBuffer:lw.w13_scale offset:0 atIndex:4];
                    [enc setBuffer:lw.w13_min offset:0 atIndex:5];
                    [enc setBuffer:b_moe_act offset:0 atIndex:6];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake((mp.inter + 3u) / 4u, mp.T * mp.K, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:pso_moe_q2_w2];
                    [enc setBuffer:b_moe_act offset:0 atIndex:0];
                    [enc setBuffer:b_topk_ids offset:0 atIndex:1];
                    [enc setBuffer:lw.w2_weight offset:0 atIndex:2];
                    [enc setBuffer:lw.w2_scale offset:0 atIndex:3];
                    [enc setBuffer:lw.w2_min offset:0 atIndex:4];
                    [enc setBuffer:b_mlp_raw offset:0 atIndex:5];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake((D + 3u) / 4u, mp.T, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    flush_stage(prof_moe_nvfp4_ms);
                } else if (lw.is_mxfp8 && T >= 16) {
                    MoeGroupParams gp{(uint32_t)(T * cfg.num_experts_per_tok), (uint32_t)cfg.n_routed_experts};
                    [enc setComputePipelineState:pso_moe_group];
                    [enc setBuffer:b_topk_ids offset:0 atIndex:0];
                    [enc setBuffer:b_moe_offsets offset:0 atIndex:1];
                    [enc setBuffer:b_moe_sorted_tk offset:0 atIndex:2];
                    [enc setBytes:&gp length:sizeof(gp) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:pso_moe_mxfp8_w13_grp];
                    [enc setBuffer:b_mlp_in offset:0 atIndex:0];
                    [enc setBuffer:b_topk_weights offset:0 atIndex:1];
                    [enc setBuffer:b_moe_offsets offset:0 atIndex:2];
                    [enc setBuffer:b_moe_sorted_tk offset:0 atIndex:3];
                    [enc setBuffer:lw.w13_weight offset:0 atIndex:4];
                    [enc setBuffer:lw.w13_scale offset:0 atIndex:5];
                    [enc setBuffer:b_moe_act offset:0 atIndex:6];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake((mp.inter + 15u) / 16u, (uint32_t)cfg.n_routed_experts, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:pso_moe_mxfp8_w2_grp];
                    [enc setBuffer:b_moe_act offset:0 atIndex:0];
                    [enc setBuffer:b_moe_offsets offset:0 atIndex:1];
                    [enc setBuffer:b_moe_sorted_tk offset:0 atIndex:2];
                    [enc setBuffer:lw.w2_weight offset:0 atIndex:3];
                    [enc setBuffer:lw.w2_scale offset:0 atIndex:4];
                    [enc setBuffer:b_moe_w2_out offset:0 atIndex:5];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake((D + 31u) / 32u, (uint32_t)cfg.n_routed_experts, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:pso_moe_reduce_k];
                    [enc setBuffer:b_moe_w2_out offset:0 atIndex:0];
                    [enc setBuffer:b_mlp_raw offset:0 atIndex:1];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:2];
                    [enc dispatchThreads:MTLSizeMake(D, mp.T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    flush_stage(prof_moe_mxfp8_ms);
                } else if (lw.is_mxfp8 && T >= 2) {
                    MoeGroupParams gp{(uint32_t)(T * cfg.num_experts_per_tok), (uint32_t)cfg.n_routed_experts};
                    [enc setComputePipelineState:pso_moe_group];
                    [enc setBuffer:b_topk_ids offset:0 atIndex:0];
                    [enc setBuffer:b_moe_offsets offset:0 atIndex:1];
                    [enc setBuffer:b_moe_sorted_tk offset:0 atIndex:2];
                    [enc setBytes:&gp length:sizeof(gp) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:pso_moe_mxfp8_w13_dedup];
                    [enc setBuffer:b_mlp_in offset:0 atIndex:0];
                    [enc setBuffer:b_topk_weights offset:0 atIndex:1];
                    [enc setBuffer:b_moe_offsets offset:0 atIndex:2];
                    [enc setBuffer:b_moe_sorted_tk offset:0 atIndex:3];
                    [enc setBuffer:lw.w13_weight offset:0 atIndex:4];
                    [enc setBuffer:lw.w13_scale offset:0 atIndex:5];
                    [enc setBuffer:b_moe_act offset:0 atIndex:6];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake((mp.inter + 3u) / 4u, (uint32_t)cfg.n_routed_experts, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:pso_moe_mxfp8_w2_dedup];
                    [enc setBuffer:b_moe_act offset:0 atIndex:0];
                    [enc setBuffer:b_moe_offsets offset:0 atIndex:1];
                    [enc setBuffer:b_moe_sorted_tk offset:0 atIndex:2];
                    [enc setBuffer:lw.w2_weight offset:0 atIndex:3];
                    [enc setBuffer:lw.w2_scale offset:0 atIndex:4];
                    [enc setBuffer:b_moe_w2_out offset:0 atIndex:5];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake((D + 3u) / 4u, (uint32_t)cfg.n_routed_experts, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:pso_moe_reduce_k];
                    [enc setBuffer:b_moe_w2_out offset:0 atIndex:0];
                    [enc setBuffer:b_mlp_raw offset:0 atIndex:1];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:2];
                    [enc dispatchThreads:MTLSizeMake(D, mp.T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    flush_stage(prof_moe_mxfp8_ms);
                } else if (lw.is_mxfp8) {
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
                } else if (T >= 16) {
                    MoeGroupParams gp{(uint32_t)(T * cfg.num_experts_per_tok), (uint32_t)cfg.n_routed_experts};
                    [enc setComputePipelineState:pso_moe_group];
                    [enc setBuffer:b_topk_ids offset:0 atIndex:0];
                    [enc setBuffer:b_moe_offsets offset:0 atIndex:1];
                    [enc setBuffer:b_moe_sorted_tk offset:0 atIndex:2];
                    [enc setBytes:&gp length:sizeof(gp) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:pso_moe_nvfp4_w13_grp];
                    [enc setBuffer:b_mlp_in offset:0 atIndex:0];
                    [enc setBuffer:b_topk_weights offset:0 atIndex:1];
                    [enc setBuffer:b_moe_offsets offset:0 atIndex:2];
                    [enc setBuffer:b_moe_sorted_tk offset:0 atIndex:3];
                    [enc setBuffer:lw.w13_weight offset:0 atIndex:4];
                    [enc setBuffer:lw.w13_scale offset:0 atIndex:5];
                    [enc setBuffer:lw.w13_amax offset:0 atIndex:6];
                    [enc setBuffer:lw.w13_denom offset:0 atIndex:7];
                    [enc setBuffer:b_moe_act offset:0 atIndex:8];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:9];
                    [enc dispatchThreadgroups:MTLSizeMake((mp.inter + 15u) / 16u, (uint32_t)cfg.n_routed_experts, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:pso_moe_nvfp4_w2_grp];
                    [enc setBuffer:b_moe_act offset:0 atIndex:0];
                    [enc setBuffer:b_moe_offsets offset:0 atIndex:1];
                    [enc setBuffer:b_moe_sorted_tk offset:0 atIndex:2];
                    [enc setBuffer:lw.w2_weight offset:0 atIndex:3];
                    [enc setBuffer:lw.w2_scale offset:0 atIndex:4];
                    [enc setBuffer:lw.w2_amax offset:0 atIndex:5];
                    [enc setBuffer:lw.w2_denom offset:0 atIndex:6];
                    [enc setBuffer:b_moe_w2_out offset:0 atIndex:7];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:8];
                    [enc dispatchThreadgroups:MTLSizeMake((D + 31u) / 32u, (uint32_t)cfg.n_routed_experts, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:pso_moe_reduce_k];
                    [enc setBuffer:b_moe_w2_out offset:0 atIndex:0];
                    [enc setBuffer:b_mlp_raw offset:0 atIndex:1];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:2];
                    [enc dispatchThreads:MTLSizeMake(D, mp.T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    flush_stage(prof_moe_nvfp4_ms);
                } else if (T >= 2) {
                    MoeGroupParams gp{(uint32_t)(T * cfg.num_experts_per_tok), (uint32_t)cfg.n_routed_experts};
                    [enc setComputePipelineState:pso_moe_group];
                    [enc setBuffer:b_topk_ids offset:0 atIndex:0];
                    [enc setBuffer:b_moe_offsets offset:0 atIndex:1];
                    [enc setBuffer:b_moe_sorted_tk offset:0 atIndex:2];
                    [enc setBytes:&gp length:sizeof(gp) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:pso_moe_nvfp4_w13_dedup];
                    [enc setBuffer:b_mlp_in offset:0 atIndex:0];
                    [enc setBuffer:b_topk_weights offset:0 atIndex:1];
                    [enc setBuffer:b_moe_offsets offset:0 atIndex:2];
                    [enc setBuffer:b_moe_sorted_tk offset:0 atIndex:3];
                    [enc setBuffer:lw.w13_weight offset:0 atIndex:4];
                    [enc setBuffer:lw.w13_scale offset:0 atIndex:5];
                    [enc setBuffer:lw.w13_amax offset:0 atIndex:6];
                    [enc setBuffer:lw.w13_denom offset:0 atIndex:7];
                    [enc setBuffer:b_moe_act offset:0 atIndex:8];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:9];
                    [enc dispatchThreadgroups:MTLSizeMake((mp.inter + 3u) / 4u, (uint32_t)cfg.n_routed_experts, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:pso_moe_nvfp4_w2_dedup];
                    [enc setBuffer:b_moe_act offset:0 atIndex:0];
                    [enc setBuffer:b_moe_offsets offset:0 atIndex:1];
                    [enc setBuffer:b_moe_sorted_tk offset:0 atIndex:2];
                    [enc setBuffer:lw.w2_weight offset:0 atIndex:3];
                    [enc setBuffer:lw.w2_scale offset:0 atIndex:4];
                    [enc setBuffer:lw.w2_amax offset:0 atIndex:5];
                    [enc setBuffer:lw.w2_denom offset:0 atIndex:6];
                    [enc setBuffer:b_moe_w2_out offset:0 atIndex:7];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:8];
                    [enc dispatchThreadgroups:MTLSizeMake((D + 3u) / 4u, (uint32_t)cfg.n_routed_experts, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:pso_moe_reduce_k];
                    [enc setBuffer:b_moe_w2_out offset:0 atIndex:0];
                    [enc setBuffer:b_mlp_raw offset:0 atIndex:1];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:2];
                    [enc dispatchThreads:MTLSizeMake(D, mp.T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    flush_stage(prof_moe_nvfp4_ms);
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

            for (int b = 0; b < n_seq; ++b) {
                int s_id = batched_seq ? batch_slots[b] : slot_id;
                size_t d_cs_off = (size_t)s_id * 3 * D * 2;
                size_t d_snap_off = (size_t)s_id * 8 * 3 * D * 2;
                size_t tok_d_off = (size_t)b * D * 2;
                SconvStaticParams sp{(uint32_t)t_seq, D, snap_flag};
                [enc setComputePipelineState:pso_sconv_static];
                [enc setBuffer:b_mlp_raw offset:tok_d_off atIndex:0];
                [enc setBuffer:lw.mlp_sconv_w offset:0 atIndex:1];
                [enc setBuffer:vc.mlp_conv_state offset:d_cs_off atIndex:2];
                [enc setBuffer:b_mlp_out offset:tok_d_off atIndex:3];
                [enc setBuffer:(save_conv_snaps ? vc.mlp_conv_snap : b_dummy) offset:(save_conv_snaps ? d_snap_off : 0) atIndex:4];
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
        if (ds_loaded || df2_loaded) {
            blit_copy(b_mlp_in, 0, b_aux_states,
                      (size_t)3 * stack_stride_bytes + (size_t)aux_tok_offset * D * 2,
                      (size_t)T * D * 2);
        }
        flush_stage(prof_attn_res_ctrl_ms);

        if (save_conv_snaps || batched_seq) {
            encode_linear_fp8(enc, b_mlp_in, 0, unembed_fp8, 0, unembed_sc, 0, nil, 0, b_logits, 0,
                              (uint32_t)T, (uint32_t)cfg.unpadded_vocab_size, D, true);
            BatchArgParams bap{(uint32_t)T, (uint32_t)cfg.unpadded_vocab_size};
            [enc setComputePipelineState:pso_batched_argmax];
            [enc setBuffer:b_logits offset:0 atIndex:0];
            [enc setBuffer:b_out_token offset:0 atIndex:1];
            [enc setBytes:&bap length:sizeof(bap) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((uint32_t)T, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
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
        }

        [enc endEncoding];
        commit_and_wait(cb);
        if (profile_stages) {
            prof_lm_head_ms += (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            prof_steps += 1;
        }
    }

    void load_dspark(const std::string& model_dir, const InklingConfig& cfg,
                     int max_seq, int max_ctx_chunk, const std::string& draft_vocab_file, int n_ds_slots)
    {
        auto t0 = std::chrono::steady_clock::now();
        dspark_slots = n_ds_slots;
        SafetensorsFile dst;
        dst.open(model_dir + "/mtp.safetensors");

        {
            std::vector<uint8_t> raw;
            dst.load_to_host("fuse_weight", raw);
            const uint16_t* fw = reinterpret_cast<const uint16_t*>(raw.data());
            ds_fuse_alpha = alloc_shared(16 * sizeof(float));
            float* fa = static_cast<float*>(ds_fuse_alpha.contents);
            for (int r = 0; r < 4; ++r) {
                float m = -1e30f;
                for (int c = 0; c < 4; ++c) m = std::max(m, bf16_bits_to_f32(fw[r * 4 + c]));
                float s = 0.0f;
                for (int c = 0; c < 4; ++c) {
                    fa[r * 4 + c] = std::exp(bf16_bits_to_f32(fw[r * 4 + c]) - m);
                    s += fa[r * 4 + c];
                }
                for (int c = 0; c < 4; ++c) fa[r * 4 + c] /= s;
            }
        }

        ds_mask_embed = load_tensor(dst, "mask_embed");
        ds_final_norm = load_tensor(dst, "norm.weight");

        id<MTLCommandBuffer> qcb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> qenc = [qcb computeCommandEncoder];

        auto load_and_quant = [&](const std::string& name, id<MTLBuffer> __strong& dst_fp8, id<MTLBuffer> __strong& dst_sc, uint32_t N, uint32_t K) {
            id<MTLBuffer> tmp = load_tensor(dst, name);
            dst_fp8 = alloc_shared((size_t)N * K);
            dst_sc = alloc_shared((size_t)N * sizeof(float));
            quantize_rowwise_fp8(qenc, tmp, dst_fp8, dst_sc, N, K);
        };

        uint32_t q_dim = (uint32_t)(ds_heads * ds_head_dim);
        uint32_t kv_dim = (uint32_t)(ds_kv_heads * ds_head_dim);
        ds_layers.resize(ds_num_layers);
        for (int i = 0; i < ds_num_layers; ++i) {
            auto& lw = ds_layers[i];
            lw.is_local = (i == 0 || i == 2);
            std::string p = "layers." + std::to_string(i) + ".";
            lw.dflare_norm = load_tensor(dst, p + "dflare_norm.weight");
            lw.input_ln = load_tensor(dst, p + "input_layernorm.weight");
            lw.post_attn_ln = load_tensor(dst, p + "post_attention_layernorm.weight");
            lw.k_proj_ctx = load_tensor(dst, p + "self_attn.k_proj_ctx.weight");
            lw.v_proj_ctx = load_tensor(dst, p + "self_attn.v_proj_ctx.weight");
            lw.q_norm = load_tensor(dst, p + "self_attn.q_norm.weight");
            lw.k_norm = load_tensor(dst, p + "self_attn.k_norm.weight");
            lw.k_norm_ctx = load_tensor(dst, p + "self_attn.k_norm_ctx.weight");
            load_and_quant(p + "self_attn.q_proj.weight", lw.q_fp8, lw.q_sc, q_dim, (uint32_t)ds_hidden);
            load_and_quant(p + "self_attn.k_proj.weight", lw.k_fp8, lw.k_sc, kv_dim, (uint32_t)ds_hidden);
            load_and_quant(p + "self_attn.v_proj.weight", lw.v_fp8, lw.v_sc, kv_dim, (uint32_t)ds_hidden);
            load_and_quant(p + "self_attn.o_proj.weight", lw.o_fp8, lw.o_sc, (uint32_t)ds_hidden, q_dim);
            load_and_quant(p + "mlp.gate_proj.weight", lw.gate_fp8, lw.gate_sc, (uint32_t)ds_inter, (uint32_t)ds_hidden);
            load_and_quant(p + "mlp.up_proj.weight", lw.up_fp8, lw.up_sc, (uint32_t)ds_inter, (uint32_t)ds_hidden);
            load_and_quant(p + "mlp.down_proj.weight", lw.down_fp8, lw.down_sc, (uint32_t)ds_hidden, (uint32_t)ds_inter);
        }

        ds_markov_dflare_norm = load_tensor(dst, "markov_head.dflare_norm.weight");
        ds_markov_w1 = load_tensor(dst, "markov_head.markov_w1.weight");
        ds_rnn_proj_w = load_tensor(dst, "markov_head.rnn_proj.weight");
        {
            std::vector<uint8_t> raw;
            dst.load_to_host("markov_head.rnn_proj.bias", raw);
            const uint16_t* bp = reinterpret_cast<const uint16_t*>(raw.data());
            ds_rnn_proj_b = alloc_shared((size_t)3 * ds_rank * sizeof(float));
            float* fb = static_cast<float*>(ds_rnn_proj_b.contents);
            for (int i = 0; i < 3 * ds_rank; ++i) fb[i] = bf16_bits_to_f32(bp[i]);
        }

        id<MTLBuffer> draft_unembed_src = unembed_w;
        if (!draft_vocab_file.empty()) {
            std::ifstream vf(draft_vocab_file, std::ios::binary | std::ios::ate);
            if (!vf) throw std::runtime_error("Cannot open draft vocab file: " + draft_vocab_file);
            size_t nbytes = (size_t)vf.tellg();
            vf.seekg(0);
            ds_vocab_size = (int)(nbytes / sizeof(int32_t));
            ds_draft_to_target = alloc_shared(nbytes);
            vf.read(static_cast<char*>(ds_draft_to_target.contents), (std::streamsize)nbytes);
            id<MTLBuffer> pruned = alloc_shared((size_t)ds_vocab_size * ds_hidden * 2);
            GatherParams gp{(uint32_t)ds_vocab_size, (uint32_t)ds_hidden};
            [qenc setComputePipelineState:pso_gather_bf16];
            [qenc setBuffer:unembed_w offset:0 atIndex:0];
            [qenc setBuffer:ds_draft_to_target offset:0 atIndex:1];
            [qenc setBuffer:pruned offset:0 atIndex:2];
            [qenc setBytes:&gp length:sizeof(gp) atIndex:3];
            [qenc dispatchThreads:MTLSizeMake((uint32_t)ds_hidden, (uint32_t)ds_vocab_size, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
            draft_unembed_src = pruned;
        } else {
            ds_vocab_size = cfg.unpadded_vocab_size;
            ds_draft_to_target = nil;
        }

        id<MTLBuffer> ho_w = load_tensor(dst, "markov_head.hidden_out.weight");
        id<MTLBuffer> ho_b = load_tensor(dst, "markov_head.hidden_out.bias");
        id<MTLBuffer> fold_w_bf16 = alloc_shared((size_t)ds_vocab_size * ds_rank * 2);
        ds_fold_b = alloc_shared((size_t)ds_vocab_size * sizeof(float));

        FoldParams fp{(uint32_t)ds_vocab_size, (uint32_t)ds_hidden, (uint32_t)ds_rank};
        [qenc setComputePipelineState:pso_fold_unembed];
        [qenc setBuffer:draft_unembed_src offset:0 atIndex:0];
        [qenc setBuffer:ho_w offset:0 atIndex:1];
        [qenc setBuffer:fold_w_bf16 offset:0 atIndex:2];
        [qenc setBytes:&fp length:sizeof(fp) atIndex:3];
        [qenc dispatchThreads:MTLSizeMake((uint32_t)ds_rank, (uint32_t)ds_vocab_size, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];

        encode_linear_bf16(qenc, ho_b, 0, draft_unembed_src, 0, nil, 0, ds_fold_b, 0,
                           1u, (uint32_t)ds_vocab_size, (uint32_t)ds_hidden, true);

        ds_pruned_unembed_fp8 = alloc_shared((size_t)ds_vocab_size * ds_hidden);
        ds_pruned_unembed_sc = alloc_shared((size_t)ds_vocab_size * sizeof(float));
        quantize_rowwise_fp8(qenc, draft_unembed_src, ds_pruned_unembed_fp8, ds_pruned_unembed_sc,
                             (uint32_t)ds_vocab_size, (uint32_t)ds_hidden);

        ds_fold_w_fp8 = alloc_shared((size_t)ds_vocab_size * ds_rank);
        ds_fold_w_sc = alloc_shared((size_t)ds_vocab_size * sizeof(float));
        quantize_rowwise_fp8(qenc, fold_w_bf16, ds_fold_w_fp8, ds_fold_w_sc,
                             (uint32_t)ds_vocab_size, (uint32_t)ds_rank);

        [qenc endEncoding];
        commit_and_wait(qcb);

        for (int i = 0; i < ds_num_layers; ++i) {
            ds_kv_cap[i] = ds_layers[i].is_local ? (ds_window + ds_block + 16) : (max_seq + ds_block + 16);
            size_t sz = (size_t)dspark_slots * (size_t)ds_kv_cap[i] * kv_dim * 2;
            ds_k_cache[i] = alloc_shared(sz);
            ds_v_cache[i] = alloc_shared(sz);
        }

        ds_ctx_mix = alloc_shared((size_t)max_ctx_chunk * ds_hidden * 2);
        ds_ctx_k = alloc_shared((size_t)max_ctx_chunk * kv_dim * 2);
        ds_ctx_v = alloc_shared((size_t)max_ctx_chunk * kv_dim * 2);
        ds_x = alloc_shared((size_t)ds_block * ds_hidden * 2);
        ds_res = alloc_shared((size_t)ds_block * ds_hidden * 2);
        ds_q = alloc_shared((size_t)ds_block * q_dim * 2);
        ds_k = alloc_shared((size_t)ds_block * kv_dim * 2);
        ds_v = alloc_shared((size_t)ds_block * kv_dim * 2);
        ds_attn_out = alloc_shared((size_t)ds_block * q_dim * 2);
        ds_proj_out = alloc_shared((size_t)ds_block * ds_hidden * 2);
        ds_mlp_g = alloc_shared((size_t)ds_block * ds_inter * 2);
        ds_mlp_u = alloc_shared((size_t)ds_block * ds_inter * 2);
        ds_f_rnn = alloc_shared((size_t)ds_hidden * 2);
        ds_base_logits = alloc_shared((size_t)(ds_block - 1) * ds_vocab_size * 2);
        ds_rnn_in = alloc_shared((size_t)(2 * ds_rank + 2 * ds_hidden) * 2);
        ds_rnn_raw = alloc_shared((size_t)3 * ds_rank * sizeof(float));
        ds_rnn_state = alloc_shared((size_t)ds_rank * 2);
        ds_rnn_latent = alloc_shared((size_t)ds_rank * 2);
        ds_markov_bias = alloc_shared((size_t)ds_vocab_size * sizeof(float));
        ds_draft_tok = alloc_shared(8 * sizeof(int32_t));
        ds_anchor_tok = alloc_shared(sizeof(int32_t));

        ds_loaded = true;
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[warpgrid-metal] Loaded DSpark drafter (FP8 W8A16, draft_vocab=%d, slots=%d) in %.2f s\n",
                ds_vocab_size, dspark_slots, sec);
    }

    void store_dspark_context_kv(size_t aux_off_bytes, uint32_t aux_stride, int T, int start_pos, int slot_id) {
        uint32_t kv_dim = (uint32_t)(ds_kv_heads * ds_head_dim);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (int i = 0; i < ds_num_layers; ++i) {
            const auto& lw = ds_layers[i];
            size_t slot_off = (size_t)slot_id * (size_t)ds_kv_cap[i] * kv_dim * 2;
            DflareMixParams dmp{aux_stride, (uint32_t)T, (uint32_t)ds_hidden, ds_eps};
            [enc setComputePipelineState:pso_dflare_mix];
            [enc setBuffer:b_aux_states offset:aux_off_bytes atIndex:0];
            [enc setBuffer:ds_fuse_alpha offset:(size_t)i * 4 * sizeof(float) atIndex:1];
            [enc setBuffer:lw.dflare_norm offset:0 atIndex:2];
            [enc setBuffer:ds_ctx_mix offset:0 atIndex:3];
            [enc setBytes:&dmp length:sizeof(dmp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((uint32_t)T, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            encode_linear_bf16(enc, ds_ctx_mix, 0, lw.k_proj_ctx, 0, ds_ctx_k, 0, nil, 0, (uint32_t)T, kv_dim, (uint32_t)ds_hidden, false);
            encode_linear_bf16(enc, ds_ctx_mix, 0, lw.v_proj_ctx, 0, ds_ctx_v, 0, nil, 0, (uint32_t)T, kv_dim, (uint32_t)ds_hidden, false);

            QkRopeParams qrp{(uint32_t)start_pos, (uint32_t)T, (uint32_t)ds_kv_heads, (uint32_t)ds_head_dim, ds_rope_theta, ds_eps};
            [enc setComputePipelineState:pso_qk_rope];
            [enc setBuffer:ds_ctx_k offset:0 atIndex:0];
            [enc setBuffer:lw.k_norm_ctx offset:0 atIndex:1];
            [enc setBytes:&qrp length:sizeof(qrp) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((uint32_t)ds_kv_heads, (uint32_t)T, 1)
                threadsPerThreadgroup:MTLSizeMake((uint32_t)(ds_head_dim / 2), 1, 1)];

            DraftWriteKvParams wkp{(uint32_t)start_pos, (uint32_t)T, (uint32_t)ds_kv_cap[i], lw.is_local ? 1u : 0u, kv_dim};
            [enc setComputePipelineState:pso_draft_write_kv];
            [enc setBuffer:ds_ctx_k offset:0 atIndex:0];
            [enc setBuffer:ds_ctx_v offset:0 atIndex:1];
            [enc setBuffer:ds_k_cache[i] offset:slot_off atIndex:2];
            [enc setBuffer:ds_v_cache[i] offset:slot_off atIndex:3];
            [enc setBytes:&wkp length:sizeof(wkp) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(kv_dim, (uint32_t)T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [enc endEncoding];
        commit_and_wait(cb);
    }

    void run_dspark_draft(int32_t anchor_tok, int anchor_pos, int num_draft, int32_t* h_out_drafts, int slot_id) {
        static_cast<int32_t*>(ds_anchor_tok.contents)[0] = anchor_tok;
        uint32_t B = (uint32_t)ds_block;
        uint32_t D = (uint32_t)ds_hidden;
        uint32_t q_dim = (uint32_t)(ds_heads * ds_head_dim);
        uint32_t kv_dim = (uint32_t)(ds_kv_heads * ds_head_dim);

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        RmsParams erp{B, D, ds_eps};
        [enc setComputePipelineState:pso_embed_draft];
        [enc setBuffer:ds_anchor_tok offset:0 atIndex:0];
        [enc setBuffer:embed_w offset:0 atIndex:1];
        [enc setBuffer:embed_norm_w offset:0 atIndex:2];
        [enc setBuffer:ds_mask_embed offset:0 atIndex:3];
        [enc setBuffer:ds_res offset:0 atIndex:4];
        [enc setBytes:&erp length:sizeof(erp) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        for (int i = 0; i < ds_num_layers; ++i) {
            const auto& lw = ds_layers[i];
            size_t slot_off = (size_t)slot_id * (size_t)ds_kv_cap[i] * kv_dim * 2;
            if (i == 0) {
                [enc setComputePipelineState:pso_rmsnorm];
                [enc setBuffer:ds_res offset:0 atIndex:0];
                [enc setBuffer:lw.input_ln offset:0 atIndex:1];
                [enc setBuffer:ds_x offset:0 atIndex:2];
                [enc setBytes:&erp length:sizeof(erp) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                [enc setComputePipelineState:pso_add_rmsnorm];
                [enc setBuffer:ds_proj_out offset:0 atIndex:0];
                [enc setBuffer:ds_res offset:0 atIndex:1];
                [enc setBuffer:lw.input_ln offset:0 atIndex:2];
                [enc setBuffer:ds_x offset:0 atIndex:3];
                [enc setBytes:&erp length:sizeof(erp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            encode_linear_fp8(enc, ds_x, 0, lw.q_fp8, 0, lw.q_sc, 0, ds_q, 0, nil, 0, B, q_dim, D, false);
            encode_linear_fp8(enc, ds_x, 0, lw.k_fp8, 0, lw.k_sc, 0, ds_k, 0, nil, 0, B, kv_dim, D, false);
            encode_linear_fp8(enc, ds_x, 0, lw.v_fp8, 0, lw.v_sc, 0, ds_v, 0, nil, 0, B, kv_dim, D, false);

            QkRopeParams qrp_q{(uint32_t)anchor_pos, B, (uint32_t)ds_heads, (uint32_t)ds_head_dim, ds_rope_theta, ds_eps};
            [enc setComputePipelineState:pso_qk_rope];
            [enc setBuffer:ds_q offset:0 atIndex:0];
            [enc setBuffer:lw.q_norm offset:0 atIndex:1];
            [enc setBytes:&qrp_q length:sizeof(qrp_q) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((uint32_t)ds_heads, B, 1)
                threadsPerThreadgroup:MTLSizeMake((uint32_t)(ds_head_dim / 2), 1, 1)];

            QkRopeParams qrp_k{(uint32_t)anchor_pos, B, (uint32_t)ds_kv_heads, (uint32_t)ds_head_dim, ds_rope_theta, ds_eps};
            [enc setBuffer:ds_k offset:0 atIndex:0];
            [enc setBuffer:lw.k_norm offset:0 atIndex:1];
            [enc setBytes:&qrp_k length:sizeof(qrp_k) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((uint32_t)ds_kv_heads, B, 1)
                threadsPerThreadgroup:MTLSizeMake((uint32_t)(ds_head_dim / 2), 1, 1)];

            DraftWriteKvParams wkp{(uint32_t)anchor_pos, B, (uint32_t)ds_kv_cap[i], lw.is_local ? 1u : 0u, kv_dim};
            [enc setComputePipelineState:pso_draft_write_kv];
            [enc setBuffer:ds_k offset:0 atIndex:0];
            [enc setBuffer:ds_v offset:0 atIndex:1];
            [enc setBuffer:ds_k_cache[i] offset:slot_off atIndex:2];
            [enc setBuffer:ds_v_cache[i] offset:slot_off atIndex:3];
            [enc setBytes:&wkp length:sizeof(wkp) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(kv_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            DraftAttnParams dap{(uint32_t)anchor_pos, B, (uint32_t)ds_kv_cap[i], lw.is_local ? 1u : 0u,
                                (uint32_t)ds_window, (uint32_t)ds_heads, (uint32_t)ds_kv_heads, (uint32_t)ds_head_dim};
            [enc setComputePipelineState:pso_draft_attn];
            [enc setBuffer:ds_q offset:0 atIndex:0];
            [enc setBuffer:ds_k_cache[i] offset:slot_off atIndex:1];
            [enc setBuffer:ds_v_cache[i] offset:slot_off atIndex:2];
            [enc setBuffer:ds_attn_out offset:0 atIndex:3];
            [enc setBytes:&dap length:sizeof(dap) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((uint32_t)ds_heads, B, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            encode_linear_fp8(enc, ds_attn_out, 0, lw.o_fp8, 0, lw.o_sc, 0, ds_proj_out, 0, nil, 0, B, D, q_dim, false);

            [enc setComputePipelineState:pso_add_rmsnorm];
            [enc setBuffer:ds_proj_out offset:0 atIndex:0];
            [enc setBuffer:ds_res offset:0 atIndex:1];
            [enc setBuffer:lw.post_attn_ln offset:0 atIndex:2];
            [enc setBuffer:ds_x offset:0 atIndex:3];
            [enc setBytes:&erp length:sizeof(erp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            encode_linear_fp8(enc, ds_x, 0, lw.gate_fp8, 0, lw.gate_sc, 0, ds_mlp_g, 0, nil, 0, B, (uint32_t)ds_inter, D, false);
            encode_linear_fp8(enc, ds_x, 0, lw.up_fp8, 0, lw.up_sc, 0, ds_mlp_u, 0, nil, 0, B, (uint32_t)ds_inter, D, false);

            uint32_t silu_n = B * (uint32_t)ds_inter;
            [enc setComputePipelineState:pso_silu_mul];
            [enc setBuffer:ds_mlp_g offset:0 atIndex:0];
            [enc setBuffer:ds_mlp_u offset:0 atIndex:1];
            [enc setBytes:&silu_n length:sizeof(silu_n) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(silu_n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            encode_linear_fp8(enc, ds_mlp_g, 0, lw.down_fp8, 0, lw.down_sc, 0, ds_proj_out, 0, nil, 0, B, D, (uint32_t)ds_inter, false);
        }

        [enc setComputePipelineState:pso_add_rmsnorm];
        [enc setBuffer:ds_proj_out offset:0 atIndex:0];
        [enc setBuffer:ds_res offset:0 atIndex:1];
        [enc setBuffer:ds_final_norm offset:0 atIndex:2];
        [enc setBuffer:ds_x offset:0 atIndex:3];
        [enc setBytes:&erp length:sizeof(erp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        encode_linear_fp8(enc, ds_x, (size_t)D * 2, ds_pruned_unembed_fp8, 0, ds_pruned_unembed_sc, 0,
                          ds_base_logits, 0, nil, 0, (uint32_t)num_draft, (uint32_t)ds_vocab_size, D, false);

        DflareMixParams fmp{D, 1u, D, ds_eps};
        [enc setComputePipelineState:pso_dflare_mix];
        [enc setBuffer:b_last_aux offset:(size_t)slot_id * 4 * D * 2 atIndex:0];
        [enc setBuffer:ds_fuse_alpha offset:(size_t)3 * 4 * sizeof(float) atIndex:1];
        [enc setBuffer:ds_markov_dflare_norm offset:0 atIndex:2];
        [enc setBuffer:ds_f_rnn offset:0 atIndex:3];
        [enc setBytes:&fmp length:sizeof(fmp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        uint32_t z_rank = (uint32_t)ds_rank;
        [enc setComputePipelineState:pso_zero_bf16];
        [enc setBuffer:ds_rnn_state offset:0 atIndex:0];
        [enc setBytes:&z_rank length:sizeof(z_rank) atIndex:1];
        [enc dispatchThreads:MTLSizeMake(z_rank, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        uint32_t rnn_in_dim = (uint32_t)(2 * ds_rank + 2 * ds_hidden);
        MarkovInParams mip{(uint32_t)ds_rank, D};
        MarkovArgParams map{(uint32_t)ds_vocab_size, ds_draft_to_target ? 1u : 0u};

        for (int s = 0; s < num_draft; ++s) {
            id<MTLBuffer> prev_buf = (s == 0) ? ds_anchor_tok : ds_draft_tok;
            size_t prev_off = (s == 0) ? 0 : (size_t)(s - 1) * sizeof(int32_t);
            [enc setComputePipelineState:pso_assemble_markov];
            [enc setBuffer:ds_rnn_state offset:0 atIndex:0];
            [enc setBuffer:ds_markov_w1 offset:0 atIndex:1];
            [enc setBuffer:prev_buf offset:prev_off atIndex:2];
            [enc setBuffer:ds_x offset:(size_t)(s + 1) * D * 2 atIndex:3];
            [enc setBuffer:ds_f_rnn offset:0 atIndex:4];
            [enc setBuffer:ds_rnn_in offset:0 atIndex:5];
            [enc setBytes:&mip length:sizeof(mip) atIndex:6];
            [enc dispatchThreads:MTLSizeMake(rnn_in_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            encode_linear_bf16(enc, ds_rnn_in, 0, ds_rnn_proj_w, 0, nil, 0, ds_rnn_raw, 0,
                               1u, (uint32_t)(3 * ds_rank), rnn_in_dim, true);

            [enc setComputePipelineState:pso_markov_gate];
            [enc setBuffer:ds_rnn_raw offset:0 atIndex:0];
            [enc setBuffer:ds_rnn_proj_b offset:0 atIndex:1];
            [enc setBuffer:ds_rnn_state offset:0 atIndex:2];
            [enc setBuffer:ds_rnn_latent offset:0 atIndex:3];
            [enc setBytes:&z_rank length:sizeof(z_rank) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(z_rank, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            encode_linear_fp8(enc, ds_rnn_latent, 0, ds_fold_w_fp8, 0, ds_fold_w_sc, 0, nil, 0, ds_markov_bias, 0,
                              1u, (uint32_t)ds_vocab_size, (uint32_t)ds_rank, true);

            [enc setComputePipelineState:pso_markov_argmax];
            [enc setBuffer:ds_base_logits offset:(size_t)s * (size_t)ds_vocab_size * 2 atIndex:0];
            [enc setBuffer:ds_fold_b offset:0 atIndex:1];
            [enc setBuffer:ds_markov_bias offset:0 atIndex:2];
            [enc setBuffer:(ds_draft_to_target ? ds_draft_to_target : b_dummy) offset:0 atIndex:3];
            [enc setBuffer:ds_draft_tok offset:(size_t)s * sizeof(int32_t) atIndex:4];
            [enc setBytes:&map length:sizeof(map) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        [enc endEncoding];
        commit_and_wait(cb);
        memcpy(h_out_drafts, ds_draft_tok.contents, (size_t)num_draft * sizeof(int32_t));
    }

    void load_dflash2(const std::string& df2_dir, const InklingConfig& cfg,
                      int max_ctx_chunk, int n_ds_slots)
    {
        auto t0 = std::chrono::steady_clock::now();
        dspark_slots = n_ds_slots;
        df2_vocab_unpadded = cfg.unpadded_vocab_size;
        df2_hidden = cfg.hidden_size;
        SafetensorsFile dfst;
        dfst.open(df2_dir + "/model.safetensors");

        df2_fc = load_tensor(dfst, "fc.weight");
        df2_hidden_norm = load_tensor(dfst, "hidden_norm.weight");
        df2_final_norm = load_tensor(dfst, "norm.weight");
        df2_sel_hproj = load_tensor(dfst, "candidate_selector.hidden_projection.weight");
        df2_sel_pred_cb = load_tensor(dfst, "candidate_selector.predecessor_codebook");
        df2_sel_succ_cb = load_tensor(dfst, "candidate_selector.successor_codebook");

        int q_dim = df2_heads * df2_head_dim;
        int kv_dim = df2_kv_heads * df2_head_dim;
        df2_layers.resize(df2_num_layers);

        id<MTLCommandBuffer> qcb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> qenc = [qcb computeCommandEncoder];
        auto load_q_fp8 = [&](const std::string& name, uint32_t N, uint32_t K,
                              id<MTLBuffer> __strong& fp8, id<MTLBuffer> __strong& sc) {
            id<MTLBuffer> raw = load_tensor(dfst, name);
            fp8 = alloc_shared((size_t)N * K);
            sc = alloc_shared((size_t)N * sizeof(float));
            quantize_rowwise_fp8(qenc, raw, fp8, sc, N, K);
        };

        for (int i = 0; i < df2_num_layers; ++i) {
            auto& lw = df2_layers[i];
            std::string p = "layers." + std::to_string(i) + ".";
            lw.input_ln = load_tensor(dfst, p + "input_layernorm.weight");
            lw.post_ln = load_tensor(dfst, p + "post_attention_layernorm.weight");
            lw.q_norm = load_tensor(dfst, p + "self_attn.q_norm.weight");
            lw.k_norm = load_tensor(dfst, p + "self_attn.k_norm.weight");
            lw.attn_conv_base = load_tensor(dfst, p + "attention_conv.base_kernel");
            lw.attn_conv_proj = load_tensor(dfst, p + "attention_conv.kernel_projection.weight");
            lw.mlp_conv_base = load_tensor(dfst, p + "mlp_conv.base_kernel");
            lw.mlp_conv_proj = load_tensor(dfst, p + "mlp_conv.kernel_projection.weight");
            lw.k_proj_bf16 = load_tensor(dfst, p + "self_attn.k_proj.weight");
            lw.v_proj_bf16 = load_tensor(dfst, p + "self_attn.v_proj.weight");

            load_q_fp8(p + "self_attn.q_proj.weight", (uint32_t)q_dim, (uint32_t)df2_hidden, lw.q_fp8, lw.q_sc);
            lw.k_fp8 = alloc_shared((size_t)kv_dim * df2_hidden);
            lw.k_sc = alloc_shared((size_t)kv_dim * sizeof(float));
            quantize_rowwise_fp8(qenc, lw.k_proj_bf16, lw.k_fp8, lw.k_sc, (uint32_t)kv_dim, (uint32_t)df2_hidden);
            lw.v_fp8 = alloc_shared((size_t)kv_dim * df2_hidden);
            lw.v_sc = alloc_shared((size_t)kv_dim * sizeof(float));
            quantize_rowwise_fp8(qenc, lw.v_proj_bf16, lw.v_fp8, lw.v_sc, (uint32_t)kv_dim, (uint32_t)df2_hidden);
            load_q_fp8(p + "self_attn.o_proj.weight", (uint32_t)df2_hidden, (uint32_t)q_dim, lw.o_fp8, lw.o_sc);
            load_q_fp8(p + "mlp.gate_proj.weight", (uint32_t)df2_inter, (uint32_t)df2_hidden, lw.g_fp8, lw.g_sc);
            load_q_fp8(p + "mlp.up_proj.weight", (uint32_t)df2_inter, (uint32_t)df2_hidden, lw.u_fp8, lw.u_sc);
            load_q_fp8(p + "mlp.down_proj.weight", (uint32_t)df2_hidden, (uint32_t)df2_inter, lw.d_fp8, lw.d_sc);
        }
        [qenc endEncoding];
        commit_and_wait(qcb);

        df2_kv_cap = df2_window + df2_block + 16;
        df2_k_cache.resize(df2_num_layers);
        df2_v_cache.resize(df2_num_layers);
        for (int i = 0; i < df2_num_layers; ++i) {
            size_t sz = (size_t)dspark_slots * (size_t)df2_kv_cap * kv_dim * 2;
            df2_k_cache[i] = alloc_shared(sz);
            df2_v_cache[i] = alloc_shared(sz);
        }

        df2_ctx_cat = alloc_shared((size_t)max_ctx_chunk * 4 * df2_hidden * 2);
        df2_ctx_fc = alloc_shared((size_t)max_ctx_chunk * df2_hidden * 2);
        df2_ctx_norm = alloc_shared((size_t)max_ctx_chunk * df2_hidden * 2);
        df2_ctx_k = alloc_shared((size_t)max_ctx_chunk * kv_dim * 2);
        df2_ctx_v = alloc_shared((size_t)max_ctx_chunk * kv_dim * 2);
        df2_tok_ids = alloc_shared((size_t)df2_block * sizeof(int32_t));
        df2_res = alloc_shared((size_t)df2_block * df2_hidden * 2);
        df2_x = alloc_shared((size_t)df2_block * df2_hidden * 2);
        df2_xc = alloc_shared((size_t)df2_block * df2_hidden * 2);
        df2_dyn = alloc_shared((size_t)df2_block * 4 * (df2_hidden / df2_group_size) * 2);
        df2_q = alloc_shared((size_t)df2_block * q_dim * 2);
        df2_k = alloc_shared((size_t)df2_block * kv_dim * 2);
        df2_v = alloc_shared((size_t)df2_block * kv_dim * 2);
        df2_attn_out = alloc_shared((size_t)df2_block * q_dim * 2);
        df2_proj_out = alloc_shared((size_t)df2_block * df2_hidden * 2);
        df2_mlp_g = alloc_shared((size_t)df2_block * df2_inter * 2);
        df2_mlp_u = alloc_shared((size_t)df2_block * df2_inter * 2);
        df2_h_sel = alloc_shared((size_t)df2_block * df2_sel_rank * 2);
        df2_top_v = alloc_shared((size_t)df2_block * 16 * sizeof(float));
        df2_top_i = alloc_shared((size_t)df2_block * 16 * sizeof(int32_t));

        df2_loaded = true;
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[warpgrid-metal] Loaded DFlash2 drafter (%d layers, W8A16 FP8, block=%d, SWA=%d, slots=%d) in %.2f s\n",
                df2_num_layers, df2_block, df2_window, dspark_slots, sec);
    }

    void store_dflash2_context_kv(size_t aux_off_bytes, uint32_t aux_stride, int T, int start_pos, int slot_id) {
        uint32_t D = (uint32_t)df2_hidden;
        if (T > df2_window) {
            int skip = T - df2_window;
            aux_off_bytes += (size_t)skip * D * 2;
            start_pos += skip;
            T = df2_window;
        }
        uint32_t kv_dim = (uint32_t)(df2_kv_heads * df2_head_dim);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        Df2ConcatParams cp{(uint32_t)T, D, aux_stride};
        [enc setComputePipelineState:pso_df2_concat];
        [enc setBuffer:b_aux_states offset:aux_off_bytes atIndex:0];
        [enc setBuffer:df2_ctx_cat offset:0 atIndex:1];
        [enc setBytes:&cp length:sizeof(cp) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(4u * D, (uint32_t)T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        encode_linear_bf16(enc, df2_ctx_cat, 0, df2_fc, 0, df2_ctx_fc, 0, nil, 0, (uint32_t)T, D, 4u * D, false);

        RmsParams rp{(uint32_t)T, D, df2_eps};
        [enc setComputePipelineState:pso_rmsnorm];
        [enc setBuffer:df2_ctx_fc offset:0 atIndex:0];
        [enc setBuffer:df2_hidden_norm offset:0 atIndex:1];
        [enc setBuffer:df2_ctx_norm offset:0 atIndex:2];
        [enc setBytes:&rp length:sizeof(rp) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((uint32_t)T, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        for (int i = 0; i < df2_num_layers; ++i) {
            const auto& lw = df2_layers[i];
            size_t slot_off = (size_t)slot_id * (size_t)df2_kv_cap * kv_dim * 2;
            if (T <= 8) {
                encode_linear_fp8(enc, df2_ctx_norm, 0, lw.k_fp8, 0, lw.k_sc, 0, df2_ctx_k, 0, nil, 0, (uint32_t)T, kv_dim, D, false);
                encode_linear_fp8(enc, df2_ctx_norm, 0, lw.v_fp8, 0, lw.v_sc, 0, df2_ctx_v, 0, nil, 0, (uint32_t)T, kv_dim, D, false);
            } else {
                encode_linear_bf16(enc, df2_ctx_norm, 0, lw.k_proj_bf16, 0, df2_ctx_k, 0, nil, 0, (uint32_t)T, kv_dim, D, false);
                encode_linear_bf16(enc, df2_ctx_norm, 0, lw.v_proj_bf16, 0, df2_ctx_v, 0, nil, 0, (uint32_t)T, kv_dim, D, false);
            }

            QkRopeParams qrp{(uint32_t)start_pos, (uint32_t)T, (uint32_t)df2_kv_heads, (uint32_t)df2_head_dim, df2_rope_theta, df2_eps};
            [enc setComputePipelineState:pso_qk_rope];
            [enc setBuffer:df2_ctx_k offset:0 atIndex:0];
            [enc setBuffer:lw.k_norm offset:0 atIndex:1];
            [enc setBytes:&qrp length:sizeof(qrp) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((uint32_t)df2_kv_heads, (uint32_t)T, 1)
                threadsPerThreadgroup:MTLSizeMake((uint32_t)(df2_head_dim / 2), 1, 1)];

            DraftWriteKvParams wkp{(uint32_t)start_pos, (uint32_t)T, (uint32_t)df2_kv_cap, 1u, kv_dim};
            [enc setComputePipelineState:pso_draft_write_kv];
            [enc setBuffer:df2_ctx_k offset:0 atIndex:0];
            [enc setBuffer:df2_ctx_v offset:0 atIndex:1];
            [enc setBuffer:df2_k_cache[i] offset:slot_off atIndex:2];
            [enc setBuffer:df2_v_cache[i] offset:slot_off atIndex:3];
            [enc setBytes:&wkp length:sizeof(wkp) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(kv_dim, (uint32_t)T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [enc endEncoding];
        commit_and_wait(cb);
    }

    void run_dflash2_draft(int32_t anchor_tok, int anchor_pos, int num_draft, int32_t* h_out_drafts, int slot_id) {
        int nd = std::max(1, std::min(df2_block - 1, num_draft));
        uint32_t B = (uint32_t)(nd + 1);
        uint32_t D = (uint32_t)df2_hidden;
        uint32_t q_dim = (uint32_t)(df2_heads * df2_head_dim);
        uint32_t kv_dim = (uint32_t)(df2_kv_heads * df2_head_dim);
        uint32_t dyn_dim = 4u * (D / (uint32_t)df2_group_size);

        int32_t* tids = static_cast<int32_t*>(df2_tok_ids.contents);
        tids[0] = anchor_tok;
        for (uint32_t i = 1; i < B; ++i) tids[i] = df2_mask_id;

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        RmsParams erp{B, D, df2_eps};
        [enc setComputePipelineState:pso_embed_norm];
        [enc setBuffer:df2_tok_ids offset:0 atIndex:0];
        [enc setBuffer:embed_w offset:0 atIndex:1];
        [enc setBuffer:embed_norm_w offset:0 atIndex:2];
        [enc setBuffer:df2_res offset:0 atIndex:3];
        [enc setBytes:&erp length:sizeof(erp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        for (int i = 0; i < df2_num_layers; ++i) {
            const auto& lw = df2_layers[i];
            size_t slot_off = (size_t)slot_id * (size_t)df2_kv_cap * kv_dim * 2;

            [enc setComputePipelineState:pso_rmsnorm];
            [enc setBuffer:df2_res offset:0 atIndex:0];
            [enc setBuffer:lw.input_ln offset:0 atIndex:1];
            [enc setBuffer:df2_x offset:0 atIndex:2];
            [enc setBytes:&erp length:sizeof(erp) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            encode_linear_bf16(enc, df2_x, 0, lw.attn_conv_proj, 0, df2_dyn, 0, nil, 0, B, dyn_dim, D, false);
            Df2ConvParams cv0{B, D, (uint32_t)df2_group_size, 0u};
            [enc setComputePipelineState:pso_df2_conv];
            [enc setBuffer:df2_x offset:0 atIndex:0];
            [enc setBuffer:lw.attn_conv_base offset:0 atIndex:1];
            [enc setBuffer:df2_dyn offset:0 atIndex:2];
            [enc setBuffer:df2_xc offset:0 atIndex:3];
            [enc setBytes:&cv0 length:sizeof(cv0) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(D, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            encode_linear_fp8(enc, df2_xc, 0, lw.q_fp8, 0, lw.q_sc, 0, df2_q, 0, nil, 0, B, q_dim, D, false);
            encode_linear_fp8(enc, df2_xc, 0, lw.k_fp8, 0, lw.k_sc, 0, df2_k, 0, nil, 0, B, kv_dim, D, false);
            encode_linear_fp8(enc, df2_xc, 0, lw.v_fp8, 0, lw.v_sc, 0, df2_v, 0, nil, 0, B, kv_dim, D, false);

            QkRopeParams qrp_q{(uint32_t)anchor_pos, B, (uint32_t)df2_heads, (uint32_t)df2_head_dim, df2_rope_theta, df2_eps};
            [enc setComputePipelineState:pso_qk_rope];
            [enc setBuffer:df2_q offset:0 atIndex:0];
            [enc setBuffer:lw.q_norm offset:0 atIndex:1];
            [enc setBytes:&qrp_q length:sizeof(qrp_q) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((uint32_t)df2_heads, B, 1)
                threadsPerThreadgroup:MTLSizeMake((uint32_t)(df2_head_dim / 2), 1, 1)];

            QkRopeParams qrp_k{(uint32_t)anchor_pos, B, (uint32_t)df2_kv_heads, (uint32_t)df2_head_dim, df2_rope_theta, df2_eps};
            [enc setComputePipelineState:pso_qk_rope];
            [enc setBuffer:df2_k offset:0 atIndex:0];
            [enc setBuffer:lw.k_norm offset:0 atIndex:1];
            [enc setBytes:&qrp_k length:sizeof(qrp_k) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((uint32_t)df2_kv_heads, B, 1)
                threadsPerThreadgroup:MTLSizeMake((uint32_t)(df2_head_dim / 2), 1, 1)];

            DraftAttnParams dap{(uint32_t)anchor_pos, B, (uint32_t)df2_kv_cap, 1u, (uint32_t)df2_window,
                                (uint32_t)df2_heads, (uint32_t)df2_kv_heads, (uint32_t)df2_head_dim};
            [enc setComputePipelineState:pso_df2_attn];
            [enc setBuffer:df2_q offset:0 atIndex:0];
            [enc setBuffer:df2_k offset:0 atIndex:1];
            [enc setBuffer:df2_v offset:0 atIndex:2];
            [enc setBuffer:df2_k_cache[i] offset:slot_off atIndex:3];
            [enc setBuffer:df2_v_cache[i] offset:slot_off atIndex:4];
            [enc setBuffer:df2_attn_out offset:0 atIndex:5];
            [enc setBytes:&dap length:sizeof(dap) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((uint32_t)df2_heads, B, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            encode_linear_fp8(enc, df2_attn_out, 0, lw.o_fp8, 0, lw.o_sc, 0, df2_proj_out, 0, nil, 0, B, D, q_dim, false);

            Df2ConvParams cv1{B, D, (uint32_t)df2_group_size, 1u};
            [enc setComputePipelineState:pso_df2_conv];
            [enc setBuffer:df2_proj_out offset:0 atIndex:0];
            [enc setBuffer:lw.attn_conv_base offset:(size_t)2 * D * 2 atIndex:1];
            [enc setBuffer:df2_dyn offset:0 atIndex:2];
            [enc setBuffer:df2_xc offset:0 atIndex:3];
            [enc setBytes:&cv1 length:sizeof(cv1) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(D, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc setComputePipelineState:pso_add_rmsnorm];
            [enc setBuffer:df2_xc offset:0 atIndex:0];
            [enc setBuffer:df2_res offset:0 atIndex:1];
            [enc setBuffer:lw.post_ln offset:0 atIndex:2];
            [enc setBuffer:df2_x offset:0 atIndex:3];
            [enc setBytes:&erp length:sizeof(erp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            encode_linear_bf16(enc, df2_x, 0, lw.mlp_conv_proj, 0, df2_dyn, 0, nil, 0, B, dyn_dim, D, false);
            [enc setComputePipelineState:pso_df2_conv];
            [enc setBuffer:df2_x offset:0 atIndex:0];
            [enc setBuffer:lw.mlp_conv_base offset:0 atIndex:1];
            [enc setBuffer:df2_dyn offset:0 atIndex:2];
            [enc setBuffer:df2_xc offset:0 atIndex:3];
            [enc setBytes:&cv0 length:sizeof(cv0) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(D, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            encode_linear_fp8(enc, df2_xc, 0, lw.g_fp8, 0, lw.g_sc, 0, df2_mlp_g, 0, nil, 0, B, (uint32_t)df2_inter, D, false);
            encode_linear_fp8(enc, df2_xc, 0, lw.u_fp8, 0, lw.u_sc, 0, df2_mlp_u, 0, nil, 0, B, (uint32_t)df2_inter, D, false);

            uint32_t n_inter = B * (uint32_t)df2_inter;
            [enc setComputePipelineState:pso_silu_mul];
            [enc setBuffer:df2_mlp_g offset:0 atIndex:0];
            [enc setBuffer:df2_mlp_u offset:0 atIndex:1];
            [enc setBytes:&n_inter length:sizeof(n_inter) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(n_inter, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            encode_linear_fp8(enc, df2_mlp_g, 0, lw.d_fp8, 0, lw.d_sc, 0, df2_proj_out, 0, nil, 0, B, D, (uint32_t)df2_inter, false);

            [enc setComputePipelineState:pso_df2_conv];
            [enc setBuffer:df2_proj_out offset:0 atIndex:0];
            [enc setBuffer:lw.mlp_conv_base offset:(size_t)2 * D * 2 atIndex:1];
            [enc setBuffer:df2_dyn offset:0 atIndex:2];
            [enc setBuffer:df2_xc offset:0 atIndex:3];
            [enc setBytes:&cv1 length:sizeof(cv1) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(D, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            uint32_t bd_n = B * D;
            [enc setComputePipelineState:pso_add_bf16];
            [enc setBuffer:df2_res offset:0 atIndex:0];
            [enc setBuffer:df2_xc offset:0 atIndex:1];
            [enc setBuffer:df2_res offset:0 atIndex:2];
            [enc setBytes:&bd_n length:sizeof(bd_n) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(bd_n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        RmsParams nrp{(uint32_t)nd, D, df2_eps};
        [enc setComputePipelineState:pso_rmsnorm];
        [enc setBuffer:df2_res offset:(size_t)D * 2 atIndex:0];
        [enc setBuffer:df2_final_norm offset:0 atIndex:1];
        [enc setBuffer:df2_x offset:0 atIndex:2];
        [enc setBytes:&nrp length:sizeof(nrp) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((uint32_t)nd, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        encode_linear_fp8(enc, df2_x, 0, unembed_fp8, 0, unembed_sc, 0, nil, 0, b_logits, 0,
                          (uint32_t)nd, (uint32_t)df2_vocab_unpadded, D, true);
        encode_linear_bf16(enc, df2_x, 0, df2_sel_hproj, 0, df2_h_sel, 0, nil, 0,
                           (uint32_t)nd, (uint32_t)df2_sel_rank, D, false);

        Df2TopKParams tkp{(uint32_t)nd, (uint32_t)df2_vocab_unpadded};
        [enc setComputePipelineState:pso_df2_topk16];
        [enc setBuffer:b_logits offset:0 atIndex:0];
        [enc setBuffer:df2_top_v offset:0 atIndex:1];
        [enc setBuffer:df2_top_i offset:0 atIndex:2];
        [enc setBytes:&tkp length:sizeof(tkp) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((uint32_t)nd, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc endEncoding];
        commit_and_wait(cb);

        const float* top_v = static_cast<const float*>(df2_top_v.contents);
        const int32_t* top_i = static_cast<const int32_t*>(df2_top_i.contents);
        const uint16_t* h_sel = static_cast<const uint16_t*>(df2_h_sel.contents);
        const uint16_t* pred_cb = static_cast<const uint16_t*>(df2_sel_pred_cb.contents);
        const uint16_t* succ_cb = static_cast<const uint16_t*>(df2_sel_succ_cb.contents);

        int32_t pred = anchor_tok;
        float pv[256];
        for (int pos = 0; pos < nd; ++pos) {
            const uint16_t* pr = pred_cb + (size_t)pred * df2_sel_rank;
            const uint16_t* hr = h_sel + (size_t)pos * df2_sel_rank;
            for (int r = 0; r < df2_sel_rank; ++r) {
                pv[r] = bf16_bits_to_f32(pr[r]) * bf16_bits_to_f32(hr[r]);
            }
            float best_s = -1e30f;
            int32_t best_tok = top_i[pos * 16];
            for (int k = 0; k < 16; ++k) {
                int32_t cid = top_i[pos * 16 + k];
                const uint16_t* sr = succ_cb + (size_t)cid * df2_sel_rank;
                float corr = 0.0f;
                for (int r = 0; r < df2_sel_rank; ++r) {
                    corr += pv[r] * bf16_bits_to_f32(sr[r]);
                }
                float sc = top_v[pos * 16 + k] + corr;
                if (sc > best_s) {
                    best_s = sc;
                    best_tok = cid;
                }
            }
            h_out_drafts[pos] = best_tok;
            pred = best_tok;
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

        {
            int n_global_visits = 0;
            int n_local_visits = 0;
            for (int i = 0; i < cfg.loop_core_start; ++i) (cfg.is_local(i) ? n_local_visits : n_global_visits)++;
            for (int p = 0; p < cfg.loop_num_passes; ++p)
                for (int i = cfg.loop_core_start; i < cfg.loop_core_end; ++i) (cfg.is_local(i) ? n_local_visits : n_global_visits)++;
            for (int i = cfg.loop_core_end; i < cfg.num_hidden_layers; ++i) (cfg.is_local(i) ? n_local_visits : n_global_visits)++;

            int kv_dim_est = cfg.num_key_value_heads * cfg.head_dim;
            double weight_gib = 21.2 + (draft_k > 0 ? 0.8 : 0.0);
            double avail_kv_gib = std::max(0.5, impl_->gpu_ws_gib * 0.90 - weight_gib);
            auto est_kv_gib = [&](int slots, int seq) -> double {
                double g_bytes = (double)n_global_visits * slots * seq * (2.0 * kv_dim_est + 2.0 * cfg.num_key_value_heads * 4.0);
                double l_bytes = (double)n_local_visits * slots * (cfg.sliding_window_size + max_batch_tokens) * 4.0 * kv_dim_est * 2.0;
                double d_bytes = (draft_k > 0) ? ((double)std::min(2, slots) * (seq + 24) * 2.0 * kv_dim_est * 2.0) : 0.0;
                return (g_bytes + l_bytes + d_bytes) / (1024.0 * 1024.0 * 1024.0);
            };
            int orig_slots = max_slots;
            int orig_seq = max_seq_len;
            while (max_slots > 1 && est_kv_gib(max_slots, max_seq_len) > avail_kv_gib) {
                max_slots -= 1;
            }
            while (max_seq_len > 2048 && est_kv_gib(max_slots, max_seq_len) > avail_kv_gib) {
                max_seq_len = std::max(2048, max_seq_len / 2);
            }
            fprintf(stderr, "[warpgrid-metal] GPU shape: %s (ws=%.2f GiB, max_tg=%u, apple9=%d, low_mem=%d)\n",
                    impl_->device.name.UTF8String, impl_->gpu_ws_gib, impl_->gpu_max_tg,
                    impl_->gpu_is_apple9 ? 1 : 0, impl_->gpu_low_mem ? 1 : 0);
            if (max_slots != orig_slots || max_seq_len != orig_seq) {
                fprintf(stderr, "[warpgrid-metal] Memory budget clamped slots %d->%d, max_seq %d->%d (est KV %.2f GiB <= %.2f GiB)\n",
                        orig_slots, max_slots, orig_seq, max_seq_len,
                        est_kv_gib(max_slots, max_seq_len), avail_kv_gib);
            }
        }

        NSError* err = nil;
#ifdef WARPGRID_METAL_PRECOMPILED
        dispatch_data_t data = dispatch_data_create(
            kMetalLibBytes, kMetalLibLen, dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        impl_->library = [impl_->device newLibraryWithData:data error:&err];
#else
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion3_1;
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeSafe;
            opts.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
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
        impl_->pso_attention_fp8_splitk = impl_->make_pso("attention_fp8_splitk_kernel");
        impl_->pso_attention_fp8_reduce = impl_->make_pso("attention_fp8_reduce_kernel");
        impl_->pso_attention_fp8_prefill4 = impl_->make_pso("attention_fp8_prefill4_kernel");
        impl_->pso_router_logits = impl_->make_pso("moe_router_logits_kernel");
        impl_->pso_router_topk = impl_->make_pso("moe_router_topk_kernel");
        impl_->pso_moe_group = impl_->make_pso("moe_group_tokens_kernel");
        impl_->pso_moe_nvfp4_w13 = impl_->make_pso("moe_nvfp4_w13_kernel");
        impl_->pso_moe_nvfp4_w2 = impl_->make_pso("moe_nvfp4_w2_kernel");
        impl_->pso_moe_nvfp4_w13_grp = impl_->make_pso("moe_nvfp4_w13_grouped_kernel");
        impl_->pso_moe_nvfp4_w2_grp = impl_->make_pso("moe_nvfp4_w2_grouped_kernel");
        impl_->pso_moe_nvfp4_w13_dedup = impl_->make_pso("moe_nvfp4_w13_dedup_kernel");
        impl_->pso_moe_nvfp4_w2_dedup = impl_->make_pso("moe_nvfp4_w2_dedup_kernel");
        impl_->pso_moe_reduce_k = impl_->make_pso("moe_reduce_k_kernel");
        impl_->pso_moe_mxfp8_w13 = impl_->make_pso("moe_mxfp8_w13_kernel");
        impl_->pso_moe_mxfp8_w2 = impl_->make_pso("moe_mxfp8_w2_kernel");
        impl_->pso_moe_mxfp8_w13_grp = impl_->make_pso("moe_mxfp8_w13_grouped_kernel");
        impl_->pso_moe_mxfp8_w2_grp = impl_->make_pso("moe_mxfp8_w2_grouped_kernel");
        impl_->pso_moe_mxfp8_w13_dedup = impl_->make_pso("moe_mxfp8_w13_dedup_kernel");
        impl_->pso_moe_mxfp8_w2_dedup = impl_->make_pso("moe_mxfp8_w2_dedup_kernel");
        impl_->pso_bf16_gemv = impl_->make_pso("bf16_gemv_kernel");
        impl_->pso_bf16_gemv_1sg = impl_->make_pso("bf16_gemv_1sg_kernel");
        impl_->pso_quant_fp8 = impl_->make_pso("quantize_bf16_to_fp8_rowwise_kernel");
        impl_->pso_fp8_gemv = impl_->make_pso("fp8_gemv_kernel");
        impl_->pso_fp8_small_gemm = impl_->make_pso("fp8_small_gemm_kernel");
        impl_->pso_bf16_gemm_tiled = impl_->make_pso("bf16_gemm_tiled_kernel");
        impl_->pso_softcap_argmax = impl_->make_pso("softcap_argmax_kernel");
        impl_->pso_batched_argmax = impl_->make_pso("batched_argmax_kernel");
        impl_->pso_dflare_mix = impl_->make_pso("dflare_mix_rmsnorm_kernel");
        impl_->pso_qk_rope = impl_->make_pso("qk_norm_rope_kernel");
        impl_->pso_draft_write_kv = impl_->make_pso("draft_write_kv_kernel");
        impl_->pso_draft_attn = impl_->make_pso("draft_block_attn_kernel");
        impl_->pso_add_rmsnorm = impl_->make_pso("add_rmsnorm_kernel");
        impl_->pso_silu_mul = impl_->make_pso("silu_mul_kernel");
        impl_->pso_embed_draft = impl_->make_pso("embed_draft_block_kernel");
        impl_->pso_assemble_markov = impl_->make_pso("assemble_markov_in_dev_kernel");
        impl_->pso_markov_gate = impl_->make_pso("markov_gate_kernel");
        impl_->pso_markov_argmax = impl_->make_pso("markov_argmax_kernel");
        impl_->pso_gather_bf16 = impl_->make_pso("gather_bf16_rows_kernel");
        impl_->pso_fold_unembed = impl_->make_pso("fold_unembed_ho_kernel");
        impl_->pso_zero_bf16 = impl_->make_pso("zero_bf16_kernel");
        impl_->pso_df2_concat = impl_->make_pso("dflash2_concat_taps_kernel");
        impl_->pso_df2_conv = impl_->make_pso("dflash2_grouped_dyn_conv_kernel");
        impl_->pso_df2_attn = impl_->make_pso("dflash2_block_attn_kernel");
        impl_->pso_df2_topk16 = impl_->make_pso("dflash2_topk16_kernel");
        impl_->pso_moe_q2_w13 = impl_->make_pso("moe_q2_w13_kernel");
        impl_->pso_moe_q2_w2 = impl_->make_pso("moe_q2_w2_kernel");
        impl_->pso_moe_q2_w13_dedup = impl_->make_pso("moe_q2_w13_dedup_kernel");
        impl_->pso_moe_q2_w2_dedup = impl_->make_pso("moe_q2_w2_dedup_kernel");

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

            lw.is_q2 = st.has(p + "mlp.experts.w13_q2_weight");
            if (lw.is_q2) {
                lw.w13_weight = impl_->load_tensor(st, p + "mlp.experts.w13_q2_weight");
                lw.w13_scale = impl_->load_tensor(st, p + "mlp.experts.w13_q2_scale");
                lw.w13_min = impl_->load_tensor(st, p + "mlp.experts.w13_q2_min");
                lw.w2_weight = impl_->load_tensor(st, p + "mlp.experts.w2_q2_weight");
                lw.w2_scale = impl_->load_tensor(st, p + "mlp.experts.w2_q2_scale");
                lw.w2_min = impl_->load_tensor(st, p + "mlp.experts.w2_q2_min");
            } else {
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

            int n_ds = std::min(2, max_slots);
            vc.k_conv_snap = impl_->alloc_shared((size_t)n_ds * 8 * 3 * kv_dim * 2);
            vc.v_conv_snap = impl_->alloc_shared((size_t)n_ds * 8 * 3 * kv_dim * 2);
            vc.attn_conv_snap = impl_->alloc_shared((size_t)n_ds * 8 * 3 * D * 2);
            vc.mlp_conv_snap = impl_->alloc_shared((size_t)n_ds * 8 * 3 * D * 2);

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
        impl_->b_attn_split_ml = impl_->alloc_shared((size_t)8 * cfg.num_attention_heads * 64 * 2 * sizeof(float));
        impl_->b_attn_split_acc = impl_->alloc_shared((size_t)8 * cfg.num_attention_heads * 64 * cfg.head_dim * sizeof(float));
        impl_->b_attn_proj = impl_->alloc_shared((size_t)T * D * 2);
        impl_->b_attn_out = impl_->alloc_shared((size_t)T * D * 2);
        impl_->b_mlp_raw = impl_->alloc_shared((size_t)T * D * 2);
        impl_->b_mlp_out = impl_->alloc_shared((size_t)T * D * 2);
        impl_->b_topk_ids = impl_->alloc_shared((size_t)T * cfg.num_experts_per_tok * sizeof(int32_t));
        impl_->b_topk_weights = impl_->alloc_shared((size_t)T * cfg.num_experts_per_tok * sizeof(float));
        impl_->b_router_logits = impl_->alloc_shared((size_t)T * cfg.n_routed_experts * sizeof(float));
        impl_->b_moe_act = impl_->alloc_shared((size_t)T * cfg.num_experts_per_tok * cfg.intermediate_size * sizeof(float));
        impl_->b_moe_offsets = impl_->alloc_shared((size_t)(cfg.n_routed_experts + 1) * sizeof(uint32_t));
        impl_->b_moe_sorted_tk = impl_->alloc_shared((size_t)T * cfg.num_experts_per_tok * sizeof(uint32_t));
        impl_->b_moe_w2_out = impl_->alloc_shared((size_t)T * cfg.num_experts_per_tok * D * 2);
        impl_->b_logits = impl_->alloc_shared((size_t)8 * cfg.vocab_size * sizeof(float));
        impl_->b_out_token = impl_->alloc_shared(16 * sizeof(int32_t));
        impl_->b_aux_states = impl_->alloc_shared((size_t)4 * T * D * 2);
        int n_ds_slots = std::min(2, max_slots);
        impl_->b_last_aux = impl_->alloc_shared((size_t)n_ds_slots * 4 * D * 2);

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

        if (!dflash2_dir.empty()) {
            if (draft_k <= 0) draft_k = 7;
            impl_->load_dflash2(dflash2_dir, cfg, max_batch_tokens, n_ds_slots);
        } else if (draft_k > 0) {
            impl_->load_dspark(model_dir, cfg, max_seq_len, max_batch_tokens, draft_vocab_file, n_ds_slots);
        }
        impl_->unembed_w = nil;
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
        impl_->forward_internal(cfg, max_batch_tokens, h_tokens, T, start_pos, slot_id, true, profile_stages, false);
        return static_cast<const int32_t*>(impl_->b_out_token.contents)[0];
    }
}

void Inkling2Model::forward_batch_decode_greedy(
    const int32_t* h_tokens, const int32_t* h_slots, const int32_t* h_positions,
    int B, int32_t* h_out_tokens)
{
    @autoreleasepool {
        if (B <= 0) return;
        if (B == 1) {
            h_out_tokens[0] = forward_and_greedy(&h_tokens[0], 1, h_positions[0], h_slots[0]);
            return;
        }
        int off = 0;
        int max_b = std::max(1, std::min(8, max_batch_tokens));
        while (off < B) {
            int chunk = std::min(max_b, B - off);
            if (chunk == 1) {
                h_out_tokens[off] = forward_and_greedy(&h_tokens[off], 1, h_positions[off], h_slots[off]);
            } else {
                impl_->forward_internal(cfg, max_batch_tokens, h_tokens + off, chunk, 0, 0, true,
                                        profile_stages, false, 0, h_slots + off, h_positions + off);
                memcpy(h_out_tokens + off, impl_->b_out_token.contents, (size_t)chunk * sizeof(int32_t));
            }
            off += chunk;
        }
    }
}

void Inkling2Model::forward_verify_greedy(const int32_t* h_tokens, int T, int start_pos, int32_t* h_out_tokens) {
    @autoreleasepool {
        impl_->forward_internal(cfg, max_batch_tokens, h_tokens, T, start_pos, 0, true, profile_stages, true);
        memcpy(h_out_tokens, impl_->b_out_token.contents, (size_t)T * sizeof(int32_t));
    }
}

void Inkling2Model::forward_batch_verify_greedy(
    const int32_t* h_tokens, const int32_t* h_slots, const int32_t* h_positions,
    int B, int L, int32_t* h_out_tokens)
{
    @autoreleasepool {
        for (int b = 0; b < B; ++b) {
            impl_->forward_internal(cfg, max_batch_tokens, h_tokens + b * L, L, h_positions[b], h_slots[b], true, profile_stages, true, b * L);
            memcpy(h_out_tokens + b * L, impl_->b_out_token.contents, (size_t)L * sizeof(int32_t));
        }
    }
}

void Inkling2Model::rollback_conv_states(int accepted_len, int slot_id) {
    int idx = accepted_len - 1;
    int kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    int D = cfg.hidden_size;
    size_t k_sz = (size_t)3 * kv_dim * 2;
    size_t d_sz = (size_t)3 * D * 2;
    size_t k_dst = (size_t)slot_id * k_sz;
    size_t d_dst = (size_t)slot_id * d_sz;
    size_t k_src = ((size_t)slot_id * 8 + (size_t)idx) * k_sz;
    size_t d_src = ((size_t)slot_id * 8 + (size_t)idx) * d_sz;
    for (auto& vc : impl_->caches) {
        memcpy(static_cast<uint8_t*>(vc.k_conv_state.contents) + k_dst,
               static_cast<uint8_t*>(vc.k_conv_snap.contents) + k_src, k_sz);
        memcpy(static_cast<uint8_t*>(vc.v_conv_state.contents) + k_dst,
               static_cast<uint8_t*>(vc.v_conv_snap.contents) + k_src, k_sz);
        memcpy(static_cast<uint8_t*>(vc.attn_conv_state.contents) + d_dst,
               static_cast<uint8_t*>(vc.attn_conv_snap.contents) + d_src, d_sz);
        memcpy(static_cast<uint8_t*>(vc.mlp_conv_state.contents) + d_dst,
               static_cast<uint8_t*>(vc.mlp_conv_snap.contents) + d_src, d_sz);
    }
}

void Inkling2Model::commit_dspark_context(int count, int start_pos, int slot_id, int aux_tok_offset) {
    if ((!impl_->ds_loaded && !impl_->df2_loaded) || count <= 0 || slot_id >= impl_->dspark_slots) return;
    @autoreleasepool {
        int D = cfg.hidden_size;
        uint32_t stack_stride = (uint32_t)(max_batch_tokens * D);
        if (impl_->df2_loaded) {
            impl_->store_dflash2_context_kv((size_t)aux_tok_offset * D * 2, stack_stride, count, start_pos, slot_id);
        } else {
            impl_->store_dspark_context_kv((size_t)aux_tok_offset * D * 2, stack_stride, count, start_pos, slot_id);
        }
        uint8_t* dst_aux = static_cast<uint8_t*>(impl_->b_last_aux.contents) + (size_t)slot_id * 4 * D * 2;
        const uint8_t* src_base = static_cast<const uint8_t*>(impl_->b_aux_states.contents);
        size_t last_off = (size_t)(aux_tok_offset + count - 1) * D * 2;
        for (int tap = 0; tap < 4; ++tap) {
            memcpy(dst_aux + (size_t)tap * D * 2,
                   src_base + (size_t)tap * (size_t)stack_stride * 2 + last_off,
                   (size_t)D * 2);
        }
    }
}

int Inkling2Model::max_dspark_slots() const {
    return impl_->dspark_slots;
}

void Inkling2Model::draft_tokens(int32_t anchor_tok, int anchor_pos, int num_draft, int32_t* h_out_drafts, int slot_id) {
    @autoreleasepool {
        if (impl_->df2_loaded) {
            impl_->run_dflash2_draft(anchor_tok, anchor_pos, num_draft, h_out_drafts, slot_id);
        } else {
            impl_->run_dspark_draft(anchor_tok, anchor_pos, num_draft, h_out_drafts, slot_id);
        }
    }
}

void Inkling2Model::forward_logits(const int32_t* h_tokens, int T, int start_pos, std::vector<float>& h_logits, int slot_id) {
    @autoreleasepool {
        impl_->forward_internal(cfg, max_batch_tokens, h_tokens, T, start_pos, slot_id, true, profile_stages, false);
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
