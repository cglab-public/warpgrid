#pragma once
#ifdef WARPGRID_METAL
#include "metal/model_metal.hpp"
#else
#include "config.hpp"
#include "dspark.hpp"
#include "safetensors.hpp"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ink {

struct LayerWeights {
    bool is_local = true;
    bool is_loop_core = false;
    bool is_mxfp8 = false;

    // Norms
    __nv_bfloat16* attn_norm = nullptr; // [D] (local) or [3, D] stacked (q, k, v) for global
    __nv_bfloat16* mlp_norm = nullptr;  // [D]

    // Attention projections (BF16 for prefill + per-row FP8 E4M3 for M=1..8 decode/verify!)
    __nv_bfloat16* w_qrgs = nullptr;    // [6552, 2304] stacked [wq(3072), wr(384), gate_attn(3072), sink_proj(24)]
    __nv_bfloat16* w_kv = nullptr;      // [2048, 2304] stacked [wk(1024), wv(1024)]
    uint8_t* w_qrgs_fp8 = nullptr;      // [6552, 2304] FP8 E4M3
    float*   w_qrgs_sc  = nullptr;      // [6552]
    uint8_t* w_kv_fp8   = nullptr;      // [2048, 2304] FP8 E4M3 (wk = +0, wv = +1024*D)
    float*   w_kv_sc    = nullptr;      // [2048]
    uint8_t* wo_fp8     = nullptr;      // [2304, 3072] FP8 E4M3
    float*   wo_sc      = nullptr;      // [2304]
    __nv_bfloat16* wq = nullptr;        // [3072, 2304] (aliases w_qrgs)
    __nv_bfloat16* wk = nullptr;        // [1024, 2304] (aliases w_kv)
    __nv_bfloat16* wv = nullptr;        // [1024, 2304] (aliases w_kv + 1024*D)
    __nv_bfloat16* wr = nullptr;        // [384, 2304]  (aliases w_qrgs + 3072*D)
    __nv_bfloat16* wo = nullptr;        // [2304, 3072]
    __nv_bfloat16* gate_attn = nullptr; // [3072, 2304] (aliases w_qrgs + 3456*D)
    __nv_bfloat16* sink_proj = nullptr; // [24, 2304]   (aliases w_qrgs + 6528*D)

    // Attention clips & rel bias
    __nv_bfloat16* q_clip_scale = nullptr;   // [24, 128]
    __nv_bfloat16* rel_clip_scale = nullptr; // [24, 16]
    __nv_bfloat16* rel_proj = nullptr;       // [16, 1024]

    // Short conv weights
    __nv_bfloat16* k_sconv_w = nullptr;      // [1024, 4] (local) or static_w [4, 1024] (global)
    __nv_bfloat16* k_sconv_proj = nullptr;   // [32, 2304] (global only)
    __nv_bfloat16* v_sconv_w = nullptr;      // [1024, 4] (local) or static_w [4, 1024] (global)
    __nv_bfloat16* v_sconv_proj = nullptr;   // [32, 2304] (global only)
    __nv_bfloat16* attn_sconv_w = nullptr;   // [2304, 4]
    __nv_bfloat16* mlp_sconv_w = nullptr;    // [2304, 4]

    // Loop controllers (for core layers 6..17, 4 streams: 0=attn, 1=k, 2=v, 3=mlp)
    __nv_bfloat16* ctrl_down[4] = {};   // [64, 4608]
    __nv_bfloat16* ctrl_gain[4] = {};   // [2304, 64]
    __nv_bfloat16* ctrl_forget[4] = {}; // [2304, 64]

    // MoE router
    __nv_bfloat16* gate_weight = nullptr; // [128, 2304]
    float* gate_bias = nullptr;           // [128]

    // MoE experts (NVFP4 or MXFP8)
    uint8_t* w13_weight = nullptr;
    uint8_t* w13_scale = nullptr;
    uint8_t* w13_scale_swiz = nullptr; // Blackwell 128x4 swizzled scales for cuBLASLt NVFP4
    float* w13_amax = nullptr;
    float* w13_denom = nullptr;
    std::vector<float> h_w13_gscale;   // [128] host amax/denom for zero-sync cuBLASLt alpha
    uint8_t* w2_weight = nullptr;
    uint8_t* w2_scale = nullptr;
    uint8_t* w2_scale_swiz = nullptr;
    float* w2_amax = nullptr;
    float* w2_denom = nullptr;
    std::vector<float> h_w2_gscale;    // [128]
};

struct VisitCache {
    bool is_local = true;
    int cap = 0; // 1024 for local, max_seq_len for global
    __nv_bfloat16* k_cache = nullptr;     // [max_slots, cap, 8, 128] (local SWA BF16)
    __nv_bfloat16* v_cache = nullptr;     // [max_slots, cap, 8, 128] (local SWA BF16)
    uint8_t* k_cache_fp8 = nullptr;       // [max_slots, cap, 8, 128] (global FP8 E4M3)
    uint8_t* v_cache_fp8 = nullptr;       // [max_slots, cap, 8, 128] (global FP8 E4M3)
    float* k_scale_fp8 = nullptr;         // [max_slots, cap, 8] per-(pos, head) scale
    float* v_scale_fp8 = nullptr;         // [max_slots, cap, 8]
    __nv_bfloat16* k_conv_state = nullptr;    // [3, 1024]
    __nv_bfloat16* v_conv_state = nullptr;    // [3, 1024]
    __nv_bfloat16* attn_conv_state = nullptr; // [3, 2304]
    __nv_bfloat16* mlp_conv_state = nullptr;  // [3, 2304]

    // Speculative verify snapshots [8, 3, C]
    __nv_bfloat16* k_conv_snap = nullptr;
    __nv_bfloat16* v_conv_snap = nullptr;
    __nv_bfloat16* attn_conv_snap = nullptr;
    __nv_bfloat16* mlp_conv_snap = nullptr;

    // Prefix-cache end-of-prefill snapshots per slot (local SWA ring + conv states)
    __nv_bfloat16* local_k_prefill_snap = nullptr; // [max_slots, cap, 8, 128] (local only)
    __nv_bfloat16* local_v_prefill_snap = nullptr;
    __nv_bfloat16* k_conv_prefill_snap = nullptr;    // [max_slots, 3, 1024]
    __nv_bfloat16* v_conv_prefill_snap = nullptr;    // [max_slots, 3, 1024]
    __nv_bfloat16* attn_conv_prefill_snap = nullptr; // [max_slots, 3, 2304]
    __nv_bfloat16* mlp_conv_prefill_snap = nullptr;  // [max_slots, 3, 2304]
};

class Inkling2Model {
public:
    InklingConfig cfg;
    SafetensorsFile st;
    cublasHandle_t cublas = nullptr;
    cublasLtHandle_t cublas_lt = nullptr;
    cublasLtMatmulDesc_t fp4_op_desc = nullptr;
    void* d_lt_workspace = nullptr;
    size_t lt_workspace_bytes = 32 * 1024 * 1024;
    struct Fp4Plan {
        cublasLtMatrixLayout_t layoutA = nullptr;
        cublasLtMatrixLayout_t layoutB = nullptr;
        cublasLtMatrixLayout_t layoutD = nullptr;
        cublasLtMatmulAlgo_t algo{};
    };
    std::unordered_map<uint64_t, Fp4Plan> fp4_plan_cache;
    cudaStream_t stream = nullptr;

    // Top-level weights
    __nv_bfloat16* embed_w = nullptr;      // [201024, 2304]
    __nv_bfloat16* embed_norm_w = nullptr; // [2304]
    __nv_bfloat16* final_norm_w = nullptr; // [2304]
    __nv_bfloat16* unembed_w = nullptr;    // [201024, 2304]
    uint8_t* unembed_fp8 = nullptr;        // [200058, 2304] FP8 E4M3
    float*   unembed_sc  = nullptr;        // [200058]

    // AttnRes queries (FP32, pre-scaled by 1/sqrt(D))
    std::vector<float*> attn_queries; // size 23: index i-1 for layer i (1..23)
    std::vector<float*> mlp_queries;  // size 24: index i for layer i (0..23)
    float* output_query = nullptr;    // [2304]
    // Loop cross-pass message queries: [12, 4, 2304] FP32 pre-scaled
    float* message_queries = nullptr;

    std::vector<LayerWeights> layers;
    std::vector<std::pair<int, int>> visits; // (layer_id, pass_idx), length 48
    std::vector<VisitCache> caches;          // length 48

    // Extra GPU allocations owned by model
    std::vector<void*> extra_allocs;

    // Runtime workspace
    int max_slots = 8;
    int max_seq_len = 32768;
    int max_batch_tokens = 2048;

    int32_t* d_tokens = nullptr;
    int32_t* d_pos = nullptr;                // [1] device pos for CUDA graph decode
    int32_t* d_batch_slots = nullptr;        // [max_slots] for batched decode
    int32_t* d_batch_pos = nullptr;          // [max_slots] for batched decode
    __nv_bfloat16* d_stack = nullptr;        // [6, max_T, D]
    __nv_bfloat16* d_prelude_part = nullptr; // [max_T, D]
    __nv_bfloat16* d_bank = nullptr;         // [2, max_T, D]
    float* d_prelude_dest = nullptr;         // [max_T, D]
    float* d_messages = nullptr;             // [4, max_T, D]
    float* d_fp32_mix = nullptr;             // [3, max_T, D]
    __nv_bfloat16* d_attn_in = nullptr;      // [3, max_T, D]
    __nv_bfloat16* d_mlp_in = nullptr;       // [max_T, D]
    __nv_bfloat16* d_pair = nullptr;         // [max_T, 2*D]
    __nv_bfloat16* d_ctrl_h = nullptr;       // [max_T, 64]
    float* d_ctrl_gain = nullptr;            // [max_T, D]
    float* d_ctrl_forget = nullptr;          // [max_T, D]

    __nv_bfloat16* d_qrgs = nullptr;       // [max_T, 6552]
    __nv_bfloat16* d_kv_out = nullptr;     // [max_T, 2048]
    __nv_bfloat16* d_Q = nullptr;          // [max_T, 3072]
    __nv_bfloat16* d_K = nullptr;          // [max_T, 1024]
    __nv_bfloat16* d_V = nullptr;          // [max_T, 1024]
    __nv_bfloat16* d_K_conv = nullptr;     // [max_T, 1024]
    __nv_bfloat16* d_V_conv = nullptr;     // [max_T, 1024]
    __nv_bfloat16* d_dyn_w = nullptr;      // [max_T, 32]
    __nv_bfloat16* d_R = nullptr;          // [max_T, 384]
    __nv_bfloat16* d_rel_logits = nullptr; // [max_T, 24, 1024]
    __nv_bfloat16* d_gate = nullptr;       // [max_T, 3072]
    __nv_bfloat16* d_sinks = nullptr;      // [max_T, 24]
    __nv_bfloat16* d_attn_raw = nullptr;   // [max_T, 3072]
    __nv_bfloat16* d_attn_proj = nullptr;  // [max_T, 2304]
    __nv_bfloat16* d_attn_out = nullptr;   // [max_T, 2304]
    __nv_bfloat16* d_mlp_raw = nullptr;    // [max_T, 2304]
    __nv_bfloat16* d_mlp_out = nullptr;    // [max_T, 2304]
    float* d_split_scratch = nullptr;      // [24 * 16 * 130]

    int32_t* d_topk_ids = nullptr;     // [max_T, 8]
    float* d_topk_weights = nullptr;   // [max_T, 8]
    float* d_router_logits = nullptr;  // [max_T, 128]
    float* d_moe_act = nullptr;        // [max_T * 8 * 1152]
    int32_t* d_moe_uniq = nullptr;     // [1024] GPU scratch for small-T deduplicated MoE

    // Grouped-expert prefill scratch buffers (only used when T > 1)
    __nv_bfloat16* d_exp_w13_bf16 = nullptr; // [2*inter, D] = 10.6 MB
    __nv_bfloat16* d_exp_w2_bf16 = nullptr;  // [D, inter] = 5.3 MB
    __nv_bfloat16* d_exp_x = nullptr;        // [max_T, D]
    __nv_bfloat16* d_exp_gu = nullptr;       // [max_T, 2*inter]
    __nv_bfloat16* d_exp_act = nullptr;      // [max_T, inter]
    __nv_bfloat16* d_exp_down = nullptr;     // [max_T, D]
    uint8_t* d_exp_x_fp4 = nullptr;          // [max_T_pad, D/2]
    uint8_t* d_exp_x_scale = nullptr;        // [max_T_pad128, D/16]
    uint8_t* d_exp_act_fp4 = nullptr;        // [max_T_pad, inter/2]
    uint8_t* d_exp_act_scale = nullptr;      // [max_T_pad128, inter/16]
    float* d_moe_out_fp32 = nullptr;         // [max_T, D]
    int32_t* d_exp_tok_idx = nullptr;        // [max_T * 8]
    float* d_exp_gammas = nullptr;           // [max_T * 8]
    float* d_logits = nullptr;         // [8 * vocab_size] (supports batched verify up to 8)
    int32_t* d_out_token = nullptr;    // [8]

    // DSpark speculative decoding
    DSparkDrafter dspark;
    __nv_bfloat16* d_aux_states = nullptr; // [4, max_T, D]
    __nv_bfloat16* d_last_aux = nullptr;   // [4, 1, D] (last committed token's aux features)
    int draft_k = 0;                       // 0 = disabled, 1..7 = DSpark speculative tokens
    std::string draft_vocab_file;          // optional frequency-ranked pruned draft vocab binary
    std::string dflash2_dir;
    int moe_group_thresh = 64;             // T >= thresh uses grouped cuBLAS prefill MoE (T < 64 stays pure GPU!)
    bool save_conv_snaps = false;

    bool use_bf16_gemv = true;
    bool use_cuda_graph = true;
    bool profile_stages = false;

    cudaGraph_t decode_graph = nullptr;
    cudaGraphExec_t decode_graph_exec = nullptr;
    bool graph_captured = false;

    cudaGraph_t verify_graphs[9] = {};
    cudaGraphExec_t verify_graph_execs[9] = {};

    cudaGraph_t batch_graphs[9] = {};
    cudaGraphExec_t batch_graph_execs[9] = {};

    cudaGraph_t batch_verify_graphs[9][6] = {};
    cudaGraphExec_t batch_verify_graph_execs[9][6] = {};

    // Per-stage accumulated GPU ms (for profiling with graphs off)
    double prof_attn_res_ctrl_ms = 0.0;
    double prof_dense_proj_ms = 0.0;
    double prof_attn_sconv_ms = 0.0;
    double prof_router_ms = 0.0;
    double prof_moe_nvfp4_ms = 0.0;
    double prof_moe_mxfp8_ms = 0.0;
    double prof_lm_head_ms = 0.0;
    int prof_steps = 0;
    void reset_profile();
    void print_profile() const;

    Inkling2Model() = default;
    ~Inkling2Model();

    void load(const std::string& model_dir, int max_seq = 32768, int max_chunk = 512, int n_slots = 8);
    void reset_cache(int slot_id = 0);
    void save_prefill_snapshot(int slot_id);
    void restore_prefill_snapshot(int slot_id);

    // Forward a chunk of tokens for `slot_id` starting at start_pos, returns greedy next token
    int32_t forward_and_greedy(const int32_t* h_tokens, int T, int start_pos, int slot_id = 0);

    // Parallel batched decode: B active sequences (1 token each), returns greedy next token for all B
    void forward_batch_decode_greedy(
        const int32_t* h_tokens, const int32_t* h_slots, const int32_t* h_positions,
        int B, int32_t* h_out_tokens);

    // Speculative verify (slot 0): forward T (<=8) tokens, return greedy next token for ALL T positions
    void forward_verify_greedy(const int32_t* h_tokens, int T, int start_pos, int32_t* h_out_tokens);

    // Parallel batched speculative verify: B slots (2..4) x L tokens (2..5), total T = B*L <= 16
    void forward_batch_verify_greedy(
        const int32_t* h_tokens, const int32_t* h_slots, const int32_t* h_positions,
        int B, int L, int32_t* h_out_tokens);

    // Roll back conv states for `slot_id` to post-step `accepted_len - 1` (1 <= accepted_len <= L)
    void rollback_conv_states(int accepted_len, int slot_id = 0);

    // Save aux states of token index `aux_tok_offset + count - 1` into d_last_aux[slot_id]
    // and store context KV for `[aux_tok_offset .. aux_tok_offset + count - 1]` into dspark slot_id
    void commit_dspark_context(int count, int start_pos, int slot_id = 0, int aux_tok_offset = 0);

    int max_dspark_slots() const { return dspark.max_dspark_slots; }

    void draft_tokens(int32_t anchor_tok, int anchor_pos, int num_draft, int32_t* h_out_drafts, int slot_id = 0) {
        dspark.draft_tokens(anchor_tok, anchor_pos,
                            d_last_aux + (int64_t)slot_id * 4 * cfg.hidden_size,
                            cfg.hidden_size, num_draft, h_out_drafts, slot_id);
    }

    void forward_logits(const int32_t* h_tokens, int T, int start_pos, std::vector<float>& h_logits, int slot_id = 0);

private:
    void* alloc_gpu(size_t bytes);
    void linear_bf16(const __nv_bfloat16* X, const __nv_bfloat16* W, __nv_bfloat16* Y, int M, int N, int K);
    void linear_bf16_to_fp32(const __nv_bfloat16* X, const __nv_bfloat16* W, float* Y, int M, int N, int K);
    void linear_nvfp4_lt(const uint8_t* X_fp4, const uint8_t* X_scale,
                         const uint8_t* W_fp4, const uint8_t* W_scale,
                         __nv_bfloat16* Y_bf16, int M_pad, int N, int K, float alpha);
    void run_controller(const LayerWeights& lw, int s, const float* dest, const float* msg,
                        const __nv_bfloat16* norm_w, __nv_bfloat16* out, int T);
    void run_moe_grouped_prefill(const LayerWeights& lw, const __nv_bfloat16* x, __nv_bfloat16* out, int T);
    void forward_internal(const int32_t* h_tokens, int T, int start_pos, int slot_id = 0);
    void forward_batch_decode_internal(int B);
    void forward_batch_verify_internal(int B, int L);
};

}
#endif
