#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace ink {

void launch_embed_norm(
    const int32_t* token_ids,
    const __nv_bfloat16* embed_w,
    const __nv_bfloat16* norm_w,
    __nv_bfloat16* out,
    int T, int D, float eps, cudaStream_t stream);

void launch_rmsnorm(
    const __nv_bfloat16* x,
    const __nv_bfloat16* w,
    __nv_bfloat16* out,
    int T, int D, float eps, cudaStream_t stream);

void launch_add_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* out,
    int N, cudaStream_t stream);

void launch_split_qrgs(
    const __nv_bfloat16* qrgs, // [T, q_dim + r_dim + q_dim + n_heads]
    __nv_bfloat16* Q, __nv_bfloat16* R, __nv_bfloat16* gate, __nv_bfloat16* sinks,
    int T, int q_dim, int r_dim, int n_heads, cudaStream_t stream);

void launch_split_kv(
    const __nv_bfloat16* kv, // [T, 2 * kv_dim]
    __nv_bfloat16* K, __nv_bfloat16* V,
    int T, int kv_dim, cudaStream_t stream);

// Block AttnRes: mixes stack[0..num_stack-1] + (last + optional addend)
// queries: [num_q, D] FP32 (pre-scaled by 1/sqrt(D))
// If norm_w != nullptr: writes BF16 out_bf16[num_q, T, D]
// Else: writes FP32 out_fp32[num_q, T, D]
void launch_attn_res(
    const __nv_bfloat16* stack,
    int num_stack,
    int stack_stride, // stride between blocks in elements (max_T * D)
    const __nv_bfloat16* last,
    const __nv_bfloat16* addend, // may be nullptr
    const float* queries,
    int num_q,
    const __nv_bfloat16* norm_w, // [num_q, D] or nullptr
    __nv_bfloat16* out_bf16,
    float* out_fp32,
    int T, int D, float eps, cudaStream_t stream);

// Loop controller step 1: RMSNorm dest and message independently, concat into BF16 [T, 2*D]
void launch_pair_rms_norm(
    const float* dest,
    const float* message,
    __nv_bfloat16* pair_out,
    int T, int D, float eps, cudaStream_t stream);

// Loop controller step 2 (fused): gain = h @ W_gain^T, forget = h @ W_forget^T,
// (1 - tanh(forget))*dest + tanh(gain)*message -> RMSNorm -> BF16
void launch_fused_ctrl_blend_rmsnorm(
    const __nv_bfloat16* ctrl_h,      // [T, 64]
    const __nv_bfloat16* gain_w,      // [D, 64]
    const __nv_bfloat16* forget_w,    // [D, 64]
    const float* dest,                // [T, D]
    const float* message,             // [T, D]
    const __nv_bfloat16* norm_w,      // [D]
    __nv_bfloat16* out,               // [T, D]
    int T, int D, int rank, float eps, cudaStream_t stream,
    float* tmp_fp32 = nullptr);

// Static causal depthwise short conv (k=4), updates state[3, C] in-place.
// If snap != nullptr, also saves post-step state [T, 3, C] for speculative rollback.
void launch_sconv_static(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight, // [C, 4]
    __nv_bfloat16* state,        // [3, C]
    __nv_bfloat16* out,
    int T, int C, cudaStream_t stream,
    __nv_bfloat16* snap = nullptr);

// Dynamic causal depthwise short conv (k=4) for global K/V, updates state[3, C].
void launch_sconv_dynamic(
    const __nv_bfloat16* x,
    const __nv_bfloat16* static_w, // [4, C]
    const __nv_bfloat16* dyn_w,    // [T, num_kv_heads * 4]
    __nv_bfloat16* state,          // [3, C]
    __nv_bfloat16* out,
    int T, int num_kv_heads, int head_dim, cudaStream_t stream,
    __nv_bfloat16* snap = nullptr);

// Attention prep (if d_pos != nullptr, reads start_pos from *d_pos for CUDA graph capture)
void launch_attn_prep(
    __nv_bfloat16* Q,             // [T, n_heads, head_dim] in-place
    const __nv_bfloat16* K_in,    // [T, n_kv_heads, head_dim]
    const __nv_bfloat16* V_in,    // [T, n_kv_heads, head_dim]
    __nv_bfloat16* R,             // [T, n_heads, d_rel] in-place
    const __nv_bfloat16* q_scale, // [n_heads, head_dim]
    const __nv_bfloat16* r_scale, // [n_heads, d_rel]
    __nv_bfloat16* k_cache,       // [cache_len, n_kv_heads, head_dim] (local BF16)
    __nv_bfloat16* v_cache,       // [cache_len, n_kv_heads, head_dim] (local BF16)
    int start_pos, const int32_t* d_pos, int T, int cache_cap, bool is_local,
    int n_heads, int n_kv_heads, int head_dim, int d_rel,
    float qk_clip, float rel_clip, float log_n_floor, float log_alpha, float eps,
    cudaStream_t stream,
    uint8_t* k_cache_fp8 = nullptr, uint8_t* v_cache_fp8 = nullptr,
    float* k_scale_fp8 = nullptr, float* v_scale_fp8 = nullptr,
    const int32_t* block_table = nullptr);

// Attention forward (if d_pos != nullptr & T==1, reads q_pos from *d_pos with fixed splits for CUDA graph)
void launch_attention(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    const __nv_bfloat16* rel_logits,
    const __nv_bfloat16* gate,
    const __nv_bfloat16* sinks,
    __nv_bfloat16* out,
    float* split_scratch,
    int start_pos, const int32_t* d_pos, int T, int cache_cap, bool is_local, int window_size,
    int n_heads, int n_kv_heads, int head_dim, int rel_extent,
    cudaStream_t stream,
    const uint8_t* k_cache_fp8 = nullptr, const uint8_t* v_cache_fp8 = nullptr,
    const float* k_scale_fp8 = nullptr, const float* v_scale_fp8 = nullptr,
    const int32_t* block_table = nullptr);

// MoE router: top-K from sigmoid(x @ W.T) + bias, normalized * route_scale
void launch_moe_router(
    const __nv_bfloat16* x,      // [T, D]
    const __nv_bfloat16* weight, // [E, D]
    const float* bias,           // [E]
    float* logits_scratch,       // [T, E]
    int32_t* topk_ids,           // [T, K]
    float* topk_weights,         // [T, K]
    int T, int D, int E, int K, float route_scale,
    cudaStream_t stream);

// MoE NVFP4 forward (decode M=1 fast path + general T path)
void launch_moe_nvfp4(
    const __nv_bfloat16* x,          // [T, D]
    const int32_t* topk_ids,         // [T, K]
    const float* topk_weights,       // [T, K]
    const uint8_t* w13_u8,           // [E, 2*inter, D/2]
    const uint8_t* w13_scale_e4m3,   // [E, 2*inter, D/16]
    const float* w13_amax,           // [E]
    const float* w13_denom,          // [1]
    const uint8_t* w2_u8,            // [E, D, inter/2]
    const uint8_t* w2_scale_e4m3,    // [E, D, inter/16]
    const float* w2_amax,            // [E]
    const float* w2_denom,           // [1]
    float* act_scratch,              // [T * K * inter]
    __nv_bfloat16* out,              // [T, D]
    int T, int D, int inter, int K,
    float gate_cap, float up_cap,
    cudaStream_t stream);

// MoE MXFP8 forward (L20-L23)
void launch_moe_mxfp8(
    const __nv_bfloat16* x,          // [T, D]
    const int32_t* topk_ids,         // [T, K]
    const float* topk_weights,       // [T, K]
    const uint8_t* w13_e4m3,         // [E, 2*inter, D]
    const uint8_t* w13_scale_ue8m0,  // [E, 2*inter, D/32]
    const uint8_t* w2_e4m3,          // [E, D, inter]
    const uint8_t* w2_scale_ue8m0,   // [E, D, inter/32]
    float* act_scratch,              // [T * K * inter]
    __nv_bfloat16* out,              // [T, D]
    int T, int D, int inter, int K,
    float gate_cap, float up_cap,
    cudaStream_t stream);

// Small-batch (T = 2..8) GPU-deduplicated MoE (reads each unique expert ONCE per verify block!)
void launch_moe_small_batch_nvfp4(
    const __nv_bfloat16* x,          // [T, D]
    const int32_t* topk_ids,         // [T, K]
    const float* topk_weights,       // [T, K]
    const uint8_t* w13_u8,
    const uint8_t* w13_scale_e4m3,
    const float* w13_amax,
    const float* w13_denom,
    const uint8_t* w2_u8,
    const uint8_t* w2_scale_e4m3,
    const float* w2_amax,
    const float* w2_denom,
    int32_t* uniq_scratch,           // [64 * 10 + 1] GPU scratch for deduplicated expert map
    float* act_scratch,              // [T * K * inter]
    __nv_bfloat16* out,              // [T, D]
    int T, int D, int inter, int K,
    float gate_cap, float up_cap,
    cudaStream_t stream);

void launch_moe_small_batch_mxfp8(
    const __nv_bfloat16* x,
    const int32_t* topk_ids,
    const float* topk_weights,
    const uint8_t* w13_e4m3,
    const uint8_t* w13_scale_ue8m0,
    const uint8_t* w2_e4m3,
    const uint8_t* w2_scale_ue8m0,
    int32_t* uniq_scratch,
    float* act_scratch,
    __nv_bfloat16* out,
    int T, int D, int inter, int K,
    float gate_cap, float up_cap,
    cudaStream_t stream);

// Grouped-expert prefill helpers (for T > 1 with cuBLAS Tensor Cores)
void launch_dequant_nvfp4_matrix(
    const uint8_t* w_u8, const uint8_t* scale_e4m3,
    const float* amax_ptr, const float* denom_ptr, int expert,
    __nv_bfloat16* out_bf16, int N, int K, cudaStream_t stream);

void launch_dequant_mxfp8_matrix(
    const uint8_t* w_e4m3, const uint8_t* scale_ue8m0, int expert,
    __nv_bfloat16* out_bf16, int N, int K, cudaStream_t stream);

void launch_gather_bf16(
    const __nv_bfloat16* src, const int32_t* idx, __nv_bfloat16* dst,
    int M, int D, cudaStream_t stream);

void launch_swiglu_softcap_bf16(
    const __nv_bfloat16* gate_up, const float* gammas, __nv_bfloat16* act_out,
    int M, int inter, float gate_cap, float up_cap, cudaStream_t stream);

void launch_scatter_add_bf16_to_fp32(
    const __nv_bfloat16* src, const int32_t* idx, float* dst_fp32,
    int M, int D, cudaStream_t stream);

void launch_fp32_to_bf16(
    const float* src, __nv_bfloat16* dst, int N, cudaStream_t stream);

// Native Blackwell NVFP4 cuBLASLt helpers (128x4 swizzled VEC16_UE4M3 scales)
void launch_swizzle_nvfp4_scales(
    const uint8_t* src, uint8_t* dst, int E, int N, int K_s, cudaStream_t stream);

void launch_gather_quant_nvfp4(
    const __nv_bfloat16* x, const int32_t* idx,
    uint8_t* out_fp4, uint8_t* out_scale_swiz,
    int M, int M_pad, int K, cudaStream_t stream);

void launch_swiglu_softcap_quant_nvfp4(
    const __nv_bfloat16* gate_up, const float* gammas,
    uint8_t* out_fp4, uint8_t* out_scale_swiz,
    int M, int M_pad, int inter, float gate_cap, float up_cap, cudaStream_t stream);

// High-bandwidth vectorized BF16 GEMV/Small-GEMM for M = 1..8: Y[M, N] = X[M, K] @ W[N, K]^T
void launch_bf16_gemv(
    const __nv_bfloat16* x, // [K]
    const __nv_bfloat16* W, // [N, K]
    __nv_bfloat16* y_bf16,  // [N] (either y_bf16 or y_fp32 non-null)
    float* y_fp32,          // [N]
    int N, int K, cudaStream_t stream);

void launch_bf16_small_gemm(
    const __nv_bfloat16* X, // [M, K] (M in 2..8)
    const __nv_bfloat16* W, // [N, K]
    __nv_bfloat16* Y_bf16,  // [M, N] or nullptr
    float* Y_fp32,          // [M, N] or nullptr
    int M, int N, int K, cudaStream_t stream);

// On-the-fly per-row FP8 E4M3 quantization + W8A16 high-bandwidth GEMV/Small-GEMM (M = 1..8)
void launch_quantize_bf16_to_fp8_rowwise(
    const __nv_bfloat16* W_bf16, // [N, K]
    uint8_t* W_fp8,              // [N, K]
    float* W_scale,              // [N]
    int N, int K, cudaStream_t stream);

void launch_fp8_small_gemm(
    const __nv_bfloat16* X, // [M, K] (M in 1..8)
    const uint8_t* W_fp8,   // [N, K]
    const float* W_scale,   // [N]
    __nv_bfloat16* Y_bf16,  // [M, N] or nullptr
    float* Y_fp32,          // [M, N] or nullptr
    int M, int N, int K, cudaStream_t stream);

// Multi-slot batched decode kernels (B active sequences, 1 token each, slot_ids[b] & positions[b])
void launch_sconv_static_batched(
    const __nv_bfloat16* x,          // [B, C]
    const __nv_bfloat16* weight,     // [C, 4]
    __nv_bfloat16* state_pool,       // [max_slots, 3, C]
    const int32_t* d_slot_ids,       // [B]
    __nv_bfloat16* out,              // [B, C]
    int B, int C, cudaStream_t stream);

void launch_sconv_dynamic_batched(
    const __nv_bfloat16* x,          // [B, C]
    const __nv_bfloat16* static_w,   // [4, C]
    const __nv_bfloat16* dyn_w,      // [B, num_kv_heads * 4]
    __nv_bfloat16* state_pool,       // [max_slots, 3, C]
    const int32_t* d_slot_ids,       // [B]
    __nv_bfloat16* out,              // [B, C]
    int B, int num_kv_heads, int head_dim, cudaStream_t stream);

void launch_attn_prep_batched(
    __nv_bfloat16* Q, const __nv_bfloat16* K_in, const __nv_bfloat16* V_in, __nv_bfloat16* R,
    const __nv_bfloat16* q_scale, const __nv_bfloat16* r_scale,
    __nv_bfloat16* k_cache_pool, __nv_bfloat16* v_cache_pool, // [max_slots, cap, kv_heads, head_dim]
    const int32_t* d_slot_ids, const int32_t* d_positions,
    int B, int cache_cap, bool is_local,
    int n_heads, int n_kv_heads, int head_dim, int d_rel,
    float qk_clip, float rel_clip, float log_n_floor, float log_alpha, float eps,
    cudaStream_t stream,
    uint8_t* k_cache_fp8_pool = nullptr, uint8_t* v_cache_fp8_pool = nullptr,
    float* k_scale_fp8_pool = nullptr, float* v_scale_fp8_pool = nullptr,
    const int32_t* block_tables = nullptr, int max_blocks_per_seq = 0);

void launch_attention_batched(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* k_cache_pool, const __nv_bfloat16* v_cache_pool,
    const __nv_bfloat16* rel_logits, const __nv_bfloat16* gate, const __nv_bfloat16* sinks,
    __nv_bfloat16* out,
    const int32_t* d_slot_ids, const int32_t* d_positions,
    int B, int cache_cap, bool is_local, int window_size,
    int n_heads, int n_kv_heads, int head_dim, int rel_extent,
    cudaStream_t stream,
    const uint8_t* k_cache_fp8_pool = nullptr, const uint8_t* v_cache_fp8_pool = nullptr,
    const float* k_scale_fp8_pool = nullptr, const float* v_scale_fp8_pool = nullptr,
    const int32_t* block_tables = nullptr, int max_blocks_per_seq = 0);

// Multi-slot batched speculative verify kernels (B active sequences, L tokens each, total T = B * L)
void launch_sconv_static_batch_verify(
    const __nv_bfloat16* x,          // [B * L, C]
    const __nv_bfloat16* weight,     // [C, 4]
    __nv_bfloat16* state_pool,       // [max_slots, 3, C]
    __nv_bfloat16* snap_pool,        // [max_dspark_slots * 8, 3, C]
    const int32_t* d_slot_ids,       // [B]
    __nv_bfloat16* out,              // [B * L, C]
    int B, int L, int C, cudaStream_t stream);

void launch_sconv_dynamic_batch_verify(
    const __nv_bfloat16* x,          // [B * L, C]
    const __nv_bfloat16* static_w,   // [4, C]
    const __nv_bfloat16* dyn_w,      // [B * L, num_kv_heads * 4]
    __nv_bfloat16* state_pool,       // [max_slots, 3, C]
    __nv_bfloat16* snap_pool,        // [max_dspark_slots * 8, 3, C]
    const int32_t* d_slot_ids,       // [B]
    __nv_bfloat16* out,              // [B * L, C]
    int B, int L, int num_kv_heads, int head_dim, cudaStream_t stream);

void launch_attn_prep_batch_verify(
    __nv_bfloat16* Q, const __nv_bfloat16* K_in, const __nv_bfloat16* V_in, __nv_bfloat16* R,
    const __nv_bfloat16* q_scale, const __nv_bfloat16* r_scale,
    __nv_bfloat16* k_cache_pool, __nv_bfloat16* v_cache_pool,
    const int32_t* d_slot_ids, const int32_t* d_positions,
    int B, int L, int cache_cap, bool is_local,
    int n_heads, int n_kv_heads, int head_dim, int d_rel,
    float qk_clip, float rel_clip, float log_n_floor, float log_alpha, float eps,
    cudaStream_t stream,
    uint8_t* k_cache_fp8_pool = nullptr, uint8_t* v_cache_fp8_pool = nullptr,
    float* k_scale_fp8_pool = nullptr, float* v_scale_fp8_pool = nullptr,
    const int32_t* block_tables = nullptr, int max_blocks_per_seq = 0);

void launch_attention_batch_verify(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* k_cache_pool, const __nv_bfloat16* v_cache_pool,
    const __nv_bfloat16* rel_logits, const __nv_bfloat16* gate, const __nv_bfloat16* sinks,
    __nv_bfloat16* out,
    const int32_t* d_slot_ids, const int32_t* d_positions,
    int B, int L, int cache_cap, bool is_local, int window_size,
    int n_heads, int n_kv_heads, int head_dim, int rel_extent,
    cudaStream_t stream,
    const uint8_t* k_cache_fp8_pool = nullptr, const uint8_t* v_cache_fp8_pool = nullptr,
    const float* k_scale_fp8_pool = nullptr, const float* v_scale_fp8_pool = nullptr,
    float* split_scratch = nullptr,
    const int32_t* block_tables = nullptr, int max_blocks_per_seq = 0);

// Final logit softcap + greedy argmax on GPU
void launch_softcap_argmax(
    float* logits, // [unpadded_vocab] in-place softcapped
    int32_t* out_token,
    int vocab_size, float softcap, cudaStream_t stream);

// Batched greedy argmax over M rows of FP32 logits [M, vocab_size]
void launch_batched_argmax(
    const float* logits, int32_t* out_tokens,
    int M, int vocab_size, cudaStream_t stream);

} // namespace ink
