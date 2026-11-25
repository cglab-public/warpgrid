#pragma once
#include "../gguf.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace ink::cuda_gguf {

void launch_gemv(GgmlType type, const float* x, const uint8_t* w, float* y,
                 uint32_t M, uint32_t N, uint32_t K, cudaStream_t stream);

void launch_grouped_gemv(GgmlType type, const float* x, const uint8_t* w, float* y,
                         uint32_t T, uint32_t G, uint32_t N, uint32_t K,
                         uint32_t group_stride_bytes, cudaStream_t stream);

void launch_moe_indexed(GgmlType type, const float* x, const uint8_t* w,
                        const int32_t* topk_ids, float* y,
                        uint32_t T, uint32_t N, uint32_t K, uint32_t top_k,
                        uint32_t expert_stride_bytes, cudaStream_t stream);

void launch_moe_down_accum(GgmlType type, const float* x, const uint8_t* w,
                           const int32_t* topk_ids, const float* topk_w, float* y,
                           uint32_t T, uint32_t N, uint32_t K, uint32_t top_k,
                           uint32_t expert_stride_bytes, cudaStream_t stream);

void launch_moe_gate_up_swiglu(GgmlType type, const float* x, const uint8_t* wg,
                               const uint8_t* wu, const int32_t* topk_ids, float* y,
                               uint32_t T, uint32_t N, uint32_t K, uint32_t top_k,
                               uint32_t expert_stride_bytes, float clamp_val, cudaStream_t stream);

void launch_mhc_expand(const float* x, float* streams, uint32_t T, uint32_t D, uint32_t hc, cudaStream_t stream);

void launch_rmsnorm(const float* x, const float* w, float* y, uint32_t T, uint32_t D, float eps, cudaStream_t stream);

void launch_rmsnorm_unweighted(const float* x, float* y, uint32_t T, uint32_t D, float eps, cudaStream_t stream);

void launch_hc_mixes(const float* raw, const float* scale, const float* base,
                     float* out_pre, float* out_post, float* out_comb,
                     uint32_t T, uint32_t hc, uint32_t sinkhorn_iters, float eps, cudaStream_t stream);

void launch_hc_pre(const float* streams, const float* pre_w, float* out_hidden,
                   uint32_t T, uint32_t D, uint32_t hc, cudaStream_t stream);

void launch_hc_post(const float* sub, const float* residual, const float* post_w,
                    const float* comb_w, float* out_streams,
                    uint32_t T, uint32_t D, uint32_t hc, cudaStream_t stream);

void launch_engram_apply(float* streams, const float* ekv, const float* gate,
                         uint32_t T, uint32_t D, uint32_t hc, float eps, cudaStream_t stream);

void launch_rope_partial(float* x, uint32_t T, uint32_t start_pos, uint32_t pos_stride,
                         uint32_t n_heads, uint32_t head_dim, uint32_t n_nope,
                         uint32_t n_rot, uint32_t inverse, float freq_base, cudaStream_t stream);

void launch_comp_group_reduce(const float* raw_kv, const float* raw_score, const float* norm_w,
                              float* comp_kv, uint32_t n_blocks, uint32_t ratio,
                              uint32_t head_dim, float eps, cudaStream_t stream);

void launch_indexer_score_topk(const float* idx_q, const float* comp_ik, const float* idx_w,
                               float* scratch_scores, int32_t* out_topk,
                               uint32_t T, uint32_t start_pos, uint32_t n_comp, uint32_t max_comp,
                               uint32_t n_idx_head, uint32_t idx_dim, uint32_t ratio,
                               uint32_t sliding_window, uint32_t top_k, float iw_scale, cudaStream_t stream);

void launch_mqa_sink_attn(const float* q, const float* kv_cache, const float* sinks, float* out,
                          uint32_t T, uint32_t start_pos, uint32_t n_heads, uint32_t head_dim,
                          uint32_t sliding_window, float scale, cudaStream_t stream);

void launch_mqa_sink_comp_attn(const float* q, const float* kv_cache, const float* comp_kv,
                               const int32_t* topk_sel, const float* sinks, float* out,
                               uint32_t T, uint32_t start_pos, uint32_t n_heads, uint32_t head_dim,
                               uint32_t sliding_window, uint32_t n_comp_sel, float scale, cudaStream_t stream);

void launch_router_sqrtsoftplus_topk(const float* logits, const float* bias,
                                     int32_t* out_ids, float* out_w,
                                     uint32_t T, uint32_t E, uint32_t top_k,
                                     uint32_t has_bias, float route_scale, cudaStream_t stream);

void launch_swiglu(const float* gate, const float* up, float* out, uint32_t n, float clamp_val, cudaStream_t stream);

void launch_zero(float* dst, uint32_t n, cudaStream_t stream);

void launch_argmax(const float* logits, int32_t* out_tok, uint32_t vocab_size, cudaStream_t stream);

}
