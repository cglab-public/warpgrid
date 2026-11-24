#include "dspark.hpp"
#include "kernels.cuh"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <chrono>

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    throw std::runtime_error(cudaGetErrorString(e)); } } while(0)

#define CHECK_CUBLAS(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
    throw std::runtime_error("cuBLAS error"); } } while(0)

namespace ink {

// ============================================================================
// DSpark CUDA Kernels
// ============================================================================

__device__ __forceinline__ float ds_warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ float ds_block_sum(float v) {
    __shared__ float sm[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = ds_warp_sum(v);
    if (lane == 0) sm[wid] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nw) ? sm[lane] : 0.0f;
    if (wid == 0) v = ds_warp_sum(v);
    return v;
}

// Mix 4 target aux states [4, stride, D] with alpha[4] + RMSNorm(weight) -> out[T, D]
__global__ void dflare_mix_rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ aux, // [4, aux_stride] where row t of tap j is aux[j*aux_stride + t*D]
    int aux_stride,
    const float* __restrict__ alpha,       // [4]
    const __nv_bfloat16* __restrict__ w,   // [D]
    __nv_bfloat16* __restrict__ out,       // [T, D]
    int T, int D, float eps)
{
    int t = blockIdx.x;
    if (t >= T) return;
    float a0 = alpha[0], a1 = alpha[1], a2 = alpha[2], a3 = alpha[3];
    float reg[12];
    float sum_sq = 0.0f;
    int k = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        int64_t off = (int64_t)t * D + d;
        float v = a0 * __bfloat162float(aux[0LL * aux_stride + off])
                + a1 * __bfloat162float(aux[1LL * aux_stride + off])
                + a2 * __bfloat162float(aux[2LL * aux_stride + off])
                + a3 * __bfloat162float(aux[3LL * aux_stride + off]);
        reg[k++] = v;
        sum_sq += v * v;
    }
    sum_sq = ds_block_sum(sum_sq);
    __shared__ float s_rstd;
    if (threadIdx.x == 0) s_rstd = rsqrtf(sum_sq / (float)D + eps);
    __syncthreads();
    float rstd = s_rstd;
    k = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        out[(int64_t)t * D + d] = __float2bfloat16(reg[k++] * rstd * __bfloat162float(w[d]));
    }
}

// Per-head RMSNorm + NeoX RoPE (in-place on x[T, n_heads, head_dim])
__global__ void qk_norm_rope_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ norm_w, // [head_dim] or nullptr
    int start_pos, const int32_t* __restrict__ d_start_pos, int T, int n_heads, int head_dim, float rope_theta, float eps)
{
    int h = blockIdx.x;
    int t = blockIdx.y;
    if (h >= n_heads || t >= T) return;
    int half = head_dim / 2; // 64
    int i = threadIdx.x;     // 64 threads

    __nv_bfloat16* row = x + ((int64_t)t * n_heads + h) * head_dim;
    float x1 = __bfloat162float(row[i]);
    float x2 = __bfloat162float(row[half + i]);

    if (norm_w) {
        float sq = x1 * x1 + x2 * x2;
        sq = ds_block_sum(sq);
        __shared__ float s_inv;
        if (i == 0) s_inv = rsqrtf(sq / (float)head_dim + eps);
        __syncthreads();
        float inv = s_inv;
        x1 = __bfloat162float(__float2bfloat16(x1 * inv * __bfloat162float(norm_w[i])));
        x2 = __bfloat162float(__float2bfloat16(x2 * inv * __bfloat162float(norm_w[half + i])));
    }
    int sp = d_start_pos ? *d_start_pos : start_pos;
    float pos = (float)(sp + t);
    float inv_freq = powf(rope_theta, -2.0f * (float)i / (float)head_dim);
    float ang = pos * inv_freq;
    float c = cosf(ang), s = sinf(ang);
    row[i]        = __float2bfloat16(x1 * c - x2 * s);
    row[half + i] = __float2bfloat16(x2 * c + x1 * s);
}

// Write K/V [T, kv_heads, head_dim] into draft KV cache at positions [start_pos .. start_pos+T-1]
__global__ void draft_write_kv_kernel(
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    int start_pos, const int32_t* __restrict__ d_start_pos, int T, int cap, bool is_local, int kv_dim)
{
    int t = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T || d >= kv_dim) return;
    int sp = d_start_pos ? *d_start_pos : start_pos;
    int pos = sp + t;
    int slot = is_local ? (pos % cap) : pos;
    k_cache[(int64_t)slot * kv_dim + d] = k_in[(int64_t)t * kv_dim + d];
    v_cache[(int64_t)slot * kv_dim + d] = v_in[(int64_t)t * kv_dim + d];
}

// Multi-warp bidirectional block attention for draft block [anchor_pos .. anchor_pos + B - 1]
// 4 warps (128 threads) per (h, b) for high bandwidth even at 130K+ context!
__global__ void draft_block_attn_kernel(
    const __nv_bfloat16* __restrict__ Q,       // [B, 24, 128]
    const __nv_bfloat16* __restrict__ k_cache, // [cap, 8, 128]
    const __nv_bfloat16* __restrict__ v_cache, // [cap, 8, 128]
    __nv_bfloat16* __restrict__ out,           // [B, 24, 128]
    int anchor_pos, const int32_t* __restrict__ d_anchor_pos, int B, int cap, bool is_local, int window_size,
    int n_heads, int n_kv_heads, int head_dim)
{
    int h = blockIdx.x;
    int b = blockIdx.y;
    if (h >= n_heads || b >= B) return;
    int kv_h = h / (n_heads / n_kv_heads);
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;

    int ap = d_anchor_pos ? *d_anchor_pos : anchor_pos;
    int k_end = ap + B;
    int k_start = is_local ? max(0, ap - window_size) : 0;

    const __nv_bfloat16* q_ptr = Q + ((int64_t)b * n_heads + h) * head_dim;
    float qv[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) qv[i] = __bfloat162float(q_ptr[lane + i * 32]);
    const float inv_sqrt_d = rsqrtf((float)head_dim);

    float m_val = -1e30f;
    float l_val = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int kv_stride = n_kv_heads * head_dim;

    for (int kp = k_start + wid; kp < k_end; kp += nwarps) {
        int slot = is_local ? (kp % cap) : kp;
        const __nv_bfloat16* k_ptr = k_cache + (int64_t)slot * kv_stride + kv_h * head_dim;
        const __nv_bfloat16* v_ptr = v_cache + (int64_t)slot * kv_stride + kv_h * head_dim;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dot = fmaf(qv[i], __bfloat162float(k_ptr[lane + i * 32]), dot);
        dot = __shfl_sync(0xffffffff, ds_warp_sum(dot), 0) * inv_sqrt_d;
        float m_new = fmaxf(m_val, dot);
        float alpha = expf(m_val - m_new);
        float p = expf(dot - m_new);
        l_val = l_val * alpha + p;
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = acc[i] * alpha + p * __bfloat162float(v_ptr[lane + i * 32]);
        m_val = m_new;
    }

    __shared__ float sm_m[8], sm_l[8], sm_acc[8][128];
    if (lane == 0) { sm_m[wid] = m_val; sm_l[wid] = l_val; }
    #pragma unroll
    for (int i = 0; i < 4; ++i) sm_acc[wid][lane + i * 32] = acc[i];
    __syncthreads();

    if (wid == 0) {
        float gm = -1e30f;
        for (int w = 0; w < nwarps; ++w) if (sm_l[w] > 0.0f) gm = fmaxf(gm, sm_m[w]);
        float gl = 0.0f;
        float gacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int w = 0; w < nwarps; ++w) {
            if (sm_l[w] > 0.0f) {
                float sc = expf(sm_m[w] - gm);
                gl += sm_l[w] * sc;
                #pragma unroll
                for (int i = 0; i < 4; ++i) gacc[i] += sm_acc[w][lane + i * 32] * sc;
            }
        }
        float inv_l = 1.0f / gl;
        __nv_bfloat16* o_ptr = out + ((int64_t)b * n_heads + h) * head_dim;
        #pragma unroll
        for (int i = 0; i < 4; ++i) o_ptr[lane + i * 32] = __float2bfloat16(gacc[i] * inv_l);
    }
}

// Fused add + rmsnorm: res += x; out = rmsnorm(res, w)
__global__ void add_rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ res,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ out,
    int T, int D, float eps)
{
    int t = blockIdx.x;
    if (t >= T) return;
    int64_t off = (int64_t)t * D;
    float reg[12];
    float sum_sq = 0.0f;
    int k = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = __bfloat162float(res[off + d]) + __bfloat162float(x[off + d]);
        res[off + d] = __float2bfloat16(v);
        reg[k++] = v;
        sum_sq += v * v;
    }
    sum_sq = ds_block_sum(sum_sq);
    __shared__ float s_rstd;
    if (threadIdx.x == 0) s_rstd = rsqrtf(sum_sq / (float)D + eps);
    __syncthreads();
    float rstd = s_rstd;
    k = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        out[off + d] = __float2bfloat16(reg[k++] * rstd * __bfloat162float(w[d]));
    }
}

__global__ void silu_mul_kernel(
    __nv_bfloat16* __restrict__ g,
    const __nv_bfloat16* __restrict__ u,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float gv = __bfloat162float(g[i]);
        float uv = __bfloat162float(u[i]);
        float s = gv / (1.0f + expf(-gv));
        g[i] = __float2bfloat16(s * uv);
    }
}

__global__ void embed_draft_block_kernel(
    const int32_t* __restrict__ d_anchor_tok,
    const __nv_bfloat16* __restrict__ embed_w,
    const __nv_bfloat16* __restrict__ embed_norm_w,
    const __nv_bfloat16* __restrict__ mask_embed,
    __nv_bfloat16* __restrict__ out,
    int B, int D, float eps)
{
    int b = blockIdx.x;
    if (b >= B) return;
    __nv_bfloat16* dst = out + (int64_t)b * D;
    if (b > 0) {
        for (int d = threadIdx.x; d < D; d += blockDim.x) dst[d] = mask_embed[d];
        return;
    }
    int32_t anchor_tok = *d_anchor_tok;
    const __nv_bfloat16* row = embed_w + (int64_t)anchor_tok * D;
    float sum_sq = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = __bfloat162float(row[d]);
        sum_sq += v * v;
    }
    sum_sq = ds_block_sum(sum_sq);
    __shared__ float s_rstd;
    if (threadIdx.x == 0) s_rstd = rsqrtf(sum_sq / (float)D + eps);
    __syncthreads();
    float rstd = s_rstd;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        dst[d] = __float2bfloat16(__bfloat162float(row[d]) * rstd * __bfloat162float(embed_norm_w[d]));
    }
}

// Assemble Markov RNN input: [state(256), markov_w1[*prev_tok_ptr](256), draft_h(2304), f_rnn(2304)] -> [5120]
__global__ void assemble_markov_in_dev_kernel(
    const __nv_bfloat16* __restrict__ state,
    const __nv_bfloat16* __restrict__ markov_w1,
    const int32_t* __restrict__ prev_tok_ptr,
    const __nv_bfloat16* __restrict__ draft_h,
    const __nv_bfloat16* __restrict__ f_rnn,
    __nv_bfloat16* __restrict__ out,
    int rank, int D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 2 * rank + 2 * D;
    if (i >= total) return;
    int32_t prev_tok = *prev_tok_ptr;
    if (i < rank) out[i] = state[i];
    else if (i < 2 * rank) out[i] = markov_w1[(int64_t)prev_tok * rank + (i - rank)];
    else if (i < 2 * rank + D) out[i] = draft_h[i - 2 * rank];
    else out[i] = f_rnn[i - 2 * rank - D];
}

// Markov gate: raw[3*rank] + bias[3*rank], updates state[rank], writes latent[rank]
__global__ void markov_gate_kernel(
    const float* __restrict__ raw,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ latent,
    int rank)
{
    int c = threadIdx.x;
    if (c >= rank) return;
    float g = __bfloat162float(__float2bfloat16(raw[c] + bias[c]));
    float cand = __bfloat162float(__float2bfloat16(raw[rank + c] + bias[rank + c]));
    float o = __bfloat162float(__float2bfloat16(raw[2 * rank + c] + bias[2 * rank + c]));
    float st = __bfloat162float(state[c]);
    g = 1.0f / (1.0f + expf(-g));
    cand = tanhf(cand);
    state[c] = __float2bfloat16(g * st + (1.0f - g) * cand);
    latent[c] = __float2bfloat16(tanhf(o));
}

// Argmax of base_logits[V] (BF16) + fold_b[V] (FP32) + markov_bias[V] (FP32)
// If draft_to_target != nullptr, maps winning pruned index back to full target token ID.
__global__ void markov_argmax_kernel(
    const __nv_bfloat16* __restrict__ base_logits,
    const float* __restrict__ fold_b,
    const float* __restrict__ markov_bias,
    const int32_t* __restrict__ draft_to_target,
    int32_t* __restrict__ out_tok,
    int V)
{
    float best_v = -1e30f;
    int best_i = 0;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float v = __bfloat162float(base_logits[i]) + fold_b[i] + markov_bias[i];
        if (v > best_v) { best_v = v; best_i = i; }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffff, best_v, off);
        int oi = __shfl_down_sync(0xffffffff, best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    __shared__ float sm_v[32];
    __shared__ int sm_i[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { sm_v[wid] = best_v; sm_i[wid] = best_i; }
    __syncthreads();
    if (wid == 0) {
        int nw = blockDim.x >> 5;
        best_v = (lane < nw) ? sm_v[lane] : -1e30f;
        best_i = (lane < nw) ? sm_i[lane] : 0;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffff, best_v, off);
            int oi = __shfl_down_sync(0xffffffff, best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0) *out_tok = draft_to_target ? draft_to_target[best_i] : best_i;
    }
}

__global__ void dflash2_embed_block_kernel(
    const int32_t* __restrict__ d_anchor_tok,
    int32_t mask_id,
    const __nv_bfloat16* __restrict__ embed_w,
    const __nv_bfloat16* __restrict__ embed_norm_w,
    __nv_bfloat16* __restrict__ out,
    int B, int D, float eps)
{
    int b = blockIdx.x;
    if (b >= B) return;
    int32_t tok = (b == 0) ? *d_anchor_tok : mask_id;
    const __nv_bfloat16* row = embed_w + (int64_t)tok * D;
    __nv_bfloat16* dst = out + (int64_t)b * D;
    float sum_sq = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = __bfloat162float(row[d]);
        sum_sq += v * v;
    }
    sum_sq = ds_block_sum(sum_sq);
    __shared__ float s_rstd;
    if (threadIdx.x == 0) s_rstd = rsqrtf(sum_sq / (float)D + eps);
    __syncthreads();
    float rstd = s_rstd;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        dst[d] = __float2bfloat16(__bfloat162float(row[d]) * rstd * __bfloat162float(embed_norm_w[d]));
    }
}

__global__ void dflash2_concat_taps_kernel(
    const __nv_bfloat16* __restrict__ aux_states,
    int aux_stride,
    __nv_bfloat16* __restrict__ out,
    int T, int D)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (t >= T || col >= 4 * D) return;
    int tap = col / D;
    int d = col - tap * D;
    out[(int64_t)t * 4 * D + col] = aux_states[(int64_t)tap * aux_stride + (int64_t)t * D + d];
}

__global__ void dflash2_grouped_dyn_conv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ base_k,
    const __nv_bfloat16* __restrict__ dyn_k,
    __nv_bfloat16* __restrict__ out,
    int T, int D, int group_size, int half_idx)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (t >= T || d >= D) return;
    int nG = D / group_size;
    int g = d / group_size;
    int64_t d_base = (int64_t)t * 4 * nG + (int64_t)half_idx * 2 * nG + g;
    float w0 = __bfloat162float(base_k[d]) + __bfloat162float(dyn_k[d_base]);
    float w1 = __bfloat162float(base_k[D + d]) + __bfloat162float(dyn_k[d_base + nG]);
    float x0 = __bfloat162float(x[(int64_t)t * D + d]);
    float x1 = (t > 0) ? __bfloat162float(x[(int64_t)(t - 1) * D + d]) : 0.0f;
    out[(int64_t)t * D + d] = __float2bfloat16(fmaf(w0, x0, w1 * x1));
}

__global__ void dflash2_block_attn_kernel(
    const __nv_bfloat16* __restrict__ q_blk,
    const __nv_bfloat16* __restrict__ k_blk,
    const __nv_bfloat16* __restrict__ v_blk,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ out,
    int anchor_pos, const int32_t* __restrict__ d_anchor_pos,
    int B, int cap, int window_size,
    int n_heads, int n_kv_heads, int head_dim)
{
    int h = blockIdx.x;
    int t = blockIdx.y;
    if (h >= n_heads || t >= B) return;
    int kv_h = h / (n_heads / n_kv_heads);
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;

    int ap = d_anchor_pos ? *d_anchor_pos : anchor_pos;
    int q_pos = ap + t;
    int c_start = (q_pos + 1 > window_size) ? (q_pos + 1 - window_size) : 0;
    int c_end = ap;

    const __nv_bfloat16* q_ptr = q_blk + ((int64_t)t * n_heads + h) * head_dim;
    float qv[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) qv[i] = __bfloat162float(q_ptr[lane + i * 32]);
    const float inv_sqrt_d = rsqrtf((float)head_dim);

    float m_val = -1e30f;
    float l_val = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int kv_stride = n_kv_heads * head_dim;

    for (int kp = c_start + wid; kp < c_end; kp += nwarps) {
        int slot = kp % cap;
        const __nv_bfloat16* k_ptr = k_cache + (int64_t)slot * kv_stride + kv_h * head_dim;
        const __nv_bfloat16* v_ptr = v_cache + (int64_t)slot * kv_stride + kv_h * head_dim;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dot = fmaf(qv[i], __bfloat162float(k_ptr[lane + i * 32]), dot);
        dot = __shfl_sync(0xffffffff, ds_warp_sum(dot), 0) * inv_sqrt_d;
        float m_new = fmaxf(m_val, dot);
        float alpha = (l_val > 0.0f) ? expf(m_val - m_new) : 0.0f;
        float p = expf(dot - m_new);
        l_val = l_val * alpha + p;
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = acc[i] * alpha + p * __bfloat162float(v_ptr[lane + i * 32]);
        m_val = m_new;
    }

    for (int j = wid; j < B; j += nwarps) {
        const __nv_bfloat16* k_ptr = k_blk + (int64_t)j * kv_stride + kv_h * head_dim;
        const __nv_bfloat16* v_ptr = v_blk + (int64_t)j * kv_stride + kv_h * head_dim;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dot = fmaf(qv[i], __bfloat162float(k_ptr[lane + i * 32]), dot);
        dot = __shfl_sync(0xffffffff, ds_warp_sum(dot), 0) * inv_sqrt_d;
        float m_new = fmaxf(m_val, dot);
        float alpha = (l_val > 0.0f) ? expf(m_val - m_new) : 0.0f;
        float p = expf(dot - m_new);
        l_val = l_val * alpha + p;
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = acc[i] * alpha + p * __bfloat162float(v_ptr[lane + i * 32]);
        m_val = m_new;
    }

    __shared__ float sm_m[8], sm_l[8], sm_acc[8][128];
    if (lane == 0) { sm_m[wid] = m_val; sm_l[wid] = l_val; }
    #pragma unroll
    for (int i = 0; i < 4; ++i) sm_acc[wid][lane + i * 32] = acc[i];
    __syncthreads();

    if (wid == 0) {
        float gm = -1e30f;
        for (int w = 0; w < nwarps; ++w) if (sm_l[w] > 0.0f) gm = fmaxf(gm, sm_m[w]);
        float gl = 0.0f;
        float gacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int w = 0; w < nwarps; ++w) {
            if (sm_l[w] > 0.0f) {
                float sc = expf(sm_m[w] - gm);
                gl += sm_l[w] * sc;
                #pragma unroll
                for (int i = 0; i < 4; ++i) gacc[i] += sm_acc[w][lane + i * 32] * sc;
            }
        }
        float inv_l = 1.0f / gl;
        __nv_bfloat16* o_ptr = out + ((int64_t)t * n_heads + h) * head_dim;
        #pragma unroll
        for (int i = 0; i < 4; ++i) o_ptr[lane + i * 32] = __float2bfloat16(gacc[i] * inv_l);
    }
}

__global__ void dflash2_topk16_kernel(
    float* __restrict__ logits,
    float* __restrict__ top_vals,
    int32_t* __restrict__ top_ids,
    const int32_t* __restrict__ draft_to_target,
    int M, int V)
{
    int m = blockIdx.x;
    if (m >= M) return;
    float* row = logits + (int64_t)m * V;
    __shared__ float sm_v[32];
    __shared__ int sm_i[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int nw = (blockDim.x + 31) >> 5;

    for (int k = 0; k < 16; ++k) {
        float best_v = -1e30f;
        int best_i = 0;
        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            float v = row[i];
            if (v > best_v) {
                best_v = v;
                best_i = i;
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffff, best_v, off);
            int oi = __shfl_down_sync(0xffffffff, best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (lane == 0) {
            sm_v[wid] = best_v;
            sm_i[wid] = best_i;
        }
        __syncthreads();
        if (wid == 0) {
            best_v = (lane < nw) ? sm_v[lane] : -1e30f;
            best_i = (lane < nw) ? sm_i[lane] : 0;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float ov = __shfl_down_sync(0xffffffff, best_v, off);
                int oi = __shfl_down_sync(0xffffffff, best_i, off);
                if (ov > best_v || (ov == best_v && oi < best_i)) {
                    best_v = ov;
                    best_i = oi;
                }
            }
            if (lane == 0) {
                top_vals[(int64_t)m * 16 + k] = best_v;
                top_ids[(int64_t)m * 16 + k] = draft_to_target ? draft_to_target[best_i] : best_i;
                row[best_i] = -1e30f;
            }
        }
        __threadfence_block();
        __syncthreads();
    }
}

__global__ void dflash2_select_path_kernel(
    const int32_t* __restrict__ d_anchor_tok,
    const float* __restrict__ top_vals,
    const int32_t* __restrict__ top_ids,
    const __nv_bfloat16* __restrict__ h_sel,
    const __nv_bfloat16* __restrict__ pred_cb,
    const __nv_bfloat16* __restrict__ succ_cb,
    int32_t* __restrict__ out_tok,
    int nd, int rank)
{
    int r = threadIdx.x;
    __shared__ int32_t s_pred;
    if (r == 0) s_pred = *d_anchor_tok;
    __syncthreads();

    for (int pos = 0; pos < nd; ++pos) {
        int32_t pred = s_pred;
        float pv = (r < rank)
            ? (__bfloat162float(pred_cb[(int64_t)pred * rank + r]) * __bfloat162float(h_sel[(int64_t)pos * rank + r]))
            : 0.0f;
        float best_s = -1e30f;
        int32_t best_tok = top_ids[pos * 16];
        for (int k = 0; k < 16; ++k) {
            int32_t cid = top_ids[pos * 16 + k];
            float prod = (r < rank) ? (pv * __bfloat162float(succ_cb[(int64_t)cid * rank + r])) : 0.0f;
            float corr = ds_block_sum(prod);
            if (r == 0) {
                float sc = top_vals[pos * 16 + k] + corr;
                if (sc > best_s) {
                    best_s = sc;
                    best_tok = cid;
                }
            }
            __syncthreads();
        }
        if (r == 0) {
            out_tok[pos] = best_tok;
            s_pred = best_tok;
        }
        __syncthreads();
    }
}

void launch_dflash2_concat_taps(
    const __nv_bfloat16* aux_states, int aux_stride,
    __nv_bfloat16* out, int T, int D, cudaStream_t stream)
{
    dflash2_concat_taps_kernel<<<dim3((4 * D + 255) / 256, T), 256, 0, stream>>>(
        aux_states, aux_stride, out, T, D);
}

void launch_dflash2_grouped_dyn_conv(
    const __nv_bfloat16* x, const __nv_bfloat16* base_k, const __nv_bfloat16* dyn_k,
    __nv_bfloat16* out, int T, int D, int group_size, int half_idx, cudaStream_t stream)
{
    dflash2_grouped_dyn_conv_kernel<<<dim3((D + 255) / 256, T), 256, 0, stream>>>(
        x, base_k, dyn_k, out, T, D, group_size, half_idx);
}

void launch_dflash2_block_attn(
    const __nv_bfloat16* q_blk, const __nv_bfloat16* k_blk, const __nv_bfloat16* v_blk,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, __nv_bfloat16* out,
    int anchor_pos, const int32_t* d_anchor_pos, int B, int cap, int window_size,
    int n_heads, int n_kv_heads, int head_dim, cudaStream_t stream)
{
    dflash2_block_attn_kernel<<<dim3(n_heads, B), 128, 0, stream>>>(
        q_blk, k_blk, v_blk, k_cache, v_cache, out,
        anchor_pos, d_anchor_pos, B, cap, window_size, n_heads, n_kv_heads, head_dim);
}

void launch_dflash2_topk16(
    float* logits, float* top_vals, int32_t* top_ids,
    const int32_t* draft_to_target, int M, int V, cudaStream_t stream)
{
    dflash2_topk16_kernel<<<M, 512, 0, stream>>>(logits, top_vals, top_ids, draft_to_target, M, V);
}

void launch_dflash2_select_path(
    const int32_t* d_anchor_tok, const float* top_vals, const int32_t* top_ids,
    const __nv_bfloat16* h_sel, const __nv_bfloat16* pred_cb, const __nv_bfloat16* succ_cb,
    int32_t* out_tok, int nd, int rank, cudaStream_t stream)
{
    dflash2_select_path_kernel<<<1, 256, 0, stream>>>(
        d_anchor_tok, top_vals, top_ids, h_sel, pred_cb, succ_cb, out_tok, nd, rank);
}

// ============================================================================
// DSparkDrafter Implementation
// ============================================================================

DSparkDrafter::~DSparkDrafter() {
    for (int s = 0; s < 8; ++s) {
        for (int k = 0; k < 8; ++k) {
            if (draft_graph_execs[s][k]) cudaGraphExecDestroy(draft_graph_execs[s][k]);
            if (draft_graphs[s][k]) cudaGraphDestroy(draft_graphs[s][k]);
        }
    }
    for (void* p : allocs) cudaFree(p);
}

void* DSparkDrafter::alloc_gpu(size_t bytes) {
    void* p = nullptr;
    CHECK_CUDA(cudaMalloc(&p, bytes));
    allocs.push_back(p);
    return p;
}

void DSparkDrafter::linear_bf16(
    const __nv_bfloat16* X, const __nv_bfloat16* W, __nv_bfloat16* Y, int M, int N, int K)
{
    if ((K & 7) == 0) {
        if (M == 1) {
            launch_bf16_gemv(X, W, Y, nullptr, N, K, stream);
            return;
        }
        if (M >= 2 && M <= 8) {
            launch_bf16_small_gemm(X, W, Y, nullptr, M, N, K, stream);
            return;
        }
    }
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasGemmEx(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
        &alpha, W, CUDA_R_16BF, K, X, CUDA_R_16BF, K,
        &beta, Y, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

void DSparkDrafter::linear_bf16_to_fp32(
    const __nv_bfloat16* X, const __nv_bfloat16* W, float* Y, int M, int N, int K)
{
    if (M == 1) {
        launch_bf16_gemv(X, W, nullptr, Y, N, K, stream);
        return;
    }
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasGemmEx(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
        &alpha, W, CUDA_R_16BF, K, X, CUDA_R_16BF, K,
        &beta, Y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

void DSparkDrafter::load(
    const std::string& model_dir,
    const __nv_bfloat16* embed_w, const __nv_bfloat16* embed_norm_w, const __nv_bfloat16* unembed_w,
    cublasHandle_t shared_cublas, cudaStream_t shared_stream, int max_seq, int max_ctx_chunk,
    const std::string& draft_vocab_file, int n_dspark_slots)
{
    auto t0 = std::chrono::steady_clock::now();
    target_embed_w = embed_w;
    target_embed_norm_w = embed_norm_w;
    target_unembed_w = unembed_w;
    cublas = shared_cublas;
    stream = shared_stream;
    max_seq_len = max_seq;
    max_dspark_slots = n_dspark_slots;

    st.open(model_dir + "/mtp.safetensors");
    std::vector<uint8_t> scratch;

    // fuse_weight [4, 4] BF16 -> softmax per row -> FP32 d_fuse_alpha [4, 4]
    {
        std::vector<uint8_t> raw;
        st.load_to_host("fuse_weight", raw);
        const __nv_bfloat16* fw = reinterpret_cast<const __nv_bfloat16*>(raw.data());
        float fa[16];
        for (int r = 0; r < 4; ++r) {
            float m = -1e30f;
            for (int c = 0; c < 4; ++c) m = fmaxf(m, __bfloat162float(fw[r * 4 + c]));
            float s = 0.0f;
            for (int c = 0; c < 4; ++c) { fa[r * 4 + c] = expf(__bfloat162float(fw[r * 4 + c]) - m); s += fa[r * 4 + c]; }
            for (int c = 0; c < 4; ++c) fa[r * 4 + c] /= s;
        }
        d_fuse_alpha = (float*)alloc_gpu(16 * sizeof(float));
        CHECK_CUDA(cudaMemcpy(d_fuse_alpha, fa, sizeof(fa), cudaMemcpyHostToDevice));
    }

    d_mask_embed = (__nv_bfloat16*)st.load_to_gpu("mask_embed", scratch);
    d_final_norm = (__nv_bfloat16*)st.load_to_gpu("norm.weight", scratch);

    layers.resize(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        auto& lw = layers[i];
        lw.is_local = (i == 0 || i == 2);
        std::string p = "layers." + std::to_string(i) + ".";
        lw.dflare_norm  = (__nv_bfloat16*)st.load_to_gpu(p + "dflare_norm.weight", scratch);
        lw.input_ln     = (__nv_bfloat16*)st.load_to_gpu(p + "input_layernorm.weight", scratch);
        lw.post_attn_ln = (__nv_bfloat16*)st.load_to_gpu(p + "post_attention_layernorm.weight", scratch);
        lw.q_proj       = (__nv_bfloat16*)st.load_to_gpu(p + "self_attn.q_proj.weight", scratch);
        lw.k_proj       = (__nv_bfloat16*)st.load_to_gpu(p + "self_attn.k_proj.weight", scratch);
        lw.v_proj       = (__nv_bfloat16*)st.load_to_gpu(p + "self_attn.v_proj.weight", scratch);
        lw.k_proj_ctx   = (__nv_bfloat16*)st.load_to_gpu(p + "self_attn.k_proj_ctx.weight", scratch);
        lw.v_proj_ctx   = (__nv_bfloat16*)st.load_to_gpu(p + "self_attn.v_proj_ctx.weight", scratch);
        lw.o_proj       = (__nv_bfloat16*)st.load_to_gpu(p + "self_attn.o_proj.weight", scratch);
        lw.q_norm       = (__nv_bfloat16*)st.load_to_gpu(p + "self_attn.q_norm.weight", scratch);
        lw.k_norm       = (__nv_bfloat16*)st.load_to_gpu(p + "self_attn.k_norm.weight", scratch);
        lw.k_norm_ctx   = (__nv_bfloat16*)st.load_to_gpu(p + "self_attn.k_norm_ctx.weight", scratch);
        lw.gate_proj    = (__nv_bfloat16*)st.load_to_gpu(p + "mlp.gate_proj.weight", scratch);
        lw.up_proj      = (__nv_bfloat16*)st.load_to_gpu(p + "mlp.up_proj.weight", scratch);
        lw.down_proj    = (__nv_bfloat16*)st.load_to_gpu(p + "mlp.down_proj.weight", scratch);

        // Quantize draft layer weights to per-row FP8 E4M3 for 2x draft bandwidth!
        auto qfp8 = [&](const __nv_bfloat16* src, uint8_t*& dst_w, float*& dst_s, int N, int K) {
            dst_w = (uint8_t*)alloc_gpu((size_t)N * K);
            dst_s = (float*)alloc_gpu((size_t)N * sizeof(float));
            launch_quantize_bf16_to_fp8_rowwise(src, dst_w, dst_s, N, K, stream);
        };
        int q_dim = num_heads * head_dim;
        int kv_dim = num_kv_heads * head_dim;
        qfp8(lw.q_proj,    lw.q_fp8,    lw.q_sc,    q_dim, hidden_size);
        qfp8(lw.k_proj,    lw.k_fp8,    lw.k_sc,    kv_dim, hidden_size);
        qfp8(lw.v_proj,    lw.v_fp8,    lw.v_sc,    kv_dim, hidden_size);
        qfp8(lw.o_proj,    lw.o_fp8,    lw.o_sc,    hidden_size, q_dim);
        qfp8(lw.gate_proj, lw.gate_fp8, lw.gate_sc, inter_size, hidden_size);
        qfp8(lw.up_proj,   lw.up_fp8,   lw.up_sc,   inter_size, hidden_size);
        qfp8(lw.down_proj, lw.down_fp8, lw.down_sc, hidden_size, inter_size);
    }

    // Markov head weights
    d_markov_dflare_norm = (__nv_bfloat16*)st.load_to_gpu("markov_head.dflare_norm.weight", scratch);
    d_markov_w1 = (__nv_bfloat16*)st.load_to_gpu("markov_head.markov_w1.weight", scratch);
    d_rnn_proj_w = (__nv_bfloat16*)st.load_to_gpu("markov_head.rnn_proj.weight", scratch);
    {
        std::vector<uint8_t> raw;
        st.load_to_host("markov_head.rnn_proj.bias", raw);
        const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(raw.data());
        std::vector<float> fb(3 * markov_rank);
        for (int i = 0; i < 3 * markov_rank; ++i) fb[i] = __bfloat162float(bp[i]);
        d_rnn_proj_b = (float*)alloc_gpu(fb.size() * sizeof(float));
        CHECK_CUDA(cudaMemcpy(d_rnn_proj_b, fb.data(), fb.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Build frequency-ranked pruned draft unembed if draft_vocab_file is provided
    const __nv_bfloat16* draft_unembed_src = target_unembed_w;
    if (!draft_vocab_file.empty()) {
        std::ifstream vf(draft_vocab_file, std::ios::binary | std::ios::ate);
        if (!vf) throw std::runtime_error("Cannot open draft vocab file: " + draft_vocab_file);
        size_t nbytes = (size_t)vf.tellg();
        vf.seekg(0);
        draft_vocab_size = (int)(nbytes / sizeof(int32_t));
        std::vector<int32_t> h_map(draft_vocab_size);
        vf.read(reinterpret_cast<char*>(h_map.data()), (std::streamsize)nbytes);

        d_draft_to_target = (int32_t*)alloc_gpu(nbytes);
        CHECK_CUDA(cudaMemcpy(d_draft_to_target, h_map.data(), nbytes, cudaMemcpyHostToDevice));

        d_pruned_unembed = (__nv_bfloat16*)alloc_gpu((size_t)draft_vocab_size * hidden_size * sizeof(__nv_bfloat16));
        launch_gather_bf16(target_unembed_w, d_draft_to_target, d_pruned_unembed, draft_vocab_size, hidden_size, stream);
        draft_unembed_src = d_pruned_unembed;
    } else {
        draft_vocab_size = vocab_size;
        d_pruned_unembed = const_cast<__nv_bfloat16*>(target_unembed_w);
        d_draft_to_target = nullptr;
    }

    // Fold hidden_out.weight [2304, 256] and hidden_out.bias [2304] into draft_unembed_src [draft_vocab, 2304]
    {
        __nv_bfloat16* d_ho_w = (__nv_bfloat16*)st.load_to_gpu("markov_head.hidden_out.weight", scratch);
        __nv_bfloat16* d_ho_b = (__nv_bfloat16*)st.load_to_gpu("markov_head.hidden_out.bias", scratch);
        d_fold_w = (__nv_bfloat16*)alloc_gpu((size_t)draft_vocab_size * markov_rank * sizeof(__nv_bfloat16));
        d_fold_b = (float*)alloc_gpu((size_t)draft_vocab_size * sizeof(float));

        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUBLAS(cublasGemmEx(
            cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            markov_rank, draft_vocab_size, hidden_size,
            &alpha, d_ho_w, CUDA_R_16BF, markov_rank,
            draft_unembed_src, CUDA_R_16BF, hidden_size,
            &beta, d_fold_w, CUDA_R_16BF, markov_rank,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        launch_bf16_gemv(d_ho_b, draft_unembed_src, nullptr, d_fold_b, draft_vocab_size, hidden_size, stream);

        // Quantize pruned draft unembed & folded Markov head to per-row FP8 E4M3!
        d_pruned_unembed_fp8 = (uint8_t*)alloc_gpu((size_t)draft_vocab_size * hidden_size);
        d_pruned_unembed_sc  = (float*)alloc_gpu((size_t)draft_vocab_size * sizeof(float));
        launch_quantize_bf16_to_fp8_rowwise(draft_unembed_src, d_pruned_unembed_fp8, d_pruned_unembed_sc, draft_vocab_size, hidden_size, stream);

        d_fold_w_fp8 = (uint8_t*)alloc_gpu((size_t)draft_vocab_size * markov_rank);
        d_fold_w_sc  = (float*)alloc_gpu((size_t)draft_vocab_size * sizeof(float));
        launch_quantize_bf16_to_fp8_rowwise(d_fold_w, d_fold_w_fp8, d_fold_w_sc, draft_vocab_size, markov_rank, stream);
    }

    // Allocate draft KV caches for max_dspark_slots
    for (int i = 0; i < num_layers; ++i) {
        kv_cap[i] = layers[i].is_local ? (sliding_window + block_size + 16) : max_seq_len;
        size_t sz = (size_t)max_dspark_slots * kv_cap[i] * num_kv_heads * head_dim * sizeof(__nv_bfloat16);
        d_k_cache[i] = (__nv_bfloat16*)alloc_gpu(sz);
        d_v_cache[i] = (__nv_bfloat16*)alloc_gpu(sz);
    }

    // Allocate workspace
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    d_ctx_mix = (__nv_bfloat16*)alloc_gpu((size_t)max_ctx_chunk * hidden_size * sizeof(__nv_bfloat16));
    d_ctx_k   = (__nv_bfloat16*)alloc_gpu((size_t)max_ctx_chunk * kv_dim * sizeof(__nv_bfloat16));
    d_ctx_v   = (__nv_bfloat16*)alloc_gpu((size_t)max_ctx_chunk * kv_dim * sizeof(__nv_bfloat16));

    d_x        = (__nv_bfloat16*)alloc_gpu((size_t)block_size * hidden_size * sizeof(__nv_bfloat16));
    d_res      = (__nv_bfloat16*)alloc_gpu((size_t)block_size * hidden_size * sizeof(__nv_bfloat16));
    d_q        = (__nv_bfloat16*)alloc_gpu((size_t)block_size * q_dim * sizeof(__nv_bfloat16));
    d_k        = (__nv_bfloat16*)alloc_gpu((size_t)block_size * kv_dim * sizeof(__nv_bfloat16));
    d_v        = (__nv_bfloat16*)alloc_gpu((size_t)block_size * kv_dim * sizeof(__nv_bfloat16));
    d_attn_out = (__nv_bfloat16*)alloc_gpu((size_t)block_size * q_dim * sizeof(__nv_bfloat16));
    d_proj_out = (__nv_bfloat16*)alloc_gpu((size_t)block_size * hidden_size * sizeof(__nv_bfloat16));
    d_mlp_g    = (__nv_bfloat16*)alloc_gpu((size_t)block_size * inter_size * sizeof(__nv_bfloat16));
    d_mlp_u    = (__nv_bfloat16*)alloc_gpu((size_t)block_size * inter_size * sizeof(__nv_bfloat16));
    d_f_rnn    = (__nv_bfloat16*)alloc_gpu((size_t)hidden_size * sizeof(__nv_bfloat16));
    d_base_logits = (__nv_bfloat16*)alloc_gpu((size_t)(block_size - 1) * draft_vocab_size * sizeof(__nv_bfloat16));
    d_rnn_in   = (__nv_bfloat16*)alloc_gpu((size_t)(2 * markov_rank + 2 * hidden_size) * sizeof(__nv_bfloat16));
    d_rnn_raw  = (float*)alloc_gpu((size_t)3 * markov_rank * sizeof(float));
    d_rnn_state  = (__nv_bfloat16*)alloc_gpu((size_t)markov_rank * sizeof(__nv_bfloat16));
    d_rnn_latent = (__nv_bfloat16*)alloc_gpu((size_t)markov_rank * sizeof(__nv_bfloat16));
    d_markov_bias = (float*)alloc_gpu((size_t)draft_vocab_size * sizeof(float));
    d_draft_tok = (int32_t*)alloc_gpu(8 * sizeof(int32_t));
    d_anchor_tok = (int32_t*)alloc_gpu(sizeof(int32_t));
    d_anchor_pos = (int32_t*)alloc_gpu(sizeof(int32_t));

    CHECK_CUDA(cudaStreamSynchronize(stream));
    loaded = true;
    auto t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "[warpgrid] Loaded DSpark drafter (FP8 W8A16, draft_vocab=%d, slots=%d) in %.2f s\n",
            draft_vocab_size, max_dspark_slots, std::chrono::duration<double>(t1 - t0).count());
}

void DSparkDrafter::load_dflash2(
    const std::string& df2_dir,
    const __nv_bfloat16* embed_w, const __nv_bfloat16* embed_norm_w, const __nv_bfloat16* unembed_w,
    cublasHandle_t shared_cublas, cudaStream_t shared_stream,
    int max_ctx_chunk, const std::string& draft_vocab_file, int n_dspark_slots)
{
    auto t0 = std::chrono::steady_clock::now();
    target_embed_w = embed_w;
    target_embed_norm_w = embed_norm_w;
    target_unembed_w = unembed_w;
    cublas = shared_cublas;
    stream = shared_stream;
    max_dspark_slots = n_dspark_slots;

    st.open(df2_dir + "/model.safetensors");
    std::vector<uint8_t> scratch;

    auto qfp8 = [&](const __nv_bfloat16* src, uint8_t*& dst_w, float*& dst_s, int N, int K) {
        dst_w = (uint8_t*)alloc_gpu((size_t)N * K);
        dst_s = (float*)alloc_gpu((size_t)N * sizeof(float));
        launch_quantize_bf16_to_fp8_rowwise(src, dst_w, dst_s, N, K, stream);
    };

    df2_fc = (__nv_bfloat16*)st.load_to_gpu("fc.weight", scratch);
    qfp8(df2_fc, df2_fc_fp8, df2_fc_sc, hidden_size, 4 * hidden_size);
    df2_hidden_norm = (__nv_bfloat16*)st.load_to_gpu("hidden_norm.weight", scratch);
    df2_final_norm = (__nv_bfloat16*)st.load_to_gpu("norm.weight", scratch);
    df2_sel_hproj = (__nv_bfloat16*)st.load_to_gpu("candidate_selector.hidden_projection.weight", scratch);
    df2_sel_pred_cb = (__nv_bfloat16*)st.load_to_gpu("candidate_selector.predecessor_codebook", scratch);
    df2_sel_succ_cb = (__nv_bfloat16*)st.load_to_gpu("candidate_selector.successor_codebook", scratch);

    int q_dim = df2_heads * df2_head_dim;
    int kv_dim = df2_kv_heads * df2_head_dim;
    df2_layers.resize(df2_num_layers);
    for (int i = 0; i < df2_num_layers; ++i) {
        auto& lw = df2_layers[i];
        std::string p = "layers." + std::to_string(i) + ".";
        lw.input_ln = (__nv_bfloat16*)st.load_to_gpu(p + "input_layernorm.weight", scratch);
        lw.post_ln = (__nv_bfloat16*)st.load_to_gpu(p + "post_attention_layernorm.weight", scratch);
        lw.q_norm = (__nv_bfloat16*)st.load_to_gpu(p + "self_attn.q_norm.weight", scratch);
        lw.k_norm = (__nv_bfloat16*)st.load_to_gpu(p + "self_attn.k_norm.weight", scratch);
        lw.attn_conv_base = (__nv_bfloat16*)st.load_to_gpu(p + "attention_conv.base_kernel", scratch);
        lw.attn_conv_proj = (__nv_bfloat16*)st.load_to_gpu(p + "attention_conv.kernel_projection.weight", scratch);
        lw.mlp_conv_base = (__nv_bfloat16*)st.load_to_gpu(p + "mlp_conv.base_kernel", scratch);
        lw.mlp_conv_proj = (__nv_bfloat16*)st.load_to_gpu(p + "mlp_conv.kernel_projection.weight", scratch);
        lw.k_proj = (__nv_bfloat16*)st.load_to_gpu(p + "self_attn.k_proj.weight", scratch);
        lw.v_proj = (__nv_bfloat16*)st.load_to_gpu(p + "self_attn.v_proj.weight", scratch);

        __nv_bfloat16* q_bf16 = (__nv_bfloat16*)st.load_to_gpu(p + "self_attn.q_proj.weight", scratch);
        __nv_bfloat16* o_bf16 = (__nv_bfloat16*)st.load_to_gpu(p + "self_attn.o_proj.weight", scratch);
        __nv_bfloat16* g_bf16 = (__nv_bfloat16*)st.load_to_gpu(p + "mlp.gate_proj.weight", scratch);
        __nv_bfloat16* u_bf16 = (__nv_bfloat16*)st.load_to_gpu(p + "mlp.up_proj.weight", scratch);
        __nv_bfloat16* d_bf16 = (__nv_bfloat16*)st.load_to_gpu(p + "mlp.down_proj.weight", scratch);

        qfp8(q_bf16, lw.q_fp8, lw.q_sc, q_dim, hidden_size);
        qfp8(lw.k_proj, lw.k_fp8, lw.k_sc, kv_dim, hidden_size);
        qfp8(lw.v_proj, lw.v_fp8, lw.v_sc, kv_dim, hidden_size);
        qfp8(o_bf16, lw.o_fp8, lw.o_sc, hidden_size, q_dim);
        qfp8(g_bf16, lw.gate_fp8, lw.gate_sc, df2_inter, hidden_size);
        qfp8(u_bf16, lw.up_fp8, lw.up_sc, df2_inter, hidden_size);
        qfp8(d_bf16, lw.down_fp8, lw.down_sc, hidden_size, df2_inter);
    }

    const __nv_bfloat16* draft_unembed_src = target_unembed_w;
    if (!draft_vocab_file.empty()) {
        std::ifstream vf(draft_vocab_file, std::ios::binary | std::ios::ate);
        if (!vf) throw std::runtime_error("Cannot open draft vocab file: " + draft_vocab_file);
        size_t nbytes = (size_t)vf.tellg();
        vf.seekg(0);
        draft_vocab_size = (int)(nbytes / sizeof(int32_t));
        std::vector<int32_t> h_map(draft_vocab_size);
        vf.read(reinterpret_cast<char*>(h_map.data()), (std::streamsize)nbytes);
        d_draft_to_target = (int32_t*)alloc_gpu(nbytes);
        CHECK_CUDA(cudaMemcpy(d_draft_to_target, h_map.data(), nbytes, cudaMemcpyHostToDevice));
        d_pruned_unembed = (__nv_bfloat16*)alloc_gpu((size_t)draft_vocab_size * hidden_size * sizeof(__nv_bfloat16));
        launch_gather_bf16(target_unembed_w, d_draft_to_target, d_pruned_unembed, draft_vocab_size, hidden_size, stream);
        draft_unembed_src = d_pruned_unembed;
    } else {
        draft_vocab_size = vocab_size;
        d_pruned_unembed = const_cast<__nv_bfloat16*>(target_unembed_w);
        d_draft_to_target = nullptr;
    }
    qfp8(draft_unembed_src, d_pruned_unembed_fp8, d_pruned_unembed_sc, draft_vocab_size, hidden_size);

    df2_kv_cap = df2_window + df2_block + 16;
    for (int i = 0; i < df2_num_layers; ++i) {
        size_t sz = (size_t)max_dspark_slots * df2_kv_cap * kv_dim * sizeof(__nv_bfloat16);
        df2_k_cache[i] = (__nv_bfloat16*)alloc_gpu(sz);
        df2_v_cache[i] = (__nv_bfloat16*)alloc_gpu(sz);
    }

    int ctx_cap = std::max(max_ctx_chunk, df2_window);
    df2_ctx_cat = (__nv_bfloat16*)alloc_gpu((size_t)ctx_cap * 4 * hidden_size * sizeof(__nv_bfloat16));
    df2_ctx_fc = (__nv_bfloat16*)alloc_gpu((size_t)ctx_cap * hidden_size * sizeof(__nv_bfloat16));
    df2_ctx_norm = (__nv_bfloat16*)alloc_gpu((size_t)ctx_cap * hidden_size * sizeof(__nv_bfloat16));
    df2_ctx_k = (__nv_bfloat16*)alloc_gpu((size_t)ctx_cap * kv_dim * sizeof(__nv_bfloat16));
    df2_ctx_v = (__nv_bfloat16*)alloc_gpu((size_t)ctx_cap * kv_dim * sizeof(__nv_bfloat16));

    int dyn_dim = 4 * (hidden_size / df2_group_size);
    df2_res = (__nv_bfloat16*)alloc_gpu((size_t)df2_block * hidden_size * sizeof(__nv_bfloat16));
    df2_x = (__nv_bfloat16*)alloc_gpu((size_t)df2_block * hidden_size * sizeof(__nv_bfloat16));
    df2_xc = (__nv_bfloat16*)alloc_gpu((size_t)df2_block * hidden_size * sizeof(__nv_bfloat16));
    df2_dyn = (__nv_bfloat16*)alloc_gpu((size_t)df2_block * dyn_dim * sizeof(__nv_bfloat16));
    df2_q = (__nv_bfloat16*)alloc_gpu((size_t)df2_block * q_dim * sizeof(__nv_bfloat16));
    df2_k = (__nv_bfloat16*)alloc_gpu((size_t)df2_block * kv_dim * sizeof(__nv_bfloat16));
    df2_v = (__nv_bfloat16*)alloc_gpu((size_t)df2_block * kv_dim * sizeof(__nv_bfloat16));
    df2_attn_out = (__nv_bfloat16*)alloc_gpu((size_t)df2_block * q_dim * sizeof(__nv_bfloat16));
    df2_proj_out = (__nv_bfloat16*)alloc_gpu((size_t)df2_block * hidden_size * sizeof(__nv_bfloat16));
    df2_mlp_g = (__nv_bfloat16*)alloc_gpu((size_t)df2_block * df2_inter * sizeof(__nv_bfloat16));
    df2_mlp_u = (__nv_bfloat16*)alloc_gpu((size_t)df2_block * df2_inter * sizeof(__nv_bfloat16));
    df2_h_sel = (__nv_bfloat16*)alloc_gpu((size_t)df2_block * df2_sel_rank * sizeof(__nv_bfloat16));
    df2_logits = (float*)alloc_gpu((size_t)(df2_block - 1) * draft_vocab_size * sizeof(float));
    df2_top_v = (float*)alloc_gpu((size_t)df2_block * 16 * sizeof(float));
    df2_top_i = (int32_t*)alloc_gpu((size_t)df2_block * 16 * sizeof(int32_t));
    d_draft_tok = (int32_t*)alloc_gpu(8 * sizeof(int32_t));
    d_anchor_tok = (int32_t*)alloc_gpu(sizeof(int32_t));
    d_anchor_pos = (int32_t*)alloc_gpu(sizeof(int32_t));

    CHECK_CUDA(cudaStreamSynchronize(stream));
    df2_mode = true;
    loaded = true;
    auto t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "[warpgrid] Loaded DFlash2 drafter (%d layers, FP8 W8A16, draft_vocab=%d, SWA=%d, slots=%d) in %.2f s\n",
            df2_num_layers, draft_vocab_size, df2_window, max_dspark_slots,
            std::chrono::duration<double>(t1 - t0).count());
}

void DSparkDrafter::store_dflash2_context_kv(
    const __nv_bfloat16* aux_states, int aux_stride, int T, int start_pos, int slot_id)
{
    int D = hidden_size;
    if (T > df2_window) {
        int skip = T - df2_window;
        aux_states += (int64_t)skip * D;
        start_pos += skip;
        T = df2_window;
    }
    int kv_dim = df2_kv_heads * df2_head_dim;
    launch_dflash2_concat_taps(aux_states, aux_stride, df2_ctx_cat, T, D, stream);
    if (T <= 8) {
        launch_fp8_small_gemm(df2_ctx_cat, df2_fc_fp8, df2_fc_sc, df2_ctx_fc, nullptr, T, D, 4 * D, stream);
    } else {
        linear_bf16(df2_ctx_cat, df2_fc, df2_ctx_fc, T, D, 4 * D);
    }
    launch_rmsnorm(df2_ctx_fc, df2_hidden_norm, df2_ctx_norm, T, D, df2_eps, stream);

    for (int i = 0; i < df2_num_layers; ++i) {
        const auto& lw = df2_layers[i];
        int64_t slot_off = (int64_t)slot_id * df2_kv_cap * kv_dim;
        if (T <= 8) {
            launch_fp8_small_gemm(df2_ctx_norm, lw.k_fp8, lw.k_sc, df2_ctx_k, nullptr, T, kv_dim, D, stream);
            launch_fp8_small_gemm(df2_ctx_norm, lw.v_fp8, lw.v_sc, df2_ctx_v, nullptr, T, kv_dim, D, stream);
        } else {
            linear_bf16(df2_ctx_norm, lw.k_proj, df2_ctx_k, T, kv_dim, D);
            linear_bf16(df2_ctx_norm, lw.v_proj, df2_ctx_v, T, kv_dim, D);
        }
        qk_norm_rope_kernel<<<dim3(df2_kv_heads, T), df2_head_dim / 2, 0, stream>>>(
            df2_ctx_k, lw.k_norm, start_pos, nullptr, T, df2_kv_heads, df2_head_dim, df2_rope_theta, df2_eps);
        draft_write_kv_kernel<<<dim3((kv_dim + 255) / 256, T), 256, 0, stream>>>(
            df2_ctx_k, df2_ctx_v, df2_k_cache[i] + slot_off, df2_v_cache[i] + slot_off,
            start_pos, nullptr, T, df2_kv_cap, true, kv_dim);
    }
}

void DSparkDrafter::draft_dflash2_tokens(
    int32_t anchor_tok, int anchor_pos, int num_draft, int32_t* h_out_drafts, int slot_id)
{
    int nd = std::max(1, std::min(df2_block - 1, num_draft));
    CHECK_CUDA(cudaMemcpyAsync(d_anchor_tok, &anchor_tok, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_anchor_pos, &anchor_pos, sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    auto run_body = [&]() {
        int B = nd + 1;
        int D = hidden_size;
        int q_dim = df2_heads * df2_head_dim;
        int kv_dim = df2_kv_heads * df2_head_dim;
        int dyn_dim = 4 * (D / df2_group_size);

        dflash2_embed_block_kernel<<<B, 256, 0, stream>>>(
            d_anchor_tok, df2_mask_id, target_embed_w, target_embed_norm_w, df2_res, B, D, df2_eps);

        for (int i = 0; i < df2_num_layers; ++i) {
            const auto& lw = df2_layers[i];
            int64_t slot_off = (int64_t)slot_id * df2_kv_cap * kv_dim;

            launch_rmsnorm(df2_res, lw.input_ln, df2_x, B, D, df2_eps, stream);
            linear_bf16(df2_x, lw.attn_conv_proj, df2_dyn, B, dyn_dim, D);
            launch_dflash2_grouped_dyn_conv(df2_x, lw.attn_conv_base, df2_dyn, df2_xc, B, D, df2_group_size, 0, stream);

            launch_fp8_small_gemm(df2_xc, lw.q_fp8, lw.q_sc, df2_q, nullptr, B, q_dim, D, stream);
            launch_fp8_small_gemm(df2_xc, lw.k_fp8, lw.k_sc, df2_k, nullptr, B, kv_dim, D, stream);
            launch_fp8_small_gemm(df2_xc, lw.v_fp8, lw.v_sc, df2_v, nullptr, B, kv_dim, D, stream);

            qk_norm_rope_kernel<<<dim3(df2_heads, B), df2_head_dim / 2, 0, stream>>>(
                df2_q, lw.q_norm, 0, d_anchor_pos, B, df2_heads, df2_head_dim, df2_rope_theta, df2_eps);
            qk_norm_rope_kernel<<<dim3(df2_kv_heads, B), df2_head_dim / 2, 0, stream>>>(
                df2_k, lw.k_norm, 0, d_anchor_pos, B, df2_kv_heads, df2_head_dim, df2_rope_theta, df2_eps);

            launch_dflash2_block_attn(
                df2_q, df2_k, df2_v, df2_k_cache[i] + slot_off, df2_v_cache[i] + slot_off, df2_attn_out,
                0, d_anchor_pos, B, df2_kv_cap, df2_window, df2_heads, df2_kv_heads, df2_head_dim, stream);

            launch_fp8_small_gemm(df2_attn_out, lw.o_fp8, lw.o_sc, df2_proj_out, nullptr, B, D, q_dim, stream);
            launch_dflash2_grouped_dyn_conv(df2_proj_out, lw.attn_conv_base + 2 * D, df2_dyn, df2_xc, B, D, df2_group_size, 1, stream);

            add_rmsnorm_kernel<<<B, 256, 0, stream>>>(df2_xc, df2_res, lw.post_ln, df2_x, B, D, df2_eps);
            linear_bf16(df2_x, lw.mlp_conv_proj, df2_dyn, B, dyn_dim, D);
            launch_dflash2_grouped_dyn_conv(df2_x, lw.mlp_conv_base, df2_dyn, df2_xc, B, D, df2_group_size, 0, stream);

            launch_fp8_small_gemm(df2_xc, lw.gate_fp8, lw.gate_sc, df2_mlp_g, nullptr, B, df2_inter, D, stream);
            launch_fp8_small_gemm(df2_xc, lw.up_fp8, lw.up_sc, df2_mlp_u, nullptr, B, df2_inter, D, stream);
            silu_mul_kernel<<<(B * df2_inter + 255) / 256, 256, 0, stream>>>(df2_mlp_g, df2_mlp_u, B * df2_inter);
            launch_fp8_small_gemm(df2_mlp_g, lw.down_fp8, lw.down_sc, df2_proj_out, nullptr, B, D, df2_inter, stream);

            launch_dflash2_grouped_dyn_conv(df2_proj_out, lw.mlp_conv_base + 2 * D, df2_dyn, df2_xc, B, D, df2_group_size, 1, stream);
            launch_add_bf16(df2_res, df2_xc, df2_res, B * D, stream);
        }

        launch_rmsnorm(df2_res + D, df2_final_norm, df2_x, nd, D, df2_eps, stream);
        launch_fp8_small_gemm(df2_x, d_pruned_unembed_fp8, d_pruned_unembed_sc, nullptr, df2_logits, nd, draft_vocab_size, D, stream);
        linear_bf16(df2_x, df2_sel_hproj, df2_h_sel, nd, df2_sel_rank, D);
        launch_dflash2_topk16(df2_logits, df2_top_v, df2_top_i, d_draft_to_target, nd, draft_vocab_size, stream);
        launch_dflash2_select_path(d_anchor_tok, df2_top_v, df2_top_i, df2_h_sel, df2_sel_pred_cb, df2_sel_succ_cb, d_draft_tok, nd, df2_sel_rank, stream);
    };

    if (use_cuda_graph && slot_id >= 0 && slot_id < 8 && nd >= 1 && nd <= 7) {
        if (!draft_graph_execs[slot_id][nd]) {
            CHECK_CUDA(cudaStreamSynchronize(stream));
            CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            run_body();
            CHECK_CUDA(cudaStreamEndCapture(stream, &draft_graphs[slot_id][nd]));
            CHECK_CUDA(cudaGraphInstantiate(&draft_graph_execs[slot_id][nd], draft_graphs[slot_id][nd], nullptr, nullptr, 0));
        }
        CHECK_CUDA(cudaGraphLaunch(draft_graph_execs[slot_id][nd], stream));
    } else {
        run_body();
    }
    CHECK_CUDA(cudaMemcpyAsync(h_out_drafts, d_draft_tok, nd * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

void DSparkDrafter::store_context_kv(
    const __nv_bfloat16* aux_states, int aux_stride, int T, int start_pos, int slot_id)
{
    if (df2_mode) {
        store_dflash2_context_kv(aux_states, aux_stride, T, start_pos, slot_id);
        return;
    }
    int kv_dim = num_kv_heads * head_dim;
    for (int i = 0; i < num_layers; ++i) {
        const auto& lw = layers[i];
        int64_t slot_off = (int64_t)slot_id * kv_cap[i] * kv_dim;
        dflare_mix_rmsnorm_kernel<<<T, 256, 0, stream>>>(
            aux_states, aux_stride, d_fuse_alpha + i * 4, lw.dflare_norm,
            d_ctx_mix, T, hidden_size, rms_norm_eps);
        linear_bf16(d_ctx_mix, lw.k_proj_ctx, d_ctx_k, T, kv_dim, hidden_size);
        linear_bf16(d_ctx_mix, lw.v_proj_ctx, d_ctx_v, T, kv_dim, hidden_size);
        qk_norm_rope_kernel<<<dim3(num_kv_heads, T), head_dim / 2, 0, stream>>>(
            d_ctx_k, lw.k_norm_ctx, start_pos, nullptr, T, num_kv_heads, head_dim, rope_theta, rms_norm_eps);
        draft_write_kv_kernel<<<dim3((kv_dim + 255) / 256, T), 256, 0, stream>>>(
            d_ctx_k, d_ctx_v, d_k_cache[i] + slot_off, d_v_cache[i] + slot_off,
            start_pos, nullptr, T, kv_cap[i], lw.is_local, kv_dim);
    }
}

void DSparkDrafter::draft_tokens(
    int32_t anchor_tok, int anchor_pos,
    const __nv_bfloat16* last_aux, int aux_stride,
    int num_draft, int32_t* h_out_drafts, int slot_id)
{
    if (df2_mode) {
        draft_dflash2_tokens(anchor_tok, anchor_pos, num_draft, h_out_drafts, slot_id);
        return;
    }
    CHECK_CUDA(cudaMemcpyAsync(d_anchor_tok, &anchor_tok, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_anchor_pos, &anchor_pos, sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    auto run_body = [&]() {
        int B = block_size; // 8
        int q_dim = num_heads * head_dim;
        int kv_dim = num_kv_heads * head_dim;

        // 1. Embed [anchor_tok, mask, mask, ..., mask] -> d_res[8, 2304]
        embed_draft_block_kernel<<<B, 256, 0, stream>>>(
            d_anchor_tok, target_embed_w, target_embed_norm_w, d_mask_embed,
            d_res, B, hidden_size, rms_norm_eps);

        // 2. Run 3 draft transformer layers
        for (int i = 0; i < num_layers; ++i) {
            const auto& lw = layers[i];
            int64_t slot_off = (int64_t)slot_id * kv_cap[i] * kv_dim;
            if (i == 0) {
                launch_rmsnorm(d_res, lw.input_ln, d_x, B, hidden_size, rms_norm_eps, stream);
            } else {
                add_rmsnorm_kernel<<<B, 256, 0, stream>>>(d_proj_out, d_res, lw.input_ln, d_x, B, hidden_size, rms_norm_eps);
            }
            launch_fp8_small_gemm(d_x, lw.q_fp8, lw.q_sc, d_q, nullptr, B, q_dim, hidden_size, stream);
            launch_fp8_small_gemm(d_x, lw.k_fp8, lw.k_sc, d_k, nullptr, B, kv_dim, hidden_size, stream);
            launch_fp8_small_gemm(d_x, lw.v_fp8, lw.v_sc, d_v, nullptr, B, kv_dim, hidden_size, stream);
            qk_norm_rope_kernel<<<dim3(num_heads, B), head_dim / 2, 0, stream>>>(
                d_q, lw.q_norm, 0, d_anchor_pos, B, num_heads, head_dim, rope_theta, rms_norm_eps);
            qk_norm_rope_kernel<<<dim3(num_kv_heads, B), head_dim / 2, 0, stream>>>(
                d_k, lw.k_norm, 0, d_anchor_pos, B, num_kv_heads, head_dim, rope_theta, rms_norm_eps);
            draft_write_kv_kernel<<<dim3((kv_dim + 255) / 256, B), 256, 0, stream>>>(
                d_k, d_v, d_k_cache[i] + slot_off, d_v_cache[i] + slot_off,
                0, d_anchor_pos, B, kv_cap[i], lw.is_local, kv_dim);

            draft_block_attn_kernel<<<dim3(num_heads, B), 128, 0, stream>>>(
                d_q, d_k_cache[i] + slot_off, d_v_cache[i] + slot_off, d_attn_out,
                0, d_anchor_pos, B, kv_cap[i], lw.is_local, sliding_window,
                num_heads, num_kv_heads, head_dim);
            launch_fp8_small_gemm(d_attn_out, lw.o_fp8, lw.o_sc, d_proj_out, nullptr, B, hidden_size, q_dim, stream);

            add_rmsnorm_kernel<<<B, 256, 0, stream>>>(d_proj_out, d_res, lw.post_attn_ln, d_x, B, hidden_size, rms_norm_eps);
            launch_fp8_small_gemm(d_x, lw.gate_fp8, lw.gate_sc, d_mlp_g, nullptr, B, inter_size, hidden_size, stream);
            launch_fp8_small_gemm(d_x, lw.up_fp8, lw.up_sc, d_mlp_u, nullptr, B, inter_size, hidden_size, stream);
            silu_mul_kernel<<<(B * inter_size + 255) / 256, 256, 0, stream>>>(d_mlp_g, d_mlp_u, B * inter_size);
            launch_fp8_small_gemm(d_mlp_g, lw.down_fp8, lw.down_sc, d_proj_out, nullptr, B, hidden_size, inter_size, stream);
        }
        // Final norm -> d_x[8, 2304]
        add_rmsnorm_kernel<<<B, 256, 0, stream>>>(d_proj_out, d_res, d_final_norm, d_x, B, hidden_size, rms_norm_eps);

        // 3. Base logits for draft steps 1..num_draft in ONE FP8 W8A16 GEMM (pruned draft vocab!)
        launch_fp8_small_gemm(d_x + hidden_size, d_pruned_unembed_fp8, d_pruned_unembed_sc, d_base_logits, nullptr, num_draft, draft_vocab_size, hidden_size, stream);

        // 4. Compute f_rnn from last_aux (row 3 of fuse_alpha)
        dflare_mix_rmsnorm_kernel<<<1, 256, 0, stream>>>(
            last_aux, aux_stride, d_fuse_alpha + 3 * 4, d_markov_dflare_norm,
            d_f_rnn, 1, hidden_size, rms_norm_eps);

        // 5. Recurrent Markov head for s = 0..num_draft-1 (zero host sync inside loop!)
        CHECK_CUDA(cudaMemsetAsync(d_rnn_state, 0, markov_rank * sizeof(__nv_bfloat16), stream));
        int rnn_in_dim = 2 * markov_rank + 2 * hidden_size;

        for (int s = 0; s < num_draft; ++s) {
            const int32_t* prev_ptr = (s == 0) ? d_anchor_tok : (d_draft_tok + s - 1);
            assemble_markov_in_dev_kernel<<<(rnn_in_dim + 255) / 256, 256, 0, stream>>>(
                d_rnn_state, d_markov_w1, prev_ptr,
                d_x + (int64_t)(s + 1) * hidden_size, d_f_rnn,
                d_rnn_in, markov_rank, hidden_size);
            linear_bf16_to_fp32(d_rnn_in, d_rnn_proj_w, d_rnn_raw, 1, 3 * markov_rank, rnn_in_dim);
            markov_gate_kernel<<<1, markov_rank, 0, stream>>>(
                d_rnn_raw, d_rnn_proj_b, d_rnn_state, d_rnn_latent, markov_rank);
            launch_fp8_small_gemm(d_rnn_latent, d_fold_w_fp8, d_fold_w_sc, nullptr, d_markov_bias, 1, draft_vocab_size, markov_rank, stream);
            markov_argmax_kernel<<<1, 1024, 0, stream>>>(
                d_base_logits + (int64_t)s * draft_vocab_size, d_fold_b, d_markov_bias,
                d_draft_to_target, d_draft_tok + s, draft_vocab_size);
        }
    };

    if (use_cuda_graph && slot_id >= 0 && slot_id < 8 && num_draft >= 1 && num_draft <= 7) {
        if (!draft_graph_execs[slot_id][num_draft]) {
            CHECK_CUDA(cudaStreamSynchronize(stream));
            CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            run_body();
            CHECK_CUDA(cudaStreamEndCapture(stream, &draft_graphs[slot_id][num_draft]));
            CHECK_CUDA(cudaGraphInstantiate(&draft_graph_execs[slot_id][num_draft], draft_graphs[slot_id][num_draft], nullptr, nullptr, 0));
        }
        CHECK_CUDA(cudaGraphLaunch(draft_graph_execs[slot_id][num_draft], stream));
    } else {
        run_body();
    }
    CHECK_CUDA(cudaMemcpyAsync(h_out_drafts, d_draft_tok, num_draft * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

} // namespace ink
