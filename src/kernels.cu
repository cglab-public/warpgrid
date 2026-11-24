#include "kernels.cuh"
#include <cmath>
#include <cstdio>
#include <cuda_fp4.h>
#include <mma.h>

namespace ink {
using namespace nvcuda;

// ============================================================================
// Helpers: warp / block reductions & FP4 / FP8 conversions
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    int nwarps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nwarps) ? smem[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
    return v;
}

// Hardware Blackwell PTX: unpack 1 byte (2 x E2M1) -> float2
__device__ __forceinline__ float2 e2m1x2_to_f32x2(uint32_t byte_val) {
    uint32_t h2;
    asm("{\n\t"
        ".reg .b8 b;\n\t"
        "cvt.u8.u32 b, %1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, b;\n\t"
        "}"
        : "=r"(h2) : "r"(byte_val));
    return __half22float2(*reinterpret_cast<const __half2*>(&h2));
}

// Fast E4M3FN byte -> float via hardware __nv_cvt_fp8_to_halfraw
__device__ __forceinline__ float e4m3_to_f32(uint8_t b) {
    __half_raw h = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
    return __half2float(__half(h));
}

// Fast UE8M0 byte -> float
__device__ __forceinline__ float ue8m0_to_f32(uint8_t e) {
    if (e == 0) return 5.877471754111438e-39f; // 2^-127
    if (e == 255) return NAN;
    uint32_t bits = uint32_t(e) << 23;
    return __uint_as_float(bits);
}

// ============================================================================
// 1. Embed + RMSNorm & standalone RMSNorm & Add
// ============================================================================

__global__ void embed_norm_kernel(
    const int32_t* __restrict__ token_ids,
    const __nv_bfloat16* __restrict__ embed_w,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ out,
    int T, int D, float eps)
{
    int t = blockIdx.x;
    if (t >= T) return;
    int64_t tid = token_ids[t];
    const __nv_bfloat16* row = embed_w + tid * D;
    __nv_bfloat16* dst = out + (int64_t)t * D;

    float sum_sq = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = __bfloat162float(row[d]);
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq);
    __shared__ float s_rstd;
    if (threadIdx.x == 0) s_rstd = rsqrtf(sum_sq / (float)D + eps);
    __syncthreads();
    float rstd = s_rstd;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = __bfloat162float(row[d]) * rstd * __bfloat162float(norm_w[d]);
        dst[d] = __float2bfloat16(v);
    }
}

void launch_embed_norm(
    const int32_t* token_ids, const __nv_bfloat16* embed_w, const __nv_bfloat16* norm_w,
    __nv_bfloat16* out, int T, int D, float eps, cudaStream_t stream)
{
    embed_norm_kernel<<<T, 256, 0, stream>>>(token_ids, embed_w, norm_w, out, T, D, eps);
}

__global__ void split_qrgs_kernel(
    const __nv_bfloat16* __restrict__ qrgs,
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ R,
    __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ sinks,
    int T, int q_dim, int r_dim, int n_heads)
{
    int stride = 2 * q_dim + r_dim + n_heads;
    int t = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T || c >= stride) return;
    __nv_bfloat16 v = qrgs[(int64_t)t * stride + c];
    if (c < q_dim) {
        Q[(int64_t)t * q_dim + c] = v;
    } else if (c < q_dim + r_dim) {
        R[(int64_t)t * r_dim + (c - q_dim)] = v;
    } else if (c < 2 * q_dim + r_dim) {
        gate[(int64_t)t * q_dim + (c - q_dim - r_dim)] = v;
    } else {
        sinks[(int64_t)t * n_heads + (c - 2 * q_dim - r_dim)] = v;
    }
}

void launch_split_qrgs(
    const __nv_bfloat16* qrgs, __nv_bfloat16* Q, __nv_bfloat16* R, __nv_bfloat16* gate, __nv_bfloat16* sinks,
    int T, int q_dim, int r_dim, int n_heads, cudaStream_t stream)
{
    int stride = 2 * q_dim + r_dim + n_heads;
    split_qrgs_kernel<<<dim3((stride + 255) / 256, T), 256, 0, stream>>>(
        qrgs, Q, R, gate, sinks, T, q_dim, r_dim, n_heads);
}

__global__ void split_kv_kernel(
    const __nv_bfloat16* __restrict__ kv,
    __nv_bfloat16* __restrict__ K,
    __nv_bfloat16* __restrict__ V,
    int T, int kv_dim)
{
    int t = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T || c >= 2 * kv_dim) return;
    __nv_bfloat16 v = kv[(int64_t)t * (2 * kv_dim) + c];
    if (c < kv_dim) K[(int64_t)t * kv_dim + c] = v;
    else V[(int64_t)t * kv_dim + (c - kv_dim)] = v;
}

void launch_split_kv(
    const __nv_bfloat16* kv, __nv_bfloat16* K, __nv_bfloat16* V,
    int T, int kv_dim, cudaStream_t stream)
{
    split_kv_kernel<<<dim3((2 * kv_dim + 255) / 256, T), 256, 0, stream>>>(kv, K, V, T, kv_dim);
}

__global__ void rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ out,
    int T, int D, float eps)
{
    int t = blockIdx.x;
    if (t >= T) return;
    const __nv_bfloat16* xr = x + (int64_t)t * D;
    __nv_bfloat16* yr = out + (int64_t)t * D;
    float sum_sq = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = __bfloat162float(xr[d]);
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq);
    __shared__ float s_rstd;
    if (threadIdx.x == 0) s_rstd = rsqrtf(sum_sq / (float)D + eps);
    __syncthreads();
    float rstd = s_rstd;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = __bfloat162float(xr[d]) * rstd * __bfloat162float(w[d]);
        yr[d] = __float2bfloat16(v);
    }
}

void launch_rmsnorm(
    const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* out,
    int T, int D, float eps, cudaStream_t stream)
{
    rmsnorm_kernel<<<T, 256, 0, stream>>>(x, w, out, T, D, eps);
}

__global__ void add_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
    }
}

void launch_add_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    int N, cudaStream_t stream)
{
    int threads = 256;
    add_bf16_kernel<<<(N + threads - 1) / threads, threads, 0, stream>>>(a, b, out, N);
}

// ============================================================================
// 2. Block AttnRes (fused with optional output RMSNorm)
// ============================================================================

__global__ void attn_res_kernel(
    const __nv_bfloat16* __restrict__ stack,
    int num_stack,
    int stack_stride,
    const __nv_bfloat16* __restrict__ last,
    const __nv_bfloat16* __restrict__ addend,
    const float* __restrict__ queries,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ out_bf16,
    float* __restrict__ out_fp32,
    int T, int D, float eps)
{
    int q_idx = blockIdx.x;
    int t = blockIdx.y;
    if (t >= T) return;

    const float* q_ptr = queries + (int64_t)q_idx * D;
    int64_t row_off = (int64_t)t * D;
    int N = num_stack + 1;

    __shared__ float s_logits[16];
    __shared__ float s_inv_rms[16];
    __shared__ float s_weights[16];
    __shared__ float s_out_rstd;

    // Pass 1: compute inv_rms and dot(q, x_n) * inv_rms for each source n
    for (int n = 0; n < N; ++n) {
        float sum_sq = 0.0f;
        float dot = 0.0f;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            float x;
            if (n < num_stack) {
                x = __bfloat162float(stack[(int64_t)n * stack_stride + row_off + d]);
            } else {
                x = __bfloat162float(last[row_off + d]);
                if (addend) x += __bfloat162float(addend[row_off + d]);
            }
            sum_sq += x * x;
            dot += q_ptr[d] * x;
        }
        sum_sq = block_reduce_sum(sum_sq);
        dot = block_reduce_sum(dot);
        if (threadIdx.x == 0) {
            float rstd = rsqrtf(sum_sq / (float)D + 1e-6f);
            s_inv_rms[n] = rstd;
            s_logits[n] = dot * rstd;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float max_l = s_logits[0];
        for (int n = 1; n < N; ++n) max_l = fmaxf(max_l, s_logits[n]);
        float sum_e = 0.0f;
        for (int n = 0; n < N; ++n) {
            float e = expf(s_logits[n] - max_l);
            s_weights[n] = e;
            sum_e += e;
        }
        float inv_sum = 1.0f / sum_e;
        // normalize_values = true -> multiply softmax prob by inv_rms[n]
        for (int n = 0; n < N; ++n) {
            s_weights[n] = s_weights[n] * inv_sum * s_inv_rms[n];
        }
    }
    __syncthreads();

    // Pass 2: compute mixed vector (and RMS if norm_w != nullptr)
    float out_sum_sq = 0.0f;
    // D=2304, blockDim=256 -> 9 elems per thread; store in registers!
    float acc_reg[12];
    int k = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            float x;
            if (n < num_stack) {
                x = __bfloat162float(stack[(int64_t)n * stack_stride + row_off + d]);
            } else {
                x = __bfloat162float(last[row_off + d]);
                if (addend) x += __bfloat162float(addend[row_off + d]);
            }
            acc = fmaf(s_weights[n], x, acc);
        }
        acc_reg[k++] = acc;
        out_sum_sq += acc * acc;
    }

    if (norm_w) {
        out_sum_sq = block_reduce_sum(out_sum_sq);
        if (threadIdx.x == 0) s_out_rstd = rsqrtf(out_sum_sq / (float)D + eps);
        __syncthreads();
        float rstd = s_out_rstd;
        const __nv_bfloat16* nw = norm_w + (int64_t)q_idx * D;
        __nv_bfloat16* dst = out_bf16 + ((int64_t)q_idx * T + t) * D;
        k = 0;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            dst[d] = __float2bfloat16(acc_reg[k++] * rstd * __bfloat162float(nw[d]));
        }
    } else {
        float* dst = out_fp32 + ((int64_t)q_idx * T + t) * D;
        k = 0;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            dst[d] = acc_reg[k++];
        }
    }
}

void launch_attn_res(
    const __nv_bfloat16* stack, int num_stack, int stack_stride,
    const __nv_bfloat16* last, const __nv_bfloat16* addend,
    const float* queries, int num_q, const __nv_bfloat16* norm_w,
    __nv_bfloat16* out_bf16, float* out_fp32,
    int T, int D, float eps, cudaStream_t stream)
{
    dim3 grid(num_q, T);
    attn_res_kernel<<<grid, 256, 0, stream>>>(
        stack, num_stack, stack_stride, last, addend, queries, norm_w,
        out_bf16, out_fp32, T, D, eps);
}

// ============================================================================
// 3. Loop Message Controller (pair_rms_norm & blend_rmsnorm)
// ============================================================================

__global__ void pair_rms_norm_kernel(
    const float* __restrict__ dest,
    const float* __restrict__ message,
    __nv_bfloat16* __restrict__ pair_out,
    int T, int D, float eps)
{
    int t = blockIdx.x;
    if (t >= T) return;
    const float* dp = dest + (int64_t)t * D;
    const float* mp = message + (int64_t)t * D;
    __nv_bfloat16* op = pair_out + (int64_t)t * (2 * D);

    float sd = 0.0f, sm = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float vd = dp[d], vm = mp[d];
        sd += vd * vd;
        sm += vm * vm;
    }
    sd = block_reduce_sum(sd);
    sm = block_reduce_sum(sm);
    __shared__ float rd, rm;
    if (threadIdx.x == 0) {
        rd = rsqrtf(sd / (float)D + eps);
        rm = rsqrtf(sm / (float)D + eps);
    }
    __syncthreads();
    float inv_d = rd, inv_m = rm;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        op[d] = __float2bfloat16(dp[d] * inv_d);
        op[D + d] = __float2bfloat16(mp[d] * inv_m);
    }
}

void launch_pair_rms_norm(
    const float* dest, const float* message, __nv_bfloat16* pair_out,
    int T, int D, float eps, cudaStream_t stream)
{
    pair_rms_norm_kernel<<<T, 256, 0, stream>>>(dest, message, pair_out, T, D, eps);
}

__global__ void fused_ctrl_blend_rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ ctrl_h,    // [T, rank=64]
    const __nv_bfloat16* __restrict__ gain_w,    // [D, rank]
    const __nv_bfloat16* __restrict__ forget_w,  // [D, rank]
    const float* __restrict__ dest,              // [T, D]
    const float* __restrict__ message,           // [T, D]
    const __nv_bfloat16* __restrict__ norm_w,    // [D]
    __nv_bfloat16* __restrict__ out,             // [T, D]
    int T, int D, int rank, float eps)
{
    int t = blockIdx.x;
    if (t >= T) return;
    __shared__ float s_h[64];
    if (threadIdx.x < rank) {
        s_h[threadIdx.x] = __bfloat162float(ctrl_h[(int64_t)t * rank + threadIdx.x]);
    }
    __syncthreads();

    int64_t off = (int64_t)t * D;
    float reg[12];
    float sum_sq = 0.0f;
    int k = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const uint4* gw4 = reinterpret_cast<const uint4*>(gain_w + (int64_t)d * rank);
        const uint4* fw4 = reinterpret_cast<const uint4*>(forget_w + (int64_t)d * rank);
        float g = 0.0f, f = 0.0f;
        // rank = 64 -> 8 uint4 loads (8 bf16 each)
        #pragma unroll
        for (int r8 = 0; r8 < 8; ++r8) {
            uint4 gv = gw4[r8];
            uint4 fv = fw4[r8];
            const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&gv);
            const __nv_bfloat162* fp = reinterpret_cast<const __nv_bfloat162*>(&fv);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 gf = __bfloat1622float2(gp[j]);
                float2 ff = __bfloat1622float2(fp[j]);
                float h0 = s_h[r8 * 8 + j * 2 + 0];
                float h1 = s_h[r8 * 8 + j * 2 + 1];
                g = fmaf(gf.x, h0, fmaf(gf.y, h1, g));
                f = fmaf(ff.x, h0, fmaf(ff.y, h1, f));
            }
        }
        float v = (1.0f - tanhf(f)) * dest[off + d] + tanhf(g) * message[off + d];
        reg[k++] = v;
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq);
    __shared__ float s_rstd;
    if (threadIdx.x == 0) s_rstd = rsqrtf(sum_sq / (float)D + eps);
    __syncthreads();
    float rstd = s_rstd;
    k = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        out[off + d] = __float2bfloat16(reg[k++] * rstd * __bfloat162float(norm_w[d]));
    }
}

// High-bandwidth 48-SM coalesced controller projection + blend (T = 1..8)
// 1 warp = 1 output channel d; 32 lanes read 64 BF16s (128B cache line) coalesced!
__global__ void ctrl_gf_blend_small_t_kernel(
    const __nv_bfloat16* __restrict__ ctrl_h,    // [T, 64]
    const __nv_bfloat16* __restrict__ gain_w,    // [D, 64]
    const __nv_bfloat16* __restrict__ forget_w,  // [D, 64]
    const float* __restrict__ dest,              // [T, D]
    const float* __restrict__ message,           // [T, D]
    float* __restrict__ out_fp32,                // [T, D]
    int T, int D)
{
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int d = blockIdx.x * 4 + wid;
    if (d >= D) return;

    __shared__ float s_h[8][64];
    for (int idx = threadIdx.x; idx < T * 64; idx += blockDim.x) {
        s_h[idx >> 6][idx & 63] = __bfloat162float(ctrl_h[idx]);
    }
    __syncthreads();

    // Each lane reads 2 contiguous BF16s of gain_w[d] and forget_w[d] (rank = 64 -> 32 lanes * 2)
    __nv_bfloat162 gw2 = reinterpret_cast<const __nv_bfloat162*>(gain_w + (int64_t)d * 64)[lane];
    __nv_bfloat162 fw2 = reinterpret_cast<const __nv_bfloat162*>(forget_w + (int64_t)d * 64)[lane];
    float2 gf = __bfloat1622float2(gw2);
    float2 ff = __bfloat1622float2(fw2);
    int r0 = lane * 2;

    for (int t = 0; t < T; ++t) {
        float h0 = s_h[t][r0];
        float h1 = s_h[t][r0 + 1];
        float g = warp_reduce_sum(fmaf(gf.x, h0, gf.y * h1));
        float f = warp_reduce_sum(fmaf(ff.x, h0, ff.y * h1));
        if (lane == 0) {
            int64_t off = (int64_t)t * D + d;
            out_fp32[off] = (1.0f - tanhf(f)) * dest[off] + tanhf(g) * message[off];
        }
    }
}

__global__ void fp32_rmsnorm_to_bf16_kernel(
    const float* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ out,
    int T, int D, float eps)
{
    int t = blockIdx.x;
    if (t >= T) return;
    const float* xr = x + (int64_t)t * D;
    __nv_bfloat16* yr = out + (int64_t)t * D;
    float sum_sq = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = xr[d];
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq);
    __shared__ float s_rstd;
    if (threadIdx.x == 0) s_rstd = rsqrtf(sum_sq / (float)D + eps);
    __syncthreads();
    float rstd = s_rstd;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        yr[d] = __float2bfloat16(xr[d] * rstd * __bfloat162float(w[d]));
    }
}

void launch_fused_ctrl_blend_rmsnorm(
    const __nv_bfloat16* ctrl_h, const __nv_bfloat16* gain_w, const __nv_bfloat16* forget_w,
    const float* dest, const float* message, const __nv_bfloat16* norm_w,
    __nv_bfloat16* out, int T, int D, int rank, float eps, cudaStream_t stream,
    float* tmp_fp32)
{
    if (T <= 8 && rank == 64 && tmp_fp32 != nullptr) {
        ctrl_gf_blend_small_t_kernel<<<(D + 3) / 4, 128, 0, stream>>>(
            ctrl_h, gain_w, forget_w, dest, message, tmp_fp32, T, D);
        fp32_rmsnorm_to_bf16_kernel<<<T, 256, 0, stream>>>(tmp_fp32, norm_w, out, T, D, eps);
    } else {
        fused_ctrl_blend_rmsnorm_kernel<<<T, 256, 0, stream>>>(
            ctrl_h, gain_w, forget_w, dest, message, norm_w, out, T, D, rank, eps);
    }
}

// ============================================================================
// 4. Short Convolutions (static k=4 and dynamic k=4)
// ============================================================================

__global__ void sconv_static_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight, // [C, 4]: w[3]=lag0, w[2]=lag1, w[1]=lag2, w[0]=lag3
    __nv_bfloat16* __restrict__ state,        // [3, C]: state[0]=t-3, state[1]=t-2, state[2]=t-1
    __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ snap,         // optional [T, 3, C]
    int T, int C)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float w0 = __bfloat162float(weight[c * 4 + 0]); // lag 3
    float w1 = __bfloat162float(weight[c * 4 + 1]); // lag 2
    float w2 = __bfloat162float(weight[c * 4 + 2]); // lag 1
    float w3 = __bfloat162float(weight[c * 4 + 3]); // lag 0
    const float inv_sqrt5 = 0.44721359549995793f;

    float s0 = __bfloat162float(state[0 * C + c]);
    float s1 = __bfloat162float(state[1 * C + c]);
    float s2 = __bfloat162float(state[2 * C + c]);

    for (int t = 0; t < T; ++t) {
        float xt = __bfloat162float(x[(int64_t)t * C + c]);
        float y = (w0 * s0 + w1 * s1 + w2 * s2 + w3 * xt + xt) * inv_sqrt5;
        out[(int64_t)t * C + c] = __float2bfloat16(y);
        s0 = s1;
        s1 = s2;
        s2 = xt;
        if (snap) {
            __nv_bfloat16* sp = snap + (int64_t)t * 3 * C;
            sp[0 * C + c] = __float2bfloat16(s0);
            sp[1 * C + c] = __float2bfloat16(s1);
            sp[2 * C + c] = __float2bfloat16(s2);
        }
    }
    state[0 * C + c] = __float2bfloat16(s0);
    state[1 * C + c] = __float2bfloat16(s1);
    state[2 * C + c] = __float2bfloat16(s2);
}

void launch_sconv_static(
    const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* state,
    __nv_bfloat16* out, int T, int C, cudaStream_t stream, __nv_bfloat16* snap)
{
    int threads = 256;
    sconv_static_kernel<<<(C + threads - 1) / threads, threads, 0, stream>>>(x, weight, state, out, snap, T, C);
}

__global__ void sconv_dynamic_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ static_w, // [4, C]: static_w[lag * C + c], lag 0 = current
    const __nv_bfloat16* __restrict__ dyn_w,    // [T, num_kv_heads * 4]
    __nv_bfloat16* __restrict__ state,          // [3, C]: state[0]=t-3, state[1]=t-2, state[2]=t-1
    __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ snap,           // optional [T, 3, C]
    int T, int num_kv_heads, int head_dim)
{
    int C = num_kv_heads * head_dim;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    int h = c / head_dim;

    float sw0 = __bfloat162float(static_w[0 * C + c]); // lag 0
    float sw1 = __bfloat162float(static_w[1 * C + c]); // lag 1
    float sw2 = __bfloat162float(static_w[2 * C + c]); // lag 2
    float sw3 = __bfloat162float(static_w[3 * C + c]); // lag 3
    const float inv_3 = 1.0f / 3.0f;

    float s0 = __bfloat162float(state[0 * C + c]); // lag 3
    float s1 = __bfloat162float(state[1 * C + c]); // lag 2
    float s2 = __bfloat162float(state[2 * C + c]); // lag 1

    int dyn_stride = num_kv_heads * 4;
    for (int t = 0; t < T; ++t) {
        float xt = __bfloat162float(x[(int64_t)t * C + c]);
        const __nv_bfloat16* dw = dyn_w + (int64_t)t * dyn_stride + h * 4;
        float tap0 = sw0 + __bfloat162float(dw[0]);
        float tap1 = sw1 + __bfloat162float(dw[1]);
        float tap2 = sw2 + __bfloat162float(dw[2]);
        float tap3 = sw3 + __bfloat162float(dw[3]);
        float y = (tap0 * xt + tap1 * s2 + tap2 * s1 + tap3 * s0 + xt) * inv_3;
        out[(int64_t)t * C + c] = __float2bfloat16(y);
        s0 = s1;
        s1 = s2;
        s2 = xt;
        if (snap) {
            __nv_bfloat16* sp = snap + (int64_t)t * 3 * C;
            sp[0 * C + c] = __float2bfloat16(s0);
            sp[1 * C + c] = __float2bfloat16(s1);
            sp[2 * C + c] = __float2bfloat16(s2);
        }
    }
    state[0 * C + c] = __float2bfloat16(s0);
    state[1 * C + c] = __float2bfloat16(s1);
    state[2 * C + c] = __float2bfloat16(s2);
}

void launch_sconv_dynamic(
    const __nv_bfloat16* x, const __nv_bfloat16* static_w, const __nv_bfloat16* dyn_w,
    __nv_bfloat16* state, __nv_bfloat16* out,
    int T, int num_kv_heads, int head_dim, cudaStream_t stream, __nv_bfloat16* snap)
{
    int C = num_kv_heads * head_dim;
    int threads = 256;
    sconv_dynamic_kernel<<<(C + threads - 1) / threads, threads, 0, stream>>>(
        x, static_w, dyn_w, state, out, snap, T, num_kv_heads, head_dim);
}

// ============================================================================
// 5. Attention Prep (Q/K/V/R clip/norm/tau + KV cache write)
// ============================================================================

__global__ void attn_prep_q_r_kernel(
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ R,
    const __nv_bfloat16* __restrict__ q_scale,
    const __nv_bfloat16* __restrict__ r_scale,
    int start_pos, const int32_t* __restrict__ d_pos, int T, bool is_local,
    int n_heads, int head_dim, int d_rel,
    float qk_clip, float rel_clip, float log_n_floor, float log_alpha)
{
    int h = blockIdx.x;
    int t = blockIdx.y;
    if (h >= n_heads || t >= T) return;

    int pos = (d_pos ? *d_pos : start_pos) + t;
    float tau = 1.0f;
    if (!is_local) {
        float ratio = fmaxf(1.0f, (float)(pos + 1) / log_n_floor);
        tau = 1.0f + log_alpha * logf(ratio);
    }

    // Q: 128 dims, 128 threads (or 32 threads x 4)
    __nv_bfloat16* q_ptr = Q + ((int64_t)t * n_heads + h) * head_dim;
    const __nv_bfloat16* qs_ptr = q_scale + h * head_dim;
    float qv[4];
    float sum_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = threadIdx.x + i * 32;
        float v = __bfloat162float(q_ptr[d]) * __bfloat162float(qs_ptr[d]);
        qv[i] = v;
        sum_sq += v * v;
    }
    sum_sq = warp_reduce_sum(sum_sq);
    float norm = sqrtf(__shfl_sync(0xffffffff, sum_sq, 0));
    float scale = fminf(1.0f, qk_clip / (norm + 1e-6f)) * tau;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = threadIdx.x + i * 32;
        q_ptr[d] = __float2bfloat16(qv[i] * scale);
    }

    // R: d_rel = 16, first 16 lanes
    __nv_bfloat16* r_ptr = R + ((int64_t)t * n_heads + h) * d_rel;
    const __nv_bfloat16* rs_ptr = r_scale + h * d_rel;
    float rv = 0.0f;
    if (threadIdx.x < d_rel) {
        rv = __bfloat162float(r_ptr[threadIdx.x]) * __bfloat162float(rs_ptr[threadIdx.x]);
    }
    float r_sq = warp_reduce_sum(rv * rv);
    float r_norm = sqrtf(__shfl_sync(0xffffffff, r_sq, 0));
    float r_sc = fminf(1.0f, rel_clip / (r_norm + 1e-6f)) * tau;
    if (threadIdx.x < d_rel) {
        r_ptr[threadIdx.x] = __float2bfloat16(rv * r_sc);
    }
}

__global__ void attn_prep_kv_kernel(
    const __nv_bfloat16* __restrict__ K_in,
    const __nv_bfloat16* __restrict__ V_in,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    uint8_t* __restrict__ k_cache_fp8,
    uint8_t* __restrict__ v_cache_fp8,
    float* __restrict__ k_scale_fp8,
    float* __restrict__ v_scale_fp8,
    int start_pos, const int32_t* __restrict__ d_pos, int T, int cache_cap, bool is_local,
    int n_kv_heads, int head_dim, float qk_clip, float eps)
{
    int h = blockIdx.x;
    int t = blockIdx.y;
    if (h >= n_kv_heads || t >= T) return;

    int pos = (d_pos ? *d_pos : start_pos) + t;
    int slot = is_local ? (pos % cache_cap) : pos;

    const __nv_bfloat16* kin = K_in + ((int64_t)t * n_kv_heads + h) * head_dim;
    const __nv_bfloat16* vin = V_in + ((int64_t)t * n_kv_heads + h) * head_dim;

    float kv[4], vv[4];
    float k_sq = 0.0f, v_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = threadIdx.x + i * 32;
        kv[i] = __bfloat162float(kin[d]);
        vv[i] = __bfloat162float(vin[d]);
        k_sq += kv[i] * kv[i];
        v_sq += vv[i] * vv[i];
    }
    k_sq = warp_reduce_sum(k_sq);
    v_sq = warp_reduce_sum(v_sq);
    float k_norm = sqrtf(__shfl_sync(0xffffffff, k_sq, 0));
    float k_sc = fminf(1.0f, qk_clip / (k_norm + 1e-6f));
    float v_rstd = rsqrtf(__shfl_sync(0xffffffff, v_sq, 0) / (float)head_dim + eps);

    if (k_cache_fp8) {
        float km = 0.0f, vm = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            kv[i] *= k_sc;
            vv[i] *= v_rstd;
            km = fmaxf(km, fabsf(kv[i]));
            vm = fmaxf(vm, fabsf(vv[i]));
        }
        float k_amax = __shfl_sync(0xffffffff, warp_reduce_max(km), 0);
        float v_amax = __shfl_sync(0xffffffff, warp_reduce_max(vm), 0);
        float ks = (k_amax > 0.0f) ? (k_amax / 448.0f) : 1.0f;
        float vs = (v_amax > 0.0f) ? (v_amax / 448.0f) : 1.0f;
        float inv_ks = 1.0f / ks, inv_vs = 1.0f / vs;
        int64_t sc_idx = (int64_t)slot * n_kv_heads + h;
        if (threadIdx.x == 0) { k_scale_fp8[sc_idx] = ks; v_scale_fp8[sc_idx] = vs; }
        uint8_t* kdst = k_cache_fp8 + sc_idx * head_dim;
        uint8_t* vdst = v_cache_fp8 + sc_idx * head_dim;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int d = threadIdx.x + i * 32;
            kdst[d] = (uint8_t)__nv_cvt_float_to_fp8(kv[i] * inv_ks, __NV_SATFINITE, __NV_E4M3);
            vdst[d] = (uint8_t)__nv_cvt_float_to_fp8(vv[i] * inv_vs, __NV_SATFINITE, __NV_E4M3);
        }
    } else {
        __nv_bfloat16* kdst = k_cache + ((int64_t)slot * n_kv_heads + h) * head_dim;
        __nv_bfloat16* vdst = v_cache + ((int64_t)slot * n_kv_heads + h) * head_dim;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int d = threadIdx.x + i * 32;
            kdst[d] = __float2bfloat16(kv[i] * k_sc);
            vdst[d] = __float2bfloat16(vv[i] * v_rstd);
        }
    }
}

void launch_attn_prep(
    __nv_bfloat16* Q, const __nv_bfloat16* K_in, const __nv_bfloat16* V_in, __nv_bfloat16* R,
    const __nv_bfloat16* q_scale, const __nv_bfloat16* r_scale,
    __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
    int start_pos, const int32_t* d_pos, int T, int cache_cap, bool is_local,
    int n_heads, int n_kv_heads, int head_dim, int d_rel,
    float qk_clip, float rel_clip, float log_n_floor, float log_alpha, float eps,
    cudaStream_t stream,
    uint8_t* k_cache_fp8, uint8_t* v_cache_fp8, float* k_scale_fp8, float* v_scale_fp8)
{
    attn_prep_q_r_kernel<<<dim3(n_heads, T), 32, 0, stream>>>(
        Q, R, q_scale, r_scale, start_pos, d_pos, T, is_local, n_heads, head_dim, d_rel,
        qk_clip, rel_clip, log_n_floor, log_alpha);
    attn_prep_kv_kernel<<<dim3(n_kv_heads, T), 32, 0, stream>>>(
        K_in, V_in, k_cache, v_cache, k_cache_fp8, v_cache_fp8, k_scale_fp8, v_scale_fp8,
        start_pos, d_pos, T, cache_cap, is_local, n_kv_heads, head_dim, qk_clip, eps);
}

// ============================================================================
// 6. Attention (Split-K FlashDecoding for T=1, Flash-style online softmax for T>1)
// ============================================================================

// Split-K partial kernel for T=1 decode: grid = (n_heads, num_splits), block = 128 (4 warps)
__global__ void attn_decode_split_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const uint8_t* __restrict__ k_cache_fp8,
    const uint8_t* __restrict__ v_cache_fp8,
    const float* __restrict__ k_scale_fp8,
    const float* __restrict__ v_scale_fp8,
    const __nv_bfloat16* __restrict__ rel_logits,
    float* __restrict__ split_scratch, // [n_heads, num_splits, head_dim + 2]
    int start_pos, const int32_t* __restrict__ d_pos, int num_splits, int cache_cap, bool is_local, int window_size,
    int n_heads, int n_kv_heads, int head_dim, int rel_extent)
{
    int h = blockIdx.x;
    int s = blockIdx.y;
    int kv_h = h / (n_heads / n_kv_heads);

    int q_pos = d_pos ? *d_pos : start_pos;
    int k_start = is_local ? max(0, q_pos - window_size + 1) : 0;
    int k_end = q_pos + 1;
    int total_k = k_end - k_start;
    int chunk = (total_k + num_splits - 1) / num_splits;
    int my_start = k_start + s * chunk;
    int my_end = min(k_end, my_start + chunk);

    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;

    const __nv_bfloat16* q_ptr = Q + h * head_dim;
    const __nv_bfloat16* r_ptr = rel_logits + h * rel_extent;
    float qv[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        qv[i] = __bfloat162float(q_ptr[lane + i * 32]);
    }
    const float inv_sqrt_d = rsqrtf((float)head_dim);

    float m_val = -1e30f;
    float l_val = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    int kv_stride = n_kv_heads * head_dim;
    for (int kp = my_start + wid; kp < my_end; kp += nwarps) {
        int slot = is_local ? (kp % cache_cap) : kp;
        float dot = 0.0f;
        float vv0, vv1, vv2, vv3;
        if (k_cache_fp8) {
            int64_t sc_idx = (int64_t)slot * n_kv_heads + kv_h;
            float ks = k_scale_fp8[sc_idx];
            float vs = v_scale_fp8[sc_idx];
            const uint8_t* k_ptr = k_cache_fp8 + sc_idx * head_dim;
            const uint8_t* v_ptr = v_cache_fp8 + sc_idx * head_dim;
            #pragma unroll
            for (int i = 0; i < 4; ++i) dot = fmaf(qv[i], e4m3_to_f32(k_ptr[lane + i * 32]), dot);
            dot = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0) * inv_sqrt_d * ks;
            vv0 = e4m3_to_f32(v_ptr[lane + 0]) * vs;
            vv1 = e4m3_to_f32(v_ptr[lane + 32]) * vs;
            vv2 = e4m3_to_f32(v_ptr[lane + 64]) * vs;
            vv3 = e4m3_to_f32(v_ptr[lane + 96]) * vs;
        } else {
            const __nv_bfloat16* k_ptr = k_cache + (int64_t)slot * kv_stride + kv_h * head_dim;
            const __nv_bfloat16* v_ptr = v_cache + (int64_t)slot * kv_stride + kv_h * head_dim;
            #pragma unroll
            for (int i = 0; i < 4; ++i) dot = fmaf(qv[i], __bfloat162float(k_ptr[lane + i * 32]), dot);
            dot = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0) * inv_sqrt_d;
            vv0 = __bfloat162float(v_ptr[lane + 0]);
            vv1 = __bfloat162float(v_ptr[lane + 32]);
            vv2 = __bfloat162float(v_ptr[lane + 64]);
            vv3 = __bfloat162float(v_ptr[lane + 96]);
        }
        int dist = q_pos - kp;
        if (dist >= 0 && dist < rel_extent) dot += __bfloat162float(r_ptr[dist]);

        float m_new = fmaxf(m_val, dot);
        float alpha = expf(m_val - m_new);
        float p = expf(dot - m_new);
        l_val = l_val * alpha + p;
        acc[0] = acc[0] * alpha + p * vv0;
        acc[1] = acc[1] * alpha + p * vv1;
        acc[2] = acc[2] * alpha + p * vv2;
        acc[3] = acc[3] * alpha + p * vv3;
        m_val = m_new;
    }

    // Reduce across warps in block
    __shared__ float sm_m[8];
    __shared__ float sm_l[8];
    __shared__ float sm_acc[8][128];
    if (lane == 0) {
        sm_m[wid] = m_val;
        sm_l[wid] = l_val;
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) sm_acc[wid][lane + i * 32] = acc[i];
    __syncthreads();

    if (wid == 0) {
        float gm = -1e30f;
        for (int w = 0; w < nwarps; ++w) gm = fmaxf(gm, sm_m[w]);
        float gl = 0.0f;
        float gacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int w = 0; w < nwarps; ++w) {
            if (sm_l[w] > 0.0f) {
                float sc = expf(sm_m[w] - gm);
                gl += sm_l[w] * sc;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    gacc[i] += sm_acc[w][lane + i * 32] * sc;
                }
            }
        }
        float* dst = split_scratch + ((int64_t)h * num_splits + s) * (head_dim + 2);
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[lane + i * 32] = gacc[i];
        if (lane == 0) {
            dst[head_dim] = gm;
            dst[head_dim + 1] = gl;
        }
    }
}

__global__ void attn_decode_merge_kernel(
    const float* __restrict__ split_scratch,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ sinks,
    __nv_bfloat16* __restrict__ out,
    int num_splits, int n_heads, int head_dim)
{
    int h = blockIdx.x;
    int lane = threadIdx.x; // 32 threads
    float sink_logit = __bfloat162float(sinks[h]);

    float gm = sink_logit;
    for (int s = 0; s < num_splits; ++s) {
        const float* src = split_scratch + ((int64_t)h * num_splits + s) * (head_dim + 2);
        if (src[head_dim + 1] > 0.0f) gm = fmaxf(gm, src[head_dim]);
    }
    float gl = expf(sink_logit - gm);
    float gacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int s = 0; s < num_splits; ++s) {
        const float* src = split_scratch + ((int64_t)h * num_splits + s) * (head_dim + 2);
        float l = src[head_dim + 1];
        if (l > 0.0f) {
            float sc = expf(src[head_dim] - gm);
            gl += l * sc;
            #pragma unroll
            for (int i = 0; i < 4; ++i) gacc[i] += src[lane + i * 32] * sc;
        }
    }
    float inv_l = 1.0f / gl;
    const __nv_bfloat16* g_ptr = gate + h * head_dim;
    __nv_bfloat16* o_ptr = out + h * head_dim;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = lane + i * 32;
        float gv = __bfloat162float(g_ptr[d]);
        float sig = 1.0f / (1.0f + expf(-gv));
        o_ptr[d] = __float2bfloat16(gacc[i] * inv_l * sig);
    }
}

// Prefill / general T attention kernel: 1 warp per (t, h)
__global__ void attn_prefill_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ rel_logits,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ sinks,
    __nv_bfloat16* __restrict__ out,
    int start_pos, const int32_t* __restrict__ d_pos, int T, int cache_cap, bool is_local, int window_size,
    int n_heads, int n_kv_heads, int head_dim, int rel_extent)
{
    int h = blockIdx.x;
    int t = blockIdx.y;
    if (h >= n_heads || t >= T) return;

    int q_pos = (d_pos ? *d_pos : start_pos) + t;
    int k_start = is_local ? max(0, q_pos - window_size + 1) : 0;
    int k_end = q_pos + 1;
    int kv_h = h / (n_heads / n_kv_heads);

    int lane = threadIdx.x; // 32 threads
    const __nv_bfloat16* q_ptr = Q + ((int64_t)t * n_heads + h) * head_dim;
    const __nv_bfloat16* r_ptr = rel_logits + ((int64_t)t * n_heads + h) * rel_extent;
    float qv[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) qv[i] = __bfloat162float(q_ptr[lane + i * 32]);
    const float inv_sqrt_d = rsqrtf((float)head_dim);

    // Initialize online softmax with attention sink (value = 0)
    float m_val = __bfloat162float(sinks[(int64_t)t * n_heads + h]);
    float l_val = 1.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    int kv_stride = n_kv_heads * head_dim;
    for (int kp = k_start; kp < k_end; ++kp) {
        int slot = is_local ? (kp % cache_cap) : kp;
        const __nv_bfloat16* k_ptr = k_cache + (int64_t)slot * kv_stride + kv_h * head_dim;
        const __nv_bfloat16* v_ptr = v_cache + (int64_t)slot * kv_stride + kv_h * head_dim;

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            dot = fmaf(qv[i], __bfloat162float(k_ptr[lane + i * 32]), dot);
        }
        dot = warp_reduce_sum(dot);
        dot = __shfl_sync(0xffffffff, dot, 0) * inv_sqrt_d;
        int dist = q_pos - kp;
        if (dist >= 0 && dist < rel_extent) {
            dot += __bfloat162float(r_ptr[dist]);
        }

        float m_new = fmaxf(m_val, dot);
        float alpha = expf(m_val - m_new);
        float p = expf(dot - m_new);
        l_val = l_val * alpha + p;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[i] = acc[i] * alpha + p * __bfloat162float(v_ptr[lane + i * 32]);
        }
        m_val = m_new;
    }

    float inv_l = 1.0f / l_val;
    const __nv_bfloat16* g_ptr = gate + ((int64_t)t * n_heads + h) * head_dim;
    __nv_bfloat16* o_ptr = out + ((int64_t)t * n_heads + h) * head_dim;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = lane + i * 32;
        float gv = __bfloat162float(g_ptr[d]);
        float sig = 1.0f / (1.0f + expf(-gv));
        o_ptr[d] = __float2bfloat16(acc[i] * inv_l * sig);
    }
}

// High-throughput SMEM-tiled FlashAttention prefill kernel for Inkling-2-Mini (GQA=3, head_dim=128)
// 1 block = 8 warps (256 threads) covering BQ=8 tokens x 3 GQA Q-heads = 24 queries per KV load!
__global__ void attn_prefill_tiled_gqa3_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ rel_logits,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ sinks,
    __nv_bfloat16* __restrict__ out,
    int start_pos, const int32_t* __restrict__ d_pos, int T, int cache_cap, bool is_local, int window_size,
    int n_heads, int n_kv_heads, int rel_extent)
{
    int kv_h = blockIdx.x;
    int t_base = blockIdx.y * 8;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int t = t_base + wid;
    bool valid_t = (t < T);

    int sp = d_pos ? *d_pos : start_pos;
    int q_pos = sp + t;
    int blk_k_start = is_local ? max(0, sp + t_base - window_size + 1) : 0;
    int blk_k_end = sp + min(T, t_base + 8);
    int kv_stride = n_kv_heads * 128;

    float qv[3][4];
    float m_val[3], l_val[3];
    float acc[3][4];
    const __nv_bfloat16* r_ptr[3];

    if (valid_t) {
        #pragma unroll
        for (int g = 0; g < 3; ++g) {
            int h = kv_h * 3 + g;
            const __nv_bfloat16* q_ptr = Q + ((int64_t)t * n_heads + h) * 128;
            r_ptr[g] = rel_logits + ((int64_t)t * n_heads + h) * rel_extent;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                qv[g][i] = __bfloat162float(q_ptr[lane + i * 32]);
                acc[g][i] = 0.0f;
            }
            m_val[g] = __bfloat162float(sinks[(int64_t)t * n_heads + h]);
            l_val[g] = 1.0f;
        }
    }

    __shared__ __nv_bfloat16 sm_k[32][128];
    __shared__ __nv_bfloat16 sm_v[32][128];
    const float inv_sqrt_d = rsqrtf(128.0f);

    int my_k_start = is_local ? max(0, q_pos - window_size + 1) : 0;
    int my_k_end = q_pos + 1;

    for (int kb = blk_k_start; kb < blk_k_end; kb += 32) {
        int k_len = min(32, blk_k_end - kb);
        int total_u4 = k_len * 16; // 16 uint4s = 128 bf16 per row
        for (int idx = threadIdx.x; idx < total_u4; idx += 256) {
            int bk = idx >> 4;
            int d8 = idx & 15;
            int kp = kb + bk;
            int slot = is_local ? (kp % cache_cap) : kp;
            const uint4* ks = reinterpret_cast<const uint4*>(k_cache + (int64_t)slot * kv_stride + kv_h * 128);
            const uint4* vs = reinterpret_cast<const uint4*>(v_cache + (int64_t)slot * kv_stride + kv_h * 128);
            reinterpret_cast<uint4*>(sm_k[bk])[d8] = ks[d8];
            reinterpret_cast<uint4*>(sm_v[bk])[d8] = vs[d8];
        }
        __syncthreads();

        if (valid_t) {
            int b_lo = max(0, my_k_start - kb);
            int b_hi = min(k_len, my_k_end - kb);
            for (int bk = b_lo; bk < b_hi; ++bk) {
                float kv0 = __bfloat162float(sm_k[bk][lane + 0]);
                float kv1 = __bfloat162float(sm_k[bk][lane + 32]);
                float kv2 = __bfloat162float(sm_k[bk][lane + 64]);
                float kv3 = __bfloat162float(sm_k[bk][lane + 96]);
                float vv0 = __bfloat162float(sm_v[bk][lane + 0]);
                float vv1 = __bfloat162float(sm_v[bk][lane + 32]);
                float vv2 = __bfloat162float(sm_v[bk][lane + 64]);
                float vv3 = __bfloat162float(sm_v[bk][lane + 96]);
                int dist = q_pos - (kb + bk);
                bool has_rel = (dist >= 0 && dist < rel_extent);

                #pragma unroll
                for (int g = 0; g < 3; ++g) {
                    float dot = fmaf(qv[g][0], kv0, fmaf(qv[g][1], kv1, fmaf(qv[g][2], kv2, qv[g][3] * kv3)));
                    dot = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0) * inv_sqrt_d;
                    if (has_rel) dot += __bfloat162float(r_ptr[g][dist]);
                    float m_new = fmaxf(m_val[g], dot);
                    float alpha = expf(m_val[g] - m_new);
                    float p = expf(dot - m_new);
                    l_val[g] = l_val[g] * alpha + p;
                    acc[g][0] = fmaf(p, vv0, acc[g][0] * alpha);
                    acc[g][1] = fmaf(p, vv1, acc[g][1] * alpha);
                    acc[g][2] = fmaf(p, vv2, acc[g][2] * alpha);
                    acc[g][3] = fmaf(p, vv3, acc[g][3] * alpha);
                    m_val[g] = m_new;
                }
            }
        }
        __syncthreads();
    }

    if (valid_t) {
        #pragma unroll
        for (int g = 0; g < 3; ++g) {
            int h = kv_h * 3 + g;
            float inv_l = 1.0f / l_val[g];
            const __nv_bfloat16* g_ptr = gate + ((int64_t)t * n_heads + h) * 128;
            __nv_bfloat16* o_ptr = out + ((int64_t)t * n_heads + h) * 128;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int d = lane + i * 32;
                float sig = 1.0f / (1.0f + expf(-__bfloat162float(g_ptr[d])));
                o_ptr[d] = __float2bfloat16(acc[g][i] * inv_l * sig);
            }
        }
    }
}

// Multi-warp (8 warps = 256 threads) verify / small-T attention kernel (T = 2..15)
// Saturates memory bandwidth even at 130K+ context!
__global__ void attn_small_t_multiwarp_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const uint8_t* __restrict__ k_cache_fp8,
    const uint8_t* __restrict__ v_cache_fp8,
    const float* __restrict__ k_scale_fp8,
    const float* __restrict__ v_scale_fp8,
    const __nv_bfloat16* __restrict__ rel_logits,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ sinks,
    __nv_bfloat16* __restrict__ out,
    int start_pos, const int32_t* __restrict__ d_pos, int T, int cache_cap, bool is_local, int window_size,
    int n_heads, int n_kv_heads, int head_dim, int rel_extent)
{
    int h = blockIdx.x;
    int t = blockIdx.y;
    if (h >= n_heads || t >= T) return;

    int q_pos = (d_pos ? *d_pos : start_pos) + t;
    int k_start = is_local ? max(0, q_pos - window_size + 1) : 0;
    int k_end = q_pos + 1;
    int kv_h = h / (n_heads / n_kv_heads);

    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;

    const __nv_bfloat16* q_ptr = Q + ((int64_t)t * n_heads + h) * head_dim;
    const __nv_bfloat16* r_ptr = rel_logits + ((int64_t)t * n_heads + h) * rel_extent;
    float qv[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) qv[i] = __bfloat162float(q_ptr[lane + i * 32]);
    const float inv_sqrt_d = rsqrtf((float)head_dim);

    float m_val = -1e30f;
    float l_val = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int kv_stride = n_kv_heads * head_dim;

    for (int kp = k_start + wid; kp < k_end; kp += nwarps) {
        int slot = is_local ? (kp % cache_cap) : kp;
        float dot = 0.0f;
        float vv0, vv1, vv2, vv3;
        if (k_cache_fp8) {
            int64_t sc_idx = (int64_t)slot * n_kv_heads + kv_h;
            float ks = k_scale_fp8[sc_idx];
            float vs = v_scale_fp8[sc_idx];
            const uint8_t* k_ptr = k_cache_fp8 + sc_idx * head_dim;
            const uint8_t* v_ptr = v_cache_fp8 + sc_idx * head_dim;
            #pragma unroll
            for (int i = 0; i < 4; ++i) dot = fmaf(qv[i], e4m3_to_f32(k_ptr[lane + i * 32]), dot);
            dot = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0) * inv_sqrt_d * ks;
            vv0 = e4m3_to_f32(v_ptr[lane + 0]) * vs;
            vv1 = e4m3_to_f32(v_ptr[lane + 32]) * vs;
            vv2 = e4m3_to_f32(v_ptr[lane + 64]) * vs;
            vv3 = e4m3_to_f32(v_ptr[lane + 96]) * vs;
        } else {
            const __nv_bfloat16* k_ptr = k_cache + (int64_t)slot * kv_stride + kv_h * head_dim;
            const __nv_bfloat16* v_ptr = v_cache + (int64_t)slot * kv_stride + kv_h * head_dim;
            #pragma unroll
            for (int i = 0; i < 4; ++i) dot = fmaf(qv[i], __bfloat162float(k_ptr[lane + i * 32]), dot);
            dot = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0) * inv_sqrt_d;
            vv0 = __bfloat162float(v_ptr[lane + 0]);
            vv1 = __bfloat162float(v_ptr[lane + 32]);
            vv2 = __bfloat162float(v_ptr[lane + 64]);
            vv3 = __bfloat162float(v_ptr[lane + 96]);
        }
        int dist = q_pos - kp;
        if (dist >= 0 && dist < rel_extent) dot += __bfloat162float(r_ptr[dist]);

        float m_new = fmaxf(m_val, dot);
        float alpha = expf(m_val - m_new);
        float p = expf(dot - m_new);
        l_val = l_val * alpha + p;
        acc[0] = acc[0] * alpha + p * vv0;
        acc[1] = acc[1] * alpha + p * vv1;
        acc[2] = acc[2] * alpha + p * vv2;
        acc[3] = acc[3] * alpha + p * vv3;
        m_val = m_new;
    }

    __shared__ float sm_m[8], sm_l[8], sm_acc[8][128];
    if (lane == 0) { sm_m[wid] = m_val; sm_l[wid] = l_val; }
    #pragma unroll
    for (int i = 0; i < 4; ++i) sm_acc[wid][lane + i * 32] = acc[i];
    __syncthreads();

    if (wid == 0) {
        float sink_logit = __bfloat162float(sinks[(int64_t)t * n_heads + h]);
        float gm = sink_logit;
        for (int w = 0; w < nwarps; ++w) if (sm_l[w] > 0.0f) gm = fmaxf(gm, sm_m[w]);
        float gl = expf(sink_logit - gm);
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
        const __nv_bfloat16* g_ptr = gate + ((int64_t)t * n_heads + h) * head_dim;
        __nv_bfloat16* o_ptr = out + ((int64_t)t * n_heads + h) * head_dim;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int d = lane + i * 32;
            float sig = 1.0f / (1.0f + expf(-__bfloat162float(g_ptr[d])));
            o_ptr[d] = __float2bfloat16(gacc[i] * inv_l * sig);
        }
    }
}

// GQA=3 Shared-KV + Shared-T Split-K Verify Attention (T = 2..15, head_dim = 128, n_heads = 3*n_kv_heads)
// Each block (kv_h, s) cooperatively loads BK=32 KV tokens ONCE into SMEM and serves all T warps x 3 GQA heads
// -> Cuts global KV DRAM traffic by 3*T (24x at T=8) while saturating all 48 SMs via Split-K!
__global__ void attn_small_t_gqa3_split_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const uint8_t* __restrict__ k_cache_fp8,
    const uint8_t* __restrict__ v_cache_fp8,
    const float* __restrict__ k_scale_fp8,
    const float* __restrict__ v_scale_fp8,
    const __nv_bfloat16* __restrict__ rel_logits,
    float* __restrict__ split_scratch, // [T, n_heads, num_splits, 130]
    int start_pos, const int32_t* __restrict__ d_pos, int T, int num_splits,
    int cache_cap, bool is_local, int window_size,
    int n_heads, int n_kv_heads, int rel_extent)
{
    int kv_h = blockIdx.x;
    int s = blockIdx.y;
    int wid = threadIdx.x >> 5;   // t = 0..T-1
    int lane = threadIdx.x & 31;
    if (kv_h >= n_kv_heads || s >= num_splits) return;

    int sp = d_pos ? *d_pos : start_pos;
    int blk_k_start = is_local ? max(0, sp - window_size + 1) : 0;
    int blk_k_end = sp + T;
    int total_k = blk_k_end - blk_k_start;
    int chunk = (total_k + num_splits - 1) / num_splits;
    int s_start = blk_k_start + s * chunk;
    int s_end = min(blk_k_end, s_start + chunk);

    int t = wid;
    int q_pos = sp + t;
    int my_k_start = is_local ? max(0, q_pos - window_size + 1) : 0;
    int my_k_end = q_pos + 1;

    float qv[3][4];
    const __nv_bfloat16* r_ptr[3];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        int h = kv_h * 3 + g;
        const __nv_bfloat16* q_ptr = Q + ((int64_t)t * n_heads + h) * 128;
        r_ptr[g] = rel_logits + ((int64_t)t * n_heads + h) * rel_extent;
        #pragma unroll
        for (int i = 0; i < 4; ++i) qv[g][i] = __bfloat162float(q_ptr[lane + i * 32]);
    }
    const float inv_sqrt_d = rsqrtf(128.0f);
    int kv_stride = n_kv_heads * 128;

    float m_val[3] = {-1e30f, -1e30f, -1e30f};
    float l_val[3] = {0.0f, 0.0f, 0.0f};
    float acc[3][4];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[g][i] = 0.0f;
    }

    __shared__ __align__(16) union { uint8_t u8[32][128]; __nv_bfloat16 bf16[32][128]; } sm_k, sm_v;
    __shared__ float sm_ks[32], sm_vs[32];

    for (int k_base = s_start; k_base < s_end; k_base += 32) {
        int tile_len = min(32, s_end - k_base);
        if (k_cache_fp8) {
            int total_u4 = tile_len * 8;
            for (int idx = threadIdx.x; idx < total_u4; idx += blockDim.x) {
                int kk = idx >> 3;
                int col = idx & 7;
                int kp = k_base + kk;
                int slot = is_local ? (kp % cache_cap) : kp;
                int64_t sc_idx = (int64_t)slot * n_kv_heads + kv_h;
                reinterpret_cast<uint4*>(sm_k.u8[kk])[col] = reinterpret_cast<const uint4*>(k_cache_fp8 + sc_idx * 128)[col];
                reinterpret_cast<uint4*>(sm_v.u8[kk])[col] = reinterpret_cast<const uint4*>(v_cache_fp8 + sc_idx * 128)[col];
            }
            if (threadIdx.x < tile_len) {
                int kp = k_base + threadIdx.x;
                int slot = is_local ? (kp % cache_cap) : kp;
                int64_t sc_idx = (int64_t)slot * n_kv_heads + kv_h;
                sm_ks[threadIdx.x] = k_scale_fp8[sc_idx];
                sm_vs[threadIdx.x] = v_scale_fp8[sc_idx];
            }
        } else {
            int total_u4 = tile_len * 16;
            for (int idx = threadIdx.x; idx < total_u4; idx += blockDim.x) {
                int kk = idx >> 4;
                int col = idx & 15;
                int kp = k_base + kk;
                int slot = is_local ? (kp % cache_cap) : kp;
                int64_t base = (int64_t)slot * kv_stride + kv_h * 128;
                reinterpret_cast<uint4*>(sm_k.bf16[kk])[col] = reinterpret_cast<const uint4*>(k_cache + base)[col];
                reinterpret_cast<uint4*>(sm_v.bf16[kk])[col] = reinterpret_cast<const uint4*>(v_cache + base)[col];
            }
        }
        __syncthreads();

        for (int kk = 0; kk < tile_len; ++kk) {
            int kp = k_base + kk;
            if (kp < my_k_start || kp >= my_k_end) continue;
            float kv0, kv1, kv2, kv3;
            float vv0, vv1, vv2, vv3;
            float k_mult = inv_sqrt_d;
            if (k_cache_fp8) {
                k_mult *= sm_ks[kk];
                float vs = sm_vs[kk];
                kv0 = e4m3_to_f32(sm_k.u8[kk][lane + 0]);
                kv1 = e4m3_to_f32(sm_k.u8[kk][lane + 32]);
                kv2 = e4m3_to_f32(sm_k.u8[kk][lane + 64]);
                kv3 = e4m3_to_f32(sm_k.u8[kk][lane + 96]);
                vv0 = e4m3_to_f32(sm_v.u8[kk][lane + 0]) * vs;
                vv1 = e4m3_to_f32(sm_v.u8[kk][lane + 32]) * vs;
                vv2 = e4m3_to_f32(sm_v.u8[kk][lane + 64]) * vs;
                vv3 = e4m3_to_f32(sm_v.u8[kk][lane + 96]) * vs;
            } else {
                kv0 = __bfloat162float(sm_k.bf16[kk][lane + 0]);
                kv1 = __bfloat162float(sm_k.bf16[kk][lane + 32]);
                kv2 = __bfloat162float(sm_k.bf16[kk][lane + 64]);
                kv3 = __bfloat162float(sm_k.bf16[kk][lane + 96]);
                vv0 = __bfloat162float(sm_v.bf16[kk][lane + 0]);
                vv1 = __bfloat162float(sm_v.bf16[kk][lane + 32]);
                vv2 = __bfloat162float(sm_v.bf16[kk][lane + 64]);
                vv3 = __bfloat162float(sm_v.bf16[kk][lane + 96]);
            }
            int dist = q_pos - kp;
            #pragma unroll
            for (int g = 0; g < 3; ++g) {
                float dot = fmaf(qv[g][0], kv0, fmaf(qv[g][1], kv1, fmaf(qv[g][2], kv2, qv[g][3] * kv3)));
                dot = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0) * k_mult;
                if (dist >= 0 && dist < rel_extent) dot += __bfloat162float(r_ptr[g][dist]);
                float m_new = fmaxf(m_val[g], dot);
                float alpha = expf(m_val[g] - m_new);
                float p = expf(dot - m_new);
                l_val[g] = l_val[g] * alpha + p;
                acc[g][0] = acc[g][0] * alpha + p * vv0;
                acc[g][1] = acc[g][1] * alpha + p * vv1;
                acc[g][2] = acc[g][2] * alpha + p * vv2;
                acc[g][3] = acc[g][3] * alpha + p * vv3;
                m_val[g] = m_new;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        int h = kv_h * 3 + g;
        float* dst = split_scratch + (((int64_t)t * n_heads + h) * num_splits + s) * 130;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[lane + i * 32] = acc[g][i];
        if (lane == 0) {
            dst[128] = m_val[g];
            dst[129] = l_val[g];
        }
    }
}

__global__ void attn_small_t_gqa3_merge_kernel(
    const float* __restrict__ split_scratch,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ sinks,
    __nv_bfloat16* __restrict__ out,
    int num_splits, int n_heads)
{
    int h = blockIdx.x;
    int t = blockIdx.y;
    int lane = threadIdx.x; // 32 threads
    float sink_logit = __bfloat162float(sinks[(int64_t)t * n_heads + h]);

    float gm = sink_logit;
    for (int s = 0; s < num_splits; ++s) {
        const float* src = split_scratch + (((int64_t)t * n_heads + h) * num_splits + s) * 130;
        if (src[129] > 0.0f) gm = fmaxf(gm, src[128]);
    }
    float gl = expf(sink_logit - gm);
    float gacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int s = 0; s < num_splits; ++s) {
        const float* src = split_scratch + (((int64_t)t * n_heads + h) * num_splits + s) * 130;
        float l = src[129];
        if (l > 0.0f) {
            float sc = expf(src[128] - gm);
            gl += l * sc;
            #pragma unroll
            for (int i = 0; i < 4; ++i) gacc[i] += src[lane + i * 32] * sc;
        }
    }
    float inv_l = 1.0f / gl;
    const __nv_bfloat16* g_ptr = gate + ((int64_t)t * n_heads + h) * 128;
    __nv_bfloat16* o_ptr = out + ((int64_t)t * n_heads + h) * 128;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = lane + i * 32;
        float sig = 1.0f / (1.0f + expf(-__bfloat162float(g_ptr[d])));
        o_ptr[d] = __float2bfloat16(gacc[i] * inv_l * sig);
    }
}

// Tensor Core WMMA FlashAttention prefill kernel (BQ = 64 via 4 warps = 128 threads, BK = 32, head_dim = 128)
// Runs Q@K^T and P@V on Blackwell BF16 Tensor Cores (~50+ TFLOPS) with exact Inkling rel_logits + sinks + gate!
__global__ void attn_prefill_wmma_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const uint8_t* __restrict__ k_cache_fp8,
    const uint8_t* __restrict__ v_cache_fp8,
    const float* __restrict__ k_scale_fp8,
    const float* __restrict__ v_scale_fp8,
    const __nv_bfloat16* __restrict__ rel_logits,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ sinks,
    __nv_bfloat16* __restrict__ out,
    int start_pos, const int32_t* __restrict__ d_pos, int T, int cache_cap, bool is_local, int window_size,
    int n_heads, int n_kv_heads, int rel_extent)
{
    extern __shared__ uint8_t smem_raw[];
    // Dynamic SMEM layout (60 KiB total):
    //   sm_k: [32][128] bf16 = 8192 B
    //   sm_v: [32][128] bf16 = 8192 B
    //   sm_o: [4][16][128] fp32 = 32768 B
    //   sm_s: [4][16][32] fp32 = 8192 B
    //   sm_p: [4][16][32] bf16 = 4096 B
    //   Total = 61440 B (60 KiB)
    __nv_bfloat16* sm_k = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sm_v = sm_k + 32 * 128;
    float*         sm_o = reinterpret_cast<float*>(sm_v + 32 * 128);
    float*         sm_s = sm_o + 4 * 16 * 128;
    __nv_bfloat16* sm_p = reinterpret_cast<__nv_bfloat16*>(sm_s + 4 * 16 * 32);

    int h = blockIdx.x;
    int t_blk = blockIdx.y * 64;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int t_warp = t_blk + wid * 16;
    int kv_h = h / (n_heads / n_kv_heads);
    int kv_stride = n_kv_heads * 128;

    int sp = d_pos ? *d_pos : start_pos;
    int blk_k_start = is_local ? max(0, sp + t_blk - window_size + 1) : 0;
    int blk_k_end = sp + min(T, t_blk + 64);

    // Load Q[16, 128] for this warp into sm_k (temporarily via sm_o? No, load directly into sm_p?
    // Even cleaner: stage this warp's Q[16, 128] into sm_k/sm_v before K/V loop!
    // 4 warps * 16 rows = 64 rows * 128 bf16 = 16384 B = EXACT size of sm_k + sm_v!)
    {
        __nv_bfloat16* sm_q_all = sm_k; // [64, 128] bf16 = 16384 B
        for (int idx = threadIdx.x; idx < 64 * 16; idx += 128) {
            int r = idx >> 4;
            int d8 = idx & 15;
            int t = t_blk + r;
            if (t < T) {
                const uint4* qs = reinterpret_cast<const uint4*>(Q + ((int64_t)t * n_heads + h) * 128);
                reinterpret_cast<uint4*>(sm_q_all + r * 128)[d8] = qs[d8];
            } else {
                reinterpret_cast<uint4*>(sm_q_all + r * 128)[d8] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();
    }

    // Load Q fragments [8] (16 x 128) into registers for the entire K/V loop!
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag[8];
    const __nv_bfloat16* my_sm_q = sm_k + wid * 16 * 128;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        wmma::load_matrix_sync(q_frag[k], my_sm_q + k * 16, 128);
    }
    __syncthreads(); // Done reading Q from sm_k/sm_v

    // Initialize sm_o[wid][16][128] = 0 and per-row online softmax state m_row, l_row
    float* my_o = sm_o + wid * 16 * 128;
    float* my_s = sm_s + wid * 16 * 32;
    __nv_bfloat16* my_p = sm_p + wid * 16 * 32;
    for (int idx = lane; idx < 16 * 128; idx += 32) my_o[idx] = 0.0f;

    __shared__ float sm_m_row[4][16];
    __shared__ float sm_l_row[4][16];
    __shared__ float sm_alpha_row[4][16];
    if (lane < 16) {
        int t = t_warp + lane;
        sm_m_row[wid][lane] = (t < T) ? __bfloat162float(sinks[(int64_t)t * n_heads + h]) : -1e30f;
        sm_l_row[wid][lane] = 1.0f;
    }
    __syncwarp();

    const float inv_sqrt_d = rsqrtf(128.0f);

    for (int kb = blk_k_start; kb < blk_k_end; kb += 32) {
        int k_len = min(32, blk_k_end - kb);
        // Cooperative coalesced uint4 load of K[32, 128] and V[32, 128] into SMEM
        for (int idx = threadIdx.x; idx < 32 * 16; idx += 128) {
            int bk = idx >> 4;
            int d8 = idx & 15;
            if (bk < k_len) {
                int kp = kb + bk;
                int slot = is_local ? (kp % cache_cap) : kp;
                if (k_cache_fp8) {
                    int64_t sc_idx = (int64_t)slot * n_kv_heads + kv_h;
                    float ks = k_scale_fp8[sc_idx];
                    float vs = v_scale_fp8[sc_idx];
                    uint2 ku = reinterpret_cast<const uint2*>(k_cache_fp8 + sc_idx * 128)[d8];
                    uint2 vu = reinterpret_cast<const uint2*>(v_cache_fp8 + sc_idx * 128)[d8];
                    const __nv_fp8x2_storage_t* kp2 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&ku);
                    const __nv_fp8x2_storage_t* vp2 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&vu);
                    __nv_bfloat162* kdst2 = reinterpret_cast<__nv_bfloat162*>(sm_k + bk * 128 + d8 * 8);
                    __nv_bfloat162* vdst2 = reinterpret_cast<__nv_bfloat162*>(sm_v + bk * 128 + d8 * 8);
                    #pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        __half2_raw kh = __nv_cvt_fp8x2_to_halfraw2(kp2[p], __NV_E4M3);
                        float2 kf = __half22float2(*reinterpret_cast<const __half2*>(&kh));
                        kdst2[p] = __floats2bfloat162_rn(kf.x * ks, kf.y * ks);
                        __half2_raw vh = __nv_cvt_fp8x2_to_halfraw2(vp2[p], __NV_E4M3);
                        float2 vf = __half22float2(*reinterpret_cast<const __half2*>(&vh));
                        vdst2[p] = __floats2bfloat162_rn(vf.x * vs, vf.y * vs);
                    }
                } else {
                    const uint4* ks = reinterpret_cast<const uint4*>(k_cache + (int64_t)slot * kv_stride + kv_h * 128);
                    const uint4* vs = reinterpret_cast<const uint4*>(v_cache + (int64_t)slot * kv_stride + kv_h * 128);
                    reinterpret_cast<uint4*>(sm_k + bk * 128)[d8] = ks[d8];
                    reinterpret_cast<uint4*>(sm_v + bk * 128)[d8] = vs[d8];
                }
            } else {
                reinterpret_cast<uint4*>(sm_k + bk * 128)[d8] = make_uint4(0, 0, 0, 0);
                reinterpret_cast<uint4*>(sm_v + bk * 128)[d8] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // Skip warp compute if all 16 queries of this warp are masked out for this K tile
        int warp_max_q = sp + min(T - 1, t_warp + 15);
        int warp_min_k = is_local ? max(0, sp + t_warp - window_size + 1) : 0;
        if (t_warp < T && kb <= warp_max_q && (kb + k_len) > warp_min_k) {
            // 1. S = Q[16, 128] @ K[32, 128]^T -> my_s[16, 32] via Tensor Cores!
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
                wmma::fill_fragment(s_frag, 0.0f);
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag;
                    wmma::load_matrix_sync(k_frag, sm_k + (n * 16) * 128 + k * 16, 128);
                    wmma::mma_sync(s_frag, q_frag[k], k_frag, s_frag);
                }
                wmma::store_matrix_sync(my_s + n * 16, s_frag, 32, wmma::mem_row_major);
            }
            __syncwarp();

            // 2. Online softmax per row r = 0..15 (32 lanes handle 32 columns of row r in parallel!)
            for (int r = 0; r < 16; ++r) {
                int t = t_warp + r;
                if (t >= T) {
                    my_p[r * 32 + lane] = __float2bfloat16(0.0f);
                    if (lane == 0) sm_alpha_row[wid][r] = 1.0f;
                    continue;
                }
                int q_pos = sp + t;
                int kp = kb + lane;
                int my_k_start = is_local ? max(0, q_pos - window_size + 1) : 0;
                bool valid_k = (lane < k_len && kp >= my_k_start && kp <= q_pos);
                float val = -1e30f;
                if (valid_k) {
                    val = my_s[r * 32 + lane] * inv_sqrt_d;
                    int dist = q_pos - kp;
                    if (dist >= 0 && dist < rel_extent) {
                        val += __bfloat162float(rel_logits[((int64_t)t * n_heads + h) * rel_extent + dist]);
                    }
                }
                // Warp max over 32 columns
                float row_max = val;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, off));
                }
                float m_old = sm_m_row[wid][r];
                float m_new = fmaxf(m_old, row_max);
                float alpha = expf(m_old - m_new);
                float p = valid_k ? expf(val - m_new) : 0.0f;
                float p_sum = warp_reduce_sum(p);
                my_p[r * 32 + lane] = __float2bfloat16(p);
                if (lane == 0) {
                    sm_m_row[wid][r] = m_new;
                    sm_l_row[wid][r] = sm_l_row[wid][r] * alpha + p_sum;
                    sm_alpha_row[wid][r] = alpha;
                }
            }
            __syncwarp();

            // 3. Rescale sm_o[wid][16][128] by alpha[r]
            for (int r = 0; r < 16; ++r) {
                float a = sm_alpha_row[wid][r];
                if (a != 1.0f) {
                    float* orow = my_o + r * 128;
                    orow[lane + 0]  *= a;
                    orow[lane + 32] *= a;
                    orow[lane + 64] *= a;
                    orow[lane + 96] *= a;
                }
            }
            __syncwarp();

            // 4. O += P[16, 32] @ V[32, 128] via Tensor Cores!
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> p_frag0, p_frag1;
            wmma::load_matrix_sync(p_frag0, my_p + 0, 32);
            wmma::load_matrix_sync(p_frag1, my_p + 16, 32);
            #pragma unroll
            for (int col = 0; col < 128; col += 16) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag;
                wmma::load_matrix_sync(o_frag, my_o + col, 128, wmma::mem_row_major);
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> v_frag0, v_frag1;
                wmma::load_matrix_sync(v_frag0, sm_v + 0 * 128 + col, 128);
                wmma::load_matrix_sync(v_frag1, sm_v + 16 * 128 + col, 128);
                wmma::mma_sync(o_frag, p_frag0, v_frag0, o_frag);
                wmma::mma_sync(o_frag, p_frag1, v_frag1, o_frag);
                wmma::store_matrix_sync(my_o + col, o_frag, 128, wmma::mem_row_major);
            }
            __syncwarp();
        }
        __syncthreads();
    }

    // Finalize: out = (sm_o / l_row) * sigmoid(gate)
    for (int r = 0; r < 16; ++r) {
        int t = t_warp + r;
        if (t >= T) break;
        float inv_l = 1.0f / sm_l_row[wid][r];
        const __nv_bfloat16* g_ptr = gate + ((int64_t)t * n_heads + h) * 128;
        __nv_bfloat16* o_ptr = out + ((int64_t)t * n_heads + h) * 128;
        const float* orow = my_o + r * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int d = lane + i * 32;
            float sig = 1.0f / (1.0f + expf(-__bfloat162float(g_ptr[d])));
            o_ptr[d] = __float2bfloat16(orow[d] * inv_l * sig);
        }
    }
}

// GQA=3 Shared-KV BK=64 Tensor Core WMMA FlashAttention prefill kernel
// 1 block = 3 warps (96 threads) = 3 GQA Q-heads x BQ=16 tokens, BK=64 K/V tile in SMEM!
// 3x less HBM K/V traffic (shared across GQA=3) + 2x fewer sm_o load/stores & syncthreads (BK=64)!
__global__ void attn_prefill_wmma_gqa3_bk64_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const uint8_t* __restrict__ k_cache_fp8,
    const uint8_t* __restrict__ v_cache_fp8,
    const float* __restrict__ k_scale_fp8,
    const float* __restrict__ v_scale_fp8,
    const __nv_bfloat16* __restrict__ rel_logits,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ sinks,
    __nv_bfloat16* __restrict__ out,
    int start_pos, const int32_t* __restrict__ d_pos, int T, int cache_cap, bool is_local, int window_size,
    int n_heads, int n_kv_heads, int rel_extent)
{
    extern __shared__ uint8_t smem_raw[];
    // Dynamic SMEM layout (74 KiB total):
    //   sm_k: [64][128] bf16 = 16384 B
    //   sm_v: [64][128] bf16 = 16384 B
    //   sm_o: [3][16][128] fp32 = 24576 B
    //   sm_s: [3][16][64] fp32 = 12288 B
    //   sm_p: [3][16][64] bf16 = 6144 B
    //   Total = 75776 B (74 KiB)
    __nv_bfloat16* sm_k = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sm_v = sm_k + 64 * 128;
    float*         sm_o = reinterpret_cast<float*>(sm_v + 64 * 128);
    float*         sm_s = sm_o + 3 * 16 * 128;
    __nv_bfloat16* sm_p = reinterpret_cast<__nv_bfloat16*>(sm_s + 3 * 16 * 64);

    int kv_h = blockIdx.x;
    int t_blk = blockIdx.y * 16;
    int wid = threadIdx.x >> 5;   // 0, 1, 2 -> GQA sub-head g!
    int lane = threadIdx.x & 31;
    int h = kv_h * 3 + wid;
    int kv_stride = n_kv_heads * 128;

    int sp = d_pos ? *d_pos : start_pos;
    int blk_k_start = is_local ? max(0, sp + t_blk - window_size + 1) : 0;
    int blk_k_end = sp + min(T, t_blk + 16);

    // Stage Q[3][16][128] into sm_k (48*128 bf16 = 12288 B < 16384 B)
    {
        for (int idx = threadIdx.x; idx < 3 * 16 * 16; idx += 96) {
            int g = idx >> 8;           // 0..2
            int r = (idx >> 4) & 15;    // 0..15
            int d8 = idx & 15;          // 0..15
            int t = t_blk + r;
            __nv_bfloat16* dst = sm_k + (g * 16 + r) * 128;
            if (t < T) {
                const uint4* qs = reinterpret_cast<const uint4*>(Q + ((int64_t)t * n_heads + (kv_h * 3 + g)) * 128);
                reinterpret_cast<uint4*>(dst)[d8] = qs[d8];
            } else {
                reinterpret_cast<uint4*>(dst)[d8] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();
    }

    // Load this warp's Q[16, 128] into WMMA fragments for the entire K/V loop!
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag[8];
    const __nv_bfloat16* my_sm_q = sm_k + wid * 16 * 128;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        wmma::load_matrix_sync(q_frag[k], my_sm_q + k * 16, 128);
    }
    __syncthreads();

    float* my_o = sm_o + wid * 16 * 128;
    float* my_s = sm_s + wid * 16 * 64;
    __nv_bfloat16* my_p = sm_p + wid * 16 * 64;
    for (int idx = lane; idx < 16 * 128; idx += 32) my_o[idx] = 0.0f;

    __shared__ float sm_m_row[3][16];
    __shared__ float sm_l_row[3][16];
    __shared__ float sm_alpha_row[3][16];
    if (lane < 16) {
        int t = t_blk + lane;
        sm_m_row[wid][lane] = (t < T) ? __bfloat162float(sinks[(int64_t)t * n_heads + h]) : -1e30f;
        sm_l_row[wid][lane] = 1.0f;
    }
    __syncwarp();

    const float inv_sqrt_d = rsqrtf(128.0f);

    for (int kb = blk_k_start; kb < blk_k_end; kb += 64) {
        int k_len = min(64, blk_k_end - kb);
        // Cooperative load of K[64, 128] & V[64, 128] ONCE for all 3 GQA Q-heads!
        for (int idx = threadIdx.x; idx < 64 * 16; idx += 96) {
            int bk = idx >> 4;
            int d8 = idx & 15;
            if (bk < k_len) {
                int kp = kb + bk;
                int slot = is_local ? (kp % cache_cap) : kp;
                if (k_cache_fp8) {
                    int64_t sc_idx = (int64_t)slot * n_kv_heads + kv_h;
                    float ks = k_scale_fp8[sc_idx];
                    float vs = v_scale_fp8[sc_idx];
                    uint2 ku = reinterpret_cast<const uint2*>(k_cache_fp8 + sc_idx * 128)[d8];
                    uint2 vu = reinterpret_cast<const uint2*>(v_cache_fp8 + sc_idx * 128)[d8];
                    const __nv_fp8x2_storage_t* kp2 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&ku);
                    const __nv_fp8x2_storage_t* vp2 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&vu);
                    __nv_bfloat162* kdst2 = reinterpret_cast<__nv_bfloat162*>(sm_k + bk * 128 + d8 * 8);
                    __nv_bfloat162* vdst2 = reinterpret_cast<__nv_bfloat162*>(sm_v + bk * 128 + d8 * 8);
                    #pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        __half2_raw kh = __nv_cvt_fp8x2_to_halfraw2(kp2[p], __NV_E4M3);
                        float2 kf = __half22float2(*reinterpret_cast<const __half2*>(&kh));
                        kdst2[p] = __floats2bfloat162_rn(kf.x * ks, kf.y * ks);
                        __half2_raw vh = __nv_cvt_fp8x2_to_halfraw2(vp2[p], __NV_E4M3);
                        float2 vf = __half22float2(*reinterpret_cast<const __half2*>(&vh));
                        vdst2[p] = __floats2bfloat162_rn(vf.x * vs, vf.y * vs);
                    }
                } else {
                    const uint4* ks = reinterpret_cast<const uint4*>(k_cache + (int64_t)slot * kv_stride + kv_h * 128);
                    const uint4* vs = reinterpret_cast<const uint4*>(v_cache + (int64_t)slot * kv_stride + kv_h * 128);
                    reinterpret_cast<uint4*>(sm_k + bk * 128)[d8] = ks[d8];
                    reinterpret_cast<uint4*>(sm_v + bk * 128)[d8] = vs[d8];
                }
            } else {
                reinterpret_cast<uint4*>(sm_k + bk * 128)[d8] = make_uint4(0, 0, 0, 0);
                reinterpret_cast<uint4*>(sm_v + bk * 128)[d8] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // 1. S = Q[16, 128] @ K[64, 128]^T -> my_s[16, 64] via Tensor Cores!
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
            wmma::fill_fragment(s_frag, 0.0f);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag;
                wmma::load_matrix_sync(k_frag, sm_k + (n * 16) * 128 + k * 16, 128);
                wmma::mma_sync(s_frag, q_frag[k], k_frag, s_frag);
            }
            wmma::store_matrix_sync(my_s + n * 16, s_frag, 64, wmma::mem_row_major);
        }
        __syncwarp();

        // 2. Online softmax per row r = 0..15 (32 lanes x 2 cols = 64 cols!)
        for (int r = 0; r < 16; ++r) {
            int t = t_blk + r;
            if (t >= T) {
                my_p[r * 64 + lane]      = __float2bfloat16(0.0f);
                my_p[r * 64 + lane + 32] = __float2bfloat16(0.0f);
                if (lane == 0) sm_alpha_row[wid][r] = 1.0f;
                continue;
            }
            int q_pos = sp + t;
            int my_k_start = is_local ? max(0, q_pos - window_size + 1) : 0;
            const __nv_bfloat16* rel_row = rel_logits + ((int64_t)t * n_heads + h) * rel_extent;

            int kp0 = kb + lane;
            int kp1 = kb + lane + 32;
            bool v0 = (lane < k_len && kp0 >= my_k_start && kp0 <= q_pos);
            bool v1 = ((lane + 32) < k_len && kp1 >= my_k_start && kp1 <= q_pos);
            float val0 = -1e30f, val1 = -1e30f;
            if (v0) {
                val0 = my_s[r * 64 + lane] * inv_sqrt_d;
                int d0 = q_pos - kp0;
                if (d0 >= 0 && d0 < rel_extent) val0 += __bfloat162float(rel_row[d0]);
            }
            if (v1) {
                val1 = my_s[r * 64 + lane + 32] * inv_sqrt_d;
                int d1 = q_pos - kp1;
                if (d1 >= 0 && d1 < rel_extent) val1 += __bfloat162float(rel_row[d1]);
            }
            float row_max = fmaxf(val0, val1);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, off));
            }
            float m_old = sm_m_row[wid][r];
            float m_new = fmaxf(m_old, row_max);
            float alpha = expf(m_old - m_new);
            float p0 = v0 ? expf(val0 - m_new) : 0.0f;
            float p1 = v1 ? expf(val1 - m_new) : 0.0f;
            float p_sum = warp_reduce_sum(p0 + p1);
            my_p[r * 64 + lane]      = __float2bfloat16(p0);
            my_p[r * 64 + lane + 32] = __float2bfloat16(p1);
            if (lane == 0) {
                sm_m_row[wid][r] = m_new;
                sm_l_row[wid][r] = sm_l_row[wid][r] * alpha + p_sum;
                sm_alpha_row[wid][r] = alpha;
            }
        }
        __syncwarp();

        // 3. Rescale sm_o[wid][16][128] by alpha[r]
        for (int r = 0; r < 16; ++r) {
            float a = sm_alpha_row[wid][r];
            if (a != 1.0f) {
                float* orow = my_o + r * 128;
                orow[lane + 0]  *= a;
                orow[lane + 32] *= a;
                orow[lane + 64] *= a;
                orow[lane + 96] *= a;
            }
        }
        __syncwarp();

        // 4. O += P[16, 64] @ V[64, 128] via Tensor Cores (4 mma_sync per load/store!)
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> p_frag[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) wmma::load_matrix_sync(p_frag[k], my_p + k * 16, 64);
        #pragma unroll
        for (int col = 0; col < 128; col += 16) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag;
            wmma::load_matrix_sync(o_frag, my_o + col, 128, wmma::mem_row_major);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> v_frag;
                wmma::load_matrix_sync(v_frag, sm_v + (k * 16) * 128 + col, 128);
                wmma::mma_sync(o_frag, p_frag[k], v_frag, o_frag);
            }
            wmma::store_matrix_sync(my_o + col, o_frag, 128, wmma::mem_row_major);
        }
        __syncwarp();
        __syncthreads();
    }

    // Finalize: out = (sm_o / l_row) * sigmoid(gate)
    for (int r = 0; r < 16; ++r) {
        int t = t_blk + r;
        if (t >= T) break;
        float inv_l = 1.0f / sm_l_row[wid][r];
        const __nv_bfloat16* g_ptr = gate + ((int64_t)t * n_heads + h) * 128;
        __nv_bfloat16* o_ptr = out + ((int64_t)t * n_heads + h) * 128;
        const float* orow = my_o + r * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int d = lane + i * 32;
            float sig = 1.0f / (1.0f + expf(-__bfloat162float(g_ptr[d])));
            o_ptr[d] = __float2bfloat16(orow[d] * inv_l * sig);
        }
    }
}

void launch_attention(
    const __nv_bfloat16* Q, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
    const __nv_bfloat16* rel_logits, const __nv_bfloat16* gate, const __nv_bfloat16* sinks,
    __nv_bfloat16* out, float* split_scratch,
    int start_pos, const int32_t* d_pos, int T, int cache_cap, bool is_local, int window_size,
    int n_heads, int n_kv_heads, int head_dim, int rel_extent, cudaStream_t stream,
    const uint8_t* k_cache_fp8, const uint8_t* v_cache_fp8,
    const float* k_scale_fp8, const float* v_scale_fp8)
{
    if (T == 1) {
        int num_splits = d_pos ? (is_local ? 4 : 8) : 1;
        if (!d_pos) {
            int k_start = is_local ? max(0, start_pos - window_size + 1) : 0;
            int total_k = (start_pos + 1) - k_start;
            if (total_k > 128) num_splits = min(16, (total_k + 63) / 64);
        }
        attn_decode_split_kernel<<<dim3(n_heads, num_splits), 128, 0, stream>>>(
            Q, k_cache, v_cache, k_cache_fp8, v_cache_fp8, k_scale_fp8, v_scale_fp8,
            rel_logits, split_scratch,
            start_pos, d_pos, num_splits, cache_cap, is_local, window_size,
            n_heads, n_kv_heads, head_dim, rel_extent);
        attn_decode_merge_kernel<<<n_heads, 32, 0, stream>>>(
            split_scratch, gate, sinks, out, num_splits, n_heads, head_dim);
    } else if (T >= 16 && head_dim == 128 && n_heads == n_kv_heads * 3) {
        static bool smem_set_gqa3 = false;
        const size_t smem_bytes = 75776;
        if (!smem_set_gqa3) {
            cudaFuncSetAttribute(attn_prefill_wmma_gqa3_bk64_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
            smem_set_gqa3 = true;
        }
        attn_prefill_wmma_gqa3_bk64_kernel<<<dim3(n_kv_heads, (T + 15) / 16), 96, smem_bytes, stream>>>(
            Q, k_cache, v_cache, k_cache_fp8, v_cache_fp8, k_scale_fp8, v_scale_fp8,
            rel_logits, gate, sinks, out,
            start_pos, d_pos, T, cache_cap, is_local, window_size,
            n_heads, n_kv_heads, rel_extent);
    } else if (T >= 16 && head_dim == 128) {
        static bool smem_set = false;
        const size_t smem_bytes = 61440;
        if (!smem_set) {
            cudaFuncSetAttribute(attn_prefill_wmma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
            smem_set = true;
        }
        attn_prefill_wmma_kernel<<<dim3(n_heads, (T + 63) / 64), 128, smem_bytes, stream>>>(
            Q, k_cache, v_cache, k_cache_fp8, v_cache_fp8, k_scale_fp8, v_scale_fp8,
            rel_logits, gate, sinks, out,
            start_pos, d_pos, T, cache_cap, is_local, window_size,
            n_heads, n_kv_heads, rel_extent);
    } else if (T >= 2 && T <= 15 && head_dim == 128 && n_heads == n_kv_heads * 3 && split_scratch != nullptr) {
        int num_splits = is_local ? 4 : 12;
        attn_small_t_gqa3_split_kernel<<<dim3(n_kv_heads, num_splits), T * 32, 0, stream>>>(
            Q, k_cache, v_cache, k_cache_fp8, v_cache_fp8, k_scale_fp8, v_scale_fp8,
            rel_logits, split_scratch,
            start_pos, d_pos, T, num_splits, cache_cap, is_local, window_size,
            n_heads, n_kv_heads, rel_extent);
        attn_small_t_gqa3_merge_kernel<<<dim3(n_heads, T), 32, 0, stream>>>(
            split_scratch, gate, sinks, out, num_splits, n_heads);
    } else {
        attn_small_t_multiwarp_kernel<<<dim3(n_heads, T), 256, 0, stream>>>(
            Q, k_cache, v_cache, k_cache_fp8, v_cache_fp8, k_scale_fp8, v_scale_fp8,
            rel_logits, gate, sinks, out,
            start_pos, d_pos, T, cache_cap, is_local, window_size,
            n_heads, n_kv_heads, head_dim, rel_extent);
    }
}

// ============================================================================
// 7. MoE Router (E=128, topK=8, multi-SM uint4 GEMV + topK)
// ============================================================================

// grid = (E/4, T), block = 128 (4 warps, 1 warp per expert, uint4 vectorized)
__global__ void moe_router_logits_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight, // [E, D]
    float* __restrict__ logits,               // [T, E]
    int T, int D, int E)
{
    int t = blockIdx.y;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int e = blockIdx.x * 4 + wid;
    if (e >= E || t >= T) return;

    const uint4* w4 = reinterpret_cast<const uint4*>(weight + (int64_t)e * D);
    const uint4* x4 = reinterpret_cast<const uint4*>(x + (int64_t)t * D);
    int k8 = D >> 3;
    float acc = 0.0f;
    for (int i = lane; i < k8; i += 32) {
        uint4 wv = w4[i];
        uint4 xv = x4[i];
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wv);
        const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 wf = __bfloat1622float2(wp[j]);
            float2 xf = __bfloat1622float2(xp[j]);
            acc = fmaf(wf.x, xf.x, acc);
            acc = fmaf(wf.y, xf.y, acc);
        }
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) logits[(int64_t)t * E + e] = acc;
}

__global__ void moe_router_topk_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int32_t* __restrict__ topk_ids,
    float* __restrict__ topk_weights,
    int T, int E, int K, float route_scale)
{
    int t = blockIdx.x;
    if (t >= T) return;
    int e = threadIdx.x;
    __shared__ float s_scores[128];
    __shared__ float s_sel[128];
    if (e < E) {
        float sig = 1.0f / (1.0f + expf(-logits[(int64_t)t * E + e]));
        s_scores[e] = sig;
        s_sel[e] = sig + bias[e];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int32_t ids[8];
        float wts[8];
        float wsum = 0.0f;
        for (int k = 0; k < K; ++k) {
            int best_e = -1;
            float best_v = -1e30f;
            for (int i = 0; i < E; ++i) {
                if (s_sel[i] > best_v) { best_v = s_sel[i]; best_e = i; }
            }
            ids[k] = best_e;
            wts[k] = s_scores[best_e];
            wsum += wts[k];
            s_sel[best_e] = -1e30f;
        }
        float inv_sum = route_scale / (wsum + 1e-20f);
        for (int k = 0; k < K; ++k) {
            topk_ids[(int64_t)t * K + k] = ids[k];
            topk_weights[(int64_t)t * K + k] = wts[k] * inv_sum;
        }
    }
}

void launch_moe_router(
    const __nv_bfloat16* x, const __nv_bfloat16* weight, const float* bias,
    float* logits_scratch, int32_t* topk_ids, float* topk_weights,
    int T, int D, int E, int K, float route_scale, cudaStream_t stream)
{
    moe_router_logits_kernel<<<dim3((E + 3) / 4, T), 128, 0, stream>>>(x, weight, logits_scratch, T, D, E);
    moe_router_topk_kernel<<<T, 128, 0, stream>>>(logits_scratch, bias, topk_ids, topk_weights, T, E, K, route_scale);
}

// ============================================================================
// 8. MoE NVFP4 Kernels (Blackwell cvt.rn.f16x2.e2m1x2 + uint4 loads)
// ============================================================================

__global__ void moe_nvfp4_w13_kernel(
    const __nv_bfloat16* __restrict__ x,        // [T, D]
    const int32_t* __restrict__ topk_ids,       // [T, K]
    const float* __restrict__ topk_weights,     // [T, K]
    const uint8_t* __restrict__ w13_u8,         // [E, 2*inter, D/2]
    const uint8_t* __restrict__ w13_scale,      // [E, 2*inter, D/16]
    const float* __restrict__ w13_amax,         // [E]
    const float* __restrict__ w13_denom,        // [1]
    float* __restrict__ act_out,                // [T * K, inter]
    int T, int D, int inter, int K, float gate_cap, float up_cap)
{
    int tk = blockIdx.y;
    int t = tk / K;
    int expert = topk_ids[tk];
    float gamma = topk_weights[tk];

    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int j = blockIdx.x * 4 + wid;
    if (j >= inter) return;

    float gscale = w13_amax[expert] / w13_denom[0];
    int row_g = 2 * j;
    int row_u = 2 * j + 1;
    int64_t half_D = D / 2;
    int64_t scale_D = D / 16;

    const uint8_t* wg = w13_u8 + ((int64_t)expert * (2 * inter) + row_g) * half_D;
    const uint8_t* wu = w13_u8 + ((int64_t)expert * (2 * inter) + row_u) * half_D;
    const uint8_t* sg = w13_scale + ((int64_t)expert * (2 * inter) + row_g) * scale_D;
    const uint8_t* su = w13_scale + ((int64_t)expert * (2 * inter) + row_u) * scale_D;
    const uint4* xr4 = reinterpret_cast<const uint4*>(x + (int64_t)t * D);

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (int blk = lane; blk < scale_D; blk += 32) {
        float bsg = e4m3_to_f32(sg[blk]);
        float bsu = e4m3_to_f32(su[blk]);
        uint2 bg = *reinterpret_cast<const uint2*>(wg + blk * 8);
        uint2 bu = *reinterpret_cast<const uint2*>(wu + blk * 8);
        uint4 x0 = xr4[blk * 2 + 0];
        uint4 x1 = xr4[blk * 2 + 1];
        const __nv_bfloat162* xp0 = reinterpret_cast<const __nv_bfloat162*>(&x0);
        const __nv_bfloat162* xp1 = reinterpret_cast<const __nv_bfloat162*>(&x1);

        float lg = 0.0f, lu = 0.0f;
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            float2 xf = __bfloat1622float2(xp0[b]);
            float2 gf = e2m1x2_to_f32x2((bg.x >> (b * 8)) & 0xFF);
            float2 uf = e2m1x2_to_f32x2((bu.x >> (b * 8)) & 0xFF);
            lg = fmaf(gf.x, xf.x, fmaf(gf.y, xf.y, lg));
            lu = fmaf(uf.x, xf.x, fmaf(uf.y, xf.y, lu));
        }
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            float2 xf = __bfloat1622float2(xp1[b]);
            float2 gf = e2m1x2_to_f32x2((bg.y >> (b * 8)) & 0xFF);
            float2 uf = e2m1x2_to_f32x2((bu.y >> (b * 8)) & 0xFF);
            lg = fmaf(gf.x, xf.x, fmaf(gf.y, xf.y, lg));
            lu = fmaf(uf.x, xf.x, fmaf(uf.y, xf.y, lu));
        }
        acc_g = fmaf(lg, bsg, acc_g);
        acc_u = fmaf(lu, bsu, acc_u);
    }
    acc_g = warp_reduce_sum(acc_g) * gscale;
    acc_u = warp_reduce_sum(acc_u) * gscale;

    if (lane == 0) {
        float sig = 1.0f / (1.0f + expf(-acc_g));
        float g_cap = tanhf(acc_g / gate_cap) * gate_cap;
        float u_cap = tanhf(acc_u / up_cap) * up_cap;
        float y = (g_cap * sig * u_cap) * gamma;
        act_out[(int64_t)tk * inter + j] = __bfloat162float(__float2bfloat16(y));
    }
}

__global__ void moe_nvfp4_w2_kernel(
    const float* __restrict__ act_in,       // [T * K, inter]
    const int32_t* __restrict__ topk_ids,   // [T, K]
    const uint8_t* __restrict__ w2_u8,      // [E, D, inter/2]
    const uint8_t* __restrict__ w2_scale,   // [E, D, inter/16]
    const float* __restrict__ w2_amax,      // [E]
    const float* __restrict__ w2_denom,     // [1]
    __nv_bfloat16* __restrict__ out,        // [T, D]
    int T, int D, int inter, int K)
{
    int t = blockIdx.y;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int d = blockIdx.x * 4 + wid;
    if (d >= D) return;

    float inv_denom = 1.0f / w2_denom[0];
    int64_t half_I = inter / 2;
    int64_t scale_I = inter / 16;

    float total = 0.0f;
    for (int k = 0; k < K; ++k) {
        int tk = t * K + k;
        int expert = topk_ids[tk];
        float gscale = w2_amax[expert] * inv_denom;
        const uint8_t* w2r = w2_u8 + ((int64_t)expert * D + d) * half_I;
        const uint8_t* s2r = w2_scale + ((int64_t)expert * D + d) * scale_I;
        const float4* ar4 = reinterpret_cast<const float4*>(act_in + (int64_t)tk * inter);

        float acc = 0.0f;
        for (int blk = lane; blk < scale_I; blk += 32) {
            float bs = e4m3_to_f32(s2r[blk]);
            uint2 bw = *reinterpret_cast<const uint2*>(w2r + blk * 8);
            float4 a0 = ar4[blk * 4 + 0];
            float4 a1 = ar4[blk * 4 + 1];
            float4 a2 = ar4[blk * 4 + 2];
            float4 a3 = ar4[blk * 4 + 3];
            float loc = 0.0f;
            float2 f0 = e2m1x2_to_f32x2(bw.x & 0xFF);
            float2 f1 = e2m1x2_to_f32x2((bw.x >> 8) & 0xFF);
            float2 f2 = e2m1x2_to_f32x2((bw.x >> 16) & 0xFF);
            float2 f3 = e2m1x2_to_f32x2((bw.x >> 24) & 0xFF);
            float2 f4 = e2m1x2_to_f32x2(bw.y & 0xFF);
            float2 f5 = e2m1x2_to_f32x2((bw.y >> 8) & 0xFF);
            float2 f6 = e2m1x2_to_f32x2((bw.y >> 16) & 0xFF);
            float2 f7 = e2m1x2_to_f32x2((bw.y >> 24) & 0xFF);
            loc = fmaf(f0.x, a0.x, fmaf(f0.y, a0.y, fmaf(f1.x, a0.z, fmaf(f1.y, a0.w, loc))));
            loc = fmaf(f2.x, a1.x, fmaf(f2.y, a1.y, fmaf(f3.x, a1.z, fmaf(f3.y, a1.w, loc))));
            loc = fmaf(f4.x, a2.x, fmaf(f4.y, a2.y, fmaf(f5.x, a2.z, fmaf(f5.y, a2.w, loc))));
            loc = fmaf(f6.x, a3.x, fmaf(f6.y, a3.y, fmaf(f7.x, a3.z, fmaf(f7.y, a3.w, loc))));
            acc = fmaf(loc, bs, acc);
        }
        acc = warp_reduce_sum(acc) * gscale;
        if (lane == 0) total += acc;
    }
    if (lane == 0) {
        out[(int64_t)t * D + d] = __float2bfloat16(total);
    }
}

void launch_moe_nvfp4(
    const __nv_bfloat16* x, const int32_t* topk_ids, const float* topk_weights,
    const uint8_t* w13_u8, const uint8_t* w13_scale_e4m3, const float* w13_amax, const float* w13_denom,
    const uint8_t* w2_u8, const uint8_t* w2_scale_e4m3, const float* w2_amax, const float* w2_denom,
    float* act_scratch, __nv_bfloat16* out,
    int T, int D, int inter, int K, float gate_cap, float up_cap, cudaStream_t stream)
{
    moe_nvfp4_w13_kernel<<<dim3((inter + 3) / 4, T * K), 128, 0, stream>>>(
        x, topk_ids, topk_weights, w13_u8, w13_scale_e4m3, w13_amax, w13_denom,
        act_scratch, T, D, inter, K, gate_cap, up_cap);
    moe_nvfp4_w2_kernel<<<dim3((D + 3) / 4, T), 128, 0, stream>>>(
        act_scratch, topk_ids, w2_u8, w2_scale_e4m3, w2_amax, w2_denom,
        out, T, D, inter, K);
}

// ============================================================================
// 9. MoE MXFP8 Kernels (L20-L23, uint4 coalesced loads)
// ============================================================================

__global__ void moe_mxfp8_w13_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    const uint8_t* __restrict__ w13_e4m3,      // [E, 2*inter, D]
    const uint8_t* __restrict__ w13_scale_u8,  // [E, 2*inter, D/32]
    float* __restrict__ act_out,               // [T * K, inter]
    int T, int D, int inter, int K, float gate_cap, float up_cap)
{
    int tk = blockIdx.y;
    int t = tk / K;
    int expert = topk_ids[tk];
    float gamma = topk_weights[tk];

    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int j = blockIdx.x * 4 + wid;
    if (j >= inter) return;

    int row_g = 2 * j;
    int row_u = 2 * j + 1;
    int64_t scale_D = D / 32;

    const uint4* wg4 = reinterpret_cast<const uint4*>(w13_e4m3 + ((int64_t)expert * (2 * inter) + row_g) * D);
    const uint4* wu4 = reinterpret_cast<const uint4*>(w13_e4m3 + ((int64_t)expert * (2 * inter) + row_u) * D);
    const uint8_t* sg = w13_scale_u8 + ((int64_t)expert * (2 * inter) + row_g) * scale_D;
    const uint8_t* su = w13_scale_u8 + ((int64_t)expert * (2 * inter) + row_u) * scale_D;
    const uint4* xr4 = reinterpret_cast<const uint4*>(x + (int64_t)t * D);

    float acc_g = 0.0f, acc_u = 0.0f;
    for (int blk = lane; blk < scale_D; blk += 32) {
        float bsg = ue8m0_to_f32(sg[blk]);
        float bsu = ue8m0_to_f32(su[blk]);
        float lg = 0.0f, lu = 0.0f;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint4 bg = wg4[blk * 2 + h];
            uint4 bu = wu4[blk * 2 + h];
            uint4 xv0 = xr4[blk * 4 + h * 2 + 0];
            uint4 xv1 = xr4[blk * 4 + h * 2 + 1];
            const uint8_t* gbytes = reinterpret_cast<const uint8_t*>(&bg);
            const uint8_t* ubytes = reinterpret_cast<const uint8_t*>(&bu);
            const __nv_bfloat16* x0 = reinterpret_cast<const __nv_bfloat16*>(&xv0);
            const __nv_bfloat16* x1 = reinterpret_cast<const __nv_bfloat16*>(&xv1);
            #pragma unroll
            for (int n = 0; n < 8; ++n) {
                float xf = __bfloat162float(x0[n]);
                lg = fmaf(e4m3_to_f32(gbytes[n]), xf, lg);
                lu = fmaf(e4m3_to_f32(ubytes[n]), xf, lu);
            }
            #pragma unroll
            for (int n = 0; n < 8; ++n) {
                float xf = __bfloat162float(x1[n]);
                lg = fmaf(e4m3_to_f32(gbytes[8 + n]), xf, lg);
                lu = fmaf(e4m3_to_f32(ubytes[8 + n]), xf, lu);
            }
        }
        acc_g = fmaf(lg, bsg, acc_g);
        acc_u = fmaf(lu, bsu, acc_u);
    }
    acc_g = warp_reduce_sum(acc_g);
    acc_u = warp_reduce_sum(acc_u);
    if (lane == 0) {
        float sig = 1.0f / (1.0f + expf(-acc_g));
        float g_cap = tanhf(acc_g / gate_cap) * gate_cap;
        float u_cap = tanhf(acc_u / up_cap) * up_cap;
        float y = (g_cap * sig * u_cap) * gamma;
        act_out[(int64_t)tk * inter + j] = __bfloat162float(__float2bfloat16(y));
    }
}

__global__ void moe_mxfp8_w2_kernel(
    const float* __restrict__ act_in,
    const int32_t* __restrict__ topk_ids,
    const uint8_t* __restrict__ w2_e4m3,       // [E, D, inter]
    const uint8_t* __restrict__ w2_scale_u8,   // [E, D, inter/32]
    __nv_bfloat16* __restrict__ out,
    int T, int D, int inter, int K)
{
    int t = blockIdx.y;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int d = blockIdx.x * 4 + wid;
    if (d >= D) return;

    int64_t scale_I = inter / 32;
    float total = 0.0f;
    for (int k = 0; k < K; ++k) {
        int tk = t * K + k;
        int expert = topk_ids[tk];
        const uint4* w2_4 = reinterpret_cast<const uint4*>(w2_e4m3 + ((int64_t)expert * D + d) * inter);
        const uint8_t* s2r = w2_scale_u8 + ((int64_t)expert * D + d) * scale_I;
        const float4* ar4 = reinterpret_cast<const float4*>(act_in + (int64_t)tk * inter);
        float acc = 0.0f;
        for (int blk = lane; blk < scale_I; blk += 32) {
            float bs = ue8m0_to_f32(s2r[blk]);
            float loc = 0.0f;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint4 bw = w2_4[blk * 2 + h];
                const uint8_t* wb = reinterpret_cast<const uint8_t*>(&bw);
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    float4 av = ar4[blk * 8 + h * 4 + q];
                    loc = fmaf(e4m3_to_f32(wb[q * 4 + 0]), av.x, loc);
                    loc = fmaf(e4m3_to_f32(wb[q * 4 + 1]), av.y, loc);
                    loc = fmaf(e4m3_to_f32(wb[q * 4 + 2]), av.z, loc);
                    loc = fmaf(e4m3_to_f32(wb[q * 4 + 3]), av.w, loc);
                }
            }
            acc = fmaf(loc, bs, acc);
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[(int64_t)t * D + d] = __float2bfloat16(total);
}

void launch_moe_mxfp8(
    const __nv_bfloat16* x, const int32_t* topk_ids, const float* topk_weights,
    const uint8_t* w13_e4m3, const uint8_t* w13_scale_ue8m0,
    const uint8_t* w2_e4m3, const uint8_t* w2_scale_ue8m0,
    float* act_scratch, __nv_bfloat16* out,
    int T, int D, int inter, int K, float gate_cap, float up_cap, cudaStream_t stream)
{
    moe_mxfp8_w13_kernel<<<dim3((inter + 3) / 4, T * K), 128, 0, stream>>>(
        x, topk_ids, topk_weights, w13_e4m3, w13_scale_ue8m0,
        act_scratch, T, D, inter, K, gate_cap, up_cap);
    moe_mxfp8_w2_kernel<<<dim3((D + 3) / 4, T), 128, 0, stream>>>(
        act_scratch, topk_ids, w2_e4m3, w2_scale_ue8m0,
        out, T, D, inter, K);
}

// ============================================================================
// 9a. Small-Batch (T = 2..16) GPU-Deduplicated MoE (Zero Host Sync!)
// ============================================================================

__global__ void build_small_batch_expert_map_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ uniq_scratch, // [0]=num_unique, [1 + u*18] = {expert, count, tk0..tk15}
    float* __restrict__ out_fp32,
    int T, int K, int out_len)
{
    for (int i = threadIdx.x; i < out_len; i += blockDim.x) {
        out_fp32[i] = 0.0f;
    }
    if (threadIdx.x != 0) return;
    int num_u = 0;
    int total = T * K;
    for (int tk = 0; tk < total; ++tk) {
        int e = topk_ids[tk];
        int found = -1;
        for (int u = 0; u < num_u; ++u) {
            if (uniq_scratch[1 + u * 18] == e) { found = u; break; }
        }
        if (found < 0) {
            found = num_u++;
            uniq_scratch[1 + found * 18] = e;
            uniq_scratch[1 + found * 18 + 1] = 0;
        }
        int cnt = uniq_scratch[1 + found * 18 + 1];
        if (cnt < 16) {
            uniq_scratch[1 + found * 18 + 2 + cnt] = tk;
            uniq_scratch[1 + found * 18 + 1] = cnt + 1;
        }
    }
    uniq_scratch[0] = num_u;
}

template <int M>
__device__ __forceinline__ void moe_nvfp4_w13_fixed_m(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ topk_weights,
    const uint8_t* __restrict__ wg,
    const uint8_t* __restrict__ wu,
    const uint8_t* __restrict__ sg,
    const uint8_t* __restrict__ su,
    const int32_t* __restrict__ entry,
    float* __restrict__ act_out,
    int j, int lane, int64_t scale_D, int D, int inter, int K,
    float gscale, float gate_cap, float up_cap)
{
    int tks[M];
    const uint4* xr4[M];
    float acc_g[M];
    float acc_u[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        tks[m] = entry[2 + m];
        xr4[m] = reinterpret_cast<const uint4*>(x + (int64_t)(tks[m] / K) * D);
        acc_g[m] = 0.0f;
        acc_u[m] = 0.0f;
    }

    for (int blk = lane; blk < scale_D; blk += 32) {
        float bsg = e4m3_to_f32(sg[blk]);
        float bsu = e4m3_to_f32(su[blk]);
        uint2 bg = *reinterpret_cast<const uint2*>(wg + blk * 8);
        uint2 bu = *reinterpret_cast<const uint2*>(wu + blk * 8);

        float2 gf[8], uf[8];
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            gf[b]     = e2m1x2_to_f32x2((bg.x >> (b * 8)) & 0xFF);
            uf[b]     = e2m1x2_to_f32x2((bu.x >> (b * 8)) & 0xFF);
            gf[4 + b] = e2m1x2_to_f32x2((bg.y >> (b * 8)) & 0xFF);
            uf[4 + b] = e2m1x2_to_f32x2((bu.y >> (b * 8)) & 0xFF);
        }

        #pragma unroll
        for (int m = 0; m < M; ++m) {
            uint4 x0 = xr4[m][blk * 2 + 0];
            uint4 x1 = xr4[m][blk * 2 + 1];
            const __nv_bfloat162* xp0 = reinterpret_cast<const __nv_bfloat162*>(&x0);
            const __nv_bfloat162* xp1 = reinterpret_cast<const __nv_bfloat162*>(&x1);
            float lg = 0.0f, lu = 0.0f;
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                float2 xf = __bfloat1622float2(xp0[b]);
                lg = fmaf(gf[b].x, xf.x, fmaf(gf[b].y, xf.y, lg));
                lu = fmaf(uf[b].x, xf.x, fmaf(uf[b].y, xf.y, lu));
            }
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                float2 xf = __bfloat1622float2(xp1[b]);
                lg = fmaf(gf[4 + b].x, xf.x, fmaf(gf[4 + b].y, xf.y, lg));
                lu = fmaf(uf[4 + b].x, xf.x, fmaf(uf[4 + b].y, xf.y, lu));
            }
            acc_g[m] = fmaf(lg, bsg, acc_g[m]);
            acc_u[m] = fmaf(lu, bsu, acc_u[m]);
        }
    }

    #pragma unroll
    for (int m = 0; m < M; ++m) {
        float ag = warp_reduce_sum(acc_g[m]) * gscale;
        float au = warp_reduce_sum(acc_u[m]) * gscale;
        if (lane == 0) {
            float sig = 1.0f / (1.0f + expf(-ag));
            float gc = tanhf(ag / gate_cap) * gate_cap;
            float uc = tanhf(au / up_cap) * up_cap;
            float y = (gc * sig * uc) * topk_weights[tks[m]];
            act_out[(int64_t)tks[m] * inter + j] = __bfloat162float(__float2bfloat16(y));
        }
    }
}

__global__ void moe_small_batch_nvfp4_w13_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ topk_weights,
    const uint8_t* __restrict__ w13_u8,
    const uint8_t* __restrict__ w13_scale,
    const float* __restrict__ w13_amax,
    const float* __restrict__ w13_denom,
    const int32_t* __restrict__ uniq_scratch,
    float* __restrict__ act_out,
    int T, int D, int inter, int K, float gate_cap, float up_cap)
{
    int u = blockIdx.y;
    if (u >= uniq_scratch[0]) return;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int j = blockIdx.x * 4 + wid;
    if (j >= inter) return;

    const int32_t* entry = uniq_scratch + 1 + u * 18;
    int expert = entry[0];
    int cnt = entry[1];

    float gscale = w13_amax[expert] / w13_denom[0];
    int row_g = 2 * j;
    int row_u = 2 * j + 1;
    int64_t half_D = D / 2;
    int64_t scale_D = D / 16;

    const uint8_t* wg = w13_u8 + ((int64_t)expert * (2 * inter) + row_g) * half_D;
    const uint8_t* wu = w13_u8 + ((int64_t)expert * (2 * inter) + row_u) * half_D;
    const uint8_t* sg = w13_scale + ((int64_t)expert * (2 * inter) + row_g) * scale_D;
    const uint8_t* su = w13_scale + ((int64_t)expert * (2 * inter) + row_u) * scale_D;

    // Zero-spill templated fast paths for cnt = 1..4 (>95% of unique experts at T=4..8!)
    if (cnt == 1) { moe_nvfp4_w13_fixed_m<1>(x, topk_weights, wg, wu, sg, su, entry, act_out, j, lane, scale_D, D, inter, K, gscale, gate_cap, up_cap); return; }
    if (cnt == 2) { moe_nvfp4_w13_fixed_m<2>(x, topk_weights, wg, wu, sg, su, entry, act_out, j, lane, scale_D, D, inter, K, gscale, gate_cap, up_cap); return; }
    if (cnt == 3) { moe_nvfp4_w13_fixed_m<3>(x, topk_weights, wg, wu, sg, su, entry, act_out, j, lane, scale_D, D, inter, K, gscale, gate_cap, up_cap); return; }
    if (cnt == 4) { moe_nvfp4_w13_fixed_m<4>(x, topk_weights, wg, wu, sg, su, entry, act_out, j, lane, scale_D, D, inter, K, gscale, gate_cap, up_cap); return; }

    for (int base = 0; base < cnt; base += 8) {
        int m_cnt = min(8, cnt - base);
        int tks[8];
        const uint4* xr4[8];
        for (int m = 0; m < m_cnt; ++m) {
            tks[m] = entry[2 + base + m];
            xr4[m] = reinterpret_cast<const uint4*>(x + (int64_t)(tks[m] / K) * D);
        }

        float acc_g[8] = {0,0,0,0,0,0,0,0};
        float acc_u[8] = {0,0,0,0,0,0,0,0};

        for (int blk = lane; blk < scale_D; blk += 32) {
            float bsg = e4m3_to_f32(sg[blk]);
            float bsu = e4m3_to_f32(su[blk]);
            uint2 bg = *reinterpret_cast<const uint2*>(wg + blk * 8);
            uint2 bu = *reinterpret_cast<const uint2*>(wu + blk * 8);

            float2 gf[8], uf[8];
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                gf[b]     = e2m1x2_to_f32x2((bg.x >> (b * 8)) & 0xFF);
                uf[b]     = e2m1x2_to_f32x2((bu.x >> (b * 8)) & 0xFF);
                gf[4 + b] = e2m1x2_to_f32x2((bg.y >> (b * 8)) & 0xFF);
                uf[4 + b] = e2m1x2_to_f32x2((bu.y >> (b * 8)) & 0xFF);
            }

            for (int m = 0; m < m_cnt; ++m) {
                uint4 x0 = xr4[m][blk * 2 + 0];
                uint4 x1 = xr4[m][blk * 2 + 1];
                const __nv_bfloat162* xp0 = reinterpret_cast<const __nv_bfloat162*>(&x0);
                const __nv_bfloat162* xp1 = reinterpret_cast<const __nv_bfloat162*>(&x1);
                float lg = 0.0f, lu = 0.0f;
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    float2 xf = __bfloat1622float2(xp0[b]);
                    lg = fmaf(gf[b].x, xf.x, fmaf(gf[b].y, xf.y, lg));
                    lu = fmaf(uf[b].x, xf.x, fmaf(uf[b].y, xf.y, lu));
                }
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    float2 xf = __bfloat1622float2(xp1[b]);
                    lg = fmaf(gf[4 + b].x, xf.x, fmaf(gf[4 + b].y, xf.y, lg));
                    lu = fmaf(uf[4 + b].x, xf.x, fmaf(uf[4 + b].y, xf.y, lu));
                }
                acc_g[m] = fmaf(lg, bsg, acc_g[m]);
                acc_u[m] = fmaf(lu, bsu, acc_u[m]);
            }
        }

        for (int m = 0; m < m_cnt; ++m) {
            float ag = warp_reduce_sum(acc_g[m]) * gscale;
            float au = warp_reduce_sum(acc_u[m]) * gscale;
            if (lane == 0) {
                float sig = 1.0f / (1.0f + expf(-ag));
                float gc = tanhf(ag / gate_cap) * gate_cap;
                float uc = tanhf(au / up_cap) * up_cap;
                float y = (gc * sig * uc) * topk_weights[tks[m]];
                act_out[(int64_t)tks[m] * inter + j] = __bfloat162float(__float2bfloat16(y));
            }
        }
    }
}

template <int M>
__device__ __forceinline__ void moe_nvfp4_w2_fixed_m(
    const float* __restrict__ act_in,
    const uint8_t* __restrict__ w2r,
    const uint8_t* __restrict__ s2r,
    const int32_t* __restrict__ entry,
    float* __restrict__ out_fp32,
    int d, int lane, int64_t scale_I, int D, int inter, int K, float gscale)
{
    int tks[M];
    const float4* ar4[M];
    float acc[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        tks[m] = entry[2 + m];
        ar4[m] = reinterpret_cast<const float4*>(act_in + (int64_t)tks[m] * inter);
        acc[m] = 0.0f;
    }
    for (int blk = lane; blk < scale_I; blk += 32) {
        float bs = e4m3_to_f32(s2r[blk]);
        uint2 bw = *reinterpret_cast<const uint2*>(w2r + blk * 8);
        float2 f[8];
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            f[b]     = e2m1x2_to_f32x2((bw.x >> (b * 8)) & 0xFF);
            f[4 + b] = e2m1x2_to_f32x2((bw.y >> (b * 8)) & 0xFF);
        }
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            float4 a0 = ar4[m][blk * 4 + 0];
            float4 a1 = ar4[m][blk * 4 + 1];
            float4 a2 = ar4[m][blk * 4 + 2];
            float4 a3 = ar4[m][blk * 4 + 3];
            float loc = 0.0f;
            loc = fmaf(f[0].x, a0.x, fmaf(f[0].y, a0.y, fmaf(f[1].x, a0.z, fmaf(f[1].y, a0.w, loc))));
            loc = fmaf(f[2].x, a1.x, fmaf(f[2].y, a1.y, fmaf(f[3].x, a1.z, fmaf(f[3].y, a1.w, loc))));
            loc = fmaf(f[4].x, a2.x, fmaf(f[4].y, a2.y, fmaf(f[5].x, a2.z, fmaf(f[5].y, a2.w, loc))));
            loc = fmaf(f[6].x, a3.x, fmaf(f[6].y, a3.y, fmaf(f[7].x, a3.z, fmaf(f[7].y, a3.w, loc))));
            acc[m] = fmaf(loc, bs, acc[m]);
        }
    }
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        float a = warp_reduce_sum(acc[m]) * gscale;
        if (lane == 0) atomicAdd(&out_fp32[(int64_t)(tks[m] / K) * D + d], a);
    }
}

__global__ void moe_small_batch_nvfp4_w2_kernel(
    const float* __restrict__ act_in,
    const uint8_t* __restrict__ w2_u8,
    const uint8_t* __restrict__ w2_scale,
    const float* __restrict__ w2_amax,
    const float* __restrict__ w2_denom,
    const int32_t* __restrict__ uniq_scratch,
    float* __restrict__ out_fp32,
    int T, int D, int inter, int K)
{
    int u = blockIdx.y;
    if (u >= uniq_scratch[0]) return;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int d = blockIdx.x * 4 + wid;
    if (d >= D) return;

    const int32_t* entry = uniq_scratch + 1 + u * 18;
    int expert = entry[0];
    int cnt = entry[1];

    float gscale = w2_amax[expert] / w2_denom[0];
    int64_t half_I = inter / 2;
    int64_t scale_I = inter / 16;
    const uint8_t* w2r = w2_u8 + ((int64_t)expert * D + d) * half_I;
    const uint8_t* s2r = w2_scale + ((int64_t)expert * D + d) * scale_I;

    if (cnt == 1) { moe_nvfp4_w2_fixed_m<1>(act_in, w2r, s2r, entry, out_fp32, d, lane, scale_I, D, inter, K, gscale); return; }
    if (cnt == 2) { moe_nvfp4_w2_fixed_m<2>(act_in, w2r, s2r, entry, out_fp32, d, lane, scale_I, D, inter, K, gscale); return; }
    if (cnt == 3) { moe_nvfp4_w2_fixed_m<3>(act_in, w2r, s2r, entry, out_fp32, d, lane, scale_I, D, inter, K, gscale); return; }
    if (cnt == 4) { moe_nvfp4_w2_fixed_m<4>(act_in, w2r, s2r, entry, out_fp32, d, lane, scale_I, D, inter, K, gscale); return; }

    for (int base = 0; base < cnt; base += 8) {
        int m_cnt = min(8, cnt - base);
        int tks[8];
        const float4* ar4[8];
        for (int m = 0; m < m_cnt; ++m) {
            tks[m] = entry[2 + base + m];
            ar4[m] = reinterpret_cast<const float4*>(act_in + (int64_t)tks[m] * inter);
        }

        float acc[8] = {0,0,0,0,0,0,0,0};
        for (int blk = lane; blk < scale_I; blk += 32) {
            float bs = e4m3_to_f32(s2r[blk]);
            uint2 bw = *reinterpret_cast<const uint2*>(w2r + blk * 8);
            float2 f[8];
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                f[b]     = e2m1x2_to_f32x2((bw.x >> (b * 8)) & 0xFF);
                f[4 + b] = e2m1x2_to_f32x2((bw.y >> (b * 8)) & 0xFF);
            }
            for (int m = 0; m < m_cnt; ++m) {
                float4 a0 = ar4[m][blk * 4 + 0];
                float4 a1 = ar4[m][blk * 4 + 1];
                float4 a2 = ar4[m][blk * 4 + 2];
                float4 a3 = ar4[m][blk * 4 + 3];
                float loc = 0.0f;
                loc = fmaf(f[0].x, a0.x, fmaf(f[0].y, a0.y, fmaf(f[1].x, a0.z, fmaf(f[1].y, a0.w, loc))));
                loc = fmaf(f[2].x, a1.x, fmaf(f[2].y, a1.y, fmaf(f[3].x, a1.z, fmaf(f[3].y, a1.w, loc))));
                loc = fmaf(f[4].x, a2.x, fmaf(f[4].y, a2.y, fmaf(f[5].x, a2.z, fmaf(f[5].y, a2.w, loc))));
                loc = fmaf(f[6].x, a3.x, fmaf(f[6].y, a3.y, fmaf(f[7].x, a3.z, fmaf(f[7].y, a3.w, loc))));
                acc[m] = fmaf(loc, bs, acc[m]);
            }
        }
        for (int m = 0; m < m_cnt; ++m) {
            float a = warp_reduce_sum(acc[m]) * gscale;
            if (lane == 0) {
                int t = tks[m] / K;
                atomicAdd(&out_fp32[(int64_t)t * D + d], a);
            }
        }
    }
}

__global__ void moe_small_batch_mxfp8_w13_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ topk_weights,
    const uint8_t* __restrict__ w13_e4m3,
    const uint8_t* __restrict__ w13_scale_u8,
    const int32_t* __restrict__ uniq_scratch,
    float* __restrict__ act_out,
    int T, int D, int inter, int K, float gate_cap, float up_cap)
{
    int u = blockIdx.y;
    if (u >= uniq_scratch[0]) return;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int j = blockIdx.x * 4 + wid;
    if (j >= inter) return;

    const int32_t* entry = uniq_scratch + 1 + u * 18;
    int expert = entry[0];
    int cnt = entry[1];
    int row_g = 2 * j, row_u = 2 * j + 1;
    int64_t scale_D = D / 32;
    const uint4* wg4 = reinterpret_cast<const uint4*>(w13_e4m3 + ((int64_t)expert * (2 * inter) + row_g) * D);
    const uint4* wu4 = reinterpret_cast<const uint4*>(w13_e4m3 + ((int64_t)expert * (2 * inter) + row_u) * D);
    const uint8_t* sg = w13_scale_u8 + ((int64_t)expert * (2 * inter) + row_g) * scale_D;
    const uint8_t* su = w13_scale_u8 + ((int64_t)expert * (2 * inter) + row_u) * scale_D;

    for (int base = 0; base < cnt; base += 8) {
        int m_cnt = min(8, cnt - base);
        int tks[8];
        const uint4* xr4[8];
        for (int m = 0; m < m_cnt; ++m) {
            tks[m] = entry[2 + base + m];
            xr4[m] = reinterpret_cast<const uint4*>(x + (int64_t)(tks[m] / K) * D);
        }

        float acc_g[8] = {0,0,0,0,0,0,0,0};
        float acc_u[8] = {0,0,0,0,0,0,0,0};

        for (int blk = lane; blk < scale_D; blk += 32) {
            float bsg = ue8m0_to_f32(sg[blk]);
            float bsu = ue8m0_to_f32(su[blk]);
            float gv[32], uv[32];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint4 bg = wg4[blk * 2 + h];
                uint4 bu = wu4[blk * 2 + h];
                const uint8_t* gb = reinterpret_cast<const uint8_t*>(&bg);
                const uint8_t* ub = reinterpret_cast<const uint8_t*>(&bu);
                #pragma unroll
                for (int n = 0; n < 16; ++n) {
                    gv[h * 16 + n] = e4m3_to_f32(gb[n]);
                    uv[h * 16 + n] = e4m3_to_f32(ub[n]);
                }
            }
            for (int m = 0; m < m_cnt; ++m) {
                float lg = 0.0f, lu = 0.0f;
                #pragma unroll
                for (int h = 0; h < 4; ++h) {
                    uint4 xv = xr4[m][blk * 4 + h];
                    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
                    #pragma unroll
                    for (int n = 0; n < 8; ++n) {
                        float xf = __bfloat162float(xp[n]);
                        lg = fmaf(gv[h * 8 + n], xf, lg);
                        lu = fmaf(uv[h * 8 + n], xf, lu);
                    }
                }
                acc_g[m] = fmaf(lg, bsg, acc_g[m]);
                acc_u[m] = fmaf(lu, bsu, acc_u[m]);
            }
        }
        for (int m = 0; m < m_cnt; ++m) {
            float ag = warp_reduce_sum(acc_g[m]);
            float au = warp_reduce_sum(acc_u[m]);
            if (lane == 0) {
                float sig = 1.0f / (1.0f + expf(-ag));
                float gc = tanhf(ag / gate_cap) * gate_cap;
                float uc = tanhf(au / up_cap) * up_cap;
                float y = (gc * sig * uc) * topk_weights[tks[m]];
                act_out[(int64_t)tks[m] * inter + j] = __bfloat162float(__float2bfloat16(y));
            }
        }
    }
}

__global__ void moe_small_batch_mxfp8_w2_kernel(
    const float* __restrict__ act_in,
    const uint8_t* __restrict__ w2_e4m3,
    const uint8_t* __restrict__ w2_scale_u8,
    const int32_t* __restrict__ uniq_scratch,
    float* __restrict__ out_fp32,
    int T, int D, int inter, int K)
{
    int u = blockIdx.y;
    if (u >= uniq_scratch[0]) return;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int d = blockIdx.x * 4 + wid;
    if (d >= D) return;

    const int32_t* entry = uniq_scratch + 1 + u * 18;
    int expert = entry[0];
    int cnt = entry[1];
    int64_t scale_I = inter / 32;
    const uint4* w2_4 = reinterpret_cast<const uint4*>(w2_e4m3 + ((int64_t)expert * D + d) * inter);
    const uint8_t* s2r = w2_scale_u8 + ((int64_t)expert * D + d) * scale_I;

    for (int base = 0; base < cnt; base += 8) {
        int m_cnt = min(8, cnt - base);
        int tks[8];
        const float4* ar4[8];
        for (int m = 0; m < m_cnt; ++m) {
            tks[m] = entry[2 + base + m];
            ar4[m] = reinterpret_cast<const float4*>(act_in + (int64_t)tks[m] * inter);
        }

        float acc[8] = {0,0,0,0,0,0,0,0};
        for (int blk = lane; blk < scale_I; blk += 32) {
            float bs = ue8m0_to_f32(s2r[blk]);
            float wv[32];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint4 bw = w2_4[blk * 2 + h];
                const uint8_t* wb = reinterpret_cast<const uint8_t*>(&bw);
                #pragma unroll
                for (int n = 0; n < 16; ++n) wv[h * 16 + n] = e4m3_to_f32(wb[n]);
            }
            for (int m = 0; m < m_cnt; ++m) {
                float loc = 0.0f;
                #pragma unroll
                for (int q = 0; q < 8; ++q) {
                    float4 av = ar4[m][blk * 8 + q];
                    loc = fmaf(wv[q * 4 + 0], av.x, loc);
                    loc = fmaf(wv[q * 4 + 1], av.y, loc);
                    loc = fmaf(wv[q * 4 + 2], av.z, loc);
                    loc = fmaf(wv[q * 4 + 3], av.w, loc);
                }
                acc[m] = fmaf(loc, bs, acc[m]);
            }
        }
        for (int m = 0; m < m_cnt; ++m) {
            float a = warp_reduce_sum(acc[m]);
            if (lane == 0) {
                int t = tks[m] / K;
                atomicAdd(&out_fp32[(int64_t)t * D + d], a);
            }
        }
    }
}

// Fused single-launch NVFP4 MoE (w13 + SwiGLU + w2) per unique active expert!
// 1 block (8 warps = 256 threads) per unique expert -> keeps sm_act[cnt][inter] (BF16) 100% in SMEM,
// eliminates act_scratch DRAM write+read, eliminates w2 launch, and reduces 40,960 blocks -> ~48 blocks!
template <int M>
__device__ __forceinline__ void moe_fused_nvfp4_body_fixed_m(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ topk_weights,
    const uint8_t* __restrict__ w13_u8,
    const uint8_t* __restrict__ w13_scale,
    const uint8_t* __restrict__ w2_u8,
    const uint8_t* __restrict__ w2_scale,
    const int32_t* __restrict__ entry,
    __nv_bfloat16* __restrict__ sm_act,
    float* __restrict__ out_fp32,
    int expert, int wid, int lane, int D, int inter, int K,
    float gscale13, float gscale2, float gate_cap, float up_cap)
{
    int tks[M];
    const uint4* xr4[M];
    float tw[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        tks[m] = entry[2 + m];
        xr4[m] = reinterpret_cast<const uint4*>(x + (int64_t)(tks[m] / K) * D);
        tw[m] = topk_weights[tks[m]];
    }

    int64_t half_D = D / 2;
    int64_t scale_D = D / 16;
    const uint8_t* w13_exp = w13_u8 + (int64_t)expert * (2 * inter) * half_D;
    const uint8_t* s13_exp = w13_scale + (int64_t)expert * (2 * inter) * scale_D;

    // Phase 1: w13 + SwiGLU -> sm_act[m][j] (BF16 in SMEM)
    for (int j = wid; j < inter; j += 8) {
        const uint8_t* wg = w13_exp + (int64_t)(2 * j) * half_D;
        const uint8_t* wu = wg + half_D;
        const uint8_t* sg = s13_exp + (int64_t)(2 * j) * scale_D;
        const uint8_t* su = sg + scale_D;

        float acc_g[M], acc_u[M];
        #pragma unroll
        for (int m = 0; m < M; ++m) { acc_g[m] = 0.0f; acc_u[m] = 0.0f; }

        for (int blk = lane; blk < scale_D; blk += 32) {
            float bsg = e4m3_to_f32(sg[blk]);
            float bsu = e4m3_to_f32(su[blk]);
            uint2 bg = *reinterpret_cast<const uint2*>(wg + blk * 8);
            uint2 bu = *reinterpret_cast<const uint2*>(wu + blk * 8);
            float2 gf[8], uf[8];
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                gf[b]     = e2m1x2_to_f32x2((bg.x >> (b * 8)) & 0xFF);
                uf[b]     = e2m1x2_to_f32x2((bu.x >> (b * 8)) & 0xFF);
                gf[4 + b] = e2m1x2_to_f32x2((bg.y >> (b * 8)) & 0xFF);
                uf[4 + b] = e2m1x2_to_f32x2((bu.y >> (b * 8)) & 0xFF);
            }
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                uint4 x0 = xr4[m][blk * 2 + 0];
                uint4 x1 = xr4[m][blk * 2 + 1];
                const __nv_bfloat162* xp0 = reinterpret_cast<const __nv_bfloat162*>(&x0);
                const __nv_bfloat162* xp1 = reinterpret_cast<const __nv_bfloat162*>(&x1);
                float lg = 0.0f, lu = 0.0f;
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    float2 xf = __bfloat1622float2(xp0[b]);
                    lg = fmaf(gf[b].x, xf.x, fmaf(gf[b].y, xf.y, lg));
                    lu = fmaf(uf[b].x, xf.x, fmaf(uf[b].y, xf.y, lu));
                }
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    float2 xf = __bfloat1622float2(xp1[b]);
                    lg = fmaf(gf[4 + b].x, xf.x, fmaf(gf[4 + b].y, xf.y, lg));
                    lu = fmaf(uf[4 + b].x, xf.x, fmaf(uf[4 + b].y, xf.y, lu));
                }
                acc_g[m] = fmaf(lg, bsg, acc_g[m]);
                acc_u[m] = fmaf(lu, bsu, acc_u[m]);
            }
        }
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            float ag = warp_reduce_sum(acc_g[m]) * gscale13;
            float au = warp_reduce_sum(acc_u[m]) * gscale13;
            if (lane == 0) {
                float sig = 1.0f / (1.0f + expf(-ag));
                float gc = tanhf(ag / gate_cap) * gate_cap;
                float uc = tanhf(au / up_cap) * up_cap;
                sm_act[m * inter + j] = __float2bfloat16((gc * sig * uc) * tw[m]);
            }
        }
    }
    __syncthreads();

    // Phase 2: w2 reading sm_act[m] directly from SMEM -> atomicAdd to out_fp32
    int64_t half_I = inter / 2;
    int64_t scale_I = inter / 16;
    const uint8_t* w2_exp = w2_u8 + (int64_t)expert * D * half_I;
    const uint8_t* s2_exp = w2_scale + (int64_t)expert * D * scale_I;
    const uint4* sm4[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) sm4[m] = reinterpret_cast<const uint4*>(sm_act + m * inter);

    for (int d = wid; d < D; d += 8) {
        const uint8_t* w2r = w2_exp + (int64_t)d * half_I;
        const uint8_t* s2r = s2_exp + (int64_t)d * scale_I;
        float acc[M];
        #pragma unroll
        for (int m = 0; m < M; ++m) acc[m] = 0.0f;

        for (int blk = lane; blk < scale_I; blk += 32) {
            float bs = e4m3_to_f32(s2r[blk]);
            uint2 bw = *reinterpret_cast<const uint2*>(w2r + blk * 8);
            float2 f[8];
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                f[b]     = e2m1x2_to_f32x2((bw.x >> (b * 8)) & 0xFF);
                f[4 + b] = e2m1x2_to_f32x2((bw.y >> (b * 8)) & 0xFF);
            }
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                uint4 a0 = sm4[m][blk * 2 + 0];
                uint4 a1 = sm4[m][blk * 2 + 1];
                const __nv_bfloat162* ap0 = reinterpret_cast<const __nv_bfloat162*>(&a0);
                const __nv_bfloat162* ap1 = reinterpret_cast<const __nv_bfloat162*>(&a1);
                float loc = 0.0f;
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    float2 af = __bfloat1622float2(ap0[b]);
                    loc = fmaf(f[b].x, af.x, fmaf(f[b].y, af.y, loc));
                }
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    float2 af = __bfloat1622float2(ap1[b]);
                    loc = fmaf(f[4 + b].x, af.x, fmaf(f[4 + b].y, af.y, loc));
                }
                acc[m] = fmaf(loc, bs, acc[m]);
            }
        }
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            float a = warp_reduce_sum(acc[m]) * gscale2;
            if (lane == 0) atomicAdd(&out_fp32[(int64_t)(tks[m] / K) * D + d], a);
        }
    }
}

__global__ void moe_fused_w13_w2_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ topk_weights,
    const uint8_t* __restrict__ w13_u8,
    const uint8_t* __restrict__ w13_scale,
    const float* __restrict__ w13_amax,
    const float* __restrict__ w13_denom,
    const uint8_t* __restrict__ w2_u8,
    const uint8_t* __restrict__ w2_scale,
    const float* __restrict__ w2_amax,
    const float* __restrict__ w2_denom,
    const int32_t* __restrict__ uniq_scratch,
    float* __restrict__ out_fp32,
    int T, int D, int inter, int K, float gate_cap, float up_cap)
{
    extern __shared__ __nv_bfloat16 sm_act[]; // [cnt, inter] BF16 (up to 8 * inter)
    int u = blockIdx.x;
    if (u >= uniq_scratch[0]) return;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    const int32_t* entry = uniq_scratch + 1 + u * 18;
    int expert = entry[0];
    int cnt = entry[1];
    float gscale13 = w13_amax[expert] / w13_denom[0];
    float gscale2  = w2_amax[expert] / w2_denom[0];

    if (cnt == 1) { moe_fused_nvfp4_body_fixed_m<1>(x, topk_weights, w13_u8, w13_scale, w2_u8, w2_scale, entry, sm_act, out_fp32, expert, wid, lane, D, inter, K, gscale13, gscale2, gate_cap, up_cap); return; }
    if (cnt == 2) { moe_fused_nvfp4_body_fixed_m<2>(x, topk_weights, w13_u8, w13_scale, w2_u8, w2_scale, entry, sm_act, out_fp32, expert, wid, lane, D, inter, K, gscale13, gscale2, gate_cap, up_cap); return; }
    if (cnt == 3) { moe_fused_nvfp4_body_fixed_m<3>(x, topk_weights, w13_u8, w13_scale, w2_u8, w2_scale, entry, sm_act, out_fp32, expert, wid, lane, D, inter, K, gscale13, gscale2, gate_cap, up_cap); return; }
    if (cnt == 4) { moe_fused_nvfp4_body_fixed_m<4>(x, topk_weights, w13_u8, w13_scale, w2_u8, w2_scale, entry, sm_act, out_fp32, expert, wid, lane, D, inter, K, gscale13, gscale2, gate_cap, up_cap); return; }

    // General cnt = 5..8 path
    int tks[8];
    const uint4* xr4[8];
    float tw[8];
    for (int m = 0; m < cnt; ++m) {
        tks[m] = entry[2 + m];
        xr4[m] = reinterpret_cast<const uint4*>(x + (int64_t)(tks[m] / K) * D);
        tw[m] = topk_weights[tks[m]];
    }
    int64_t half_D = D / 2, scale_D = D / 16;
    const uint8_t* w13_exp = w13_u8 + (int64_t)expert * (2 * inter) * half_D;
    const uint8_t* s13_exp = w13_scale + (int64_t)expert * (2 * inter) * scale_D;

    for (int j = wid; j < inter; j += 8) {
        const uint8_t* wg = w13_exp + (int64_t)(2 * j) * half_D;
        const uint8_t* wu = wg + half_D;
        const uint8_t* sg = s13_exp + (int64_t)(2 * j) * scale_D;
        const uint8_t* su = sg + scale_D;
        float acc_g[8] = {0,0,0,0,0,0,0,0};
        float acc_u[8] = {0,0,0,0,0,0,0,0};
        for (int blk = lane; blk < scale_D; blk += 32) {
            float bsg = e4m3_to_f32(sg[blk]);
            float bsu = e4m3_to_f32(su[blk]);
            uint2 bg = *reinterpret_cast<const uint2*>(wg + blk * 8);
            uint2 bu = *reinterpret_cast<const uint2*>(wu + blk * 8);
            float2 gf[8], uf[8];
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                gf[b]     = e2m1x2_to_f32x2((bg.x >> (b * 8)) & 0xFF);
                uf[b]     = e2m1x2_to_f32x2((bu.x >> (b * 8)) & 0xFF);
                gf[4 + b] = e2m1x2_to_f32x2((bg.y >> (b * 8)) & 0xFF);
                uf[4 + b] = e2m1x2_to_f32x2((bu.y >> (b * 8)) & 0xFF);
            }
            for (int m = 0; m < cnt; ++m) {
                uint4 x0 = xr4[m][blk * 2 + 0];
                uint4 x1 = xr4[m][blk * 2 + 1];
                const __nv_bfloat162* xp0 = reinterpret_cast<const __nv_bfloat162*>(&x0);
                const __nv_bfloat162* xp1 = reinterpret_cast<const __nv_bfloat162*>(&x1);
                float lg = 0.0f, lu = 0.0f;
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    float2 xf = __bfloat1622float2(xp0[b]);
                    lg = fmaf(gf[b].x, xf.x, fmaf(gf[b].y, xf.y, lg));
                    lu = fmaf(uf[b].x, xf.x, fmaf(uf[b].y, xf.y, lu));
                }
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    float2 xf = __bfloat1622float2(xp1[b]);
                    lg = fmaf(gf[4 + b].x, xf.x, fmaf(gf[4 + b].y, xf.y, lg));
                    lu = fmaf(uf[4 + b].x, xf.x, fmaf(uf[4 + b].y, xf.y, lu));
                }
                acc_g[m] = fmaf(lg, bsg, acc_g[m]);
                acc_u[m] = fmaf(lu, bsu, acc_u[m]);
            }
        }
        for (int m = 0; m < cnt; ++m) {
            float ag = warp_reduce_sum(acc_g[m]) * gscale13;
            float au = warp_reduce_sum(acc_u[m]) * gscale13;
            if (lane == 0) {
                float sig = 1.0f / (1.0f + expf(-ag));
                float gc = tanhf(ag / gate_cap) * gate_cap;
                float uc = tanhf(au / up_cap) * up_cap;
                sm_act[m * inter + j] = __float2bfloat16((gc * sig * uc) * tw[m]);
            }
        }
    }
    __syncthreads();

    int64_t half_I = inter / 2, scale_I = inter / 16;
    const uint8_t* w2_exp = w2_u8 + (int64_t)expert * D * half_I;
    const uint8_t* s2_exp = w2_scale + (int64_t)expert * D * scale_I;
    for (int d = wid; d < D; d += 8) {
        const uint8_t* w2r = w2_exp + (int64_t)d * half_I;
        const uint8_t* s2r = s2_exp + (int64_t)d * scale_I;
        float acc[8] = {0,0,0,0,0,0,0,0};
        for (int blk = lane; blk < scale_I; blk += 32) {
            float bs = e4m3_to_f32(s2r[blk]);
            uint2 bw = *reinterpret_cast<const uint2*>(w2r + blk * 8);
            float2 f[8];
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                f[b]     = e2m1x2_to_f32x2((bw.x >> (b * 8)) & 0xFF);
                f[4 + b] = e2m1x2_to_f32x2((bw.y >> (b * 8)) & 0xFF);
            }
            for (int m = 0; m < cnt; ++m) {
                const uint4* sm4m = reinterpret_cast<const uint4*>(sm_act + m * inter);
                uint4 a0 = sm4m[blk * 2 + 0];
                uint4 a1 = sm4m[blk * 2 + 1];
                const __nv_bfloat162* ap0 = reinterpret_cast<const __nv_bfloat162*>(&a0);
                const __nv_bfloat162* ap1 = reinterpret_cast<const __nv_bfloat162*>(&a1);
                float loc = 0.0f;
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    float2 af = __bfloat1622float2(ap0[b]);
                    loc = fmaf(f[b].x, af.x, fmaf(f[b].y, af.y, loc));
                }
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    float2 af = __bfloat1622float2(ap1[b]);
                    loc = fmaf(f[4 + b].x, af.x, fmaf(f[4 + b].y, af.y, loc));
                }
                acc[m] = fmaf(loc, bs, acc[m]);
            }
        }
        for (int m = 0; m < cnt; ++m) {
            float a = warp_reduce_sum(acc[m]) * gscale2;
            if (lane == 0) atomicAdd(&out_fp32[(int64_t)(tks[m] / K) * D + d], a);
        }
    }
}

void launch_moe_small_batch_nvfp4(
    const __nv_bfloat16* x, const int32_t* topk_ids, const float* topk_weights,
    const uint8_t* w13_u8, const uint8_t* w13_scale_e4m3, const float* w13_amax, const float* w13_denom,
    const uint8_t* w2_u8, const uint8_t* w2_scale_e4m3, const float* w2_amax, const float* w2_denom,
    int32_t* uniq_scratch, float* act_scratch, __nv_bfloat16* out,
    int T, int D, int inter, int K, float gate_cap, float up_cap, cudaStream_t stream)
{
    int max_u = min(128, T * K);
    float* out_fp32 = act_scratch + (int64_t)T * K * inter;
    build_small_batch_expert_map_kernel<<<1, 256, 0, stream>>>(topk_ids, uniq_scratch, out_fp32, T, K, T * D);
    moe_small_batch_nvfp4_w13_kernel<<<dim3((inter + 3) / 4, max_u), 128, 0, stream>>>(
        x, topk_weights, w13_u8, w13_scale_e4m3, w13_amax, w13_denom,
        uniq_scratch, act_scratch, T, D, inter, K, gate_cap, up_cap);
    moe_small_batch_nvfp4_w2_kernel<<<dim3((D + 3) / 4, max_u), 128, 0, stream>>>(
        act_scratch, w2_u8, w2_scale_e4m3, w2_amax, w2_denom,
        uniq_scratch, out_fp32, T, D, inter, K);
    launch_fp32_to_bf16(out_fp32, out, T * D, stream);
}

void launch_moe_small_batch_mxfp8(
    const __nv_bfloat16* x, const int32_t* topk_ids, const float* topk_weights,
    const uint8_t* w13_e4m3, const uint8_t* w13_scale_ue8m0,
    const uint8_t* w2_e4m3, const uint8_t* w2_scale_ue8m0,
    int32_t* uniq_scratch, float* act_scratch, __nv_bfloat16* out,
    int T, int D, int inter, int K, float gate_cap, float up_cap, cudaStream_t stream)
{
    int max_u = min(128, T * K);
    float* out_fp32 = act_scratch + (int64_t)T * K * inter;
    build_small_batch_expert_map_kernel<<<1, 256, 0, stream>>>(topk_ids, uniq_scratch, out_fp32, T, K, T * D);
    moe_small_batch_mxfp8_w13_kernel<<<dim3((inter + 3) / 4, max_u), 128, 0, stream>>>(
        x, topk_weights, w13_e4m3, w13_scale_ue8m0,
        uniq_scratch, act_scratch, T, D, inter, K, gate_cap, up_cap);
    moe_small_batch_mxfp8_w2_kernel<<<dim3((D + 3) / 4, max_u), 128, 0, stream>>>(
        act_scratch, w2_e4m3, w2_scale_ue8m0,
        uniq_scratch, out_fp32, T, D, inter, K);
    launch_fp32_to_bf16(out_fp32, out, T * D, stream);
}

// ============================================================================
// 9b. Grouped-Expert Prefill Helpers (dequant + gather + SwiGLU + scatter-add)
// ============================================================================

__global__ void dequant_nvfp4_matrix_kernel(
    const uint8_t* __restrict__ w_u8,
    const uint8_t* __restrict__ scale_e4m3,
    const float* __restrict__ amax_ptr,
    const float* __restrict__ denom_ptr,
    int expert,
    __nv_bfloat16* __restrict__ out_bf16,
    int N, int K)
{
    int64_t total_blocks = (int64_t)N * (K / 16);
    int64_t blk_id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (blk_id >= total_blocks) return;

    float gscale = amax_ptr[expert] / denom_ptr[0];
    const uint8_t* w_base = w_u8 + (int64_t)expert * N * (K / 2);
    const uint8_t* s_base = scale_e4m3 + (int64_t)expert * N * (K / 16);

    float sc = e4m3_to_f32(s_base[blk_id]) * gscale;
    uint2 bw = *reinterpret_cast<const uint2*>(w_base + blk_id * 8);
    __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(out_bf16 + blk_id * 16);

    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        float2 f = e2m1x2_to_f32x2((bw.x >> (b * 8)) & 0xFF);
        dst[b] = __floats2bfloat162_rn(f.x * sc, f.y * sc);
    }
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        float2 f = e2m1x2_to_f32x2((bw.y >> (b * 8)) & 0xFF);
        dst[4 + b] = __floats2bfloat162_rn(f.x * sc, f.y * sc);
    }
}

void launch_dequant_nvfp4_matrix(
    const uint8_t* w_u8, const uint8_t* scale_e4m3,
    const float* amax_ptr, const float* denom_ptr, int expert,
    __nv_bfloat16* out_bf16, int N, int K, cudaStream_t stream)
{
    int64_t total_blocks = (int64_t)N * (K / 16);
    int threads = 256;
    dequant_nvfp4_matrix_kernel<<<(int)((total_blocks + threads - 1) / threads), threads, 0, stream>>>(
        w_u8, scale_e4m3, amax_ptr, denom_ptr, expert, out_bf16, N, K);
}

__global__ void dequant_mxfp8_matrix_kernel(
    const uint8_t* __restrict__ w_e4m3,
    const uint8_t* __restrict__ scale_ue8m0,
    int expert,
    __nv_bfloat16* __restrict__ out_bf16,
    int N, int K)
{
    int64_t total_blocks = (int64_t)N * (K / 32);
    int64_t blk_id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (blk_id >= total_blocks) return;

    const uint8_t* w_base = w_e4m3 + (int64_t)expert * N * K;
    const uint8_t* s_base = scale_ue8m0 + (int64_t)expert * N * (K / 32);
    float sc = ue8m0_to_f32(s_base[blk_id]);

    const uint4* w4 = reinterpret_cast<const uint4*>(w_base + blk_id * 32);
    __nv_bfloat16* dst = out_bf16 + blk_id * 32;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        uint4 v = w4[h];
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            dst[h * 16 + i] = __float2bfloat16(e4m3_to_f32(b[i]) * sc);
        }
    }
}

void launch_dequant_mxfp8_matrix(
    const uint8_t* w_e4m3, const uint8_t* scale_ue8m0, int expert,
    __nv_bfloat16* out_bf16, int N, int K, cudaStream_t stream)
{
    int64_t total_blocks = (int64_t)N * (K / 32);
    int threads = 256;
    dequant_mxfp8_matrix_kernel<<<(int)((total_blocks + threads - 1) / threads), threads, 0, stream>>>(
        w_e4m3, scale_ue8m0, expert, out_bf16, N, K);
}

__global__ void gather_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    const int32_t* __restrict__ idx,
    __nv_bfloat16* __restrict__ dst,
    int M, int D)
{
    int m = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M && d < D) {
        int t = idx[m];
        dst[(int64_t)m * D + d] = src[(int64_t)t * D + d];
    }
}

void launch_gather_bf16(
    const __nv_bfloat16* src, const int32_t* idx, __nv_bfloat16* dst,
    int M, int D, cudaStream_t stream)
{
    int threads = 256;
    gather_bf16_kernel<<<dim3((D + threads - 1) / threads, M), threads, 0, stream>>>(src, idx, dst, M, D);
}

__global__ void swiglu_softcap_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate_up, // [M, 2*inter] interleaved
    const float* __restrict__ gammas,          // [M]
    __nv_bfloat16* __restrict__ act_out,       // [M, inter]
    int M, int inter, float gate_cap, float up_cap)
{
    int m = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || j >= inter) return;
    float g = __bfloat162float(gate_up[(int64_t)m * (2 * inter) + 2 * j]);
    float u = __bfloat162float(gate_up[(int64_t)m * (2 * inter) + 2 * j + 1]);
    float sig = 1.0f / (1.0f + expf(-g));
    float gc = tanhf(g / gate_cap) * gate_cap;
    float uc = tanhf(u / up_cap) * up_cap;
    act_out[(int64_t)m * inter + j] = __float2bfloat16((gc * sig * uc) * gammas[m]);
}

void launch_swiglu_softcap_bf16(
    const __nv_bfloat16* gate_up, const float* gammas, __nv_bfloat16* act_out,
    int M, int inter, float gate_cap, float up_cap, cudaStream_t stream)
{
    int threads = 256;
    swiglu_softcap_bf16_kernel<<<dim3((inter + threads - 1) / threads, M), threads, 0, stream>>>(
        gate_up, gammas, act_out, M, inter, gate_cap, up_cap);
}

__global__ void scatter_add_bf16_to_fp32_kernel(
    const __nv_bfloat16* __restrict__ src,
    const int32_t* __restrict__ idx,
    float* __restrict__ dst_fp32,
    int M, int D)
{
    int m = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M && d < D) {
        int t = idx[m];
        atomicAdd(&dst_fp32[(int64_t)t * D + d], __bfloat162float(src[(int64_t)m * D + d]));
    }
}

void launch_scatter_add_bf16_to_fp32(
    const __nv_bfloat16* src, const int32_t* idx, float* dst_fp32,
    int M, int D, cudaStream_t stream)
{
    int threads = 256;
    scatter_add_bf16_to_fp32_kernel<<<dim3((D + threads - 1) / threads, M), threads, 0, stream>>>(
        src, idx, dst_fp32, M, D);
}

__global__ void fp32_to_bf16_kernel(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = __float2bfloat16(src[i]);
}

void launch_fp32_to_bf16(const float* src, __nv_bfloat16* dst, int N, cudaStream_t stream) {
    int threads = 256;
    fp32_to_bf16_kernel<<<(N + threads - 1) / threads, threads, 0, stream>>>(src, dst, N);
}

// ============================================================================
// 9c. Native Blackwell NVFP4 cuBLASLt Helpers (128x4 Swizzled VEC16_UE4M3)
// ============================================================================

__device__ __forceinline__ size_t nvfp4_swizzle_idx(int row, int col, int K_s_pad) {
    int tm = row >> 7;
    int tk = col >> 2;
    int m_in = row & 127;
    int k_in = col & 3;
    return ((size_t)tm * (K_s_pad >> 2) + tk) * 512 + (m_in & 31) * 16 + (m_in >> 5) * 4 + k_in;
}

__global__ void swizzle_nvfp4_scales_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int E, int N, int K_s)
{
    int e = blockIdx.y;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * K_s;
    if (e >= E || idx >= total) return;
    int row = (int)(idx / K_s);
    int col = (int)(idx % K_s);
    size_t out_off = (size_t)e * (size_t)total + nvfp4_swizzle_idx(row, col, K_s);
    dst[out_off] = src[(size_t)e * (size_t)total + idx];
}

void launch_swizzle_nvfp4_scales(
    const uint8_t* src, uint8_t* dst, int E, int N, int K_s, cudaStream_t stream)
{
    int64_t total = (int64_t)N * K_s;
    swizzle_nvfp4_scales_kernel<<<dim3((int)((total + 255) / 256), E), 256, 0, stream>>>(
        src, dst, E, N, K_s);
}

__global__ void gather_quant_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ idx,
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_scale_swiz,
    int M, int M_pad, int K)
{
    int K_s = K >> 4;
    int m = blockIdx.y;
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M_pad || blk >= K_s) return;

    uint8_t* dst_u8 = out_fp4 + (int64_t)m * (K >> 1) + blk * 8;
    if (m >= M) {
        *reinterpret_cast<uint2*>(dst_u8) = make_uint2(0, 0);
        out_scale_swiz[nvfp4_swizzle_idx(m, blk, K_s)] = 0;
        return;
    }

    int tok = idx[m];
    const __nv_bfloat16* row = x + (int64_t)tok * K + blk * 16;
    float vals[16];
    float bmax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        vals[i] = __bfloat162float(row[i]);
        bmax = fmaxf(bmax, fabsf(vals[i]));
    }
    uint8_t s_u8 = 0;
    float inv_s = 0.0f;
    if (bmax > 0.0f) {
        float sc = fminf(448.0f, bmax * (1.0f / 6.0f));
        s_u8 = (uint8_t)__nv_cvt_float_to_fp8(sc, __NV_SATFINITE, __NV_E4M3);
        float s_real = e4m3_to_f32(s_u8);
        if (s_real > 0.0f) inv_s = 1.0f / s_real;
    }
    out_scale_swiz[nvfp4_swizzle_idx(m, blk, K_s)] = s_u8;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        dst_u8[p] = (uint8_t)__nv_cvt_float2_to_fp4x2(
            make_float2(vals[2 * p] * inv_s, vals[2 * p + 1] * inv_s), __NV_E2M1, cudaRoundNearest);
    }
}

void launch_gather_quant_nvfp4(
    const __nv_bfloat16* x, const int32_t* idx,
    uint8_t* out_fp4, uint8_t* out_scale_swiz,
    int M, int M_pad, int K, cudaStream_t stream)
{
    int K_s = K >> 4;
    gather_quant_nvfp4_kernel<<<dim3((K_s + 127) / 128, M_pad), 128, 0, stream>>>(
        x, idx, out_fp4, out_scale_swiz, M, M_pad, K);
}

__global__ void swiglu_softcap_quant_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ gate_up,
    const float* __restrict__ gammas,
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_scale_swiz,
    int M, int M_pad, int inter, float gate_cap, float up_cap)
{
    int K_s = inter >> 4;
    int m = blockIdx.y;
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M_pad || blk >= K_s) return;

    uint8_t* dst_u8 = out_fp4 + (int64_t)m * (inter >> 1) + blk * 8;
    if (m >= M) {
        *reinterpret_cast<uint2*>(dst_u8) = make_uint2(0, 0);
        out_scale_swiz[nvfp4_swizzle_idx(m, blk, K_s)] = 0;
        return;
    }

    float gamma = gammas[m];
    const __nv_bfloat16* gu = gate_up + (int64_t)m * (2 * inter) + blk * 32;
    float vals[16];
    float bmax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        float g = __bfloat162float(gu[2 * i]);
        float u = __bfloat162float(gu[2 * i + 1]);
        float sig = 1.0f / (1.0f + expf(-g));
        float gc = tanhf(g / gate_cap) * gate_cap;
        float uc = tanhf(u / up_cap) * up_cap;
        float y = __bfloat162float(__float2bfloat16((gc * sig * uc) * gamma));
        vals[i] = y;
        bmax = fmaxf(bmax, fabsf(y));
    }
    uint8_t s_u8 = 0;
    float inv_s = 0.0f;
    if (bmax > 0.0f) {
        float sc = fminf(448.0f, bmax * (1.0f / 6.0f));
        s_u8 = (uint8_t)__nv_cvt_float_to_fp8(sc, __NV_SATFINITE, __NV_E4M3);
        float s_real = e4m3_to_f32(s_u8);
        if (s_real > 0.0f) inv_s = 1.0f / s_real;
    }
    out_scale_swiz[nvfp4_swizzle_idx(m, blk, K_s)] = s_u8;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        dst_u8[p] = (uint8_t)__nv_cvt_float2_to_fp4x2(
            make_float2(vals[2 * p] * inv_s, vals[2 * p + 1] * inv_s), __NV_E2M1, cudaRoundNearest);
    }
}

void launch_swiglu_softcap_quant_nvfp4(
    const __nv_bfloat16* gate_up, const float* gammas,
    uint8_t* out_fp4, uint8_t* out_scale_swiz,
    int M, int M_pad, int inter, float gate_cap, float up_cap, cudaStream_t stream)
{
    int K_s = inter >> 4;
    swiglu_softcap_quant_nvfp4_kernel<<<dim3((K_s + 127) / 128, M_pad), 128, 0, stream>>>(
        gate_up, gammas, out_fp4, out_scale_swiz, M, M_pad, inter, gate_cap, up_cap);
}

// ============================================================================
// 10. High-Bandwidth Vectorized BF16 GEMV (M=1 decode)
// ============================================================================

__global__ void bf16_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ y_bf16,
    float* __restrict__ y_fp32,
    int N, int K)
{
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + wid;
    if (row >= N) return;

    const __nv_bfloat16* wr = W + (int64_t)row * K;
    float acc = 0.0f;

    int k8 = K >> 3; // 8 bf16 per uint4 (128-bit load)
    const uint4* w4 = reinterpret_cast<const uint4*>(wr);
    const uint4* x4 = reinterpret_cast<const uint4*>(x);
    for (int i = lane; i < k8; i += 32) {
        uint4 wv = w4[i];
        uint4 xv = x4[i];
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wv);
        const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 wf = __bfloat1622float2(wp[j]);
            float2 xf = __bfloat1622float2(xp[j]);
            acc = fmaf(wf.x, xf.x, acc);
            acc = fmaf(wf.y, xf.y, acc);
        }
    }
    // Tail if K % 8 != 0
    for (int k = (k8 << 3) + lane; k < K; k += 32) {
        acc = fmaf(__bfloat162float(wr[k]), __bfloat162float(x[k]), acc);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        if (y_bf16) y_bf16[row] = __float2bfloat16(acc);
        if (y_fp32) y_fp32[row] = acc;
    }
}

void launch_bf16_gemv(
    const __nv_bfloat16* x, const __nv_bfloat16* W,
    __nv_bfloat16* y_bf16, float* y_fp32,
    int N, int K, cudaStream_t stream)
{
    bf16_gemv_kernel<<<(N + 3) / 4, 128, 0, stream>>>(x, W, y_bf16, y_fp32, N, K);
}

// High-bandwidth vectorized BF16 Small-GEMM for M = 2..8: reads W[row] ONCE, multiplies M rows!
template <int M_CONST>
__global__ void bf16_small_gemm_kernel(
    const __nv_bfloat16* __restrict__ X, // [M, K]
    const __nv_bfloat16* __restrict__ W, // [N, K]
    __nv_bfloat16* __restrict__ Y_bf16,  // [M, N] or nullptr
    float* __restrict__ Y_fp32,          // [M, N] or nullptr
    int N, int K)
{
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + wid;
    if (row >= N) return;

    const uint4* w4 = reinterpret_cast<const uint4*>(W + (int64_t)row * K);
    int k8 = K >> 3;
    float acc[M_CONST];
    #pragma unroll
    for (int m = 0; m < M_CONST; ++m) acc[m] = 0.0f;

    for (int i = lane; i < k8; i += 32) {
        uint4 wv = w4[i];
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wv);
        float2 wf0 = __bfloat1622float2(wp[0]);
        float2 wf1 = __bfloat1622float2(wp[1]);
        float2 wf2 = __bfloat1622float2(wp[2]);
        float2 wf3 = __bfloat1622float2(wp[3]);
        #pragma unroll
        for (int m = 0; m < M_CONST; ++m) {
            const uint4* xm = reinterpret_cast<const uint4*>(X + (int64_t)m * K);
            uint4 xv = xm[i];
            const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
            float2 xf0 = __bfloat1622float2(xp[0]);
            float2 xf1 = __bfloat1622float2(xp[1]);
            float2 xf2 = __bfloat1622float2(xp[2]);
            float2 xf3 = __bfloat1622float2(xp[3]);
            acc[m] = fmaf(wf0.x, xf0.x, fmaf(wf0.y, xf0.y, acc[m]));
            acc[m] = fmaf(wf1.x, xf1.x, fmaf(wf1.y, xf1.y, acc[m]));
            acc[m] = fmaf(wf2.x, xf2.x, fmaf(wf2.y, xf2.y, acc[m]));
            acc[m] = fmaf(wf3.x, xf3.x, fmaf(wf3.y, xf3.y, acc[m]));
        }
    }
    #pragma unroll
    for (int m = 0; m < M_CONST; ++m) {
        float v = warp_reduce_sum(acc[m]);
        if (lane == 0) {
            if (Y_bf16) Y_bf16[(int64_t)m * N + row] = __float2bfloat16(v);
            if (Y_fp32) Y_fp32[(int64_t)m * N + row] = v;
        }
    }
}

void launch_bf16_small_gemm(
    const __nv_bfloat16* X, const __nv_bfloat16* W,
    __nv_bfloat16* Y_bf16, float* Y_fp32,
    int M, int N, int K, cudaStream_t stream)
{
    int grid = (N + 3) / 4;
    switch (M) {
        case 2: bf16_small_gemm_kernel<2><<<grid, 128, 0, stream>>>(X, W, Y_bf16, Y_fp32, N, K); break;
        case 3: bf16_small_gemm_kernel<3><<<grid, 128, 0, stream>>>(X, W, Y_bf16, Y_fp32, N, K); break;
        case 4: bf16_small_gemm_kernel<4><<<grid, 128, 0, stream>>>(X, W, Y_bf16, Y_fp32, N, K); break;
        case 5: bf16_small_gemm_kernel<5><<<grid, 128, 0, stream>>>(X, W, Y_bf16, Y_fp32, N, K); break;
        case 6: bf16_small_gemm_kernel<6><<<grid, 128, 0, stream>>>(X, W, Y_bf16, Y_fp32, N, K); break;
        case 7: bf16_small_gemm_kernel<7><<<grid, 128, 0, stream>>>(X, W, Y_bf16, Y_fp32, N, K); break;
        case 8: bf16_small_gemm_kernel<8><<<grid, 128, 0, stream>>>(X, W, Y_bf16, Y_fp32, N, K); break;
        default: break;
    }
}

// ============================================================================
// 10b. Per-Row FP8 E4M3 Weight Quantization + W8A16 Small-GEMM (M = 1..8)
// Cuts dense + LM-head HBM reads in HALF (2x memory bandwidth on GB10!)
// ============================================================================

__global__ void quantize_bf16_to_fp8_rowwise_kernel(
    const __nv_bfloat16* __restrict__ W_bf16,
    uint8_t* __restrict__ W_fp8,
    float* __restrict__ W_scale,
    int N, int K)
{
    int row = blockIdx.x;
    if (row >= N) return;
    const __nv_bfloat16* src = W_bf16 + (int64_t)row * K;
    uint8_t* dst = W_fp8 + (int64_t)row * K;

    float amax = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        amax = fmaxf(amax, fabsf(__bfloat162float(src[k])));
    }
    // Block reduce max
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, off));
    __shared__ float sm[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) sm[wid] = amax;
    __syncthreads();
    if (wid == 0) {
        int nw = blockDim.x >> 5;
        amax = (lane < nw) ? sm[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, off));
        if (lane == 0) sm[0] = (amax > 0.0f) ? (amax / 448.0f) : 1.0f;
    }
    __syncthreads();
    float sc = sm[0];
    float inv_sc = 1.0f / sc;
    if (threadIdx.x == 0) W_scale[row] = sc;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        dst[k] = (uint8_t)__nv_cvt_float_to_fp8(__bfloat162float(src[k]) * inv_sc, __NV_SATFINITE, __NV_E4M3);
    }
}

void launch_quantize_bf16_to_fp8_rowwise(
    const __nv_bfloat16* W_bf16, uint8_t* W_fp8, float* W_scale,
    int N, int K, cudaStream_t stream)
{
    quantize_bf16_to_fp8_rowwise_kernel<<<N, 256, 0, stream>>>(W_bf16, W_fp8, W_scale, N, K);
}

template <int M_CONST>
__global__ void fp8_small_gemm_kernel(
    const __nv_bfloat16* __restrict__ X, // [M, K]
    const uint8_t* __restrict__ W_fp8,   // [N, K]
    const float* __restrict__ W_scale,   // [N]
    __nv_bfloat16* __restrict__ Y_bf16,  // [M, N] or nullptr
    float* __restrict__ Y_fp32,          // [M, N] or nullptr
    int N, int K)
{
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + wid;
    if (row >= N) return;

    const uint4* w16 = reinterpret_cast<const uint4*>(W_fp8 + (int64_t)row * K);
    int k16 = K >> 4; // 16 FP8 weights per uint4 (128-bit load)
    float acc[M_CONST];
    #pragma unroll
    for (int m = 0; m < M_CONST; ++m) acc[m] = 0.0f;

    for (int i = lane; i < k16; i += 32) {
        uint4 wv = w16[i];
        const __nv_fp8x2_storage_t* wp = reinterpret_cast<const __nv_fp8x2_storage_t*>(&wv);
        float2 wf[8];
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(wp[p], __NV_E4M3);
            wf[p] = __half22float2(*reinterpret_cast<const __half2*>(&hr));
        }
        #pragma unroll
        for (int m = 0; m < M_CONST; ++m) {
            const uint4* xm = reinterpret_cast<const uint4*>(X + (int64_t)m * K);
            uint4 x0 = xm[i * 2 + 0];
            uint4 x1 = xm[i * 2 + 1];
            const __nv_bfloat162* xp0 = reinterpret_cast<const __nv_bfloat162*>(&x0);
            const __nv_bfloat162* xp1 = reinterpret_cast<const __nv_bfloat162*>(&x1);
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                float2 xf = __bfloat1622float2(xp0[p]);
                acc[m] = fmaf(wf[p].x, xf.x, fmaf(wf[p].y, xf.y, acc[m]));
            }
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                float2 xf = __bfloat1622float2(xp1[p]);
                acc[m] = fmaf(wf[4 + p].x, xf.x, fmaf(wf[4 + p].y, xf.y, acc[m]));
            }
        }
    }
    float sc = W_scale[row];
    #pragma unroll
    for (int m = 0; m < M_CONST; ++m) {
        float v = warp_reduce_sum(acc[m]) * sc;
        if (lane == 0) {
            if (Y_bf16) Y_bf16[(int64_t)m * N + row] = __float2bfloat16(v);
            if (Y_fp32) Y_fp32[(int64_t)m * N + row] = v;
        }
    }
}

void launch_fp8_small_gemm(
    const __nv_bfloat16* X, const uint8_t* W_fp8, const float* W_scale,
    __nv_bfloat16* Y_bf16, float* Y_fp32,
    int M, int N, int K, cudaStream_t stream)
{
    int grid = (N + 3) / 4;
    switch (M) {
        case 1: fp8_small_gemm_kernel<1><<<grid, 128, 0, stream>>>(X, W_fp8, W_scale, Y_bf16, Y_fp32, N, K); break;
        case 2: fp8_small_gemm_kernel<2><<<grid, 128, 0, stream>>>(X, W_fp8, W_scale, Y_bf16, Y_fp32, N, K); break;
        case 3: fp8_small_gemm_kernel<3><<<grid, 128, 0, stream>>>(X, W_fp8, W_scale, Y_bf16, Y_fp32, N, K); break;
        case 4: fp8_small_gemm_kernel<4><<<grid, 128, 0, stream>>>(X, W_fp8, W_scale, Y_bf16, Y_fp32, N, K); break;
        case 5: fp8_small_gemm_kernel<5><<<grid, 128, 0, stream>>>(X, W_fp8, W_scale, Y_bf16, Y_fp32, N, K); break;
        case 6: fp8_small_gemm_kernel<6><<<grid, 128, 0, stream>>>(X, W_fp8, W_scale, Y_bf16, Y_fp32, N, K); break;
        case 7: fp8_small_gemm_kernel<7><<<grid, 128, 0, stream>>>(X, W_fp8, W_scale, Y_bf16, Y_fp32, N, K); break;
        case 8: fp8_small_gemm_kernel<8><<<grid, 128, 0, stream>>>(X, W_fp8, W_scale, Y_bf16, Y_fp32, N, K); break;
        default: break;
    }
}

// ============================================================================
// 12. Multi-Slot Batched Decode Kernels (B concurrent sequences, 1 token each)
// ============================================================================

__global__ void sconv_static_batched_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ state_pool,
    const int32_t* __restrict__ slot_ids,
    __nv_bfloat16* __restrict__ out,
    int B, int C)
{
    int b = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || c >= C) return;
    int slot = slot_ids[b];
    __nv_bfloat16* st = state_pool + (int64_t)slot * 3 * C;

    float w0 = __bfloat162float(weight[c * 4 + 0]);
    float w1 = __bfloat162float(weight[c * 4 + 1]);
    float w2 = __bfloat162float(weight[c * 4 + 2]);
    float w3 = __bfloat162float(weight[c * 4 + 3]);
    float s0 = __bfloat162float(st[0 * C + c]);
    float s1 = __bfloat162float(st[1 * C + c]);
    float s2 = __bfloat162float(st[2 * C + c]);
    float xt = __bfloat162float(x[(int64_t)b * C + c]);

    float y = (w0 * s0 + w1 * s1 + w2 * s2 + w3 * xt + xt) * 0.44721359549995793f;
    out[(int64_t)b * C + c] = __float2bfloat16(y);
    st[0 * C + c] = __float2bfloat16(s1);
    st[1 * C + c] = __float2bfloat16(s2);
    st[2 * C + c] = __float2bfloat16(xt);
}

void launch_sconv_static_batched(
    const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* state_pool,
    const int32_t* d_slot_ids, __nv_bfloat16* out, int B, int C, cudaStream_t stream)
{
    sconv_static_batched_kernel<<<dim3((C + 255) / 256, B), 256, 0, stream>>>(
        x, weight, state_pool, d_slot_ids, out, B, C);
}

__global__ void sconv_dynamic_batched_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ static_w,
    const __nv_bfloat16* __restrict__ dyn_w,
    __nv_bfloat16* __restrict__ state_pool,
    const int32_t* __restrict__ slot_ids,
    __nv_bfloat16* __restrict__ out,
    int B, int num_kv_heads, int head_dim)
{
    int C = num_kv_heads * head_dim;
    int b = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || c >= C) return;
    int slot = slot_ids[b];
    int h = c / head_dim;
    __nv_bfloat16* st = state_pool + (int64_t)slot * 3 * C;

    float sw0 = __bfloat162float(static_w[0 * C + c]);
    float sw1 = __bfloat162float(static_w[1 * C + c]);
    float sw2 = __bfloat162float(static_w[2 * C + c]);
    float sw3 = __bfloat162float(static_w[3 * C + c]);
    const __nv_bfloat16* dw = dyn_w + (int64_t)b * (num_kv_heads * 4) + h * 4;

    float s0 = __bfloat162float(st[0 * C + c]);
    float s1 = __bfloat162float(st[1 * C + c]);
    float s2 = __bfloat162float(st[2 * C + c]);
    float xt = __bfloat162float(x[(int64_t)b * C + c]);

    float y = ((sw0 + __bfloat162float(dw[0])) * xt
             + (sw1 + __bfloat162float(dw[1])) * s2
             + (sw2 + __bfloat162float(dw[2])) * s1
             + (sw3 + __bfloat162float(dw[3])) * s0 + xt) * (1.0f / 3.0f);
    out[(int64_t)b * C + c] = __float2bfloat16(y);
    st[0 * C + c] = __float2bfloat16(s1);
    st[1 * C + c] = __float2bfloat16(s2);
    st[2 * C + c] = __float2bfloat16(xt);
}

void launch_sconv_dynamic_batched(
    const __nv_bfloat16* x, const __nv_bfloat16* static_w, const __nv_bfloat16* dyn_w,
    __nv_bfloat16* state_pool, const int32_t* d_slot_ids, __nv_bfloat16* out,
    int B, int num_kv_heads, int head_dim, cudaStream_t stream)
{
    int C = num_kv_heads * head_dim;
    sconv_dynamic_batched_kernel<<<dim3((C + 255) / 256, B), 256, 0, stream>>>(
        x, static_w, dyn_w, state_pool, d_slot_ids, out, B, num_kv_heads, head_dim);
}

__global__ void attn_prep_q_r_batched_kernel(
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ R,
    const __nv_bfloat16* __restrict__ q_scale,
    const __nv_bfloat16* __restrict__ r_scale,
    const int32_t* __restrict__ positions,
    int B, bool is_local, int n_heads, int head_dim, int d_rel,
    float qk_clip, float rel_clip, float log_n_floor, float log_alpha)
{
    int h = blockIdx.x;
    int b = blockIdx.y;
    if (h >= n_heads || b >= B) return;
    int pos = positions[b];
    float tau = 1.0f;
    if (!is_local) {
        tau = 1.0f + log_alpha * logf(fmaxf(1.0f, (float)(pos + 1) / log_n_floor));
    }
    __nv_bfloat16* q_ptr = Q + ((int64_t)b * n_heads + h) * head_dim;
    const __nv_bfloat16* qs_ptr = q_scale + h * head_dim;
    float qv[4], sum_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = threadIdx.x + i * 32;
        qv[i] = __bfloat162float(q_ptr[d]) * __bfloat162float(qs_ptr[d]);
        sum_sq += qv[i] * qv[i];
    }
    sum_sq = warp_reduce_sum(sum_sq);
    float scale = fminf(1.0f, qk_clip / (sqrtf(__shfl_sync(0xffffffff, sum_sq, 0)) + 1e-6f)) * tau;
    #pragma unroll
    for (int i = 0; i < 4; ++i) q_ptr[threadIdx.x + i * 32] = __float2bfloat16(qv[i] * scale);

    __nv_bfloat16* r_ptr = R + ((int64_t)b * n_heads + h) * d_rel;
    const __nv_bfloat16* rs_ptr = r_scale + h * d_rel;
    float rv = (threadIdx.x < d_rel) ? __bfloat162float(r_ptr[threadIdx.x]) * __bfloat162float(rs_ptr[threadIdx.x]) : 0.0f;
    float r_sq = warp_reduce_sum(rv * rv);
    float r_sc = fminf(1.0f, rel_clip / (sqrtf(__shfl_sync(0xffffffff, r_sq, 0)) + 1e-6f)) * tau;
    if (threadIdx.x < d_rel) r_ptr[threadIdx.x] = __float2bfloat16(rv * r_sc);
}

__global__ void attn_prep_kv_batched_kernel(
    const __nv_bfloat16* __restrict__ K_in,
    const __nv_bfloat16* __restrict__ V_in,
    __nv_bfloat16* __restrict__ k_cache_pool,
    __nv_bfloat16* __restrict__ v_cache_pool,
    uint8_t* __restrict__ k_cache_fp8_pool,
    uint8_t* __restrict__ v_cache_fp8_pool,
    float* __restrict__ k_scale_fp8_pool,
    float* __restrict__ v_scale_fp8_pool,
    const int32_t* __restrict__ slot_ids,
    const int32_t* __restrict__ positions,
    int B, int cache_cap, bool is_local, int n_kv_heads, int head_dim, float qk_clip, float eps)
{
    int h = blockIdx.x;
    int b = blockIdx.y;
    if (h >= n_kv_heads || b >= B) return;
    int slot = slot_ids[b];
    int pos = positions[b];
    int kpos = is_local ? (pos % cache_cap) : pos;

    const __nv_bfloat16* kin = K_in + ((int64_t)b * n_kv_heads + h) * head_dim;
    const __nv_bfloat16* vin = V_in + ((int64_t)b * n_kv_heads + h) * head_dim;

    float kv[4], vv[4], k_sq = 0.0f, v_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = threadIdx.x + i * 32;
        kv[i] = __bfloat162float(kin[d]);
        vv[i] = __bfloat162float(vin[d]);
        k_sq += kv[i] * kv[i];
        v_sq += vv[i] * vv[i];
    }
    float k_sc = fminf(1.0f, qk_clip / (sqrtf(__shfl_sync(0xffffffff, warp_reduce_sum(k_sq), 0)) + 1e-6f));
    float v_rstd = rsqrtf(__shfl_sync(0xffffffff, warp_reduce_sum(v_sq), 0) / (float)head_dim + eps);
    if (k_cache_fp8_pool) {
        float km = 0.0f, vm = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            kv[i] *= k_sc; vv[i] *= v_rstd;
            km = fmaxf(km, fabsf(kv[i])); vm = fmaxf(vm, fabsf(vv[i]));
        }
        float ks = __shfl_sync(0xffffffff, warp_reduce_max(km), 0); ks = (ks > 0.0f) ? (ks / 448.0f) : 1.0f;
        float vs = __shfl_sync(0xffffffff, warp_reduce_max(vm), 0); vs = (vs > 0.0f) ? (vs / 448.0f) : 1.0f;
        float inv_ks = 1.0f / ks, inv_vs = 1.0f / vs;
        int64_t sc_idx = ((int64_t)slot * cache_cap + kpos) * n_kv_heads + h;
        if (threadIdx.x == 0) { k_scale_fp8_pool[sc_idx] = ks; v_scale_fp8_pool[sc_idx] = vs; }
        uint8_t* kdst = k_cache_fp8_pool + sc_idx * head_dim;
        uint8_t* vdst = v_cache_fp8_pool + sc_idx * head_dim;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int d = threadIdx.x + i * 32;
            kdst[d] = (uint8_t)__nv_cvt_float_to_fp8(kv[i] * inv_ks, __NV_SATFINITE, __NV_E4M3);
            vdst[d] = (uint8_t)__nv_cvt_float_to_fp8(vv[i] * inv_vs, __NV_SATFINITE, __NV_E4M3);
        }
    } else {
        int64_t slot_stride = (int64_t)cache_cap * n_kv_heads * head_dim;
        __nv_bfloat16* kdst = k_cache_pool + slot * slot_stride + ((int64_t)kpos * n_kv_heads + h) * head_dim;
        __nv_bfloat16* vdst = v_cache_pool + slot * slot_stride + ((int64_t)kpos * n_kv_heads + h) * head_dim;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int d = threadIdx.x + i * 32;
            kdst[d] = __float2bfloat16(kv[i] * k_sc);
            vdst[d] = __float2bfloat16(vv[i] * v_rstd);
        }
    }
}

void launch_attn_prep_batched(
    __nv_bfloat16* Q, const __nv_bfloat16* K_in, const __nv_bfloat16* V_in, __nv_bfloat16* R,
    const __nv_bfloat16* q_scale, const __nv_bfloat16* r_scale,
    __nv_bfloat16* k_cache_pool, __nv_bfloat16* v_cache_pool,
    const int32_t* d_slot_ids, const int32_t* d_positions,
    int B, int cache_cap, bool is_local, int n_heads, int n_kv_heads, int head_dim, int d_rel,
    float qk_clip, float rel_clip, float log_n_floor, float log_alpha, float eps, cudaStream_t stream,
    uint8_t* k_cache_fp8_pool, uint8_t* v_cache_fp8_pool, float* k_scale_fp8_pool, float* v_scale_fp8_pool)
{
    attn_prep_q_r_batched_kernel<<<dim3(n_heads, B), 32, 0, stream>>>(
        Q, R, q_scale, r_scale, d_positions, B, is_local, n_heads, head_dim, d_rel,
        qk_clip, rel_clip, log_n_floor, log_alpha);
    attn_prep_kv_batched_kernel<<<dim3(n_kv_heads, B), 32, 0, stream>>>(
        K_in, V_in, k_cache_pool, v_cache_pool, k_cache_fp8_pool, v_cache_fp8_pool, k_scale_fp8_pool, v_scale_fp8_pool,
        d_slot_ids, d_positions, B, cache_cap, is_local, n_kv_heads, head_dim, qk_clip, eps);
}

// Batched decode attention: 1 block (4 warps) per (h, b), parallelizing over K positions
__global__ void attn_decode_batched_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ k_cache_pool,
    const __nv_bfloat16* __restrict__ v_cache_pool,
    const uint8_t* __restrict__ k_cache_fp8_pool,
    const uint8_t* __restrict__ v_cache_fp8_pool,
    const float* __restrict__ k_scale_fp8_pool,
    const float* __restrict__ v_scale_fp8_pool,
    const __nv_bfloat16* __restrict__ rel_logits,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ sinks,
    __nv_bfloat16* __restrict__ out,
    const int32_t* __restrict__ slot_ids,
    const int32_t* __restrict__ positions,
    int B, int cache_cap, bool is_local, int window_size,
    int n_heads, int n_kv_heads, int head_dim, int rel_extent)
{
    int h = blockIdx.x;
    int b = blockIdx.y;
    if (h >= n_heads || b >= B) return;

    int slot = slot_ids[b];
    int q_pos = positions[b];
    int k_start = is_local ? max(0, q_pos - window_size + 1) : 0;
    int k_end = q_pos + 1;
    int kv_h = h / (n_heads / n_kv_heads);

    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;

    const __nv_bfloat16* q_ptr = Q + ((int64_t)b * n_heads + h) * head_dim;
    const __nv_bfloat16* r_ptr = rel_logits + ((int64_t)b * n_heads + h) * rel_extent;
    float qv[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) qv[i] = __bfloat162float(q_ptr[lane + i * 32]);
    const float inv_sqrt_d = rsqrtf((float)head_dim);

    int kv_stride = n_kv_heads * head_dim;
    int64_t slot_stride = (int64_t)cache_cap * kv_stride;
    const __nv_bfloat16* kc = k_cache_pool ? (k_cache_pool + slot * slot_stride) : nullptr;
    const __nv_bfloat16* vc = v_cache_pool ? (v_cache_pool + slot * slot_stride) : nullptr;

    float m_val = -1e30f;
    float l_val = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kp = k_start + wid; kp < k_end; kp += nwarps) {
        int kslot = is_local ? (kp % cache_cap) : kp;
        float dot = 0.0f;
        float vv0, vv1, vv2, vv3;
        if (k_cache_fp8_pool) {
            int64_t sc_idx = ((int64_t)slot * cache_cap + kslot) * n_kv_heads + kv_h;
            float ks = k_scale_fp8_pool[sc_idx];
            float vs = v_scale_fp8_pool[sc_idx];
            const uint8_t* k_ptr = k_cache_fp8_pool + sc_idx * head_dim;
            const uint8_t* v_ptr = v_cache_fp8_pool + sc_idx * head_dim;
            #pragma unroll
            for (int i = 0; i < 4; ++i) dot = fmaf(qv[i], e4m3_to_f32(k_ptr[lane + i * 32]), dot);
            dot = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0) * inv_sqrt_d * ks;
            vv0 = e4m3_to_f32(v_ptr[lane + 0]) * vs;
            vv1 = e4m3_to_f32(v_ptr[lane + 32]) * vs;
            vv2 = e4m3_to_f32(v_ptr[lane + 64]) * vs;
            vv3 = e4m3_to_f32(v_ptr[lane + 96]) * vs;
        } else {
            const __nv_bfloat16* k_ptr = kc + (int64_t)kslot * kv_stride + kv_h * head_dim;
            const __nv_bfloat16* v_ptr = vc + (int64_t)kslot * kv_stride + kv_h * head_dim;
            #pragma unroll
            for (int i = 0; i < 4; ++i) dot = fmaf(qv[i], __bfloat162float(k_ptr[lane + i * 32]), dot);
            dot = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0) * inv_sqrt_d;
            vv0 = __bfloat162float(v_ptr[lane + 0]);
            vv1 = __bfloat162float(v_ptr[lane + 32]);
            vv2 = __bfloat162float(v_ptr[lane + 64]);
            vv3 = __bfloat162float(v_ptr[lane + 96]);
        }
        int dist = q_pos - kp;
        if (dist >= 0 && dist < rel_extent) dot += __bfloat162float(r_ptr[dist]);

        float m_new = fmaxf(m_val, dot);
        float alpha = expf(m_val - m_new);
        float p = expf(dot - m_new);
        l_val = l_val * alpha + p;
        acc[0] = acc[0] * alpha + p * vv0;
        acc[1] = acc[1] * alpha + p * vv1;
        acc[2] = acc[2] * alpha + p * vv2;
        acc[3] = acc[3] * alpha + p * vv3;
        m_val = m_new;
    }

    __shared__ float sm_m[8], sm_l[8], sm_acc[8][128];
    if (lane == 0) { sm_m[wid] = m_val; sm_l[wid] = l_val; }
    #pragma unroll
    for (int i = 0; i < 4; ++i) sm_acc[wid][lane + i * 32] = acc[i];
    __syncthreads();

    if (wid == 0) {
        float sink_logit = __bfloat162float(sinks[(int64_t)b * n_heads + h]);
        float gm = sink_logit;
        for (int w = 0; w < nwarps; ++w) if (sm_l[w] > 0.0f) gm = fmaxf(gm, sm_m[w]);
        float gl = expf(sink_logit - gm);
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
        const __nv_bfloat16* g_ptr = gate + ((int64_t)b * n_heads + h) * head_dim;
        __nv_bfloat16* o_ptr = out + ((int64_t)b * n_heads + h) * head_dim;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int d = lane + i * 32;
            float sig = 1.0f / (1.0f + expf(-__bfloat162float(g_ptr[d])));
            o_ptr[d] = __float2bfloat16(gacc[i] * inv_l * sig);
        }
    }
}


// GQA=3 Shared-KV Batched Decode Attention (8 warps = 256 threads per (kv_h, b), head_dim = 128)
// Cuts KV DRAM traffic by 3x for c=3..8 continuous batching!
__global__ void attn_decode_batched_gqa3_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ k_cache_pool,
    const __nv_bfloat16* __restrict__ v_cache_pool,
    const uint8_t* __restrict__ k_cache_fp8_pool,
    const uint8_t* __restrict__ v_cache_fp8_pool,
    const float* __restrict__ k_scale_fp8_pool,
    const float* __restrict__ v_scale_fp8_pool,
    const __nv_bfloat16* __restrict__ rel_logits,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ sinks,
    __nv_bfloat16* __restrict__ out,
    const int32_t* __restrict__ slot_ids,
    const int32_t* __restrict__ positions,
    int B, int cache_cap, bool is_local, int window_size,
    int n_heads, int n_kv_heads, int rel_extent)
{
    int kv_h = blockIdx.x;
    int b = blockIdx.y;
    if (kv_h >= n_kv_heads || b >= B) return;

    int slot = slot_ids[b];
    int q_pos = positions[b];
    int k_start = is_local ? max(0, q_pos - window_size + 1) : 0;
    int k_end = q_pos + 1;

    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;

    float qv[3][4];
    const __nv_bfloat16* r_ptr[3];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        int h = kv_h * 3 + g;
        const __nv_bfloat16* q_ptr = Q + ((int64_t)b * n_heads + h) * 128;
        r_ptr[g] = rel_logits + ((int64_t)b * n_heads + h) * rel_extent;
        #pragma unroll
        for (int i = 0; i < 4; ++i) qv[g][i] = __bfloat162float(q_ptr[lane + i * 32]);
    }
    const float inv_sqrt_d = rsqrtf(128.0f);

    int kv_stride = n_kv_heads * 128;
    int64_t slot_stride = (int64_t)cache_cap * kv_stride;
    const __nv_bfloat16* kc = k_cache_pool ? (k_cache_pool + slot * slot_stride) : nullptr;
    const __nv_bfloat16* vc = v_cache_pool ? (v_cache_pool + slot * slot_stride) : nullptr;

    float m_val[3] = {-1e30f, -1e30f, -1e30f};
    float l_val[3] = {0.0f, 0.0f, 0.0f};
    float acc[3][4];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[g][i] = 0.0f;
    }

    for (int kp = k_start + wid; kp < k_end; kp += nwarps) {
        int kslot = is_local ? (kp % cache_cap) : kp;
        float kv0, kv1, kv2, kv3;
        float vv0, vv1, vv2, vv3;
        float k_mult = inv_sqrt_d;
        if (k_cache_fp8_pool) {
            int64_t sc_idx = ((int64_t)slot * cache_cap + kslot) * n_kv_heads + kv_h;
            k_mult *= k_scale_fp8_pool[sc_idx];
            float vs = v_scale_fp8_pool[sc_idx];
            const uint8_t* k_ptr = k_cache_fp8_pool + sc_idx * 128;
            const uint8_t* v_ptr = v_cache_fp8_pool + sc_idx * 128;
            kv0 = e4m3_to_f32(k_ptr[lane + 0]);
            kv1 = e4m3_to_f32(k_ptr[lane + 32]);
            kv2 = e4m3_to_f32(k_ptr[lane + 64]);
            kv3 = e4m3_to_f32(k_ptr[lane + 96]);
            vv0 = e4m3_to_f32(v_ptr[lane + 0]) * vs;
            vv1 = e4m3_to_f32(v_ptr[lane + 32]) * vs;
            vv2 = e4m3_to_f32(v_ptr[lane + 64]) * vs;
            vv3 = e4m3_to_f32(v_ptr[lane + 96]) * vs;
        } else {
            const __nv_bfloat16* k_ptr = kc + (int64_t)kslot * kv_stride + kv_h * 128;
            const __nv_bfloat16* v_ptr = vc + (int64_t)kslot * kv_stride + kv_h * 128;
            kv0 = __bfloat162float(k_ptr[lane + 0]);
            kv1 = __bfloat162float(k_ptr[lane + 32]);
            kv2 = __bfloat162float(k_ptr[lane + 64]);
            kv3 = __bfloat162float(k_ptr[lane + 96]);
            vv0 = __bfloat162float(v_ptr[lane + 0]);
            vv1 = __bfloat162float(v_ptr[lane + 32]);
            vv2 = __bfloat162float(v_ptr[lane + 64]);
            vv3 = __bfloat162float(v_ptr[lane + 96]);
        }
        int dist = q_pos - kp;
        #pragma unroll
        for (int g = 0; g < 3; ++g) {
            float dot = fmaf(qv[g][0], kv0, fmaf(qv[g][1], kv1, fmaf(qv[g][2], kv2, qv[g][3] * kv3)));
            dot = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0) * k_mult;
            if (dist >= 0 && dist < rel_extent) dot += __bfloat162float(r_ptr[g][dist]);
            float m_new = fmaxf(m_val[g], dot);
            float alpha = expf(m_val[g] - m_new);
            float p = expf(dot - m_new);
            l_val[g] = l_val[g] * alpha + p;
            acc[g][0] = acc[g][0] * alpha + p * vv0;
            acc[g][1] = acc[g][1] * alpha + p * vv1;
            acc[g][2] = acc[g][2] * alpha + p * vv2;
            acc[g][3] = acc[g][3] * alpha + p * vv3;
            m_val[g] = m_new;
        }
    }

    __shared__ float sm_m[3][8], sm_l[3][8], sm_acc[3][8][128];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (lane == 0) { sm_m[g][wid] = m_val[g]; sm_l[g][wid] = l_val[g]; }
        #pragma unroll
        for (int i = 0; i < 4; ++i) sm_acc[g][wid][lane + i * 32] = acc[g][i];
    }
    __syncthreads();

    if (wid < 3) {
        int g = wid;
        int h = kv_h * 3 + g;
        float sink_logit = __bfloat162float(sinks[(int64_t)b * n_heads + h]);
        float gm = sink_logit;
        for (int w = 0; w < nwarps; ++w) if (sm_l[g][w] > 0.0f) gm = fmaxf(gm, sm_m[g][w]);
        float gl = expf(sink_logit - gm);
        float gacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int w = 0; w < nwarps; ++w) {
            if (sm_l[g][w] > 0.0f) {
                float sc = expf(sm_m[g][w] - gm);
                gl += sm_l[g][w] * sc;
                #pragma unroll
                for (int i = 0; i < 4; ++i) gacc[i] += sm_acc[g][w][lane + i * 32] * sc;
            }
        }
        float inv_l = 1.0f / gl;
        const __nv_bfloat16* g_ptr = gate + ((int64_t)b * n_heads + h) * 128;
        __nv_bfloat16* o_ptr = out + ((int64_t)b * n_heads + h) * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int d = lane + i * 32;
            float sig = 1.0f / (1.0f + expf(-__bfloat162float(g_ptr[d])));
            o_ptr[d] = __float2bfloat16(gacc[i] * inv_l * sig);
        }
    }
}

void launch_attention_batched(
    const __nv_bfloat16* Q, const __nv_bfloat16* k_cache_pool, const __nv_bfloat16* v_cache_pool,
    const __nv_bfloat16* rel_logits, const __nv_bfloat16* gate, const __nv_bfloat16* sinks,
    __nv_bfloat16* out, const int32_t* d_slot_ids, const int32_t* d_positions,
    int B, int cache_cap, bool is_local, int window_size,
    int n_heads, int n_kv_heads, int head_dim, int rel_extent, cudaStream_t stream,
    const uint8_t* k_cache_fp8_pool, const uint8_t* v_cache_fp8_pool,
    const float* k_scale_fp8_pool, const float* v_scale_fp8_pool)
{
    if (head_dim == 128 && n_heads == n_kv_heads * 3) {
        attn_decode_batched_gqa3_kernel<<<dim3(n_kv_heads, B), 256, 0, stream>>>(
            Q, k_cache_pool, v_cache_pool, k_cache_fp8_pool, v_cache_fp8_pool, k_scale_fp8_pool, v_scale_fp8_pool,
            rel_logits, gate, sinks, out, d_slot_ids, d_positions,
            B, cache_cap, is_local, window_size, n_heads, n_kv_heads, rel_extent);
    } else {
        attn_decode_batched_kernel<<<dim3(n_heads, B), 128, 0, stream>>>(
            Q, k_cache_pool, v_cache_pool, k_cache_fp8_pool, v_cache_fp8_pool, k_scale_fp8_pool, v_scale_fp8_pool,
            rel_logits, gate, sinks, out, d_slot_ids, d_positions,
            B, cache_cap, is_local, window_size, n_heads, n_kv_heads, head_dim, rel_extent);
    }
}

// ============================================================================
// 10b. Multi-Slot Batched Speculative Verify Kernels (B slots x L tokens)
// ============================================================================

__global__ void sconv_static_batch_verify_kernel(
    const __nv_bfloat16* __restrict__ x,      // [B * L, C]
    const __nv_bfloat16* __restrict__ weight, // [C, 4]
    __nv_bfloat16* __restrict__ state_pool,   // [max_slots, 3, C]
    __nv_bfloat16* __restrict__ snap_pool,    // [max_dspark_slots * 8, 3, C]
    const int32_t* __restrict__ slot_ids,     // [B]
    __nv_bfloat16* __restrict__ out,          // [B * L, C]
    int B, int L, int C)
{
    int b = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || c >= C) return;
    int slot = slot_ids[b];
    __nv_bfloat16* st = state_pool + (int64_t)slot * 3 * C;
    __nv_bfloat16* sn = snap_pool + (int64_t)slot * 8 * 3 * C;

    float w0 = __bfloat162float(weight[c * 4 + 0]);
    float w1 = __bfloat162float(weight[c * 4 + 1]);
    float w2 = __bfloat162float(weight[c * 4 + 2]);
    float w3 = __bfloat162float(weight[c * 4 + 3]);

    float s0 = __bfloat162float(st[0 * C + c]);
    float s1 = __bfloat162float(st[1 * C + c]);
    float s2 = __bfloat162float(st[2 * C + c]);

    for (int j = 0; j < L; ++j) {
        int t = b * L + j;
        float xt = __bfloat162float(x[(int64_t)t * C + c]);
        float y = (w0 * s0 + w1 * s1 + w2 * s2 + w3 * xt + xt) * 0.44721359549995793f;
        out[(int64_t)t * C + c] = __float2bfloat16(y);
        s0 = s1; s1 = s2; s2 = xt;
        __nv_bfloat16* sn_j = sn + (int64_t)j * 3 * C;
        sn_j[0 * C + c] = __float2bfloat16(s0);
        sn_j[1 * C + c] = __float2bfloat16(s1);
        sn_j[2 * C + c] = __float2bfloat16(s2);
    }
    st[0 * C + c] = __float2bfloat16(s0);
    st[1 * C + c] = __float2bfloat16(s1);
    st[2 * C + c] = __float2bfloat16(s2);
}

void launch_sconv_static_batch_verify(
    const __nv_bfloat16* x, const __nv_bfloat16* weight,
    __nv_bfloat16* state_pool, __nv_bfloat16* snap_pool, const int32_t* d_slot_ids,
    __nv_bfloat16* out, int B, int L, int C, cudaStream_t stream)
{
    sconv_static_batch_verify_kernel<<<dim3((C + 255) / 256, B), 256, 0, stream>>>(
        x, weight, state_pool, snap_pool, d_slot_ids, out, B, L, C);
}

__global__ void sconv_dynamic_batch_verify_kernel(
    const __nv_bfloat16* __restrict__ x,        // [B * L, C]
    const __nv_bfloat16* __restrict__ static_w, // [4, C]
    const __nv_bfloat16* __restrict__ dyn_w,    // [B * L, num_kv_heads * 4]
    __nv_bfloat16* __restrict__ state_pool,     // [max_slots, 3, C]
    __nv_bfloat16* __restrict__ snap_pool,      // [max_dspark_slots * 8, 3, C]
    const int32_t* __restrict__ slot_ids,       // [B]
    __nv_bfloat16* __restrict__ out,            // [B * L, C]
    int B, int L, int num_kv_heads, int head_dim)
{
    int C = num_kv_heads * head_dim;
    int b = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || c >= C) return;
    int slot = slot_ids[b];
    int h = c / head_dim;
    __nv_bfloat16* st = state_pool + (int64_t)slot * 3 * C;
    __nv_bfloat16* sn = snap_pool + (int64_t)slot * 8 * 3 * C;

    float sw0 = __bfloat162float(static_w[0 * C + c]);
    float sw1 = __bfloat162float(static_w[1 * C + c]);
    float sw2 = __bfloat162float(static_w[2 * C + c]);
    float sw3 = __bfloat162float(static_w[3 * C + c]);

    float s0 = __bfloat162float(st[0 * C + c]);
    float s1 = __bfloat162float(st[1 * C + c]);
    float s2 = __bfloat162float(st[2 * C + c]);

    for (int j = 0; j < L; ++j) {
        int t = b * L + j;
        const __nv_bfloat16* dw = dyn_w + (int64_t)t * (num_kv_heads * 4) + h * 4;
        float xt = __bfloat162float(x[(int64_t)t * C + c]);
        float y = ((sw0 + __bfloat162float(dw[0])) * xt
                 + (sw1 + __bfloat162float(dw[1])) * s2
                 + (sw2 + __bfloat162float(dw[2])) * s1
                 + (sw3 + __bfloat162float(dw[3])) * s0 + xt) * (1.0f / 3.0f);
        out[(int64_t)t * C + c] = __float2bfloat16(y);
        s0 = s1; s1 = s2; s2 = xt;
        __nv_bfloat16* sn_j = sn + (int64_t)j * 3 * C;
        sn_j[0 * C + c] = __float2bfloat16(s0);
        sn_j[1 * C + c] = __float2bfloat16(s1);
        sn_j[2 * C + c] = __float2bfloat16(s2);
    }
    st[0 * C + c] = __float2bfloat16(s0);
    st[1 * C + c] = __float2bfloat16(s1);
    st[2 * C + c] = __float2bfloat16(s2);
}

void launch_sconv_dynamic_batch_verify(
    const __nv_bfloat16* x, const __nv_bfloat16* static_w, const __nv_bfloat16* dyn_w,
    __nv_bfloat16* state_pool, __nv_bfloat16* snap_pool, const int32_t* d_slot_ids,
    __nv_bfloat16* out, int B, int L, int num_kv_heads, int head_dim, cudaStream_t stream)
{
    int C = num_kv_heads * head_dim;
    sconv_dynamic_batch_verify_kernel<<<dim3((C + 255) / 256, B), 256, 0, stream>>>(
        x, static_w, dyn_w, state_pool, snap_pool, d_slot_ids, out, B, L, num_kv_heads, head_dim);
}

__global__ void attn_prep_q_r_batch_verify_kernel(
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ R,
    const __nv_bfloat16* __restrict__ q_scale,
    const __nv_bfloat16* __restrict__ r_scale,
    const int32_t* __restrict__ positions,
    int B, int L, bool is_local, int n_heads, int head_dim, int d_rel,
    float qk_clip, float rel_clip, float log_n_floor, float log_alpha)
{
    int h = blockIdx.x;
    int t = blockIdx.y;
    if (h >= n_heads || t >= B * L) return;
    int b = t / L;
    int j = t % L;
    int pos = positions[b] + j;
    float tau = 1.0f;
    if (!is_local) {
        tau = 1.0f + log_alpha * logf(fmaxf(1.0f, (float)(pos + 1) / log_n_floor));
    }
    __nv_bfloat16* q_ptr = Q + ((int64_t)t * n_heads + h) * head_dim;
    const __nv_bfloat16* qs_ptr = q_scale + h * head_dim;
    float qv[4], sum_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = threadIdx.x + i * 32;
        qv[i] = __bfloat162float(q_ptr[d]) * __bfloat162float(qs_ptr[d]);
        sum_sq += qv[i] * qv[i];
    }
    sum_sq = warp_reduce_sum(sum_sq);
    float scale = fminf(1.0f, qk_clip / (sqrtf(__shfl_sync(0xffffffff, sum_sq, 0)) + 1e-6f)) * tau;
    #pragma unroll
    for (int i = 0; i < 4; ++i) q_ptr[threadIdx.x + i * 32] = __float2bfloat16(qv[i] * scale);

    __nv_bfloat16* r_ptr = R + ((int64_t)t * n_heads + h) * d_rel;
    const __nv_bfloat16* rs_ptr = r_scale + h * d_rel;
    float rv = (threadIdx.x < d_rel) ? __bfloat162float(r_ptr[threadIdx.x]) * __bfloat162float(rs_ptr[threadIdx.x]) : 0.0f;
    float r_sq = warp_reduce_sum(rv * rv);
    float r_sc = fminf(1.0f, rel_clip / (sqrtf(__shfl_sync(0xffffffff, r_sq, 0)) + 1e-6f)) * tau;
    if (threadIdx.x < d_rel) r_ptr[threadIdx.x] = __float2bfloat16(rv * r_sc);
}

__global__ void attn_prep_kv_batch_verify_kernel(
    const __nv_bfloat16* __restrict__ K_in,
    const __nv_bfloat16* __restrict__ V_in,
    __nv_bfloat16* __restrict__ k_cache_pool,
    __nv_bfloat16* __restrict__ v_cache_pool,
    uint8_t* __restrict__ k_cache_fp8_pool,
    uint8_t* __restrict__ v_cache_fp8_pool,
    float* __restrict__ k_scale_fp8_pool,
    float* __restrict__ v_scale_fp8_pool,
    const int32_t* __restrict__ slot_ids,
    const int32_t* __restrict__ positions,
    int B, int L, int cache_cap, bool is_local, int n_kv_heads, int head_dim, float qk_clip, float eps)
{
    int h = blockIdx.x;
    int t = blockIdx.y;
    if (h >= n_kv_heads || t >= B * L) return;
    int b = t / L;
    int j = t % L;
    int slot = slot_ids[b];
    int pos = positions[b] + j;
    int kpos = is_local ? (pos % cache_cap) : pos;

    const __nv_bfloat16* kin = K_in + ((int64_t)t * n_kv_heads + h) * head_dim;
    const __nv_bfloat16* vin = V_in + ((int64_t)t * n_kv_heads + h) * head_dim;

    float kv[4], vv[4], k_sq = 0.0f, v_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = threadIdx.x + i * 32;
        kv[i] = __bfloat162float(kin[d]);
        vv[i] = __bfloat162float(vin[d]);
        k_sq += kv[i] * kv[i];
        v_sq += vv[i] * vv[i];
    }
    float k_sc = fminf(1.0f, qk_clip / (sqrtf(__shfl_sync(0xffffffff, warp_reduce_sum(k_sq), 0)) + 1e-6f));
    float v_rstd = rsqrtf(__shfl_sync(0xffffffff, warp_reduce_sum(v_sq), 0) / (float)head_dim + eps);
    if (k_cache_fp8_pool) {
        float km = 0.0f, vm = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            kv[i] *= k_sc; vv[i] *= v_rstd;
            km = fmaxf(km, fabsf(kv[i])); vm = fmaxf(vm, fabsf(vv[i]));
        }
        float ks = __shfl_sync(0xffffffff, warp_reduce_max(km), 0); ks = (ks > 0.0f) ? (ks / 448.0f) : 1.0f;
        float vs = __shfl_sync(0xffffffff, warp_reduce_max(vm), 0); vs = (vs > 0.0f) ? (vs / 448.0f) : 1.0f;
        float inv_ks = 1.0f / ks, inv_vs = 1.0f / vs;
        int64_t sc_idx = ((int64_t)slot * cache_cap + kpos) * n_kv_heads + h;
        if (threadIdx.x == 0) { k_scale_fp8_pool[sc_idx] = ks; v_scale_fp8_pool[sc_idx] = vs; }
        uint8_t* kdst = k_cache_fp8_pool + sc_idx * head_dim;
        uint8_t* vdst = v_cache_fp8_pool + sc_idx * head_dim;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int d = threadIdx.x + i * 32;
            kdst[d] = (uint8_t)__nv_cvt_float_to_fp8(kv[i] * inv_ks, __NV_SATFINITE, __NV_E4M3);
            vdst[d] = (uint8_t)__nv_cvt_float_to_fp8(vv[i] * inv_vs, __NV_SATFINITE, __NV_E4M3);
        }
    } else {
        int64_t slot_stride = (int64_t)cache_cap * n_kv_heads * head_dim;
        __nv_bfloat16* kdst = k_cache_pool + slot * slot_stride + ((int64_t)kpos * n_kv_heads + h) * head_dim;
        __nv_bfloat16* vdst = v_cache_pool + slot * slot_stride + ((int64_t)kpos * n_kv_heads + h) * head_dim;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int d = threadIdx.x + i * 32;
            kdst[d] = __float2bfloat16(kv[i] * k_sc);
            vdst[d] = __float2bfloat16(vv[i] * v_rstd);
        }
    }
}

void launch_attn_prep_batch_verify(
    __nv_bfloat16* Q, const __nv_bfloat16* K_in, const __nv_bfloat16* V_in, __nv_bfloat16* R,
    const __nv_bfloat16* q_scale, const __nv_bfloat16* r_scale,
    __nv_bfloat16* k_cache_pool, __nv_bfloat16* v_cache_pool,
    const int32_t* d_slot_ids, const int32_t* d_positions,
    int B, int L, int cache_cap, bool is_local, int n_heads, int n_kv_heads, int head_dim, int d_rel,
    float qk_clip, float rel_clip, float log_n_floor, float log_alpha, float eps, cudaStream_t stream,
    uint8_t* k_cache_fp8_pool, uint8_t* v_cache_fp8_pool, float* k_scale_fp8_pool, float* v_scale_fp8_pool)
{
    attn_prep_q_r_batch_verify_kernel<<<dim3(n_heads, B * L), 32, 0, stream>>>(
        Q, R, q_scale, r_scale, d_positions, B, L, is_local, n_heads, head_dim, d_rel,
        qk_clip, rel_clip, log_n_floor, log_alpha);
    attn_prep_kv_batch_verify_kernel<<<dim3(n_kv_heads, B * L), 32, 0, stream>>>(
        K_in, V_in, k_cache_pool, v_cache_pool, k_cache_fp8_pool, v_cache_fp8_pool, k_scale_fp8_pool, v_scale_fp8_pool,
        d_slot_ids, d_positions, B, L, cache_cap, is_local, n_kv_heads, head_dim, qk_clip, eps);
}

__global__ void attn_batch_verify_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ k_cache_pool,
    const __nv_bfloat16* __restrict__ v_cache_pool,
    const uint8_t* __restrict__ k_cache_fp8_pool,
    const uint8_t* __restrict__ v_cache_fp8_pool,
    const float* __restrict__ k_scale_fp8_pool,
    const float* __restrict__ v_scale_fp8_pool,
    const __nv_bfloat16* __restrict__ rel_logits,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ sinks,
    __nv_bfloat16* __restrict__ out,
    const int32_t* __restrict__ slot_ids,
    const int32_t* __restrict__ positions,
    int B, int L, int cache_cap, bool is_local, int window_size,
    int n_heads, int n_kv_heads, int head_dim, int rel_extent)
{
    int h = blockIdx.x;
    int t = blockIdx.y;
    if (h >= n_heads || t >= B * L) return;

    int b = t / L;
    int j = t % L;
    int slot = slot_ids[b];
    int q_pos = positions[b] + j;
    int k_start = is_local ? max(0, q_pos - window_size + 1) : 0;
    int k_end = q_pos + 1;
    int kv_h = h / (n_heads / n_kv_heads);

    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;

    const __nv_bfloat16* q_ptr = Q + ((int64_t)t * n_heads + h) * head_dim;
    const __nv_bfloat16* r_ptr = rel_logits + ((int64_t)t * n_heads + h) * rel_extent;
    float qv[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) qv[i] = __bfloat162float(q_ptr[lane + i * 32]);
    const float inv_sqrt_d = rsqrtf((float)head_dim);

    int kv_stride = n_kv_heads * head_dim;
    int64_t slot_stride = (int64_t)cache_cap * kv_stride;
    const __nv_bfloat16* kc = k_cache_pool ? (k_cache_pool + slot * slot_stride) : nullptr;
    const __nv_bfloat16* vc = v_cache_pool ? (v_cache_pool + slot * slot_stride) : nullptr;

    float m_val = -1e30f;
    float l_val = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kp = k_start + wid; kp < k_end; kp += nwarps) {
        int kslot = is_local ? (kp % cache_cap) : kp;
        float dot = 0.0f;
        float vv0, vv1, vv2, vv3;
        if (k_cache_fp8_pool) {
            int64_t sc_idx = ((int64_t)slot * cache_cap + kslot) * n_kv_heads + kv_h;
            float ks = k_scale_fp8_pool[sc_idx];
            float vs = v_scale_fp8_pool[sc_idx];
            const uint8_t* k_ptr = k_cache_fp8_pool + sc_idx * head_dim;
            const uint8_t* v_ptr = v_cache_fp8_pool + sc_idx * head_dim;
            #pragma unroll
            for (int i = 0; i < 4; ++i) dot = fmaf(qv[i], e4m3_to_f32(k_ptr[lane + i * 32]), dot);
            dot = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0) * inv_sqrt_d * ks;
            vv0 = e4m3_to_f32(v_ptr[lane + 0]) * vs;
            vv1 = e4m3_to_f32(v_ptr[lane + 32]) * vs;
            vv2 = e4m3_to_f32(v_ptr[lane + 64]) * vs;
            vv3 = e4m3_to_f32(v_ptr[lane + 96]) * vs;
        } else {
            const __nv_bfloat16* k_ptr = kc + (int64_t)kslot * kv_stride + kv_h * head_dim;
            const __nv_bfloat16* v_ptr = vc + (int64_t)kslot * kv_stride + kv_h * head_dim;
            #pragma unroll
            for (int i = 0; i < 4; ++i) dot = fmaf(qv[i], __bfloat162float(k_ptr[lane + i * 32]), dot);
            dot = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0) * inv_sqrt_d;
            vv0 = __bfloat162float(v_ptr[lane + 0]);
            vv1 = __bfloat162float(v_ptr[lane + 32]);
            vv2 = __bfloat162float(v_ptr[lane + 64]);
            vv3 = __bfloat162float(v_ptr[lane + 96]);
        }
        int dist = q_pos - kp;
        if (dist >= 0 && dist < rel_extent) dot += __bfloat162float(r_ptr[dist]);

        float m_new = fmaxf(m_val, dot);
        float alpha = expf(m_val - m_new);
        float p = expf(dot - m_new);
        l_val = l_val * alpha + p;
        acc[0] = acc[0] * alpha + p * vv0;
        acc[1] = acc[1] * alpha + p * vv1;
        acc[2] = acc[2] * alpha + p * vv2;
        acc[3] = acc[3] * alpha + p * vv3;
        m_val = m_new;
    }

    __shared__ float sm_m[8], sm_l[8], sm_acc[8][128];
    if (lane == 0) { sm_m[wid] = m_val; sm_l[wid] = l_val; }
    #pragma unroll
    for (int i = 0; i < 4; ++i) sm_acc[wid][lane + i * 32] = acc[i];
    __syncthreads();

    if (wid == 0) {
        float sink_logit = __bfloat162float(sinks[(int64_t)t * n_heads + h]);
        float gm = sink_logit;
        for (int w = 0; w < nwarps; ++w) if (sm_l[w] > 0.0f) gm = fmaxf(gm, sm_m[w]);
        float gl = expf(sink_logit - gm);
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
        const __nv_bfloat16* g_ptr = gate + ((int64_t)t * n_heads + h) * head_dim;
        __nv_bfloat16* o_ptr = out + ((int64_t)t * n_heads + h) * head_dim;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int d = lane + i * 32;
            float sig = 1.0f / (1.0f + expf(-__bfloat162float(g_ptr[d])));
            o_ptr[d] = __float2bfloat16(gacc[i] * inv_l * sig);
        }
    }
}


// GQA=3 Shared-KV + Shared-L Split-K Batch-Verify Attention (B slots x L tokens, head_dim = 128)
// Cuts KV DRAM traffic by 3*L (12x at B=2, L=4) while saturating all 48 SMs via Split-K!
__global__ void attn_batch_verify_gqa3_split_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ k_cache_pool,
    const __nv_bfloat16* __restrict__ v_cache_pool,
    const uint8_t* __restrict__ k_cache_fp8_pool,
    const uint8_t* __restrict__ v_cache_fp8_pool,
    const float* __restrict__ k_scale_fp8_pool,
    const float* __restrict__ v_scale_fp8_pool,
    const __nv_bfloat16* __restrict__ rel_logits,
    float* __restrict__ split_scratch, // [B*L, n_heads, num_splits, 130]
    const int32_t* __restrict__ slot_ids,
    const int32_t* __restrict__ positions,
    int B, int L, int num_splits, int cache_cap, bool is_local, int window_size,
    int n_heads, int n_kv_heads, int rel_extent)
{
    int kb = blockIdx.x;
    int s = blockIdx.y;
    int b = kb / n_kv_heads;
    int kv_h = kb % n_kv_heads;
    int j = threadIdx.x >> 5;   // 0..L-1
    int lane = threadIdx.x & 31;
    if (b >= B || s >= num_splits) return;

    int slot = slot_ids[b];
    int sp = positions[b];
    int blk_k_start = is_local ? max(0, sp - window_size + 1) : 0;
    int blk_k_end = sp + L;
    int total_k = blk_k_end - blk_k_start;
    int chunk = (total_k + num_splits - 1) / num_splits;
    int s_start = blk_k_start + s * chunk;
    int s_end = min(blk_k_end, s_start + chunk);

    int t = b * L + j;
    int q_pos = sp + j;
    int my_k_start = is_local ? max(0, q_pos - window_size + 1) : 0;
    int my_k_end = q_pos + 1;

    float qv[3][4];
    const __nv_bfloat16* r_ptr[3];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        int h = kv_h * 3 + g;
        const __nv_bfloat16* q_ptr = Q + ((int64_t)t * n_heads + h) * 128;
        r_ptr[g] = rel_logits + ((int64_t)t * n_heads + h) * rel_extent;
        #pragma unroll
        for (int i = 0; i < 4; ++i) qv[g][i] = __bfloat162float(q_ptr[lane + i * 32]);
    }
    const float inv_sqrt_d = rsqrtf(128.0f);
    int kv_stride = n_kv_heads * 128;
    int64_t slot_stride = (int64_t)cache_cap * kv_stride;
    const __nv_bfloat16* kc = k_cache_pool ? (k_cache_pool + slot * slot_stride) : nullptr;
    const __nv_bfloat16* vc = v_cache_pool ? (v_cache_pool + slot * slot_stride) : nullptr;

    float m_val[3] = {-1e30f, -1e30f, -1e30f};
    float l_val[3] = {0.0f, 0.0f, 0.0f};
    float acc[3][4];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[g][i] = 0.0f;
    }

    __shared__ __align__(16) union { uint8_t u8[32][128]; __nv_bfloat16 bf16[32][128]; } sm_k, sm_v;
    __shared__ float sm_ks[32], sm_vs[32];

    for (int k_base = s_start; k_base < s_end; k_base += 32) {
        int tile_len = min(32, s_end - k_base);
        if (k_cache_fp8_pool) {
            int total_u4 = tile_len * 8;
            for (int idx = threadIdx.x; idx < total_u4; idx += blockDim.x) {
                int kk = idx >> 3;
                int col = idx & 7;
                int kp = k_base + kk;
                int kslot = is_local ? (kp % cache_cap) : kp;
                int64_t sc_idx = ((int64_t)slot * cache_cap + kslot) * n_kv_heads + kv_h;
                reinterpret_cast<uint4*>(sm_k.u8[kk])[col] = reinterpret_cast<const uint4*>(k_cache_fp8_pool + sc_idx * 128)[col];
                reinterpret_cast<uint4*>(sm_v.u8[kk])[col] = reinterpret_cast<const uint4*>(v_cache_fp8_pool + sc_idx * 128)[col];
            }
            if (threadIdx.x < tile_len) {
                int kp = k_base + threadIdx.x;
                int kslot = is_local ? (kp % cache_cap) : kp;
                int64_t sc_idx = ((int64_t)slot * cache_cap + kslot) * n_kv_heads + kv_h;
                sm_ks[threadIdx.x] = k_scale_fp8_pool[sc_idx];
                sm_vs[threadIdx.x] = v_scale_fp8_pool[sc_idx];
            }
        } else {
            int total_u4 = tile_len * 16;
            for (int idx = threadIdx.x; idx < total_u4; idx += blockDim.x) {
                int kk = idx >> 4;
                int col = idx & 15;
                int kp = k_base + kk;
                int kslot = is_local ? (kp % cache_cap) : kp;
                int64_t base = (int64_t)kslot * kv_stride + kv_h * 128;
                reinterpret_cast<uint4*>(sm_k.bf16[kk])[col] = reinterpret_cast<const uint4*>(kc + base)[col];
                reinterpret_cast<uint4*>(sm_v.bf16[kk])[col] = reinterpret_cast<const uint4*>(vc + base)[col];
            }
        }
        __syncthreads();

        for (int kk = 0; kk < tile_len; ++kk) {
            int kp = k_base + kk;
            if (kp < my_k_start || kp >= my_k_end) continue;
            float kv0, kv1, kv2, kv3;
            float vv0, vv1, vv2, vv3;
            float k_mult = inv_sqrt_d;
            if (k_cache_fp8_pool) {
                k_mult *= sm_ks[kk];
                float vs = sm_vs[kk];
                kv0 = e4m3_to_f32(sm_k.u8[kk][lane + 0]);
                kv1 = e4m3_to_f32(sm_k.u8[kk][lane + 32]);
                kv2 = e4m3_to_f32(sm_k.u8[kk][lane + 64]);
                kv3 = e4m3_to_f32(sm_k.u8[kk][lane + 96]);
                vv0 = e4m3_to_f32(sm_v.u8[kk][lane + 0]) * vs;
                vv1 = e4m3_to_f32(sm_v.u8[kk][lane + 32]) * vs;
                vv2 = e4m3_to_f32(sm_v.u8[kk][lane + 64]) * vs;
                vv3 = e4m3_to_f32(sm_v.u8[kk][lane + 96]) * vs;
            } else {
                kv0 = __bfloat162float(sm_k.bf16[kk][lane + 0]);
                kv1 = __bfloat162float(sm_k.bf16[kk][lane + 32]);
                kv2 = __bfloat162float(sm_k.bf16[kk][lane + 64]);
                kv3 = __bfloat162float(sm_k.bf16[kk][lane + 96]);
                vv0 = __bfloat162float(sm_v.bf16[kk][lane + 0]);
                vv1 = __bfloat162float(sm_v.bf16[kk][lane + 32]);
                vv2 = __bfloat162float(sm_v.bf16[kk][lane + 64]);
                vv3 = __bfloat162float(sm_v.bf16[kk][lane + 96]);
            }
            int dist = q_pos - kp;
            #pragma unroll
            for (int g = 0; g < 3; ++g) {
                float dot = fmaf(qv[g][0], kv0, fmaf(qv[g][1], kv1, fmaf(qv[g][2], kv2, qv[g][3] * kv3)));
                dot = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0) * k_mult;
                if (dist >= 0 && dist < rel_extent) dot += __bfloat162float(r_ptr[g][dist]);
                float m_new = fmaxf(m_val[g], dot);
                float alpha = expf(m_val[g] - m_new);
                float p = expf(dot - m_new);
                l_val[g] = l_val[g] * alpha + p;
                acc[g][0] = acc[g][0] * alpha + p * vv0;
                acc[g][1] = acc[g][1] * alpha + p * vv1;
                acc[g][2] = acc[g][2] * alpha + p * vv2;
                acc[g][3] = acc[g][3] * alpha + p * vv3;
                m_val[g] = m_new;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        int h = kv_h * 3 + g;
        float* dst = split_scratch + (((int64_t)t * n_heads + h) * num_splits + s) * 130;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[lane + i * 32] = acc[g][i];
        if (lane == 0) {
            dst[128] = m_val[g];
            dst[129] = l_val[g];
        }
    }
}

void launch_attention_batch_verify(
    const __nv_bfloat16* Q, const __nv_bfloat16* k_cache_pool, const __nv_bfloat16* v_cache_pool,
    const __nv_bfloat16* rel_logits, const __nv_bfloat16* gate, const __nv_bfloat16* sinks,
    __nv_bfloat16* out, const int32_t* d_slot_ids, const int32_t* d_positions,
    int B, int L, int cache_cap, bool is_local, int window_size,
    int n_heads, int n_kv_heads, int head_dim, int rel_extent, cudaStream_t stream,
    const uint8_t* k_cache_fp8_pool, const uint8_t* v_cache_fp8_pool,
    const float* k_scale_fp8_pool, const float* v_scale_fp8_pool,
    float* split_scratch)
{
    if (head_dim == 128 && n_heads == n_kv_heads * 3 && B * L <= 16 && L <= 8 && split_scratch != nullptr) {
        int num_splits = is_local ? 4 : max(2, min(12, 48 / (n_kv_heads * B)));
        attn_batch_verify_gqa3_split_kernel<<<dim3(n_kv_heads * B, num_splits), L * 32, 0, stream>>>(
            Q, k_cache_pool, v_cache_pool, k_cache_fp8_pool, v_cache_fp8_pool, k_scale_fp8_pool, v_scale_fp8_pool,
            rel_logits, split_scratch, d_slot_ids, d_positions,
            B, L, num_splits, cache_cap, is_local, window_size, n_heads, n_kv_heads, rel_extent);
        attn_small_t_gqa3_merge_kernel<<<dim3(n_heads, B * L), 32, 0, stream>>>(
            split_scratch, gate, sinks, out, num_splits, n_heads);
    } else {
        attn_batch_verify_kernel<<<dim3(n_heads, B * L), 128, 0, stream>>>(
            Q, k_cache_pool, v_cache_pool, k_cache_fp8_pool, v_cache_fp8_pool, k_scale_fp8_pool, v_scale_fp8_pool,
            rel_logits, gate, sinks, out, d_slot_ids, d_positions,
            B, L, cache_cap, is_local, window_size, n_heads, n_kv_heads, head_dim, rel_extent);
    }
}

// ============================================================================
// 11. Final Logit Softcap + GPU Greedy Argmax
// ============================================================================

__global__ void softcap_argmax_kernel(
    float* __restrict__ logits,
    int32_t* __restrict__ out_token,
    int vocab_size, float softcap)
{
    float best_v = -1e30f;
    int best_i = 0;
    float inv_cap = 1.0f / softcap;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float v = tanhf(logits[i] * inv_cap) * softcap;
        logits[i] = v;
        if (v > best_v) {
            best_v = v;
            best_i = i;
        }
    }
    // Warp reduction for argmax
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffff, best_v, off);
        int oi = __shfl_down_sync(0xffffffff, best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }
    __shared__ float sm_v[32];
    __shared__ int sm_i[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) { sm_v[wid] = best_v; sm_i[wid] = best_i; }
    __syncthreads();
    if (wid == 0) {
        int nwarps = blockDim.x >> 5;
        best_v = (lane < nwarps) ? sm_v[lane] : -1e30f;
        best_i = (lane < nwarps) ? sm_i[lane] : 0;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffff, best_v, off);
            int oi = __shfl_down_sync(0xffffffff, best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (lane == 0) *out_token = best_i;
    }
}

void launch_softcap_argmax(
    float* logits, int32_t* out_token, int vocab_size, float softcap, cudaStream_t stream)
{
    softcap_argmax_kernel<<<1, 1024, 0, stream>>>(logits, out_token, vocab_size, softcap);
}

__global__ void batched_argmax_kernel(
    const float* __restrict__ logits, int32_t* __restrict__ out_tokens, int M, int vocab_size)
{
    int m = blockIdx.x;
    if (m >= M) return;
    const float* row = logits + (int64_t)m * vocab_size;
    float best_v = -1e30f;
    int best_i = 0;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float v = row[i];
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
        if (lane == 0) out_tokens[m] = best_i;
    }
}

void launch_batched_argmax(
    const float* logits, int32_t* out_tokens, int M, int vocab_size, cudaStream_t stream)
{
    batched_argmax_kernel<<<M, 1024, 0, stream>>>(logits, out_tokens, M, vocab_size);
}

} // namespace ink
