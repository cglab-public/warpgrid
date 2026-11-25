#include "kernels_gguf.cuh"
#include "../gguf_iq_tables.hpp"
#include <cuda_fp16.h>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>

namespace ink::cuda_gguf {

namespace {

__device__ __constant__ float d_e2m1_lut[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __constant__ float d_iq4nl_lut[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
       1.0f,   13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f
};

__device__ __constant__ uint8_t d_signs_iq2xs[128];
__device__ __constant__ uint64_t d_iq2xxs_grid[256];
__device__ __constant__ uint32_t d_iq1s_grid_gpu[2048];

void ensure_iq_tables_uploaded() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        cudaMemcpyToSymbol(d_signs_iq2xs, ink::kSignsIq2xs, sizeof(ink::kSignsIq2xs));
        cudaMemcpyToSymbol(d_iq2xxs_grid, ink::kIq2xxsGrid, sizeof(ink::kIq2xxsGrid));
        cudaMemcpyToSymbol(d_iq1s_grid_gpu, ink::kIq1sGridGpu, sizeof(ink::kIq1sGridGpu));
    });
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    }
    return v;
}

__device__ __forceinline__ float load_f16(const uint8_t* p) {
    uint16_t bits = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    __half h = *reinterpret_cast<const __half*>(&bits);
    return __half2float(h);
}

__device__ __forceinline__ float e8m0_to_f32(uint8_t e) {
    if (e == 0u) return 5.877471754111438e-39f;
    return __uint_as_float((uint32_t)e << 23);
}

__device__ __forceinline__ void get_scale_min_k4(uint32_t j, const uint8_t* scales, uint8_t& sc_out, uint8_t& m_out) {
    if (j < 4u) {
        sc_out = scales[j] & 63u;
        m_out = scales[j + 4u] & 63u;
    } else {
        sc_out = (scales[j + 4u] & 0x0Fu) | ((scales[j - 4u] >> 6) << 4);
        m_out = (scales[j + 4u] >> 4) | ((scales[j] >> 6) << 4);
    }
}

__device__ __forceinline__ float dot_f32_row(const float* x_row, const uint8_t* w_row, uint32_t K, uint32_t lane) {
    const float* wp = reinterpret_cast<const float*>(w_row);
    float acc = 0.0f;
    for (uint32_t k = lane; k < K; k += 32u) acc = fmaf(x_row[k], wp[k], acc);
    return warp_sum(acc);
}

__device__ __forceinline__ float dot_bf16_row(const float* x_row, const uint8_t* w_row, uint32_t K, uint32_t lane) {
    const uint16_t* wp = reinterpret_cast<const uint16_t*>(w_row);
    float acc = 0.0f;
    for (uint32_t k = lane; k < K; k += 32u) {
        float w = __uint_as_float((uint32_t)wp[k] << 16);
        acc = fmaf(x_row[k], w, acc);
    }
    return warp_sum(acc);
}

__device__ __forceinline__ float dot_q8_0_row(const float* x_row, const uint8_t* w_row, uint32_t K, uint32_t lane) {
    uint32_t nb = K >> 5;
    float acc = 0.0f;
    for (uint32_t b = lane; b < nb; b += 32u) {
        const uint8_t* blk = w_row + (size_t)b * 34u;
        float d = load_f16(blk);
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2u);
        const float* xp = x_row + (b << 5);
        float sum = 0.0f;
        for (uint32_t i = 0; i < 32u; ++i) sum = fmaf(xp[i], (float)qs[i], sum);
        acc = fmaf(d, sum, acc);
    }
    return warp_sum(acc);
}

__device__ __forceinline__ float dot_iq4_nl_row(const float* x_row, const uint8_t* w_row, uint32_t K, uint32_t lane) {
    uint32_t nb = K >> 5;
    float acc = 0.0f;
    for (uint32_t b = lane; b < nb; b += 32u) {
        const uint8_t* blk = w_row + (size_t)b * 18u;
        float d = load_f16(blk);
        const uint8_t* qs = blk + 2u;
        const float* xp = x_row + (b << 5);
        float sum = 0.0f;
        for (uint32_t i = 0; i < 16u; ++i) {
            uint8_t q = qs[i];
            sum = fmaf(xp[i], d_iq4nl_lut[q & 0x0Fu], sum);
            sum = fmaf(xp[i + 16u], d_iq4nl_lut[q >> 4], sum);
        }
        acc = fmaf(d, sum, acc);
    }
    return warp_sum(acc);
}

__device__ __forceinline__ float dot_mxfp4_row(const float* x_row, const uint8_t* w_row, uint32_t K, uint32_t lane) {
    uint32_t nb = K >> 5;
    float acc = 0.0f;
    for (uint32_t b = lane; b < nb; b += 32u) {
        const uint8_t* blk = w_row + (size_t)b * 17u;
        float sc = e8m0_to_f32(blk[0]);
        const uint8_t* qs = blk + 1u;
        const float* xp = x_row + (b << 5);
        float sum = 0.0f;
        for (uint32_t i = 0; i < 16u; ++i) {
            uint8_t q = qs[i];
            sum = fmaf(xp[i], d_e2m1_lut[q & 0x0Fu], sum);
            sum = fmaf(xp[i + 16u], d_e2m1_lut[q >> 4], sum);
        }
        acc = fmaf(sc, sum, acc);
    }
    return warp_sum(acc);
}

__device__ __forceinline__ float dot_q4_k_row(const float* x_row, const uint8_t* w_row, uint32_t K, uint32_t lane) {
    uint32_t nb = K >> 8;
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint8_t* blk = w_row + (size_t)b * 144u;
        float d = load_f16(blk);
        float dmin = load_f16(blk + 2u);
        const uint8_t* scales = blk + 4u;
        const uint8_t* qs = blk + 16u;
        const float* xp = x_row + (b << 8);
        for (uint32_t sb = 0; sb < 4u; ++sb) {
            uint8_t sc0 = 0, m0 = 0, sc1 = 0, m1 = 0;
            get_scale_min_k4(sb * 2u, scales, sc0, m0);
            get_scale_min_k4(sb * 2u + 1u, scales, sc1, m1);
            float d1 = d * (float)sc0, min1 = dmin * (float)m0;
            float d2 = d * (float)sc1, min2 = dmin * (float)m1;
            uint8_t q = qs[sb * 32u + lane];
            acc = fmaf(xp[sb * 64u + lane], d1 * (float)(q & 0x0Fu) - min1, acc);
            acc = fmaf(xp[sb * 64u + 32u + lane], d2 * (float)(q >> 4) - min2, acc);
        }
    }
    return warp_sum(acc);
}

__device__ __forceinline__ float dot_q5_k_row(const float* x_row, const uint8_t* w_row, uint32_t K, uint32_t lane) {
    uint32_t nb = K >> 8;
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint8_t* blk = w_row + (size_t)b * 176u;
        float d = load_f16(blk);
        float dmin = load_f16(blk + 2u);
        const uint8_t* scales = blk + 4u;
        const uint8_t* qh = blk + 16u;
        const uint8_t* qs = blk + 48u;
        const float* xp = x_row + (b << 8);
        uint8_t h = qh[lane];
        uint8_t u1 = 1u, u2 = 2u;
        for (uint32_t sb = 0; sb < 4u; ++sb) {
            uint8_t sc0 = 0, m0 = 0, sc1 = 0, m1 = 0;
            get_scale_min_k4(sb * 2u, scales, sc0, m0);
            get_scale_min_k4(sb * 2u + 1u, scales, sc1, m1);
            float d1 = d * (float)sc0, min1 = dmin * (float)m0;
            float d2 = d * (float)sc1, min2 = dmin * (float)m1;
            uint8_t q = qs[sb * 32u + lane];
            float wlo = d1 * (float)((q & 0x0Fu) + ((h & u1) ? 16u : 0u)) - min1;
            float whi = d2 * (float)((q >> 4) + ((h & u2) ? 16u : 0u)) - min2;
            acc = fmaf(xp[sb * 64u + lane], wlo, acc);
            acc = fmaf(xp[sb * 64u + 32u + lane], whi, acc);
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    return warp_sum(acc);
}

__device__ __forceinline__ float dot_q6_k_row(const float* x_row, const uint8_t* w_row, uint32_t K, uint32_t lane) {
    uint32_t nb = K >> 8;
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint8_t* blk = w_row + (size_t)b * 210u;
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + 128u;
        const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192u);
        float d = load_f16(blk + 208u);
        const float* xp = x_row + (b << 8);
        for (uint32_t half_idx = 0; half_idx < 2u; ++half_idx) {
            const uint8_t* qlp = ql + half_idx * 64u;
            const uint8_t* qhp = qh + half_idx * 32u;
            const int8_t* scp = sc + half_idx * 8u;
            const float* xhp = xp + half_idx * 128u;
            uint32_t is = lane >> 4;
            uint8_t h = qhp[lane];
            int q1 = (int)((qlp[lane] & 0x0Fu) | (((h >> 0) & 3u) << 4)) - 32;
            int q2 = (int)((qlp[lane + 32u] & 0x0Fu) | (((h >> 2) & 3u) << 4)) - 32;
            int q3 = (int)((qlp[lane] >> 4) | (((h >> 4) & 3u) << 4)) - 32;
            int q4 = (int)((qlp[lane + 32u] >> 4) | (((h >> 6) & 3u) << 4)) - 32;
            float s = xhp[lane] * (float)scp[is + 0u] * (float)q1
                    + xhp[lane + 32u] * (float)scp[is + 2u] * (float)q2
                    + xhp[lane + 64u] * (float)scp[is + 4u] * (float)q3
                    + xhp[lane + 96u] * (float)scp[is + 6u] * (float)q4;
            acc = fmaf(d, s, acc);
        }
    }
    return warp_sum(acc);
}

__device__ __forceinline__ float dot_q2_k_row(const float* x_row, const uint8_t* w_row, uint32_t K, uint32_t lane) {
    uint32_t nb = K >> 8;
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint8_t* blk = w_row + (size_t)b * 84u;
        const uint8_t* scales = blk;
        const uint8_t* qs = blk + 16u;
        float d = load_f16(blk + 80u);
        float dmin = load_f16(blk + 82u);
        const float* xp = x_row + (b << 8);
        for (uint32_t half_idx = 0; half_idx < 2u; ++half_idx) {
            const uint8_t* qp = qs + half_idx * 32u;
            const float* xhp = xp + half_idx * 128u;
            uint32_t is_base = half_idx * 8u;
            uint32_t is_off = lane >> 4;
            uint8_t qval = qp[lane];
            for (uint32_t j = 0; j < 4u; ++j) {
                uint8_t sc = scales[is_base + j * 2u + is_off];
                float dl = d * (float)(sc & 0x0Fu);
                float ml = dmin * (float)(sc >> 4);
                float w = dl * (float)((qval >> (j * 2u)) & 3u) - ml;
                acc = fmaf(xhp[j * 32u + lane], w, acc);
            }
        }
    }
    return warp_sum(acc);
}

__device__ __forceinline__ float dot_iq2_xxs_row(const float* x_row, const uint8_t* w_row, uint32_t K, uint32_t lane) {
    uint32_t nb32 = K >> 5;
    float acc = 0.0f;
    for (uint32_t ib32 = lane; ib32 < nb32; ib32 += 32u) {
        uint32_t ibl = ib32 >> 3;
        uint32_t ib = ib32 & 7u;
        const uint8_t* blk = w_row + (size_t)ibl * 66u;
        float d = load_f16(blk);
        const uint8_t* qp = blk + 2u + ib * 8u;
        uint32_t aux_g = (uint32_t)qp[0] | ((uint32_t)qp[1] << 8) | ((uint32_t)qp[2] << 16) | ((uint32_t)qp[3] << 24);
        uint32_t aux_s = (uint32_t)qp[4] | ((uint32_t)qp[5] << 8) | ((uint32_t)qp[6] << 16) | ((uint32_t)qp[7] << 24);
        float db = d * (0.5f + (float)(aux_s >> 28)) * 0.25f;
        const float* xp = x_row + (ib32 << 5);
        float s = 0.0f;
        for (uint32_t l = 0; l < 4u; ++l) {
            uint64_t g64 = d_iq2xxs_grid[(aux_g >> (l * 8u)) & 0xFFu];
            uint8_t sg = d_signs_iq2xs[(aux_s >> (7u * l)) & 127u];
            for (uint32_t j = 0; j < 8u; ++j) {
                float gv = (float)((g64 >> (j * 8u)) & 0xFFull);
                float sgn = (sg & (1u << j)) ? -1.0f : 1.0f;
                s = fmaf(xp[l * 8u + j], gv * sgn, s);
            }
        }
        acc = fmaf(db, s, acc);
    }
    return warp_sum(acc);
}

__device__ __forceinline__ float dot_iq1_s_row(const float* x_row, const uint8_t* w_row, uint32_t K, uint32_t lane) {
    uint32_t nb32 = K >> 5;
    float acc = 0.0f;
    for (uint32_t ib32 = lane; ib32 < nb32; ib32 += 32u) {
        uint32_t ibl = ib32 >> 3;
        uint32_t ib = ib32 & 7u;
        const uint8_t* blk = w_row + (size_t)ibl * 50u;
        float d = load_f16(blk);
        uint16_t qh = (uint16_t)blk[34u + ib * 2u] | ((uint16_t)blk[35u + ib * 2u] << 8);
        float dl = d * (float)(2u * (((uint32_t)qh >> 12) & 7u) + 1u);
        float delta = (qh & 0x8000u) ? (-1.0f - 0.125f) : (-1.0f + 0.125f);
        const uint8_t* qs = blk + 2u + ib * 4u;
        const float* xp = x_row + (ib32 << 5);
        float s_g = 0.0f;
        float s_x = 0.0f;
        for (uint32_t l = 0; l < 4u; ++l) {
            uint32_t idx = (uint32_t)qs[l] | ((((uint32_t)qh >> (3u * l)) & 7u) << 8);
            uint32_t pk = d_iq1s_grid_gpu[idx];
            const float* xl = xp + l * 8u;
            for (uint32_t j = 0; j < 4u; ++j) {
                uint32_t bp = (pk >> (j * 8u)) & 0xFFu;
                float x0 = xl[j];
                float x1 = xl[j + 4u];
                s_g = fmaf(x0, (float)(bp & 0x0Fu), fmaf(x1, (float)(bp >> 4), s_g));
                s_x += x0 + x1;
            }
        }
        acc = fmaf(dl, fmaf(delta, s_x, s_g), acc);
    }
    return warp_sum(acc);
}

__device__ __forceinline__ size_t dev_row_bytes(GgmlType t, uint32_t K) {
    switch (t) {
        case GgmlType::F32: return (size_t)K * 4u;
        case GgmlType::BF16: return (size_t)K * 2u;
        case GgmlType::Q8_0: return (size_t)(K >> 5) * 34u;
        case GgmlType::IQ4_NL: return (size_t)(K >> 5) * 18u;
        case GgmlType::MXFP4: return (size_t)(K >> 5) * 17u;
        case GgmlType::Q4_K: return (size_t)(K >> 8) * 144u;
        case GgmlType::Q5_K: return (size_t)(K >> 8) * 176u;
        case GgmlType::Q6_K: return (size_t)(K >> 8) * 210u;
        case GgmlType::Q2_K: return (size_t)(K >> 8) * 84u;
        case GgmlType::IQ2_XXS: return (size_t)(K >> 8) * 66u;
        case GgmlType::IQ1_S: return (size_t)(K >> 8) * 50u;
        default: return 0;
    }
}

__device__ __forceinline__ float dot_dispatch(GgmlType t, const float* x_row, const uint8_t* w_row, uint32_t K, uint32_t lane) {
    switch (t) {
        case GgmlType::F32: return dot_f32_row(x_row, w_row, K, lane);
        case GgmlType::BF16: return dot_bf16_row(x_row, w_row, K, lane);
        case GgmlType::Q8_0: return dot_q8_0_row(x_row, w_row, K, lane);
        case GgmlType::IQ4_NL: return dot_iq4_nl_row(x_row, w_row, K, lane);
        case GgmlType::MXFP4: return dot_mxfp4_row(x_row, w_row, K, lane);
        case GgmlType::Q4_K: return dot_q4_k_row(x_row, w_row, K, lane);
        case GgmlType::Q5_K: return dot_q5_k_row(x_row, w_row, K, lane);
        case GgmlType::Q6_K: return dot_q6_k_row(x_row, w_row, K, lane);
        case GgmlType::Q2_K: return dot_q2_k_row(x_row, w_row, K, lane);
        case GgmlType::IQ2_XXS: return dot_iq2_xxs_row(x_row, w_row, K, lane);
        case GgmlType::IQ1_S: return dot_iq1_s_row(x_row, w_row, K, lane);
        default: return 0.0f;
    }
}

__global__ void k_gguf_gemv(GgmlType type, const float* X, const uint8_t* W, float* Y,
                            uint32_t M, uint32_t N, uint32_t K) {
    uint32_t m = blockIdx.y;
    uint32_t n = blockIdx.x * 4u + (threadIdx.x >> 5);
    uint32_t lane = threadIdx.x & 31u;
    if (m >= M || n >= N) return;
    size_t rb = dev_row_bytes(type, K);
    float s = dot_dispatch(type, X + (size_t)m * K, W + (size_t)n * rb, K, lane);
    if (lane == 0u) Y[(size_t)m * N + n] = s;
}

__global__ void k_gguf_gemv_f32_wide(const float* X, const float* W, float* Y,
                                     uint32_t M, uint32_t N, uint32_t K) {
    uint32_t m = blockIdx.y;
    uint32_t n = blockIdx.x;
    if (m >= M || n >= N) return;
    const float4* xp = reinterpret_cast<const float4*>(X + (size_t)m * K);
    const float4* wp = reinterpret_cast<const float4*>(W + (size_t)n * K);
    uint32_t k4 = K >> 2;
    float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint32_t k = threadIdx.x; k < k4; k += blockDim.x) {
        float4 xv = xp[k];
        float4 wv = wp[k];
        a.x = fmaf(xv.x, wv.x, a.x);
        a.y = fmaf(xv.y, wv.y, a.y);
        a.z = fmaf(xv.z, wv.z, a.z);
        a.w = fmaf(xv.w, wv.w, a.w);
    }
    float s = warp_sum(a.x + a.y + a.z + a.w);
    __shared__ float partials[8];
    uint32_t sg = threadIdx.x >> 5;
    uint32_t lane = threadIdx.x & 31u;
    if (lane == 0u) partials[sg] = s;
    __syncthreads();
    if (sg == 0u) {
        float v = (lane < 8u) ? partials[lane] : 0.0f;
        float tot = warp_sum(v);
        if (lane == 0u) Y[(size_t)m * N + n] = tot;
    }
}

__global__ void k_gguf_grouped_gemv(GgmlType type, const float* X, const uint8_t* W, float* Y,
                                    uint32_t T, uint32_t G, uint32_t N, uint32_t K, uint32_t group_stride_bytes) {
    uint32_t tg = blockIdx.y;
    uint32_t n = blockIdx.x * 4u + (threadIdx.x >> 5);
    uint32_t lane = threadIdx.x & 31u;
    if (tg >= T * G || n >= N) return;
    uint32_t g = tg % G;
    size_t rb = dev_row_bytes(type, K);
    const uint8_t* w_row = W + (size_t)g * group_stride_bytes + (size_t)n * rb;
    float s = dot_dispatch(type, X + (size_t)tg * K, w_row, K, lane);
    if (lane == 0u) Y[(size_t)tg * N + n] = s;
}

__global__ void k_gguf_moe_indexed(GgmlType type, const float* X, const uint8_t* W,
                                   const int32_t* topk_ids, float* Y,
                                   uint32_t T, uint32_t N, uint32_t K, uint32_t top_k, uint32_t expert_stride_bytes) {
    uint32_t tr = blockIdx.y;
    uint32_t n = blockIdx.x * 4u + (threadIdx.x >> 5);
    uint32_t lane = threadIdx.x & 31u;
    if (tr >= T * top_k || n >= N) return;
    int32_t e_id = topk_ids[tr];
    float s = 0.0f;
    if (e_id >= 0) {
        uint32_t t = tr / top_k;
        size_t rb = dev_row_bytes(type, K);
        const uint8_t* w_row = W + (size_t)(uint32_t)e_id * expert_stride_bytes + (size_t)n * rb;
        s = dot_dispatch(type, X + (size_t)t * K, w_row, K, lane);
    }
    if (lane == 0u) Y[(size_t)tr * N + n] = s;
}

__global__ void k_gguf_moe_gate_up_swiglu(GgmlType type, const float* X, const uint8_t* Wg, const uint8_t* Wu,
                                          const int32_t* topk_ids, float* Y,
                                          uint32_t T, uint32_t N, uint32_t K, uint32_t top_k,
                                          uint32_t expert_stride_bytes, float clamp_val) {
    uint32_t tr = blockIdx.y;
    uint32_t n = blockIdx.x * 4u + (threadIdx.x >> 5);
    uint32_t lane = threadIdx.x & 31u;
    if (tr >= T * top_k || n >= N) return;
    int32_t e_id = topk_ids[tr];
    if (e_id < 0) {
        if (lane == 0u) Y[(size_t)tr * N + n] = 0.0f;
        return;
    }
    uint32_t t = tr / top_k;
    size_t rb = dev_row_bytes(type, K);
    size_t off = (size_t)(uint32_t)e_id * expert_stride_bytes + (size_t)n * rb;
    const float* x_row = X + (size_t)t * K;
    float gv = dot_dispatch(type, x_row, Wg + off, K, lane);
    float uv = dot_dispatch(type, x_row, Wu + off, K, lane);
    if (lane == 0u) {
        if (clamp_val > 0.0f) {
            gv = fminf(fmaxf(gv, -clamp_val), clamp_val);
            uv = fminf(fmaxf(uv, -clamp_val), clamp_val);
        }
        Y[(size_t)tr * N + n] = (gv / (1.0f + expf(-gv))) * uv;
    }
}

__global__ void k_gguf_moe_down_accum(GgmlType type, const float* X, const uint8_t* W,
                                      const int32_t* topk_ids, const float* topk_w, float* Y,
                                      uint32_t T, uint32_t N, uint32_t K, uint32_t top_k, uint32_t expert_stride_bytes) {
    uint32_t t = blockIdx.y;
    uint32_t n = blockIdx.x * 4u + (threadIdx.x >> 5);
    uint32_t lane = threadIdx.x & 31u;
    if (t >= T || n >= N) return;
    size_t rb = dev_row_bytes(type, K);
    float total = 0.0f;
    for (uint32_t r = 0; r < top_k; ++r) {
        uint32_t tr = t * top_k + r;
        int32_t e_id = topk_ids[tr];
        if (e_id < 0) continue;
        float gw = topk_w[tr];
        const uint8_t* w_row = W + (size_t)(uint32_t)e_id * expert_stride_bytes + (size_t)n * rb;
        total = fmaf(gw, dot_dispatch(type, X + (size_t)tr * K, w_row, K, lane), total);
    }
    if (lane == 0u) Y[(size_t)t * N + n] += total;
}

__global__ void k_ds_mhc_expand(const float* X, float* S, uint32_t T, uint32_t D, uint32_t hc) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t t = blockIdx.y;
    if (t >= T || d >= D) return;
    float v = X[(size_t)t * D + d];
    for (uint32_t s = 0; s < hc; ++s) {
        S[((size_t)t * hc + s) * D + d] = v;
    }
}

__global__ void k_ds_rmsnorm(const float* X, const float* W, float* Y, uint32_t T, uint32_t D, float eps) {
    uint32_t t = blockIdx.x;
    if (t >= T) return;
    const float* xp = X + (size_t)t * D;
    float* yp = Y + (size_t)t * D;
    float ss = 0.0f;
    for (uint32_t d = threadIdx.x; d < D; d += blockDim.x) {
        float v = xp[d];
        ss = fmaf(v, v, ss);
    }
    __shared__ float partials[8];
    uint32_t sg = threadIdx.x >> 5;
    uint32_t lane = threadIdx.x & 31u;
    float sg_sum = warp_sum(ss);
    if (lane == 0u) partials[sg] = sg_sum;
    __syncthreads();
    uint32_t n_sg = (blockDim.x + 31u) >> 5;
    if (sg == 0u) {
        float v = (lane < n_sg) ? partials[lane] : 0.0f;
        float tot = warp_sum(v);
        if (lane == 0u) partials[0] = rsqrtf(tot / (float)D + eps);
    }
    __syncthreads();
    float inv = partials[0];
    if (W) {
        for (uint32_t d = threadIdx.x; d < D; d += blockDim.x) yp[d] = xp[d] * inv * W[d];
    } else {
        for (uint32_t d = threadIdx.x; d < D; d += blockDim.x) yp[d] = xp[d] * inv;
    }
}

__global__ void k_ds_hc_mixes(const float* raw, const float* scale, const float* base,
                              float* out_pre, float* out_post, float* out_comb,
                              uint32_t T, uint32_t hc_mult, uint32_t sinkhorn_iters, float eps) {
    uint32_t t = blockIdx.x;
    if (t >= T || threadIdx.x != 0u) return;
    uint32_t hc = hc_mult < 8u ? hc_mult : 8u;
    uint32_t total = (2u + hc) * hc;
    const float* rp = raw + (size_t)t * total;
    float s_pre = scale[0], s_post = scale[1], s_comb = scale[2];
    float* pre = out_pre + (size_t)t * hc;
    float* post = out_post + (size_t)t * hc;
    float* comb = out_comb + (size_t)t * hc * hc;

    for (uint32_t i = 0; i < hc; ++i) {
        float vp = rp[i] * s_pre + base[i];
        pre[i] = (1.0f / (1.0f + expf(-vp))) + eps;
        float vo = rp[hc + i] * s_post + base[hc + i];
        post[i] = 2.0f / (1.0f + expf(-vo));
    }

    float M[64];
    for (uint32_t src = 0; src < hc; ++src) {
        float mx = -1e30f;
        for (uint32_t dst = 0; dst < hc; ++dst) {
            uint32_t idx = src * hc + dst;
            float v = rp[2u * hc + idx] * s_comb + base[2u * hc + idx];
            M[idx] = v;
            mx = fmaxf(mx, v);
        }
        float sm = 0.0f;
        for (uint32_t dst = 0; dst < hc; ++dst) {
            float e = expf(M[src * hc + dst] - mx);
            M[src * hc + dst] = e;
            sm += e;
        }
        float inv = 1.0f / fmaxf(sm, 1e-20f);
        for (uint32_t dst = 0; dst < hc; ++dst) {
            M[src * hc + dst] = M[src * hc + dst] * inv + eps;
        }
    }
    for (uint32_t it = 0; it < sinkhorn_iters; ++it) {
        if (it > 0u) {
            for (uint32_t src = 0; src < hc; ++src) {
                float rs = eps;
                for (uint32_t dst = 0; dst < hc; ++dst) rs += M[src * hc + dst];
                float inv = 1.0f / rs;
                for (uint32_t dst = 0; dst < hc; ++dst) M[src * hc + dst] *= inv;
            }
        }
        for (uint32_t dst = 0; dst < hc; ++dst) {
            float cs = eps;
            for (uint32_t src = 0; src < hc; ++src) cs += M[src * hc + dst];
            float inv = 1.0f / cs;
            for (uint32_t src = 0; src < hc; ++src) M[src * hc + dst] *= inv;
        }
    }
    for (uint32_t i = 0; i < hc * hc; ++i) comb[i] = M[i];
}

__global__ void k_ds_hc_pre(const float* streams, const float* pre_w, float* out_hidden,
                            uint32_t T, uint32_t D, uint32_t hc) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t t = blockIdx.y;
    if (t >= T || d >= D) return;
    float acc = 0.0f;
    for (uint32_t s = 0; s < hc; ++s) {
        acc = fmaf(pre_w[(size_t)t * hc + s], streams[((size_t)t * hc + s) * D + d], acc);
    }
    out_hidden[(size_t)t * D + d] = acc;
}

__global__ void k_ds_hc_post(const float* sub, const float* residual, const float* post_w,
                             const float* comb_w, float* out_streams, uint32_t T, uint32_t D, uint32_t hc) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t t = blockIdx.y;
    if (t >= T || d >= D) return;
    float sv = sub[(size_t)t * D + d];
    const float* pw = post_w + (size_t)t * hc;
    const float* cw = comb_w + (size_t)t * hc * hc;
    for (uint32_t s = 0; s < hc; ++s) {
        float acc = pw[s] * sv;
        for (uint32_t r = 0; r < hc; ++r) {
            acc = fmaf(cw[r * hc + s], residual[((size_t)t * hc + r) * D + d], acc);
        }
        out_streams[((size_t)t * hc + s) * D + d] = acc;
    }
}

__global__ void k_ds_engram_apply(float* streams, const float* ekv, const float* gate,
                                  uint32_t T, uint32_t D, uint32_t hc, float eps) {
    uint32_t s = blockIdx.x;
    uint32_t t = blockIdx.y;
    uint32_t lane = threadIdx.x;
    if (s >= hc || t >= T) return;
    float* xp = streams + ((size_t)t * hc + s) * D;
    const float* kv_base = ekv + (size_t)t * (hc + 1u) * D;
    const float* kp = kv_base + (size_t)s * D;
    const float* vp = kv_base + (size_t)hc * D;
    const float* gp = gate + (size_t)s * D;

    float x_ss = 0.0f, k_ss = 0.0f;
    for (uint32_t d = lane; d < D; d += 32u) {
        x_ss = fmaf(xp[d], xp[d], x_ss);
        k_ss = fmaf(kp[d], kp[d], k_ss);
    }
    float inv_x = rsqrtf(warp_sum(x_ss) / (float)D + eps);
    float inv_k = rsqrtf(warp_sum(k_ss) / (float)D + eps);

    float dot = 0.0f;
    for (uint32_t d = lane; d < D; d += 32u) {
        dot = fmaf((xp[d] * inv_x) * (kp[d] * inv_k), gp[d], dot);
    }
    float dval = warp_sum(dot) * rsqrtf((float)D);
    float mag = sqrtf(fmaxf(fabsf(dval), 1e-6f));
    float signed_mag = (dval < 0.0f) ? -mag : mag;
    float g = 1.0f / (1.0f + expf(-signed_mag));
    for (uint32_t d = lane; d < D; d += 32u) {
        xp[d] += vp[d] * g;
    }
}

__global__ void k_ds_rope_partial(float* X, uint32_t T, uint32_t start_pos, uint32_t pos_stride,
                                  uint32_t n_heads, uint32_t head_dim, uint32_t n_nope,
                                  uint32_t n_rot, uint32_t inverse, float freq_base) {
    uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t h = blockIdx.y;
    uint32_t t = blockIdx.z;
    if (pair >= (n_rot >> 1) || h >= n_heads || t >= T) return;
    float pos = (float)(start_pos + t * pos_stride);
    float theta = pos * powf(freq_base, -2.0f * (float)pair / (float)n_rot);
    float c = cosf(theta);
    float s = sinf(theta);
    if (inverse) s = -s;
    float* p = X + ((size_t)t * n_heads + h) * head_dim + n_nope + pair * 2u;
    float x0 = p[0];
    float x1 = p[1];
    p[0] = x0 * c - x1 * s;
    p[1] = x0 * s + x1 * c;
}

__global__ void k_ds_comp_group_reduce(const float* raw_kv, const float* raw_score, const float* norm_w,
                                       float* comp_kv, uint32_t n_blocks, uint32_t ratio,
                                       uint32_t head_dim, float eps) {
    uint32_t b = blockIdx.x;
    uint32_t lane = threadIdx.x;
    if (b >= n_blocks) return;
    float* out = comp_kv + (size_t)b * head_dim;
    float ss = 0.0f;
    for (uint32_t d = lane; d < head_dim; d += 32u) {
        float acc = 0.0f;
        if (ratio == 1u) {
            acc = raw_kv[(size_t)b * head_dim + d];
        } else {
            float mx = -1e30f;
            for (uint32_t r = 0; r < ratio; ++r) {
                float sc = raw_score[((size_t)b * ratio + r) * head_dim + d];
                if (sc > mx) mx = sc;
            }
            float denom = 0.0f;
            for (uint32_t r = 0; r < ratio; ++r) {
                float w = expf(raw_score[((size_t)b * ratio + r) * head_dim + d] - mx);
                denom += w;
                acc = fmaf(w, raw_kv[((size_t)b * ratio + r) * head_dim + d], acc);
            }
            acc /= denom;
        }
        out[d] = acc;
        ss = fmaf(acc, acc, ss);
    }
    float inv = rsqrtf(warp_sum(ss) / (float)head_dim + eps);
    for (uint32_t d = lane; d < head_dim; d += 32u) {
        out[d] = out[d] * inv * norm_w[d];
    }
}

__global__ void k_ds_indexer_score_topk(const float* idx_q, const float* comp_ik, const float* idx_w,
                                        float* scratch_scores, int32_t* out_topk,
                                        uint32_t T, uint32_t start_pos, uint32_t n_comp, uint32_t max_comp,
                                        uint32_t n_idx_head, uint32_t idx_dim, uint32_t ratio,
                                        uint32_t sliding_window, uint32_t top_k, float iw_scale) {
    uint32_t t = blockIdx.x;
    uint32_t lane = threadIdx.x;
    if (t >= T) return;
    uint32_t q_pos = start_pos + t;
    uint32_t cutoff = (sliding_window > 0u && q_pos + 1u > sliding_window) ? (q_pos + 1u - sliding_window) : 0u;
    uint32_t n_elig = cutoff / ratio;
    if (n_elig > n_comp) n_elig = n_comp;

    float* sc_buf = scratch_scores + (size_t)t * max_comp;
    const float* qp = idx_q + (size_t)t * n_idx_head * idx_dim;
    const float* wp = idx_w + (size_t)t * n_idx_head;

    for (uint32_t c = 0; c < n_elig; ++c) {
        const float* kp = comp_ik + (size_t)c * idx_dim;
        float tot = 0.0f;
        for (uint32_t h = 0; h < n_idx_head; ++h) {
            const float* qh = qp + (size_t)h * idx_dim;
            float dot = 0.0f;
            for (uint32_t d = lane; d < idx_dim; d += 32u) dot = fmaf(qh[d], kp[d], dot);
            float s = warp_sum(dot);
            if (lane == 0u && s > 0.0f) tot = fmaf(wp[h] * iw_scale, s, tot);
        }
        if (lane == 0u) sc_buf[c] = tot;
    }
    __syncwarp();
    if (lane == 0u) {
        int32_t* dst = out_topk + (size_t)t * top_k;
        for (uint32_t k = 0; k < top_k; ++k) {
            if (k >= n_elig) {
                dst[k] = -1;
                continue;
            }
            int32_t best_i = -1;
            float best_v = -1e30f;
            for (uint32_t c = 0; c < n_elig; ++c) {
                if (sc_buf[c] > best_v) {
                    best_v = sc_buf[c];
                    best_i = (int32_t)c;
                }
            }
            dst[k] = best_i;
            if (best_i >= 0) sc_buf[best_i] = -1e30f;
        }
    }
}

__global__ void k_ds_mqa_sink_attn(const float* Q, const float* kv_cache, const float* comp_kv,
                                   const int32_t* topk_sel, const float* sinks, float* out,
                                   uint32_t T, uint32_t start_pos, uint32_t n_heads, uint32_t head_dim,
                                   uint32_t sliding_window, uint32_t n_comp_sel, float scale) {
    uint32_t h = blockIdx.x;
    uint32_t t = blockIdx.y;
    uint32_t lane = threadIdx.x;
    if (h >= n_heads || t >= T) return;

    const float* qp = Q + ((size_t)t * n_heads + h) * head_dim;
    float* op = out + ((size_t)t * n_heads + h) * head_dim;
    uint32_t q_pos = start_pos + t;
    uint32_t win_start = (sliding_window > 0u && q_pos + 1u > sliding_window) ? (q_pos + 1u - sliding_window) : 0u;

    float m = sinks[h];
    float l = 1.0f;
    for (uint32_t d = lane; d < head_dim; d += 32u) op[d] = 0.0f;

    if (comp_kv && topk_sel && n_comp_sel > 0u) {
        const int32_t* sel = topk_sel + (size_t)t * n_comp_sel;
        for (uint32_t idx = 0; idx < n_comp_sel; ++idx) {
            int32_t c = sel[idx];
            if (c < 0) continue;
            const float* kvp = comp_kv + (size_t)(uint32_t)c * head_dim;
            float dot = 0.0f;
            for (uint32_t d = lane; d < head_dim; d += 32u) dot = fmaf(qp[d], kvp[d], dot);
            float sc = __shfl_sync(0xFFFFFFFFu, warp_sum(dot), 0) * scale;
            float m_new = fmaxf(m, sc);
            float alpha = expf(m - m_new);
            float beta = expf(sc - m_new);
            l = l * alpha + beta;
            for (uint32_t d = lane; d < head_dim; d += 32u) op[d] = op[d] * alpha + beta * kvp[d];
            m = m_new;
        }
    }

    for (uint32_t pos = win_start; pos <= q_pos; ++pos) {
        const float* kvp = kv_cache + (size_t)pos * head_dim;
        float dot = 0.0f;
        for (uint32_t d = lane; d < head_dim; d += 32u) dot = fmaf(qp[d], kvp[d], dot);
        float sc = __shfl_sync(0xFFFFFFFFu, warp_sum(dot), 0) * scale;
        float m_new = fmaxf(m, sc);
        float alpha = expf(m - m_new);
        float beta = expf(sc - m_new);
        l = l * alpha + beta;
        for (uint32_t d = lane; d < head_dim; d += 32u) op[d] = op[d] * alpha + beta * kvp[d];
        m = m_new;
    }

    float inv_l = 1.0f / l;
    for (uint32_t d = lane; d < head_dim; d += 32u) op[d] *= inv_l;
}

__global__ void k_ds_router_sqrtsoftplus_topk(const float* logits, const float* bias,
                                              int32_t* out_ids, float* out_w,
                                              uint32_t T, uint32_t E, uint32_t top_k,
                                              uint32_t has_bias, float route_scale) {
    uint32_t t = blockIdx.x;
    if (t >= T || threadIdx.x != 0u) return;
    const float* lp = logits + (size_t)t * E;
    int32_t* ids = out_ids + (size_t)t * top_k;
    float* ws = out_w + (size_t)t * top_k;

    float picked_w[16];
    int32_t picked_id[16];
    for (uint32_t k = 0; k < top_k; ++k) {
        picked_id[k] = -1;
        picked_w[k] = 0.0f;
    }
    for (uint32_t k = 0; k < top_k; ++k) {
        int32_t best_e = -1;
        float best_s = -1e30f;
        float best_p = 0.0f;
        for (uint32_t e = 0; e < E; ++e) {
            bool used = false;
            for (uint32_t j = 0; j < k; ++j) {
                if (picked_id[j] == (int32_t)e) { used = true; break; }
            }
            if (used) continue;
            float x = lp[e];
            float sp = (x > 20.0f) ? x : log1pf(expf(x));
            float p = sqrtf(fmaxf(sp, 0.0f));
            float s = has_bias ? (p + bias[e]) : p;
            if (s > best_s) {
                best_s = s;
                best_e = (int32_t)e;
                best_p = p;
            }
        }
        picked_id[k] = best_e;
        picked_w[k] = best_p;
    }
    float denom = 1e-20f;
    for (uint32_t k = 0; k < top_k; ++k) denom += picked_w[k];
    for (uint32_t k = 0; k < top_k; ++k) {
        ids[k] = picked_id[k];
        ws[k] = (picked_w[k] / denom) * route_scale;
    }
}

__global__ void k_ds_swiglu(const float* gate, const float* up, float* out, uint32_t n, float clamp_val) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float u = up[i];
    if (clamp_val > 0.0f) {
        g = fminf(fmaxf(g, -clamp_val), clamp_val);
        u = fminf(fmaxf(u, -clamp_val), clamp_val);
    }
    out[i] = (g / (1.0f + expf(-g))) * u;
}

__global__ void k_ds_zero(float* dst, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = 0.0f;
}

__global__ void k_ds_argmax(const float* logits, int32_t* out_tok, uint32_t vocab_size) {
    float bv = -INFINITY;
    int32_t bi = 0;
    for (uint32_t i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float v = logits[i];
        if (v > bv) { bv = v; bi = (int32_t)i; }
    }
    __shared__ float s_val[256];
    __shared__ int32_t s_idx[256];
    s_val[threadIdx.x] = bv;
    s_idx[threadIdx.x] = bi;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (threadIdx.x < stride) {
            float ov = s_val[threadIdx.x + stride];
            int32_t oi = s_idx[threadIdx.x + stride];
            if (ov > s_val[threadIdx.x] || (ov == s_val[threadIdx.x] && oi < s_idx[threadIdx.x])) {
                s_val[threadIdx.x] = ov;
                s_idx[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) out_tok[0] = s_idx[0];
}

void validate_quant_type(GgmlType type) {
    switch (type) {
        case GgmlType::F32:
        case GgmlType::BF16:
        case GgmlType::Q8_0:
        case GgmlType::IQ4_NL:
        case GgmlType::MXFP4:
        case GgmlType::Q4_K:
        case GgmlType::Q5_K:
        case GgmlType::Q6_K:
        case GgmlType::Q2_K:
        case GgmlType::IQ2_XXS:
        case GgmlType::IQ1_S:
            return;
        default:
            throw std::runtime_error("Unsupported CUDA GGUF quant type: " + std::to_string((uint32_t)type));
    }
}

}

void launch_gemv(GgmlType type, const float* x, const uint8_t* w, float* y,
                 uint32_t M, uint32_t N, uint32_t K, cudaStream_t stream) {
    validate_quant_type(type);
    ensure_iq_tables_uploaded();
    if (type == GgmlType::F32 && N <= 64u && (K & 3u) == 0u &&
        ((reinterpret_cast<uintptr_t>(x) & 15u) == 0u) && ((reinterpret_cast<uintptr_t>(w) & 15u) == 0u)) {
        k_gguf_gemv_f32_wide<<<dim3(N, M, 1), 256, 0, stream>>>(x, reinterpret_cast<const float*>(w), y, M, N, K);
        return;
    }
    k_gguf_gemv<<<dim3((N + 3u) / 4u, M, 1), 128, 0, stream>>>(type, x, w, y, M, N, K);
}

void launch_grouped_gemv(GgmlType type, const float* x, const uint8_t* w, float* y,
                         uint32_t T, uint32_t G, uint32_t N, uint32_t K,
                         uint32_t group_stride_bytes, cudaStream_t stream) {
    validate_quant_type(type);
    ensure_iq_tables_uploaded();
    k_gguf_grouped_gemv<<<dim3((N + 3u) / 4u, T * G, 1), 128, 0, stream>>>(type, x, w, y, T, G, N, K, group_stride_bytes);
}

void launch_moe_indexed(GgmlType type, const float* x, const uint8_t* w,
                        const int32_t* topk_ids, float* y,
                        uint32_t T, uint32_t N, uint32_t K, uint32_t top_k,
                        uint32_t expert_stride_bytes, cudaStream_t stream) {
    validate_quant_type(type);
    ensure_iq_tables_uploaded();
    k_gguf_moe_indexed<<<dim3((N + 3u) / 4u, T * top_k, 1), 128, 0, stream>>>(type, x, w, topk_ids, y, T, N, K, top_k, expert_stride_bytes);
}

void launch_moe_down_accum(GgmlType type, const float* x, const uint8_t* w,
                           const int32_t* topk_ids, const float* topk_w, float* y,
                           uint32_t T, uint32_t N, uint32_t K, uint32_t top_k,
                           uint32_t expert_stride_bytes, cudaStream_t stream) {
    validate_quant_type(type);
    ensure_iq_tables_uploaded();
    k_gguf_moe_down_accum<<<dim3((N + 3u) / 4u, T, 1), 128, 0, stream>>>(type, x, w, topk_ids, topk_w, y, T, N, K, top_k, expert_stride_bytes);
}

void launch_moe_gate_up_swiglu(GgmlType type, const float* x, const uint8_t* wg,
                               const uint8_t* wu, const int32_t* topk_ids, float* y,
                               uint32_t T, uint32_t N, uint32_t K, uint32_t top_k,
                               uint32_t expert_stride_bytes, float clamp_val, cudaStream_t stream) {
    validate_quant_type(type);
    ensure_iq_tables_uploaded();
    k_gguf_moe_gate_up_swiglu<<<dim3((N + 3u) / 4u, T * top_k, 1), 128, 0, stream>>>(
        type, x, wg, wu, topk_ids, y, T, N, K, top_k, expert_stride_bytes, clamp_val);
}

void launch_mhc_expand(const float* x, float* streams, uint32_t T, uint32_t D, uint32_t hc, cudaStream_t stream) {
    k_ds_mhc_expand<<<dim3((D + 255u) / 256u, T, 1), 256, 0, stream>>>(x, streams, T, D, hc);
}

void launch_rmsnorm(const float* x, const float* w, float* y, uint32_t T, uint32_t D, float eps, cudaStream_t stream) {
    uint32_t tpb = (D >= 1024u) ? 256u : 32u;
    k_ds_rmsnorm<<<T, tpb, 0, stream>>>(x, w, y, T, D, eps);
}

void launch_rmsnorm_unweighted(const float* x, float* y, uint32_t T, uint32_t D, float eps, cudaStream_t stream) {
    uint32_t tpb = (D >= 1024u) ? 256u : 32u;
    k_ds_rmsnorm<<<T, tpb, 0, stream>>>(x, nullptr, y, T, D, eps);
}

void launch_hc_mixes(const float* raw, const float* scale, const float* base,
                     float* out_pre, float* out_post, float* out_comb,
                     uint32_t T, uint32_t hc, uint32_t sinkhorn_iters, float eps, cudaStream_t stream) {
    k_ds_hc_mixes<<<T, 32, 0, stream>>>(raw, scale, base, out_pre, out_post, out_comb, T, hc, sinkhorn_iters, eps);
}

void launch_hc_pre(const float* streams, const float* pre_w, float* out_hidden,
                   uint32_t T, uint32_t D, uint32_t hc, cudaStream_t stream) {
    k_ds_hc_pre<<<dim3((D + 255u) / 256u, T, 1), 256, 0, stream>>>(streams, pre_w, out_hidden, T, D, hc);
}

void launch_hc_post(const float* sub, const float* residual, const float* post_w,
                    const float* comb_w, float* out_streams,
                    uint32_t T, uint32_t D, uint32_t hc, cudaStream_t stream) {
    k_ds_hc_post<<<dim3((D + 255u) / 256u, T, 1), 256, 0, stream>>>(sub, residual, post_w, comb_w, out_streams, T, D, hc);
}

void launch_engram_apply(float* streams, const float* ekv, const float* gate,
                         uint32_t T, uint32_t D, uint32_t hc, float eps, cudaStream_t stream) {
    k_ds_engram_apply<<<dim3(hc, T, 1), 32, 0, stream>>>(streams, ekv, gate, T, D, hc, eps);
}

void launch_rope_partial(float* x, uint32_t T, uint32_t start_pos, uint32_t pos_stride,
                         uint32_t n_heads, uint32_t head_dim, uint32_t n_nope,
                         uint32_t n_rot, uint32_t inverse, float freq_base, cudaStream_t stream) {
    uint32_t pairs = n_rot >> 1;
    if (pairs == 0u) return;
    uint32_t tpb = pairs < 32u ? pairs : 32u;
    k_ds_rope_partial<<<dim3((pairs + tpb - 1u) / tpb, n_heads, T), tpb, 0, stream>>>(
        x, T, start_pos, pos_stride, n_heads, head_dim, n_nope, n_rot, inverse, freq_base);
}

void launch_comp_group_reduce(const float* raw_kv, const float* raw_score, const float* norm_w,
                              float* comp_kv, uint32_t n_blocks, uint32_t ratio,
                              uint32_t head_dim, float eps, cudaStream_t stream) {
    k_ds_comp_group_reduce<<<n_blocks, 32, 0, stream>>>(raw_kv, raw_score, norm_w, comp_kv, n_blocks, ratio, head_dim, eps);
}

void launch_indexer_score_topk(const float* idx_q, const float* comp_ik, const float* idx_w,
                               float* scratch_scores, int32_t* out_topk,
                               uint32_t T, uint32_t start_pos, uint32_t n_comp, uint32_t max_comp,
                               uint32_t n_idx_head, uint32_t idx_dim, uint32_t ratio,
                               uint32_t sliding_window, uint32_t top_k, float iw_scale, cudaStream_t stream) {
    k_ds_indexer_score_topk<<<T, 32, 0, stream>>>(
        idx_q, comp_ik, idx_w, scratch_scores, out_topk,
        T, start_pos, n_comp, max_comp, n_idx_head, idx_dim, ratio, sliding_window, top_k, iw_scale);
}

void launch_mqa_sink_attn(const float* q, const float* kv_cache, const float* sinks, float* out,
                          uint32_t T, uint32_t start_pos, uint32_t n_heads, uint32_t head_dim,
                          uint32_t sliding_window, float scale, cudaStream_t stream) {
    k_ds_mqa_sink_attn<<<dim3(n_heads, T, 1), 32, 0, stream>>>(
        q, kv_cache, nullptr, nullptr, sinks, out, T, start_pos, n_heads, head_dim, sliding_window, 0u, scale);
}

void launch_mqa_sink_comp_attn(const float* q, const float* kv_cache, const float* comp_kv,
                               const int32_t* topk_sel, const float* sinks, float* out,
                               uint32_t T, uint32_t start_pos, uint32_t n_heads, uint32_t head_dim,
                               uint32_t sliding_window, uint32_t n_comp_sel, float scale, cudaStream_t stream) {
    k_ds_mqa_sink_attn<<<dim3(n_heads, T, 1), 32, 0, stream>>>(
        q, kv_cache, comp_kv, topk_sel, sinks, out, T, start_pos, n_heads, head_dim, sliding_window, n_comp_sel, scale);
}

void launch_router_sqrtsoftplus_topk(const float* logits, const float* bias,
                                     int32_t* out_ids, float* out_w,
                                     uint32_t T, uint32_t E, uint32_t top_k,
                                     uint32_t has_bias, float route_scale, cudaStream_t stream) {
    k_ds_router_sqrtsoftplus_topk<<<T, 32, 0, stream>>>(logits, bias, out_ids, out_w, T, E, top_k, has_bias, route_scale);
}

void launch_swiglu(const float* gate, const float* up, float* out, uint32_t n, float clamp_val, cudaStream_t stream) {
    k_ds_swiglu<<<(n + 255u) / 256u, 256, 0, stream>>>(gate, up, out, n, clamp_val);
}

void launch_zero(float* dst, uint32_t n, cudaStream_t stream) {
    k_ds_zero<<<(n + 255u) / 256u, 256, 0, stream>>>(dst, n);
}

void launch_argmax(const float* logits, int32_t* out_tok, uint32_t vocab_size, cudaStream_t stream) {
    k_ds_argmax<<<1, 256, 0, stream>>>(logits, out_tok, vocab_size);
}

}
