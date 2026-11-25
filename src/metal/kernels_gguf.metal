#include <metal_stdlib>
using namespace metal;

struct GgufGemvParams {
    uint M;
    uint N;
    uint K;
};

struct GgufMoeParams {
    uint T;
    uint N;
    uint K;
    uint top_k;
    uint expert_stride_bytes;
};

constant float kGgufE2M1Lut[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

constant float kGgufIq4NlLut[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
       1.0f,   13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f
};

inline float gguf_load_f16(device const uchar* p) {
    ushort bits = ushort(p[0]) | (ushort(p[1]) << 8);
    return float(as_type<half>(bits));
}

inline float gguf_e8m0_to_f32(uchar e) {
    if (e == 0u) return 5.877471754111438e-39f;
    return as_type<float>(uint(e) << 23);
}

inline float gguf_dot_q8_0_row(device const float* x_row, device const uchar* w_row, uint K, uint lane) {
    uint nb = K >> 5;
    float acc = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        device const uchar* blk = w_row + b * 34u;
        float d = gguf_load_f16(blk);
        device const char* qs = reinterpret_cast<device const char*>(blk + 2u);
        device const float* xp = x_row + (b << 5);
        float sum = 0.0f;
        for (uint i = 0; i < 32u; ++i) {
            sum += xp[i] * float(qs[i]);
        }
        acc += d * sum;
    }
    return simd_sum(acc);
}

inline float gguf_dot_iq4_nl_row(device const float* x_row, device const uchar* w_row, uint K, uint lane) {
    uint nb = K >> 5;
    float acc = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        device const uchar* blk = w_row + b * 18u;
        float d = gguf_load_f16(blk);
        device const uchar* qs = blk + 2u;
        device const float* xp = x_row + (b << 5);
        float sum = 0.0f;
        for (uint i = 0; i < 16u; ++i) {
            uchar q = qs[i];
            sum += xp[i] * kGgufIq4NlLut[q & 0x0Fu];
            sum += xp[i + 16u] * kGgufIq4NlLut[q >> 4];
        }
        acc += d * sum;
    }
    return simd_sum(acc);
}

inline float gguf_dot_mxfp4_row(device const float* x_row, device const uchar* w_row, uint K, uint lane) {
    uint nb = K >> 5;
    float acc = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        device const uchar* blk = w_row + b * 17u;
        float sc = gguf_e8m0_to_f32(blk[0]);
        device const uchar* qs = blk + 1u;
        device const float* xp = x_row + (b << 5);
        float sum = 0.0f;
        for (uint i = 0; i < 16u; ++i) {
            uchar q = qs[i];
            sum += xp[i] * kGgufE2M1Lut[q & 0x0Fu];
            sum += xp[i + 16u] * kGgufE2M1Lut[q >> 4];
        }
        acc += sc * sum;
    }
    return simd_sum(acc);
}

inline void gguf_get_scale_min_k4(uint j, device const uchar* scales, thread uchar& sc_out, thread uchar& m_out) {
    if (j < 4u) {
        sc_out = scales[j] & 63u;
        m_out = scales[j + 4u] & 63u;
    } else {
        sc_out = (scales[j + 4u] & 0x0Fu) | ((scales[j - 4u] >> 6) << 4);
        m_out = (scales[j + 4u] >> 4) | ((scales[j] >> 6) << 4);
    }
}

inline float gguf_dot_q4_k_row(device const float* x_row, device const uchar* w_row, uint K, uint lane) {
    uint nb = K >> 8;
    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        device const uchar* blk = w_row + b * 144u;
        float d = gguf_load_f16(blk);
        float dmin = gguf_load_f16(blk + 2u);
        device const uchar* scales = blk + 4u;
        device const uchar* qs = blk + 16u;
        device const float* xp = x_row + (b << 8);

        for (uint sb = 0; sb < 4u; ++sb) {
            uchar sc0 = 0, m0 = 0, sc1 = 0, m1 = 0;
            gguf_get_scale_min_k4(sb * 2u, scales, sc0, m0);
            gguf_get_scale_min_k4(sb * 2u + 1u, scales, sc1, m1);
            float d1 = d * float(sc0);
            float min1 = dmin * float(m0);
            float d2 = d * float(sc1);
            float min2 = dmin * float(m1);
            uchar q = qs[sb * 32u + lane];
            float wlo = d1 * float(q & 0x0Fu) - min1;
            float whi = d2 * float(q >> 4) - min2;
            acc += xp[sb * 64u + lane] * wlo + xp[sb * 64u + 32u + lane] * whi;
        }
    }
    return simd_sum(acc);
}

inline float gguf_dot_q6_k_row(device const float* x_row, device const uchar* w_row, uint K, uint lane) {
    uint nb = K >> 8;
    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        device const uchar* blk = w_row + b * 210u;
        device const uchar* ql = blk;
        device const uchar* qh = blk + 128u;
        device const char* sc = reinterpret_cast<device const char*>(blk + 192u);
        float d = gguf_load_f16(blk + 208u);
        device const float* xp = x_row + (b << 8);

        for (uint half_idx = 0; half_idx < 2u; ++half_idx) {
            device const uchar* qlp = ql + half_idx * 64u;
            device const uchar* qhp = qh + half_idx * 32u;
            device const char* scp = sc + half_idx * 8u;
            device const float* xhp = xp + half_idx * 128u;

            uint is = lane >> 4;
            uchar h = qhp[lane];
            int q1 = int((qlp[lane] & 0x0Fu) | (((h >> 0) & 3u) << 4)) - 32;
            int q2 = int((qlp[lane + 32u] & 0x0Fu) | (((h >> 2) & 3u) << 4)) - 32;
            int q3 = int((qlp[lane] >> 4) | (((h >> 4) & 3u) << 4)) - 32;
            int q4 = int((qlp[lane + 32u] >> 4) | (((h >> 6) & 3u) << 4)) - 32;

            float s = xhp[lane] * float(scp[is + 0u]) * float(q1)
                    + xhp[lane + 32u] * float(scp[is + 2u]) * float(q2)
                    + xhp[lane + 64u] * float(scp[is + 4u]) * float(q3)
                    + xhp[lane + 96u] * float(scp[is + 6u]) * float(q4);
            acc += d * s;
        }
    }
    return simd_sum(acc);
}

inline float gguf_dot_q2_k_row(device const float* x_row, device const uchar* w_row, uint K, uint lane) {
    uint nb = K >> 8;
    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        device const uchar* blk = w_row + b * 84u;
        device const uchar* scales = blk;
        device const uchar* qs = blk + 16u;
        float d = gguf_load_f16(blk + 80u);
        float dmin = gguf_load_f16(blk + 82u);
        device const float* xp = x_row + (b << 8);

        for (uint half_idx = 0; half_idx < 2u; ++half_idx) {
            device const uchar* qp = qs + half_idx * 32u;
            device const float* xhp = xp + half_idx * 128u;
            uint is_base = half_idx * 8u;
            uint is_off = lane >> 4;
            uchar qval = qp[lane];
            for (uint j = 0; j < 4u; ++j) {
                uchar sc = scales[is_base + j * 2u + is_off];
                float dl = d * float(sc & 0x0Fu);
                float ml = dmin * float(sc >> 4);
                float w = dl * float((qval >> (j * 2u)) & 3u) - ml;
                acc += xhp[j * 32u + lane] * w;
            }
        }
    }
    return simd_sum(acc);
}

kernel void gguf_gemv_q8_0(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant GgufGemvParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint m = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (m >= p.M || n >= p.N) return;
    ulong row_bytes = ulong(p.K >> 5) * 34ul;
    float sum = gguf_dot_q8_0_row(X + ulong(m) * p.K, W + ulong(n) * row_bytes, p.K, lane);
    if (lane == 0u) Y[ulong(m) * p.N + n] = sum;
}

kernel void gguf_gemv_iq4_nl(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant GgufGemvParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint m = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (m >= p.M || n >= p.N) return;
    ulong row_bytes = ulong(p.K >> 5) * 18ul;
    float sum = gguf_dot_iq4_nl_row(X + ulong(m) * p.K, W + ulong(n) * row_bytes, p.K, lane);
    if (lane == 0u) Y[ulong(m) * p.N + n] = sum;
}

kernel void gguf_gemv_mxfp4(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant GgufGemvParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint m = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (m >= p.M || n >= p.N) return;
    ulong row_bytes = ulong(p.K >> 5) * 17ul;
    float sum = gguf_dot_mxfp4_row(X + ulong(m) * p.K, W + ulong(n) * row_bytes, p.K, lane);
    if (lane == 0u) Y[ulong(m) * p.N + n] = sum;
}

kernel void gguf_gemv_q4_k(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant GgufGemvParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint m = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (m >= p.M || n >= p.N) return;
    ulong row_bytes = ulong(p.K >> 8) * 144ul;
    float sum = gguf_dot_q4_k_row(X + ulong(m) * p.K, W + ulong(n) * row_bytes, p.K, lane);
    if (lane == 0u) Y[ulong(m) * p.N + n] = sum;
}

kernel void gguf_gemv_q6_k(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant GgufGemvParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint m = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (m >= p.M || n >= p.N) return;
    ulong row_bytes = ulong(p.K >> 8) * 210ul;
    float sum = gguf_dot_q6_k_row(X + ulong(m) * p.K, W + ulong(n) * row_bytes, p.K, lane);
    if (lane == 0u) Y[ulong(m) * p.N + n] = sum;
}

kernel void gguf_gemv_q2_k(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant GgufGemvParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint m = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (m >= p.M || n >= p.N) return;
    ulong row_bytes = ulong(p.K >> 8) * 84ul;
    float sum = gguf_dot_q2_k_row(X + ulong(m) * p.K, W + ulong(n) * row_bytes, p.K, lane);
    if (lane == 0u) Y[ulong(m) * p.N + n] = sum;
}

kernel void gguf_moe_gemv_q8_0(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const int* topk_ids [[buffer(2)]],
    device const float* topk_w [[buffer(3)]],
    device float* Y [[buffer(4)]],
    constant GgufMoeParams& p [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint t = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (t >= p.T || n >= p.N) return;
    ulong row_bytes = ulong(p.K >> 5) * 34ul;
    device const float* x_row = X + ulong(t) * p.K;
    float total = 0.0f;
    for (uint r = 0; r < p.top_k; ++r) {
        int e_id = topk_ids[t * p.top_k + r];
        if (e_id < 0) continue;
        float gw = topk_w[t * p.top_k + r];
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        total += gw * gguf_dot_q8_0_row(x_row, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(t) * p.N + n] = total;
}

kernel void gguf_moe_gemv_q4_k(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const int* topk_ids [[buffer(2)]],
    device const float* topk_w [[buffer(3)]],
    device float* Y [[buffer(4)]],
    constant GgufMoeParams& p [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint t = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (t >= p.T || n >= p.N) return;
    ulong row_bytes = ulong(p.K >> 8) * 144ul;
    device const float* x_row = X + ulong(t) * p.K;
    float total = 0.0f;
    for (uint r = 0; r < p.top_k; ++r) {
        int e_id = topk_ids[t * p.top_k + r];
        if (e_id < 0) continue;
        float gw = topk_w[t * p.top_k + r];
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        total += gw * gguf_dot_q4_k_row(x_row, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(t) * p.N + n] = total;
}

kernel void gguf_moe_gemv_q2_k(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const int* topk_ids [[buffer(2)]],
    device const float* topk_w [[buffer(3)]],
    device float* Y [[buffer(4)]],
    constant GgufMoeParams& p [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint t = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (t >= p.T || n >= p.N) return;
    ulong row_bytes = ulong(p.K >> 8) * 84ul;
    device const float* x_row = X + ulong(t) * p.K;
    float total = 0.0f;
    for (uint r = 0; r < p.top_k; ++r) {
        int e_id = topk_ids[t * p.top_k + r];
        if (e_id < 0) continue;
        float gw = topk_w[t * p.top_k + r];
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        total += gw * gguf_dot_q2_k_row(x_row, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(t) * p.N + n] = total;
}

kernel void gguf_moe_gemv_mxfp4(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const int* topk_ids [[buffer(2)]],
    device const float* topk_w [[buffer(3)]],
    device float* Y [[buffer(4)]],
    constant GgufMoeParams& p [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint t = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (t >= p.T || n >= p.N) return;
    ulong row_bytes = ulong(p.K >> 5) * 17ul;
    device const float* x_row = X + ulong(t) * p.K;
    float total = 0.0f;
    for (uint r = 0; r < p.top_k; ++r) {
        int e_id = topk_ids[t * p.top_k + r];
        if (e_id < 0) continue;
        float gw = topk_w[t * p.top_k + r];
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        total += gw * gguf_dot_mxfp4_row(x_row, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(t) * p.N + n] = total;
}

kernel void gguf_gemv_f32(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant GgufGemvParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint m = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (m >= p.M || n >= p.N) return;
    device const float* xp = X + ulong(m) * p.K;
    device const float* wp = W + ulong(n) * p.K;
    float acc = 0.0f;
    for (uint k = lane; k < p.K; k += 32u) acc += xp[k] * wp[k];
    float sum = simd_sum(acc);
    if (lane == 0u) Y[ulong(m) * p.N + n] = sum;
}

struct DsRmsParams {
    uint T;
    uint D;
    float eps;
};

kernel void ds_rmsnorm_f32(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant DsRmsParams& p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tgid >= p.T) return;
    device const float* xp = X + ulong(tgid) * p.D;
    device float* yp = Y + ulong(tgid) * p.D;
    float ss = 0.0f;
    for (uint d = tid; d < p.D; d += 32u) {
        float v = xp[d];
        ss += v * v;
    }
    float mean_ss = simd_sum(ss) / float(p.D);
    float inv_rms = rsqrt(mean_ss + p.eps);
    for (uint d = tid; d < p.D; d += 32u) {
        yp[d] = xp[d] * inv_rms * W[d];
    }
}

kernel void ds_rmsnorm_unweighted_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant DsRmsParams& p [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tgid >= p.T) return;
    device const float* xp = X + ulong(tgid) * p.D;
    device float* yp = Y + ulong(tgid) * p.D;
    float ss = 0.0f;
    for (uint d = tid; d < p.D; d += 32u) {
        float v = xp[d];
        ss += v * v;
    }
    float mean_ss = simd_sum(ss) / float(p.D);
    float inv_rms = rsqrt(mean_ss + p.eps);
    for (uint d = tid; d < p.D; d += 32u) {
        yp[d] = xp[d] * inv_rms;
    }
}

struct DsHcMixParams {
    uint T;
    uint hc_mult;
    uint sinkhorn_iters;
    float eps;
};

kernel void ds_hc_mixes_f32(
    device const float* raw_mixes [[buffer(0)]],
    device const float* hc_scale [[buffer(1)]],
    device const float* hc_base [[buffer(2)]],
    device float* out_pre [[buffer(3)]],
    device float* out_post [[buffer(4)]],
    device float* out_comb [[buffer(5)]],
    constant DsHcMixParams& p [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tgid >= p.T || tid != 0u) return;
    uint hc = min(p.hc_mult, 8u);
    uint mix_dim = (2u + hc) * hc;
    device const float* m = raw_mixes + ulong(tgid) * mix_dim;
    float s_pre = hc_scale[0];
    float s_post = hc_scale[1];
    float s_comb = hc_scale[2];

    for (uint i = 0; i < hc; ++i) {
        float v = m[i] * s_pre + hc_base[i];
        out_pre[ulong(tgid) * hc + i] = (1.0f / (1.0f + exp(-v))) + p.eps;
    }
    for (uint i = 0; i < hc; ++i) {
        float v = m[hc + i] * s_post + hc_base[hc + i];
        out_post[ulong(tgid) * hc + i] = 2.0f / (1.0f + exp(-v));
    }

    float mat[64];
    for (uint src = 0; src < hc; ++src) {
        float mx = -1e30f;
        for (uint dst = 0; dst < hc; ++dst) {
            uint idx = src * hc + dst;
            float v = m[2u * hc + idx] * s_comb + hc_base[2u * hc + idx];
            mat[idx] = v;
            mx = max(mx, v);
        }
        float sm = 0.0f;
        for (uint dst = 0; dst < hc; ++dst) {
            float e = exp(mat[src * hc + dst] - mx);
            mat[src * hc + dst] = e;
            sm += e;
        }
        float inv = 1.0f / max(sm, 1e-20f);
        for (uint dst = 0; dst < hc; ++dst) {
            mat[src * hc + dst] = mat[src * hc + dst] * inv + p.eps;
        }
    }

    for (uint it = 0; it < p.sinkhorn_iters; ++it) {
        if (it > 0u) {
            for (uint src = 0; src < hc; ++src) {
                float rs = p.eps;
                for (uint dst = 0; dst < hc; ++dst) rs += mat[src * hc + dst];
                float inv = 1.0f / rs;
                for (uint dst = 0; dst < hc; ++dst) mat[src * hc + dst] *= inv;
            }
        }
        for (uint dst = 0; dst < hc; ++dst) {
            float cs = p.eps;
            for (uint src = 0; src < hc; ++src) cs += mat[src * hc + dst];
            float inv = 1.0f / cs;
            for (uint src = 0; src < hc; ++src) mat[src * hc + dst] *= inv;
        }
    }
    for (uint i = 0; i < hc * hc; ++i) {
        out_comb[ulong(tgid) * (hc * hc) + i] = mat[i];
    }
}

struct DsHcApplyParams {
    uint T;
    uint D;
    uint hc_mult;
};

kernel void ds_hc_pre_f32(
    device const float* streams [[buffer(0)]],
    device const float* pre_w [[buffer(1)]],
    device float* out_hidden [[buffer(2)]],
    constant DsHcApplyParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint d = gid.x;
    uint t = gid.y;
    if (d >= p.D || t >= p.T) return;
    float acc = 0.0f;
    for (uint s = 0; s < p.hc_mult; ++s) {
        acc += streams[(ulong(t) * p.hc_mult + s) * p.D + d] * pre_w[ulong(t) * p.hc_mult + s];
    }
    out_hidden[ulong(t) * p.D + d] = acc;
}

kernel void ds_hc_post_f32(
    device const float* sublayer [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* post_w [[buffer(2)]],
    device const float* comb_w [[buffer(3)]],
    device float* out_streams [[buffer(4)]],
    constant DsHcApplyParams& p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint d = gid.x;
    uint t = gid.y;
    if (d >= p.D || t >= p.T) return;
    float sv = sublayer[ulong(t) * p.D + d];
    for (uint dst = 0; dst < p.hc_mult; ++dst) {
        float acc = sv * post_w[ulong(t) * p.hc_mult + dst];
        for (uint src = 0; src < p.hc_mult; ++src) {
            float c = comb_w[ulong(t) * (p.hc_mult * p.hc_mult) + src * p.hc_mult + dst];
            acc += residual[(ulong(t) * p.hc_mult + src) * p.D + d] * c;
        }
        out_streams[(ulong(t) * p.hc_mult + dst) * p.D + d] = acc;
    }
}

struct DsEngramApplyParams {
    uint T;
    uint D;
    uint hc_mult;
    float eps;
};

kernel void ds_engram_apply_f32(
    device float* streams [[buffer(0)]],
    device const float* engram_kv [[buffer(1)]],
    device const float* engram_gate [[buffer(2)]],
    constant DsEngramApplyParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint s = tgid.x;
    uint t = tgid.y;
    if (s >= p.hc_mult || t >= p.T) return;
    device float* xp = streams + (ulong(t) * p.hc_mult + s) * p.D;
    device const float* kv_base = engram_kv + ulong(t) * (p.hc_mult + 1u) * p.D;
    device const float* kp = kv_base + ulong(s) * p.D;
    device const float* vp = kv_base + ulong(p.hc_mult) * p.D;
    device const float* gp = engram_gate + ulong(s) * p.D;

    float x_ss = 0.0f, k_ss = 0.0f;
    for (uint d = tid; d < p.D; d += 32u) {
        x_ss += xp[d] * xp[d];
        k_ss += kp[d] * kp[d];
    }
    float inv_x = rsqrt(simd_sum(x_ss) / float(p.D) + p.eps);
    float inv_k = rsqrt(simd_sum(k_ss) / float(p.D) + p.eps);

    float dot = 0.0f;
    for (uint d = tid; d < p.D; d += 32u) {
        dot += (xp[d] * inv_x) * (kp[d] * inv_k) * gp[d];
    }
    float dval = simd_sum(dot) * rsqrt(float(p.D));
    float mag = sqrt(max(fabs(dval), 1e-6f));
    float signed_mag = (dval < 0.0f) ? -mag : mag;
    float gate = 1.0f / (1.0f + exp(-signed_mag));
    for (uint d = tid; d < p.D; d += 32u) {
        xp[d] += vp[d] * gate;
    }
}

struct DsPartialRopeParams {
    uint T;
    uint start_pos;
    uint n_heads;
    uint head_dim;
    uint n_nope;
    uint n_rot;
    uint inverse;
    float freq_base;
};

kernel void ds_rope_partial_f32(
    device float* X [[buffer(0)]],
    constant DsPartialRopeParams& p [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]) {
    uint pair = gid.x;
    uint h = gid.y;
    uint t = gid.z;
    uint half_rot = p.n_rot >> 1;
    if (pair >= half_rot || h >= p.n_heads || t >= p.T) return;
    float pos = float(p.start_pos + t);
    float freq = pow(p.freq_base, -float(2u * pair) / float(p.n_rot));
    float angle = pos * freq;
    if (p.inverse != 0u) angle = -angle;
    float cs = cos(angle);
    float sn = sin(angle);
    ulong base = (ulong(t) * p.n_heads + h) * p.head_dim + p.n_nope;
    float x0 = X[base + 2u * pair];
    float x1 = X[base + 2u * pair + 1u];
    X[base + 2u * pair] = x0 * cs - x1 * sn;
    X[base + 2u * pair + 1u] = x0 * sn + x1 * cs;
}

struct DsMqaSinkAttnParams {
    uint T;
    uint start_pos;
    uint n_heads;
    uint head_dim;
    uint sliding_window;
    uint n_comp_sel;
    float scale;
};

kernel void ds_mqa_sink_attn_f32(
    device const float* Q [[buffer(0)]],
    device const float* KV_cache [[buffer(1)]],
    device const float* sinks [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant DsMqaSinkAttnParams& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.n_heads || t >= p.T) return;
    uint seq_end = p.start_pos + t + 1u;
    uint win_start = (p.sliding_window > 0u && seq_end > p.sliding_window) ? (seq_end - p.sliding_window) : 0u;
    device const float* qp = Q + (ulong(t) * p.n_heads + h) * p.head_dim;
    device float* op = O + (ulong(t) * p.n_heads + h) * p.head_dim;

    float sink_val = sinks[h];
    float m_max = sink_val;
    for (uint s = win_start; s < seq_end; ++s) {
        device const float* kvp = KV_cache + ulong(s) * p.head_dim;
        float dot = 0.0f;
        for (uint d = tid; d < p.head_dim; d += 32u) dot += qp[d] * kvp[d];
        float sc = simd_sum(dot) * p.scale;
        m_max = max(m_max, sc);
    }

    float l_sum = exp(sink_val - m_max);
    for (uint d = tid; d < p.head_dim; d += 32u) op[d] = 0.0f;
    for (uint s = win_start; s < seq_end; ++s) {
        device const float* kvp = KV_cache + ulong(s) * p.head_dim;
        float dot = 0.0f;
        for (uint d = tid; d < p.head_dim; d += 32u) dot += qp[d] * kvp[d];
        float w = exp(simd_sum(dot) * p.scale - m_max);
        l_sum += w;
        for (uint d = tid; d < p.head_dim; d += 32u) op[d] += w * kvp[d];
    }
    float inv_l = 1.0f / max(l_sum, 1e-20f);
    for (uint d = tid; d < p.head_dim; d += 32u) op[d] *= inv_l;
}

struct DsMhcParams {
    uint T;
    uint D;
    uint hc_mult;
    uint sinkhorn_iters;
    float eps;
};

kernel void ds_mhc_sinkhorn_mix(
    device const float* streams [[buffer(0)]],
    device const float* sublayer [[buffer(1)]],
    device const float* h_logits [[buffer(2)]],
    device const float* h_post [[buffer(3)]],
    device float* out_streams [[buffer(4)]],
    constant DsMhcParams& p [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint t = tgid.y;
    uint d = tgid.x * 64u + tid;
    if (t >= p.T || d >= p.D) return;
    uint M = min(p.hc_mult, 8u);
    float mat[64];
    for (uint i = 0; i < M * M; ++i) mat[i] = exp(h_logits[i]);
    for (uint it = 0; it < p.sinkhorn_iters; ++it) {
        for (uint r = 0; r < M; ++r) {
            float s = p.eps;
            for (uint c = 0; c < M; ++c) s += mat[r * M + c];
            float inv = 1.0f / s;
            for (uint c = 0; c < M; ++c) mat[r * M + c] *= inv;
        }
        for (uint c = 0; c < M; ++c) {
            float s = p.eps;
            for (uint r = 0; r < M; ++r) s += mat[r * M + c];
            float inv = 1.0f / s;
            for (uint r = 0; r < M; ++r) mat[r * M + c] *= inv;
        }
    }
    float sv = sublayer[ulong(t) * p.D + d];
    for (uint r = 0; r < M; ++r) {
        float acc = h_post[r] * sv;
        for (uint c = 0; c < M; ++c) {
            acc += mat[r * M + c] * streams[(ulong(t) * M + c) * p.D + d];
        }
        out_streams[(ulong(t) * M + r) * p.D + d] = acc;
    }
}

struct DsRopeParams {
    uint T;
    uint start_pos;
    uint n_heads;
    uint head_dim;
    float freq_base;
};

kernel void ds_rope_f32(
    device float* X [[buffer(0)]],
    constant DsRopeParams& p [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]) {
    uint pair = gid.x;
    uint h = gid.y;
    uint t = gid.z;
    uint half_d = p.head_dim >> 1;
    if (pair >= half_d || h >= p.n_heads || t >= p.T) return;
    float pos = float(p.start_pos + t);
    float freq = pow(p.freq_base, -float(2u * pair) / float(p.head_dim));
    float angle = pos * freq;
    float cs = cos(angle);
    float sn = sin(angle);
    ulong base = (ulong(t) * p.n_heads + h) * p.head_dim;
    float x0 = X[base + 2u * pair];
    float x1 = X[base + 2u * pair + 1u];
    X[base + 2u * pair] = x0 * cs - x1 * sn;
    X[base + 2u * pair + 1u] = x0 * sn + x1 * cs;
}

struct DsRouterParams {
    uint T;
    uint E;
    uint top_k;
    uint has_bias;
    float route_scale;
};

kernel void ds_router_topk_f32(
    device const float* logits [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device int* out_ids [[buffer(2)]],
    device float* out_w [[buffer(3)]],
    constant DsRouterParams& p [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tgid >= p.T || tid != 0u || p.top_k == 0u || p.E == 0u) return;
    device const float* row = logits + ulong(tgid) * p.E;
    uint K = min(p.top_k, 16u);
    int best_id[16];
    float best_score[16];
    float best_sig[16];
    for (uint k = 0; k < K; ++k) {
        best_id[k] = -1;
        best_score[k] = -1e30f;
        best_sig[k] = 0.0f;
    }
    for (uint e = 0; e < p.E; ++e) {
        float sig = 1.0f / (1.0f + exp(-row[e]));
        float sc = sig + ((p.has_bias != 0u) ? bias[e] : 0.0f);
        if (sc > best_score[K - 1u]) {
            uint pos = K - 1u;
            while (pos > 0u && sc > best_score[pos - 1u]) {
                best_score[pos] = best_score[pos - 1u];
                best_sig[pos] = best_sig[pos - 1u];
                best_id[pos] = best_id[pos - 1u];
                --pos;
            }
            best_score[pos] = sc;
            best_sig[pos] = sig;
            best_id[pos] = int(e);
        }
    }
    float denom = 1e-20f;
    for (uint k = 0; k < K; ++k) denom += best_sig[k];
    float inv = p.route_scale / denom;
    for (uint k = 0; k < K; ++k) {
        out_ids[ulong(tgid) * p.top_k + k] = best_id[k];
        out_w[ulong(tgid) * p.top_k + k] = best_sig[k] * inv;
    }
}

struct DsAttnParams {
    uint T;
    uint start_pos;
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint max_seq;
    uint compress_ratio;
    uint local_window;
    float scale;
};

kernel void ds_csa_hca_attn_f32(
    device const float* Q [[buffer(0)]],
    device const float* K_cache [[buffer(1)]],
    device const float* V_cache [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant DsAttnParams& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.n_heads || t >= p.T) return;
    uint kvh = h / max(1u, p.n_heads / p.n_kv_heads);
    uint seq_end = p.start_pos + t + 1u;
    uint cr = max(1u, p.compress_ratio);
    uint local_start = (cr > 1u && seq_end > p.local_window) ? (seq_end - p.local_window) : 0u;
    uint comp_blocks = (cr > 1u) ? (local_start / cr) : 0u;
    uint local_begin = (cr > 1u) ? (comp_blocks * cr) : 0u;
    device const float* qp = Q + (ulong(t) * p.n_heads + h) * p.head_dim;
    device float* op = O + (ulong(t) * p.n_heads + h) * p.head_dim;

    float m_max = -1e30f;
    for (uint blk = 0; blk < comp_blocks; ++blk) {
        uint s0 = blk * cr;
        float dot = 0.0f;
        float inv_cr = 1.0f / float(cr);
        for (uint d = tid; d < p.head_dim; d += 32u) {
            float k_avg = 0.0f;
            for (uint j = 0; j < cr; ++j) {
                k_avg += K_cache[(ulong(s0 + j) * p.n_kv_heads + kvh) * p.head_dim + d];
            }
            dot += qp[d] * (k_avg * inv_cr);
        }
        float sc = simd_sum(dot) * p.scale;
        m_max = max(m_max, sc);
    }
    for (uint s = local_begin; s < seq_end; ++s) {
        device const float* kp = K_cache + (ulong(s) * p.n_kv_heads + kvh) * p.head_dim;
        float dot = 0.0f;
        for (uint d = tid; d < p.head_dim; d += 32u) dot += qp[d] * kp[d];
        float sc = simd_sum(dot) * p.scale;
        m_max = max(m_max, sc);
    }

    float l_sum = 0.0f;
    for (uint d = tid; d < p.head_dim; d += 32u) op[d] = 0.0f;
    for (uint blk = 0; blk < comp_blocks; ++blk) {
        uint s0 = blk * cr;
        float dot = 0.0f;
        float inv_cr = 1.0f / float(cr);
        for (uint d = tid; d < p.head_dim; d += 32u) {
            float k_avg = 0.0f;
            for (uint j = 0; j < cr; ++j) {
                k_avg += K_cache[(ulong(s0 + j) * p.n_kv_heads + kvh) * p.head_dim + d];
            }
            dot += qp[d] * (k_avg * inv_cr);
        }
        float w = exp(simd_sum(dot) * p.scale - m_max);
        l_sum += w;
        for (uint d = tid; d < p.head_dim; d += 32u) {
            float v_avg = 0.0f;
            for (uint j = 0; j < cr; ++j) {
                v_avg += V_cache[(ulong(s0 + j) * p.n_kv_heads + kvh) * p.head_dim + d];
            }
            op[d] += w * (v_avg * inv_cr);
        }
    }
    for (uint s = local_begin; s < seq_end; ++s) {
        device const float* kp = K_cache + (ulong(s) * p.n_kv_heads + kvh) * p.head_dim;
        device const float* vp = V_cache + (ulong(s) * p.n_kv_heads + kvh) * p.head_dim;
        float dot = 0.0f;
        for (uint d = tid; d < p.head_dim; d += 32u) dot += qp[d] * kp[d];
        float w = exp(simd_sum(dot) * p.scale - m_max);
        l_sum += w;
        for (uint d = tid; d < p.head_dim; d += 32u) op[d] += w * vp[d];
    }
    float inv_l = 1.0f / max(l_sum, 1e-20f);
    for (uint d = tid; d < p.head_dim; d += 32u) op[d] *= inv_l;
}

struct DsCompGroupParams {
    uint n_blocks;
    uint ratio;
    uint head_dim;
    float eps;
};

kernel void ds_comp_group_reduce_f32(
    device const float* state_kv [[buffer(0)]],
    device const float* state_score [[buffer(1)]],
    device const float* norm_w [[buffer(2)]],
    device float* out_comp [[buffer(3)]],
    constant DsCompGroupParams& p [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tgid >= p.n_blocks) return;
    uint base_row = tgid * p.ratio;
    device float* yp = out_comp + ulong(tgid) * p.head_dim;
    float ss = 0.0f;
    for (uint d = tid; d < p.head_dim; d += 32u) {
        float mx = -1e30f;
        for (uint r = 0; r < p.ratio; ++r) {
            mx = max(mx, state_score[ulong(base_row + r) * p.head_dim + d]);
        }
        float sm = 0.0f;
        float val = 0.0f;
        for (uint r = 0; r < p.ratio; ++r) {
            float w = exp(state_score[ulong(base_row + r) * p.head_dim + d] - mx);
            sm += w;
            val += w * state_kv[ulong(base_row + r) * p.head_dim + d];
        }
        float c = val / max(sm, 1e-20f);
        yp[d] = c;
        ss += c * c;
    }
    float inv_rms = rsqrt(simd_sum(ss) / float(p.head_dim) + p.eps);
    for (uint d = tid; d < p.head_dim; d += 32u) {
        yp[d] = yp[d] * inv_rms * norm_w[d];
    }
}

struct DsIndexerParams {
    uint T;
    uint start_pos;
    uint n_comp;
    uint n_idx_head;
    uint idx_dim;
    uint ratio;
    uint sliding_window;
    uint top_k;
    float iw_scale;
};

kernel void ds_indexer_score_topk_f32(
    device const float* iq [[buffer(0)]],
    device const float* ik_cache [[buffer(1)]],
    device const float* iw [[buffer(2)]],
    device int* out_topk [[buffer(3)]],
    constant DsIndexerParams& p [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tgid >= p.T || tid != 0u) return;
    uint t = tgid;
    uint pos = p.start_pos + t;
    uint win_start = (p.sliding_window > 0u && pos + 1u > p.sliding_window) ? (pos + 1u - p.sliding_window) : 0u;
    uint max_vis = min(p.n_comp, win_start / max(1u, p.ratio));
    uint K = min(p.top_k, 256u);

    int best_id[256];
    float best_sc[256];
    for (uint k = 0; k < K; ++k) {
        best_id[k] = -1;
        best_sc[k] = -1e30f;
    }
    device const float* iqt = iq + ulong(t) * p.n_idx_head * p.idx_dim;
    device const float* iwt = iw + ulong(t) * p.n_idx_head;

    for (uint c = 0; c < max_vis; ++c) {
        device const float* ikc = ik_cache + ulong(c) * p.idx_dim;
        float total = 0.0f;
        for (uint h = 0; h < p.n_idx_head; ++h) {
            device const float* iqh = iqt + ulong(h) * p.idx_dim;
            float dot = 0.0f;
            for (uint d = 0; d < p.idx_dim; ++d) dot += iqh[d] * ikc[d];
            if (dot > 0.0f) total += dot * (iwt[h] * p.iw_scale);
        }
        if (total > best_sc[K - 1u]) {
            uint ins = K - 1u;
            while (ins > 0u && total > best_sc[ins - 1u]) {
                best_sc[ins] = best_sc[ins - 1u];
                best_id[ins] = best_id[ins - 1u];
                --ins;
            }
            best_sc[ins] = total;
            best_id[ins] = int(c);
        }
    }
    for (uint k = 0; k < K; ++k) {
        out_topk[ulong(t) * K + k] = best_id[k];
    }
}

kernel void ds_mqa_sink_comp_attn_f32(
    device const float* Q [[buffer(0)]],
    device const float* KV_cache [[buffer(1)]],
    device const float* comp_kv_cache [[buffer(2)]],
    device const int* comp_topk [[buffer(3)]],
    device const float* sinks [[buffer(4)]],
    device float* O [[buffer(5)]],
    constant DsMqaSinkAttnParams& p [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.n_heads || t >= p.T) return;
    uint seq_end = p.start_pos + t + 1u;
    uint win_start = (p.sliding_window > 0u && seq_end > p.sliding_window) ? (seq_end - p.sliding_window) : 0u;
    device const float* qp = Q + (ulong(t) * p.n_heads + h) * p.head_dim;
    device float* op = O + (ulong(t) * p.n_heads + h) * p.head_dim;
    device const int* sel = comp_topk + ulong(t) * p.n_comp_sel;

    float sink_val = sinks[h];
    float m_max = sink_val;
    for (uint s = win_start; s < seq_end; ++s) {
        device const float* kvp = KV_cache + ulong(s) * p.head_dim;
        float dot = 0.0f;
        for (uint d = tid; d < p.head_dim; d += 32u) dot += qp[d] * kvp[d];
        m_max = max(m_max, simd_sum(dot) * p.scale);
    }
    for (uint k = 0; k < p.n_comp_sel; ++k) {
        int c = sel[k];
        if (c < 0) continue;
        device const float* kvp = comp_kv_cache + ulong(uint(c)) * p.head_dim;
        float dot = 0.0f;
        for (uint d = tid; d < p.head_dim; d += 32u) dot += qp[d] * kvp[d];
        m_max = max(m_max, simd_sum(dot) * p.scale);
    }

    float l_sum = exp(sink_val - m_max);
    for (uint d = tid; d < p.head_dim; d += 32u) op[d] = 0.0f;
    for (uint s = win_start; s < seq_end; ++s) {
        device const float* kvp = KV_cache + ulong(s) * p.head_dim;
        float dot = 0.0f;
        for (uint d = tid; d < p.head_dim; d += 32u) dot += qp[d] * kvp[d];
        float w = exp(simd_sum(dot) * p.scale - m_max);
        l_sum += w;
        for (uint d = tid; d < p.head_dim; d += 32u) op[d] += w * kvp[d];
    }
    for (uint k = 0; k < p.n_comp_sel; ++k) {
        int c = sel[k];
        if (c < 0) continue;
        device const float* kvp = comp_kv_cache + ulong(uint(c)) * p.head_dim;
        float dot = 0.0f;
        for (uint d = tid; d < p.head_dim; d += 32u) dot += qp[d] * kvp[d];
        float w = exp(simd_sum(dot) * p.scale - m_max);
        l_sum += w;
        for (uint d = tid; d < p.head_dim; d += 32u) op[d] += w * kvp[d];
    }
    float inv_l = 1.0f / max(l_sum, 1e-20f);
    for (uint d = tid; d < p.head_dim; d += 32u) op[d] *= inv_l;
}

kernel void ds_router_sqrtsoftplus_topk_f32(
    device const float* logits [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device int* out_ids [[buffer(2)]],
    device float* out_w [[buffer(3)]],
    constant DsRouterParams& p [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tgid >= p.T || tid != 0u || p.top_k == 0u || p.E == 0u) return;
    device const float* row = logits + ulong(tgid) * p.E;
    uint K = min(p.top_k, 16u);
    int best_id[16];
    float best_score[16];
    float best_val[16];
    for (uint k = 0; k < K; ++k) {
        best_id[k] = -1;
        best_score[k] = -1e30f;
        best_val[k] = 0.0f;
    }
    for (uint e = 0; e < p.E; ++e) {
        float x = row[e];
        float sp = (x > 20.0f) ? x : log(1.0f + exp(x));
        float val = sqrt(max(sp, 0.0f));
        float sc = val + ((p.has_bias != 0u) ? bias[e] : 0.0f);
        if (sc > best_score[K - 1u]) {
            uint pos = K - 1u;
            while (pos > 0u && sc > best_score[pos - 1u]) {
                best_score[pos] = best_score[pos - 1u];
                best_val[pos] = best_val[pos - 1u];
                best_id[pos] = best_id[pos - 1u];
                --pos;
            }
            best_score[pos] = sc;
            best_val[pos] = val;
            best_id[pos] = int(e);
        }
    }
    float denom = 1e-20f;
    for (uint k = 0; k < K; ++k) denom += best_val[k];
    float inv = (K > 1u) ? (p.route_scale / denom) : p.route_scale;
    for (uint k = 0; k < K; ++k) {
        out_ids[ulong(tgid) * p.top_k + k] = best_id[k];
        out_w[ulong(tgid) * p.top_k + k] = best_val[k] * inv;
    }
}

struct DsStreamCollapseParams {
    uint T;
    uint D;
    uint hc_mult;
};

kernel void ds_mhc_expand_f32(
    device const float* hidden [[buffer(0)]],
    device float* streams [[buffer(1)]],
    constant DsStreamCollapseParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint d = gid.x;
    uint t = gid.y;
    if (d >= p.D || t >= p.T) return;
    float v = hidden[ulong(t) * p.D + d];
    for (uint r = 0; r < p.hc_mult; ++r) {
        streams[(ulong(t) * p.hc_mult + r) * p.D + d] = v;
    }
}

kernel void ds_mhc_collapse_f32(
    device const float* streams [[buffer(0)]],
    device const float* h_pre [[buffer(1)]],
    device float* hidden [[buffer(2)]],
    constant DsStreamCollapseParams& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint d = gid.x;
    uint t = gid.y;
    if (d >= p.D || t >= p.T) return;
    float acc = 0.0f;
    for (uint r = 0; r < p.hc_mult; ++r) {
        acc += h_pre[r] * streams[(ulong(t) * p.hc_mult + r) * p.D + d];
    }
    hidden[ulong(t) * p.D + d] = acc;
}

kernel void ds_swiglu_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    float g = gate[gid];
    float silu = g / (1.0f + exp(-g));
    out[gid] = silu * up[gid];
}

kernel void ds_add_f32(
    device float* dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    dst[gid] += src[gid];
}

kernel void ds_zero_f32(
    device float* dst [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    dst[gid] = 0.0f;
}

struct DsScaledAddParams {
    uint n;
    uint top_k;
    uint r;
    uint D;
};

kernel void ds_moe_scaled_add_f32(
    device float* dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const float* topk_w [[buffer(2)]],
    constant DsScaledAddParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    uint t = gid / p.D;
    float w = topk_w[t * p.top_k + p.r];
    dst[gid] += w * src[gid];
}

kernel void ds_argmax_f32(
    device const float* logits [[buffer(0)]],
    device int* out_tok [[buffer(1)]],
    constant uint& vocab_size [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tid != 0u) return;
    int best = 0;
    float bv = logits[0];
    for (uint i = 1u; i < vocab_size; ++i) {
        if (logits[i] > bv) {
            bv = logits[i];
            best = int(i);
        }
    }
    out_tok[0] = best;
}
