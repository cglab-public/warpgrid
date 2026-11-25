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
