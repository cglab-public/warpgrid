#include <metal_stdlib>
using namespace metal;

struct GgufGemvParams {
    uint M;
    uint N;
    uint K;
};

struct GgufGroupedGemvParams {
    uint T;
    uint G;
    uint N;
    uint K;
    uint group_stride_bytes;
};

struct GgufMoeParams {
    uint T;
    uint N;
    uint K;
    uint top_k;
    uint expert_stride_bytes;
};

struct GgufMoeSwigluParams {
    uint T;
    uint N;
    uint K;
    uint top_k;
    uint expert_stride_bytes;
    float clamp_val;
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

inline float gguf_q6_k_sub4(
    device const uchar* b,
    uint ql_off,
    uint qh_off,
    uint sc_off,
    float4 y0,
    float4 y1,
    float4 y2,
    float4 y3) {
    uchar4 a = uchar4(*reinterpret_cast<device const packed_uchar4*>(b + ql_off));
    uchar4 c = uchar4(*reinterpret_cast<device const packed_uchar4*>(b + ql_off + 32u));
    uchar4 h = uchar4(*reinterpret_cast<device const packed_uchar4*>(b + qh_off));
    device const char* sc = reinterpret_cast<device const char*>(b + sc_off);
    float d = gguf_load_f16(b + 208u);
    float4 q0 = float4(int4((a & 0x0Fu) | ((h & 0x03u) << 4)) - 32);
    float4 q1 = float4(int4((c & 0x0Fu) | ((h & 0x0Cu) << 2)) - 32);
    float4 q2 = float4(int4((a >> 4) | (h & 0x30u)) - 32);
    float4 q3 = float4(int4((c >> 4) | ((h & 0xC0u) >> 2)) - 32);
    return d * (dot(y0, q0) * float(sc[0]) + dot(y1, q1) * float(sc[2]) +
                dot(y2, q2) * float(sc[4]) + dot(y3, q3) * float(sc[6]));
}

kernel void gguf_gemv_q6_k_fast(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant GgufGemvParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint m = tgid.y;
    uint sg = tid >> 5;
    uint tiisg = tid & 31u;
    uint n0 = tgid.x * 4u + sg * 2u;
    if (m >= p.M || n0 >= p.N) return;
    uint nb = p.K >> 8;
    ulong row_bytes = ulong(nb) * 210ul;
    device const float* yy = X + ulong(m) * p.K;
    uint lane_id = tiisg >> 1;
    uint ix = tiisg & 1u;
    uint ip = lane_id >> 3;
    uint il = lane_id & 7u;
    uint l0 = il << 2;
    uint is = (ip << 3) + (l0 >> 4);
    uint y_off = (ip << 7) + l0;
    uint ql_off = (ip << 6) + l0;
    uint qh_off = 128u + (ip << 5) + l0;
    uint sc_off = 192u + is;
    bool has_r1 = (n0 + 1u < p.N);
    device const uchar* w0 = W + ulong(n0) * row_bytes;
    device const uchar* w1 = has_r1 ? (w0 + row_bytes) : w0;
    float sumf0 = 0.0f;
    float sumf1 = 0.0f;
    for (uint i = ix; i < nb; i += 2u) {
        device const float* y = yy + (i << 8) + y_off;
        float4 y0 = *reinterpret_cast<device const float4*>(y + 0u);
        float4 y1 = *reinterpret_cast<device const float4*>(y + 32u);
        float4 y2 = *reinterpret_cast<device const float4*>(y + 64u);
        float4 y3 = *reinterpret_cast<device const float4*>(y + 96u);
        ulong boff = ulong(i) * 210ul;
        sumf0 += gguf_q6_k_sub4(w0 + boff, ql_off, qh_off, sc_off, y0, y1, y2, y3);
        if (has_r1) sumf1 += gguf_q6_k_sub4(w1 + boff, ql_off, qh_off, sc_off, y0, y1, y2, y3);
    }
    float out0 = simd_sum(sumf0);
    float out1 = simd_sum(sumf1);
    if (tiisg == 0u) {
        ulong yb = ulong(m) * p.N;
        Y[yb + n0] = out0;
        if (has_r1) Y[yb + n0 + 1u] = out1;
    }
}

kernel void gguf_gemv_q6_k_smallk(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant GgufGemvParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint m = tgid.y;
    uint sg = tid >> 5;
    uint tiisg = tid & 31u;
    uint row_sub = tiisg >> 4;
    uint lane_id = tiisg & 15u;
    uint n = tgid.x * 8u + sg * 2u + row_sub;
    if (m >= p.M || n >= p.N) return;
    uint nb = p.K >> 8;
    ulong row_bytes = ulong(nb) * 210ul;
    device const float* yy = X + ulong(m) * p.K;
    uint ip = lane_id >> 3;
    uint il = lane_id & 7u;
    uint l0 = il << 2;
    uint is = (ip << 3) + (l0 >> 4);
    uint y_off = (ip << 7) + l0;
    uint ql_off = (ip << 6) + l0;
    uint qh_off = 128u + (ip << 5) + l0;
    uint sc_off = 192u + is;
    device const uchar* w = W + ulong(n) * row_bytes;
    float sumf = 0.0f;
    for (uint i = 0; i < nb; ++i) {
        device const float* y = yy + (i << 8) + y_off;
        float4 y0 = *reinterpret_cast<device const float4*>(y + 0u);
        float4 y1 = *reinterpret_cast<device const float4*>(y + 32u);
        float4 y2 = *reinterpret_cast<device const float4*>(y + 64u);
        float4 y3 = *reinterpret_cast<device const float4*>(y + 96u);
        sumf += gguf_q6_k_sub4(w + ulong(i) * 210ul, ql_off, qh_off, sc_off, y0, y1, y2, y3);
    }
    sumf += simd_shuffle_xor(sumf, 8u);
    sumf += simd_shuffle_xor(sumf, 4u);
    sumf += simd_shuffle_xor(sumf, 2u);
    sumf += simd_shuffle_xor(sumf, 1u);
    if (lane_id == 0u) {
        Y[ulong(m) * p.N + n] = sumf;
    }
}

kernel void gguf_grouped_gemv_q6_k_fast(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant GgufGroupedGemvParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint tg_idx = tgid.y;
    uint sg = tid >> 5;
    uint tiisg = tid & 31u;
    uint n0 = tgid.x * 4u + sg * 2u;
    if (tg_idx >= p.T * p.G || n0 >= p.N) return;
    uint g = tg_idx % p.G;
    uint nb = p.K >> 8;
    ulong row_bytes = ulong(nb) * 210ul;
    device const float* yy = X + ulong(tg_idx) * p.K;
    uint lane_id = tiisg >> 1;
    uint ix = tiisg & 1u;
    uint ip = lane_id >> 3;
    uint il = lane_id & 7u;
    uint l0 = il << 2;
    uint is = (ip << 3) + (l0 >> 4);
    uint y_off = (ip << 7) + l0;
    uint ql_off = (ip << 6) + l0;
    uint qh_off = 128u + (ip << 5) + l0;
    uint sc_off = 192u + is;
    bool has_r1 = (n0 + 1u < p.N);
    device const uchar* w0 = W + ulong(g) * ulong(p.group_stride_bytes) + ulong(n0) * row_bytes;
    device const uchar* w1 = has_r1 ? (w0 + row_bytes) : w0;
    float sumf0 = 0.0f;
    float sumf1 = 0.0f;
    for (uint i = ix; i < nb; i += 2u) {
        device const float* y = yy + (i << 8) + y_off;
        float4 y0 = *reinterpret_cast<device const float4*>(y + 0u);
        float4 y1 = *reinterpret_cast<device const float4*>(y + 32u);
        float4 y2 = *reinterpret_cast<device const float4*>(y + 64u);
        float4 y3 = *reinterpret_cast<device const float4*>(y + 96u);
        ulong boff = ulong(i) * 210ul;
        sumf0 += gguf_q6_k_sub4(w0 + boff, ql_off, qh_off, sc_off, y0, y1, y2, y3);
        if (has_r1) sumf1 += gguf_q6_k_sub4(w1 + boff, ql_off, qh_off, sc_off, y0, y1, y2, y3);
    }
    float out0 = simd_sum(sumf0);
    float out1 = simd_sum(sumf1);
    if (tiisg == 0u) {
        ulong yb = ulong(tg_idx) * p.N;
        Y[yb + n0] = out0;
        if (has_r1) Y[yb + n0 + 1u] = out1;
    }
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

kernel void gguf_gemv_f32_wide(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant GgufGemvParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint m = tgid.y;
    uint n = tgid.x;
    if (m >= p.M || n >= p.N) return;
    device const float4* xp = reinterpret_cast<device const float4*>(X + ulong(m) * p.K);
    device const float4* wp = reinterpret_cast<device const float4*>(W + ulong(n) * p.K);
    uint k4 = p.K >> 2;
    float4 acc4 = float4(0.0f);
    for (uint k = tid; k < k4; k += 256u) {
        acc4 = fma(xp[k], wp[k], acc4);
    }
    float acc = acc4.x + acc4.y + acc4.z + acc4.w;
    float s = simd_sum(acc);
    threadgroup float partials[8];
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    if (lane == 0u) partials[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float v = (lane < 8u) ? partials[lane] : 0.0f;
        float total = simd_sum(v);
        if (lane == 0u) Y[ulong(m) * p.N + n] = total;
    }
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
    uint tid [[thread_index_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
    if (tgid >= p.T) return;
    device const float* xp = X + ulong(tgid) * p.D;
    device float* yp = Y + ulong(tgid) * p.D;
    float ss = 0.0f;
    if ((p.D & 3u) == 0u) {
        uint d4 = p.D >> 2;
        device const float4* xp4 = reinterpret_cast<device const float4*>(xp);
        float4 s4 = float4(0.0f);
        for (uint d = tid; d < d4; d += tptg) {
            float4 v = xp4[d];
            s4 = fma(v, v, s4);
        }
        ss = s4.x + s4.y + s4.z + s4.w;
    } else {
        for (uint d = tid; d < p.D; d += tptg) {
            float v = xp[d];
            ss = fma(v, v, ss);
        }
    }
    threadgroup float partials[8];
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    float sg_sum = simd_sum(ss);
    if (lane == 0u) partials[sg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n_sg = (tptg + 31u) >> 5;
    if (sg == 0u) {
        float v = (lane < n_sg) ? partials[lane] : 0.0f;
        float tot = simd_sum(v);
        if (lane == 0u) partials[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = partials[0];
    if ((p.D & 3u) == 0u) {
        uint d4 = p.D >> 2;
        device const float4* xp4 = reinterpret_cast<device const float4*>(xp);
        device const float4* wp4 = reinterpret_cast<device const float4*>(W);
        device float4* yp4 = reinterpret_cast<device float4*>(yp);
        for (uint d = tid; d < d4; d += tptg) {
            yp4[d] = xp4[d] * inv_rms * wp4[d];
        }
    } else {
        for (uint d = tid; d < p.D; d += tptg) {
            yp[d] = xp[d] * inv_rms * W[d];
        }
    }
}

kernel void ds_rmsnorm_unweighted_f32(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant DsRmsParams& p [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
    if (tgid >= p.T) return;
    device const float* xp = X + ulong(tgid) * p.D;
    device float* yp = Y + ulong(tgid) * p.D;
    float ss = 0.0f;
    if ((p.D & 3u) == 0u) {
        uint d4 = p.D >> 2;
        device const float4* xp4 = reinterpret_cast<device const float4*>(xp);
        float4 s4 = float4(0.0f);
        for (uint d = tid; d < d4; d += tptg) {
            float4 v = xp4[d];
            s4 = fma(v, v, s4);
        }
        ss = s4.x + s4.y + s4.z + s4.w;
    } else {
        for (uint d = tid; d < p.D; d += tptg) {
            float v = xp[d];
            ss = fma(v, v, ss);
        }
    }
    threadgroup float partials[8];
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    float sg_sum = simd_sum(ss);
    if (lane == 0u) partials[sg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n_sg = (tptg + 31u) >> 5;
    if (sg == 0u) {
        float v = (lane < n_sg) ? partials[lane] : 0.0f;
        float tot = simd_sum(v);
        if (lane == 0u) partials[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = partials[0];
    if ((p.D & 3u) == 0u) {
        uint d4 = p.D >> 2;
        device const float4* xp4 = reinterpret_cast<device const float4*>(xp);
        device float4* yp4 = reinterpret_cast<device float4*>(yp);
        for (uint d = tid; d < d4; d += tptg) {
            yp4[d] = xp4[d] * inv_rms;
        }
    } else {
        for (uint d = tid; d < p.D; d += tptg) {
            yp[d] = xp[d] * inv_rms;
        }
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
    uint pos_stride;
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
    float pos = float(p.start_pos + t * max(1u, p.pos_stride));
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

    if (p.head_dim == 512u) {
        device const float4* q4 = reinterpret_cast<device const float4*>(qp);
        float4 qv0 = q4[tid];
        float4 qv1 = q4[tid + 32u];
        float4 qv2 = q4[tid + 64u];
        float4 qv3 = q4[tid + 96u];
        float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f), acc3 = float4(0.0f);
        float m_max = sinks[h];
        float l_sum = 1.0f;
        for (uint s = win_start; s < seq_end; ++s) {
            device const float4* kv4 = reinterpret_cast<device const float4*>(KV_cache + ulong(s) * 512ul);
            float4 k0 = kv4[tid];
            float4 k1 = kv4[tid + 32u];
            float4 k2 = kv4[tid + 64u];
            float4 k3 = kv4[tid + 96u];
            float sc = simd_sum(dot(qv0, k0) + dot(qv1, k1) + dot(qv2, k2) + dot(qv3, k3)) * p.scale;
            float m_new = max(m_max, sc);
            float alpha = exp(m_max - m_new);
            float w = exp(sc - m_new);
            l_sum = fma(l_sum, alpha, w);
            acc0 = fma(acc0, alpha, w * k0);
            acc1 = fma(acc1, alpha, w * k1);
            acc2 = fma(acc2, alpha, w * k2);
            acc3 = fma(acc3, alpha, w * k3);
            m_max = m_new;
        }
        float inv_l = 1.0f / max(l_sum, 1e-20f);
        device float4* o4 = reinterpret_cast<device float4*>(op);
        o4[tid] = acc0 * inv_l;
        o4[tid + 32u] = acc1 * inv_l;
        o4[tid + 64u] = acc2 * inv_l;
        o4[tid + 96u] = acc3 * inv_l;
        return;
    }

    float sink_val = sinks[h];
    float m_max = sink_val;
    for (uint s = win_start; s < seq_end; ++s) {
        device const float* kvp = KV_cache + ulong(s) * p.head_dim;
        float dsum = 0.0f;
        for (uint d = tid; d < p.head_dim; d += 32u) dsum += qp[d] * kvp[d];
        float sc = simd_sum(dsum) * p.scale;
        m_max = max(m_max, sc);
    }

    float l_sum = exp(sink_val - m_max);
    for (uint d = tid; d < p.head_dim; d += 32u) op[d] = 0.0f;
    for (uint s = win_start; s < seq_end; ++s) {
        device const float* kvp = KV_cache + ulong(s) * p.head_dim;
        float dsum = 0.0f;
        for (uint d = tid; d < p.head_dim; d += 32u) dsum += qp[d] * kvp[d];
        float w = exp(simd_sum(dsum) * p.scale - m_max);
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
    uint max_comp;
    uint n_idx_head;
    uint idx_dim;
    uint ratio;
    uint sliding_window;
    uint top_k;
    float iw_scale;
};

inline uint gguf_f32_to_ord(float v) {
    uint u = as_type<uint>(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

kernel void ds_indexer_score_topk_f32(
    device const float* iq [[buffer(0)]],
    device const float* ik_cache [[buffer(1)]],
    device const float* iw [[buffer(2)]],
    device float* scores_buf [[buffer(3)]],
    device int* out_topk [[buffer(4)]],
    constant DsIndexerParams& p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
    if (tgid >= p.T) return;
    uint t = tgid;
    uint pos = p.start_pos + t;
    uint win_start = (p.sliding_window > 0u && pos + 1u > p.sliding_window) ? (pos + 1u - p.sliding_window) : 0u;
    uint max_vis = min(p.n_comp, win_start / max(1u, p.ratio));
    uint K = p.top_k;

    device const float* iqt = iq + ulong(t) * p.n_idx_head * p.idx_dim;
    device const float* iwt = iw + ulong(t) * p.n_idx_head;
    device float* sc = scores_buf + ulong(t) * p.max_comp;
    device int* dst = out_topk + ulong(t) * K;

    if (max_vis <= K) {
        for (uint k = tid; k < K; k += tptg) {
            dst[k] = (k < max_vis) ? int(k) : -1;
        }
        return;
    }

    for (uint c = tid; c < max_vis; c += tptg) {
        device const float* ikc = ik_cache + ulong(c) * p.idx_dim;
        float total = 0.0f;
        if ((p.idx_dim & 3u) == 0u) {
            uint d4 = p.idx_dim >> 2;
            device const float4* ik4 = reinterpret_cast<device const float4*>(ikc);
            for (uint h = 0; h < p.n_idx_head; ++h) {
                device const float4* iq4 = reinterpret_cast<device const float4*>(iqt + ulong(h) * p.idx_dim);
                float4 dsum = float4(0.0f);
                for (uint d = 0; d < d4; ++d) dsum = fma(iq4[d], ik4[d], dsum);
                float dv = dsum.x + dsum.y + dsum.z + dsum.w;
                if (dv > 0.0f) total = fma(dv, iwt[h] * p.iw_scale, total);
            }
        } else {
            for (uint h = 0; h < p.n_idx_head; ++h) {
                device const float* iqh = iqt + ulong(h) * p.idx_dim;
                float dv = 0.0f;
                for (uint d = 0; d < p.idx_dim; ++d) dv += iqh[d] * ikc[d];
                if (dv > 0.0f) total += dv * (iwt[h] * p.iw_scale);
            }
        }
        sc[c] = total;
    }
    threadgroup_barrier(mem_flags::mem_device);

    threadgroup uint sg_u[8];
    threadgroup uint b_lo;
    threadgroup uint b_hi;
    threadgroup atomic_uint w_cnt;
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint n_sg = (tptg + 31u) >> 5;

    uint lmin = 0xFFFFFFFFu;
    uint lmax = 0u;
    for (uint c = tid; c < max_vis; c += tptg) {
        uint kv = gguf_f32_to_ord(sc[c]);
        lmin = min(lmin, kv);
        lmax = max(lmax, kv);
    }
    uint smin = simd_min(lmin);
    uint smax = simd_max(lmax);
    if (lane == 0u) sg_u[sg] = smin;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint gmin = 0xFFFFFFFFu;
        for (uint s = 0; s < n_sg; ++s) gmin = min(gmin, sg_u[s]);
        b_lo = gmin;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) sg_u[sg] = smax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint gmax = 0u;
        for (uint s = 0; s < n_sg; ++s) gmax = max(gmax, sg_u[s]);
        b_hi = gmax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint it = 0; it < 32u; ++it) {
        uint lo = b_lo;
        uint hi = b_hi;
        if (lo >= hi) break;
        uint mid = lo + ((hi - lo) >> 1);
        uint cnt = 0u;
        for (uint c = tid; c < max_vis; c += tptg) {
            if (gguf_f32_to_ord(sc[c]) > mid) cnt += 1u;
        }
        uint scnt = simd_sum(cnt);
        if (lane == 0u) sg_u[sg] = scnt;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint tot = 0u;
            for (uint s = 0; s < n_sg; ++s) tot += sg_u[s];
            if (tot >= K) {
                b_lo = mid + 1u;
            } else {
                b_hi = mid;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint thresh = b_lo;
    if (tid == 0u) atomic_store_explicit(&w_cnt, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = tid; c < max_vis; c += tptg) {
        if (gguf_f32_to_ord(sc[c]) > thresh) {
            uint idx = atomic_fetch_add_explicit(&w_cnt, 1u, memory_order_relaxed);
            if (idx < K) dst[idx] = int(c);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = tid; c < max_vis; c += tptg) {
        if (gguf_f32_to_ord(sc[c]) == thresh) {
            uint cur = atomic_load_explicit(&w_cnt, memory_order_relaxed);
            if (cur < K) {
                uint idx = atomic_fetch_add_explicit(&w_cnt, 1u, memory_order_relaxed);
                if (idx < K) dst[idx] = int(c);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint filled = atomic_load_explicit(&w_cnt, memory_order_relaxed);
    for (uint k = filled + tid; k < K; k += tptg) {
        dst[k] = -1;
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

    if (p.head_dim == 512u) {
        device const float4* q4 = reinterpret_cast<device const float4*>(qp);
        float4 qv0 = q4[tid];
        float4 qv1 = q4[tid + 32u];
        float4 qv2 = q4[tid + 64u];
        float4 qv3 = q4[tid + 96u];
        float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f), acc3 = float4(0.0f);
        float m_max = sinks[h];
        float l_sum = 1.0f;
        for (uint s = win_start; s < seq_end; ++s) {
            device const float4* kv4 = reinterpret_cast<device const float4*>(KV_cache + ulong(s) * 512ul);
            float4 k0 = kv4[tid];
            float4 k1 = kv4[tid + 32u];
            float4 k2 = kv4[tid + 64u];
            float4 k3 = kv4[tid + 96u];
            float sc = simd_sum(dot(qv0, k0) + dot(qv1, k1) + dot(qv2, k2) + dot(qv3, k3)) * p.scale;
            float m_new = max(m_max, sc);
            float alpha = exp(m_max - m_new);
            float w = exp(sc - m_new);
            l_sum = fma(l_sum, alpha, w);
            acc0 = fma(acc0, alpha, w * k0);
            acc1 = fma(acc1, alpha, w * k1);
            acc2 = fma(acc2, alpha, w * k2);
            acc3 = fma(acc3, alpha, w * k3);
            m_max = m_new;
        }
        for (uint k = 0; k < p.n_comp_sel; ++k) {
            int c = sel[k];
            if (c < 0) continue;
            device const float4* kv4 = reinterpret_cast<device const float4*>(comp_kv_cache + ulong(uint(c)) * 512ul);
            float4 k0 = kv4[tid];
            float4 k1 = kv4[tid + 32u];
            float4 k2 = kv4[tid + 64u];
            float4 k3 = kv4[tid + 96u];
            float sc = simd_sum(dot(qv0, k0) + dot(qv1, k1) + dot(qv2, k2) + dot(qv3, k3)) * p.scale;
            float m_new = max(m_max, sc);
            float alpha = exp(m_max - m_new);
            float w = exp(sc - m_new);
            l_sum = fma(l_sum, alpha, w);
            acc0 = fma(acc0, alpha, w * k0);
            acc1 = fma(acc1, alpha, w * k1);
            acc2 = fma(acc2, alpha, w * k2);
            acc3 = fma(acc3, alpha, w * k3);
            m_max = m_new;
        }
        float inv_l = 1.0f / max(l_sum, 1e-20f);
        device float4* o4 = reinterpret_cast<device float4*>(op);
        o4[tid] = acc0 * inv_l;
        o4[tid + 32u] = acc1 * inv_l;
        o4[tid + 64u] = acc2 * inv_l;
        o4[tid + 96u] = acc3 * inv_l;
        return;
    }

    float sink_val = sinks[h];
    float m_max = sink_val;
    for (uint s = win_start; s < seq_end; ++s) {
        device const float* kvp = KV_cache + ulong(s) * p.head_dim;
        float dsum = 0.0f;
        for (uint d = tid; d < p.head_dim; d += 32u) dsum += qp[d] * kvp[d];
        m_max = max(m_max, simd_sum(dsum) * p.scale);
    }
    for (uint k = 0; k < p.n_comp_sel; ++k) {
        int c = sel[k];
        if (c < 0) continue;
        device const float* kvp = comp_kv_cache + ulong(uint(c)) * p.head_dim;
        float dsum = 0.0f;
        for (uint d = tid; d < p.head_dim; d += 32u) dsum += qp[d] * kvp[d];
        m_max = max(m_max, simd_sum(dsum) * p.scale);
    }

    float l_sum = exp(sink_val - m_max);
    for (uint d = tid; d < p.head_dim; d += 32u) op[d] = 0.0f;
    for (uint s = win_start; s < seq_end; ++s) {
        device const float* kvp = KV_cache + ulong(s) * p.head_dim;
        float dsum = 0.0f;
        for (uint d = tid; d < p.head_dim; d += 32u) dsum += qp[d] * kvp[d];
        float w = exp(simd_sum(dsum) * p.scale - m_max);
        l_sum += w;
        for (uint d = tid; d < p.head_dim; d += 32u) op[d] += w * kvp[d];
    }
    for (uint k = 0; k < p.n_comp_sel; ++k) {
        int c = sel[k];
        if (c < 0) continue;
        device const float* kvp = comp_kv_cache + ulong(uint(c)) * p.head_dim;
        float dsum = 0.0f;
        for (uint d = tid; d < p.head_dim; d += 32u) dsum += qp[d] * kvp[d];
        float w = exp(simd_sum(dsum) * p.scale - m_max);
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

struct DsSwigluParams {
    uint n;
    float clamp_val;
};

kernel void ds_swiglu_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant DsSwigluParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    float g = gate[gid];
    float u = up[gid];
    if (p.clamp_val > 0.0f) {
        g = clamp(g, -p.clamp_val, p.clamp_val);
        u = clamp(u, -p.clamp_val, p.clamp_val);
    }
    float silu = g / (1.0f + exp(-g));
    out[gid] = silu * u;
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
    uint tid [[thread_index_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
    float bv = -INFINITY;
    int bi = 0;
    for (uint i = tid; i < vocab_size; i += tptg) {
        float v = logits[i];
        if (v > bv) {
            bv = v;
            bi = int(i);
        }
    }
    threadgroup float s_val[256];
    threadgroup int s_idx[256];
    s_val[tid] = bv;
    s_idx[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tptg >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            float ov = s_val[tid + stride];
            int oi = s_idx[tid + stride];
            if (ov > s_val[tid] || (ov == s_val[tid] && oi < s_idx[tid])) {
                s_val[tid] = ov;
                s_idx[tid] = oi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) out_tok[0] = s_idx[0];
}

constant uchar kGgufSignsIq2xs[128] = {
    0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

constant ulong kGgufIq2xxsGrid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

constant uint kGgufIq1sGridGpu[2048] = {
    0x00000000, 0x00000002, 0x00000101, 0x00000200, 0x00000202, 0x00010001, 0x00010101, 0x00020000,
    0x00020002, 0x00020200, 0x00020202, 0x01000101, 0x01010001, 0x01010100, 0x01010102, 0x01020101,
    0x02000000, 0x02000002, 0x02000200, 0x02000202, 0x02010101, 0x02020000, 0x02020002, 0x02020200,
    0x02020202, 0x00000110, 0x00000111, 0x00010011, 0x00010110, 0x00010112, 0x00010211, 0x00010212,
    0x00020111, 0x01000011, 0x01000112, 0x01000211, 0x01010012, 0x01010111, 0x01010212, 0x01020011,
    0x01020110, 0x01020112, 0x01020210, 0x02000111, 0x02010011, 0x02010110, 0x02010112, 0x02020111,
    0x00000020, 0x00000022, 0x00000220, 0x00000222, 0x00010121, 0x00020020, 0x00020022, 0x00020220,
    0x00020222, 0x01000121, 0x01010021, 0x01010221, 0x01020120, 0x01020221, 0x02000020, 0x02000022,
    0x02000220, 0x02000222, 0x02010021, 0x02010121, 0x02010221, 0x02020020, 0x02020022, 0x02020220,
    0x02020222, 0x00011001, 0x00011100, 0x00011102, 0x00021101, 0x01001001, 0x01001201, 0x01011101,
    0x01011202, 0x01021100, 0x01021101, 0x02011001, 0x02011201, 0x02021101, 0x00001011, 0x00001110,
    0x00001111, 0x00001112, 0x00011111, 0x00011210, 0x00011212, 0x00021211, 0x01001010, 0x01001111,
    0x01001212, 0x01011010, 0x01011011, 0x01011110, 0x01011111, 0x01011112, 0x01011211, 0x01021010,
    0x01021012, 0x01021111, 0x01021210, 0x01021212, 0x02001011, 0x02011011, 0x02011111, 0x02011210,
    0x02011212, 0x02021011, 0x02021110, 0x02021111, 0x02021112, 0x02021211, 0x00011120, 0x00011221,
    0x01001021, 0x01001120, 0x01011020, 0x01011022, 0x01011121, 0x01011220, 0x01021020, 0x01021021,
    0x01021122, 0x01021221, 0x02001121, 0x02011021, 0x02011120, 0x02011221, 0x00002000, 0x00002002,
    0x00002200, 0x00002202, 0x00012101, 0x00022000, 0x00022002, 0x00022200, 0x00022202, 0x01002101,
    0x01012001, 0x01012102, 0x01022101, 0x02002000, 0x02002002, 0x02002200, 0x02002202, 0x02012101,
    0x02022000, 0x02022002, 0x02022200, 0x02022202, 0x00002111, 0x00012011, 0x00012110, 0x00012211,
    0x00022110, 0x00022111, 0x01002011, 0x01012010, 0x01012011, 0x01012111, 0x01022011, 0x01022110,
    0x01022211, 0x02012011, 0x02012110, 0x02012112, 0x02012211, 0x02022111, 0x00002020, 0x00002022,
    0x00002220, 0x00002222, 0x00012121, 0x00022020, 0x00022022, 0x00022220, 0x00022222, 0x01002121,
    0x01012021, 0x01012221, 0x01022021, 0x01022121, 0x02002020, 0x02002022, 0x02002121, 0x02002220,
    0x02002222, 0x02012121, 0x02022020, 0x02022022, 0x02022220, 0x02022222, 0x00110000, 0x00110001,
    0x00110100, 0x00110201, 0x00120100, 0x00120101, 0x01100001, 0x01100100, 0x01110000, 0x01110101,
    0x01110200, 0x01120001, 0x01120100, 0x01120101, 0x01120201, 0x02110001, 0x02110100, 0x02110102,
    0x02120001, 0x02120101, 0x00100011, 0x00100110, 0x00100112, 0x00100211, 0x00110010, 0x00110012,
    0x00110111, 0x00110210, 0x00120011, 0x00120110, 0x00120211, 0x01100111, 0x01100212, 0x01110010,
    0x01110011, 0x01110012, 0x01110110, 0x01110111, 0x01110112, 0x01110211, 0x01120010, 0x01120111,
    0x02100110, 0x02110012, 0x02110111, 0x02120011, 0x02120110, 0x00110021, 0x00110120, 0x00110122,
    0x00120121, 0x01100020, 0x01100122, 0x01100221, 0x01110022, 0x01110121, 0x01110220, 0x01110222,
    0x01120120, 0x01120122, 0x02100121, 0x02110021, 0x02110120, 0x02110122, 0x02120121, 0x00101001,
    0x00101102, 0x00101201, 0x00111100, 0x00111101, 0x00111200, 0x00111201, 0x00121001, 0x00121102,
    0x01101001, 0x01101101, 0x01101102, 0x01101200, 0x01101202, 0x01111001, 0x01111100, 0x01111101,
    0x01111102, 0x01111201, 0x01121002, 0x01121101, 0x01121200, 0x02101100, 0x02101201, 0x02111000,
    0x02111100, 0x02111101, 0x02111200, 0x02111201, 0x02111202, 0x02121001, 0x02121100, 0x02121101,
    0x02121201, 0x00101012, 0x00101111, 0x00101212, 0x00111011, 0x00111110, 0x00111111, 0x00111112,
    0x00111211, 0x00121010, 0x00121012, 0x00121111, 0x00121210, 0x00121212, 0x01101011, 0x01101110,
    0x01101111, 0x01101112, 0x01111011, 0x01111012, 0x01111110, 0x01111111, 0x01111112, 0x01111211,
    0x01111212, 0x01121011, 0x01121110, 0x01121111, 0x01121112, 0x01121211, 0x02101010, 0x02101012,
    0x02101110, 0x02101111, 0x02101210, 0x02101212, 0x02111010, 0x02111011, 0x02111110, 0x02111111,
    0x02111112, 0x02111211, 0x02111212, 0x02121010, 0x02121012, 0x02121111, 0x00101021, 0x00101120,
    0x00101121, 0x00101122, 0x00111121, 0x00111122, 0x00111220, 0x00111222, 0x00121021, 0x00121122,
    0x01101020, 0x01101022, 0x01101120, 0x01101121, 0x01101220, 0x01101222, 0x01111021, 0x01111121,
    0x01111122, 0x01111220, 0x01111221, 0x01121021, 0x01121120, 0x01121121, 0x01121220, 0x01121221,
    0x01121222, 0x02101122, 0x02101222, 0x02111022, 0x02111121, 0x02121120, 0x02121221, 0x00112001,
    0x00112102, 0x00122101, 0x01102001, 0x01102100, 0x01102102, 0x01102201, 0x01112000, 0x01112101,
    0x01112200, 0x01112202, 0x01122000, 0x01122001, 0x01122100, 0x01122102, 0x01122201, 0x02102101,
    0x02112001, 0x02112100, 0x02122101, 0x00112010, 0x00112012, 0x00112111, 0x00112212, 0x00122011,
    0x00122111, 0x01102012, 0x01102110, 0x01102111, 0x01102210, 0x01112011, 0x01112110, 0x01112111,
    0x01112112, 0x01112211, 0x01112212, 0x01122010, 0x01122111, 0x01122212, 0x02102211, 0x02112011,
    0x02112012, 0x02112111, 0x02112210, 0x02122011, 0x02122112, 0x02122211, 0x00102221, 0x00112122,
    0x00122120, 0x00122122, 0x01102120, 0x01102122, 0x01102221, 0x01112020, 0x01112022, 0x01112121,
    0x01112220, 0x01122021, 0x01122122, 0x01122221, 0x02102121, 0x02112021, 0x02112122, 0x02112222,
    0x00200000, 0x00200002, 0x00200200, 0x00200202, 0x00210101, 0x00220000, 0x00220002, 0x00220101,
    0x00220200, 0x00220202, 0x01200101, 0x01210001, 0x01210201, 0x01220001, 0x01220101, 0x02200000,
    0x02200002, 0x02200200, 0x02200202, 0x02210101, 0x02220000, 0x02220002, 0x02220101, 0x02220200,
    0x02220202, 0x00200111, 0x00210011, 0x00210110, 0x00210211, 0x00220111, 0x01200012, 0x01200110,
    0x01200211, 0x01210111, 0x01210210, 0x01210212, 0x01220011, 0x01220110, 0x01220111, 0x01220112,
    0x02200111, 0x02210010, 0x02210112, 0x02210211, 0x02220111, 0x00200021, 0x00200220, 0x00200222,
    0x00210021, 0x00210121, 0x00220020, 0x00220022, 0x00220220, 0x00220222, 0x01200121, 0x01210021,
    0x01210122, 0x01210221, 0x01220121, 0x02200021, 0x02200220, 0x02200222, 0x02210021, 0x02210121,
    0x02220020, 0x02220022, 0x02220220, 0x02220222, 0x00201101, 0x00211100, 0x00211102, 0x00211201,
    0x00221101, 0x01201100, 0x01201101, 0x01201102, 0x01201201, 0x01211002, 0x01211101, 0x01211200,
    0x01211202, 0x01221102, 0x02201101, 0x02211001, 0x02211100, 0x02211201, 0x02221001, 0x02221101,
    0x00201211, 0x00211111, 0x00221011, 0x00221211, 0x01201010, 0x01201111, 0x01201210, 0x01211011,
    0x01211110, 0x01211111, 0x01211211, 0x01221012, 0x01221111, 0x01221210, 0x02201211, 0x02211010,
    0x02211110, 0x02211111, 0x02211210, 0x02211212, 0x02221011, 0x02221110, 0x02221112, 0x02221211,
    0x00201121, 0x00211020, 0x00211022, 0x00211221, 0x00221121, 0x01201021, 0x01201221, 0x01211121,
    0x01221020, 0x01221021, 0x01221221, 0x02201120, 0x02201122, 0x02211020, 0x02211222, 0x00202000,
    0x00202002, 0x00202200, 0x00202202, 0x00212101, 0x00222000, 0x00222002, 0x00222200, 0x00222202,
    0x01202101, 0x01212001, 0x01212100, 0x01222101, 0x02202000, 0x02202002, 0x02202200, 0x02202202,
    0x02222000, 0x02222002, 0x02222200, 0x02222202, 0x00202211, 0x00212011, 0x00212110, 0x00212211,
    0x00222111, 0x01202112, 0x01202211, 0x01212012, 0x01212111, 0x01222011, 0x01222110, 0x01222112,
    0x01222211, 0x02202111, 0x02212010, 0x02212112, 0x02212211, 0x02222110, 0x02222111, 0x00202020,
    0x00202022, 0x00202220, 0x00202222, 0x00222020, 0x00222022, 0x00222220, 0x00222222, 0x01202121,
    0x01212021, 0x01212122, 0x01212221, 0x01222121, 0x02202020, 0x02202022, 0x02202220, 0x02202222,
    0x02212121, 0x02222020, 0x02222022, 0x02222220, 0x02222222, 0x10000101, 0x10010001, 0x10010102,
    0x10020101, 0x11000201, 0x11010002, 0x11010101, 0x11010200, 0x11010202, 0x11020001, 0x11020100,
    0x11020102, 0x12010100, 0x12010201, 0x12020001, 0x12020102, 0x10000010, 0x10000011, 0x10000110,
    0x10000112, 0x10000211, 0x10010012, 0x10010111, 0x10010112, 0x10010210, 0x10010212, 0x10020011,
    0x10020112, 0x10020211, 0x11000111, 0x11000210, 0x11000212, 0x11010011, 0x11010110, 0x11010111,
    0x11010112, 0x11010211, 0x11010212, 0x11020111, 0x11020210, 0x11020212, 0x12000011, 0x12000110,
    0x12000112, 0x12010010, 0x12010012, 0x12010111, 0x12020010, 0x12020011, 0x12020012, 0x10000121,
    0x10010021, 0x10010120, 0x10010122, 0x10020121, 0x11000021, 0x11010022, 0x11010121, 0x11010222,
    0x11020120, 0x11020221, 0x12000221, 0x12010120, 0x12020121, 0x10001001, 0x10011101, 0x10011201,
    0x10021201, 0x11001101, 0x11001200, 0x11001202, 0x11011001, 0x11011100, 0x11011101, 0x11011102,
    0x11021001, 0x11021002, 0x11021101, 0x11021200, 0x11021202, 0x12001001, 0x12001102, 0x12001201,
    0x12011000, 0x12011002, 0x12011101, 0x12021000, 0x12021001, 0x12021201, 0x10001011, 0x10001012,
    0x10001111, 0x10001212, 0x10011011, 0x10011110, 0x10011111, 0x10011112, 0x10011211, 0x10021010,
    0x10021111, 0x10021212, 0x11001011, 0x11001110, 0x11001111, 0x11001112, 0x11001211, 0x11011010,
    0x11011011, 0x11011110, 0x11011111, 0x11011112, 0x11011210, 0x11011211, 0x11021011, 0x11021110,
    0x11021111, 0x11021112, 0x11021211, 0x12001012, 0x12001110, 0x12001111, 0x12001210, 0x12011011,
    0x12011110, 0x12011111, 0x12011112, 0x12011211, 0x12011212, 0x12021111, 0x12021210, 0x12021212,
    0x10001021, 0x10001121, 0x10001221, 0x10011120, 0x10011121, 0x10011220, 0x10011222, 0x10021021,
    0x10021120, 0x10021221, 0x11001020, 0x11001022, 0x11001121, 0x11001220, 0x11011020, 0x11011021,
    0x11011022, 0x11011121, 0x11011122, 0x11011221, 0x11021022, 0x11021121, 0x11021220, 0x12001021,
    0x12001121, 0x12001222, 0x12011120, 0x12011121, 0x12021021, 0x12021120, 0x12021122, 0x10002101,
    0x10012001, 0x10012101, 0x10012202, 0x10022101, 0x11002002, 0x11002201, 0x11012000, 0x11012101,
    0x11012200, 0x11022001, 0x11022100, 0x11022102, 0x11022201, 0x12002101, 0x12012001, 0x12012100,
    0x12012102, 0x12012201, 0x12022101, 0x10002011, 0x10002111, 0x10002112, 0x10002212, 0x10012010,
    0x10012110, 0x10012111, 0x10012210, 0x10022011, 0x10022110, 0x10022112, 0x11002010, 0x11002111,
    0x11002212, 0x11012011, 0x11012012, 0x11012110, 0x11012111, 0x11012112, 0x11012211, 0x11022010,
    0x11022012, 0x11022111, 0x11022112, 0x11022212, 0x12002112, 0x12002211, 0x12012012, 0x12012111,
    0x12012112, 0x12012210, 0x12022011, 0x12022110, 0x12022112, 0x12022211, 0x10012122, 0x11002120,
    0x11002122, 0x11002221, 0x11012121, 0x11012220, 0x11012222, 0x11022120, 0x11022221, 0x12012120,
    0x12022121, 0x10100001, 0x10100100, 0x10100101, 0x10100102, 0x10100201, 0x10110002, 0x10110101,
    0x10110202, 0x10120001, 0x10120100, 0x10120201, 0x11100000, 0x11100101, 0x11100200, 0x11110001,
    0x11110100, 0x11110101, 0x11110102, 0x11110201, 0x11120101, 0x11120200, 0x12100102, 0x12100201,
    0x12110101, 0x12110200, 0x12120000, 0x12120001, 0x12120102, 0x12120201, 0x10100111, 0x10100210,
    0x10100211, 0x10100212, 0x10110011, 0x10110110, 0x10110111, 0x10110112, 0x10110210, 0x10110211,
    0x10120010, 0x10120111, 0x10120112, 0x10120210, 0x10120212, 0x11100011, 0x11100110, 0x11100111,
    0x11100112, 0x11100211, 0x11110010, 0x11110011, 0x11110012, 0x11110110, 0x11110111, 0x11110112,
    0x11110210, 0x11110211, 0x11110212, 0x11120011, 0x11120110, 0x11120111, 0x11120112, 0x11120211,
    0x12100012, 0x12100111, 0x12110011, 0x12110110, 0x12110111, 0x12110112, 0x12110211, 0x12120010,
    0x12120111, 0x12120212, 0x10100021, 0x10100122, 0x10110022, 0x10110121, 0x10110222, 0x10120021,
    0x10120120, 0x11100022, 0x11100121, 0x11100222, 0x11110021, 0x11110120, 0x11110121, 0x11110122,
    0x11110221, 0x11120022, 0x11120121, 0x12100121, 0x12110020, 0x12110022, 0x12110121, 0x12110221,
    0x12110222, 0x12120120, 0x10101100, 0x10101101, 0x10111001, 0x10111100, 0x10111101, 0x10111102,
    0x10111200, 0x10111201, 0x10121001, 0x10121101, 0x10121200, 0x10121202, 0x11101001, 0x11101100,
    0x11101101, 0x11101102, 0x11101201, 0x11101202, 0x11111000, 0x11111001, 0x11111100, 0x11111101,
    0x11111102, 0x11111200, 0x11111201, 0x11111202, 0x11121001, 0x11121002, 0x11121100, 0x11121101,
    0x11121102, 0x11121201, 0x12101000, 0x12101200, 0x12101202, 0x12111001, 0x12111100, 0x12111101,
    0x12111102, 0x12111201, 0x12121001, 0x12121100, 0x12121101, 0x12121202, 0x10101011, 0x10101012,
    0x10101110, 0x10101111, 0x10101112, 0x10101211, 0x10111010, 0x10111011, 0x10111012, 0x10111110,
    0x10111111, 0x10111112, 0x10111211, 0x10111212, 0x10121011, 0x10121110, 0x10121111, 0x10121112,
    0x10121211, 0x11101010, 0x11101011, 0x11101012, 0x11101110, 0x11101111, 0x11101112, 0x11101210,
    0x11101211, 0x11111010, 0x11111011, 0x11111012, 0x11111110, 0x11111111, 0x11111112, 0x11111210,
    0x11111211, 0x11111212, 0x11121010, 0x11121011, 0x11121110, 0x11121111, 0x11121112, 0x11121210,
    0x11121211, 0x11121212, 0x12101011, 0x12101110, 0x12101111, 0x12101211, 0x12101212, 0x12111010,
    0x12111011, 0x12111110, 0x12111111, 0x12111112, 0x12111210, 0x12111211, 0x12121011, 0x12121110,
    0x12121111, 0x12121112, 0x12121211, 0x10101020, 0x10101021, 0x10101022, 0x10101120, 0x10101122,
    0x10101220, 0x10101221, 0x10111021, 0x10111120, 0x10111121, 0x10111220, 0x10111221, 0x10121020,
    0x10121021, 0x10121022, 0x10121120, 0x10121121, 0x10121122, 0x10121220, 0x10121221, 0x11101021,
    0x11101121, 0x11101122, 0x11101220, 0x11101221, 0x11101222, 0x11111020, 0x11111021, 0x11111022,
    0x11111120, 0x11111121, 0x11111122, 0x11111220, 0x11111221, 0x11111222, 0x11121021, 0x11121120,
    0x11121121, 0x11121221, 0x12101022, 0x12101121, 0x12101122, 0x12101220, 0x12101221, 0x12101222,
    0x12111021, 0x12111121, 0x12111222, 0x12121022, 0x12121121, 0x12121122, 0x12121220, 0x12121221,
    0x10102100, 0x10102101, 0x10102102, 0x10102201, 0x10112000, 0x10112101, 0x10112200, 0x10122001,
    0x10122202, 0x11102101, 0x11102200, 0x11102202, 0x11112001, 0x11112100, 0x11112101, 0x11112102,
    0x11112200, 0x11112201, 0x11122000, 0x11122002, 0x11122100, 0x11122101, 0x12102002, 0x12102201,
    0x12112000, 0x12112002, 0x12112101, 0x12112200, 0x12122001, 0x12122201, 0x10102011, 0x10102012,
    0x10102111, 0x10102212, 0x10112011, 0x10112110, 0x10112111, 0x10112112, 0x10112211, 0x10122111,
    0x11102011, 0x11102110, 0x11102111, 0x11102112, 0x11102211, 0x11112010, 0x11112011, 0x11112012,
    0x11112110, 0x11112111, 0x11112112, 0x11112210, 0x11112211, 0x11112212, 0x11122011, 0x11122110,
    0x11122111, 0x11122112, 0x11122211, 0x12102011, 0x12102111, 0x12102211, 0x12112011, 0x12112110,
    0x12112111, 0x12112112, 0x12112210, 0x12112211, 0x12122111, 0x10102120, 0x10102220, 0x10112121,
    0x10112222, 0x10122020, 0x10122121, 0x10122122, 0x10122221, 0x11102121, 0x11102220, 0x11102221,
    0x11112021, 0x11112121, 0x11112122, 0x11112220, 0x11112221, 0x11122022, 0x11122121, 0x11122220,
    0x11122222, 0x12102021, 0x12102222, 0x12112022, 0x12112121, 0x12112122, 0x12112220, 0x12112222,
    0x12122021, 0x10200101, 0x10210100, 0x10210102, 0x10210201, 0x10220101, 0x11200100, 0x11210000,
    0x11210101, 0x11210102, 0x11210200, 0x11210202, 0x11220001, 0x11220100, 0x11220102, 0x11220201,
    0x12200001, 0x12210102, 0x12220101, 0x10200011, 0x10200110, 0x10200112, 0x10200211, 0x10210012,
    0x10210111, 0x10220011, 0x10220012, 0x10220112, 0x10220211, 0x11200111, 0x11200211, 0x11210011,
    0x11210111, 0x11210112, 0x11210211, 0x11220111, 0x11220112, 0x11220212, 0x12200110, 0x12200212,
    0x12210012, 0x12210111, 0x12220011, 0x12220112, 0x12220211, 0x10210021, 0x10210122, 0x10210221,
    0x11200020, 0x11200021, 0x11200122, 0x11210121, 0x11210122, 0x11210220, 0x11220020, 0x12200121,
    0x12210021, 0x12210122, 0x12220121, 0x10211001, 0x10211002, 0x10211101, 0x10211102, 0x10211202,
    0x10221001, 0x10221102, 0x10221201, 0x11201000, 0x11201002, 0x11201101, 0x11201200, 0x11201202,
    0x11211001, 0x11211100, 0x11211101, 0x11211102, 0x11211201, 0x11211202, 0x11221000, 0x11221002,
    0x11221101, 0x12201100, 0x12201101, 0x12201201, 0x12211000, 0x12211002, 0x12211100, 0x12211101,
    0x12211102, 0x12211200, 0x12211202, 0x12221001, 0x12221100, 0x12221201, 0x10201111, 0x10201210,
    0x10201212, 0x10211011, 0x10211111, 0x10211112, 0x10211211, 0x11201110, 0x11201111, 0x11201112,
    0x11201211, 0x11211010, 0x11211011, 0x11211110, 0x11211111, 0x11211112, 0x11211211, 0x11221011,
    0x11221110, 0x11221111, 0x11221112, 0x11221211, 0x12201112, 0x12201211, 0x12201212, 0x12211011,
    0x12211111, 0x12211112, 0x12211211, 0x12211212, 0x12221012, 0x12221111, 0x12221112, 0x12221210,
    0x10201022, 0x10201221, 0x10211121, 0x10221020, 0x10221122, 0x10221220, 0x10221221, 0x11201020,
    0x11201121, 0x11201220, 0x11201222, 0x11211021, 0x11211120, 0x11211121, 0x11211122, 0x11211220,
    0x11211222, 0x11221020, 0x11221121, 0x11221220, 0x12201020, 0x12201022, 0x12201121, 0x12201222,
    0x12211120, 0x12211122, 0x12211220, 0x12211221, 0x12221020, 0x12221120, 0x12221122, 0x12221222,
    0x10212102, 0x10212201, 0x10222101, 0x11202001, 0x11212002, 0x11212101, 0x11212202, 0x11222001,
    0x11222201, 0x12202101, 0x12212001, 0x12212200, 0x12222102, 0x10202011, 0x10202110, 0x10212010,
    0x10212111, 0x10222011, 0x10222110, 0x10222112, 0x10222211, 0x11202010, 0x11202011, 0x11202111,
    0x11202112, 0x11202210, 0x11212011, 0x11212110, 0x11212111, 0x11212112, 0x11212211, 0x11222010,
    0x11222111, 0x11222212, 0x12202012, 0x12202110, 0x12202212, 0x12212111, 0x12222011, 0x12222110,
    0x12222111, 0x12222211, 0x10212021, 0x10212122, 0x10212220, 0x11202021, 0x11202120, 0x11202221,
    0x11212020, 0x11212121, 0x11212220, 0x11212222, 0x11222120, 0x11222121, 0x11222221, 0x12202122,
    0x12212120, 0x12212220, 0x12212222, 0x12222122, 0x20000000, 0x20000002, 0x20000200, 0x20000202,
    0x20020000, 0x20020002, 0x20020200, 0x20020202, 0x21000101, 0x21010000, 0x21010001, 0x21010100,
    0x21010102, 0x21010201, 0x21020101, 0x22000000, 0x22000002, 0x22000200, 0x22000202, 0x22010101,
    0x22020000, 0x22020002, 0x22020200, 0x22020202, 0x20000111, 0x20010011, 0x20010110, 0x20010112,
    0x20010211, 0x20020111, 0x21000011, 0x21000110, 0x21000211, 0x21010010, 0x21010012, 0x21010111,
    0x21010112, 0x21010210, 0x21010211, 0x21020110, 0x21020112, 0x21020211, 0x22000111, 0x22000211,
    0x22010110, 0x22010112, 0x22010211, 0x22020111, 0x20000020, 0x20000022, 0x20000220, 0x20000222,
    0x20010121, 0x20020020, 0x20020022, 0x20020220, 0x20020222, 0x21010021, 0x21010120, 0x21010221,
    0x21020121, 0x22000020, 0x22000022, 0x22000220, 0x22000222, 0x22010121, 0x22020020, 0x22020022,
    0x22020220, 0x22020222, 0x20011100, 0x20011201, 0x21001001, 0x21001100, 0x21011001, 0x21011101,
    0x21011202, 0x21021001, 0x21021100, 0x21021201, 0x22011100, 0x22011201, 0x20001011, 0x20001211,
    0x20011012, 0x20011111, 0x20011212, 0x20021112, 0x20021211, 0x21001010, 0x21001011, 0x21001111,
    0x21001210, 0x21011011, 0x21011110, 0x21011111, 0x21011112, 0x21011211, 0x21011212, 0x21021111,
    0x21021112, 0x21021210, 0x21021212, 0x22001011, 0x22001110, 0x22001112, 0x22001211, 0x22011010,
    0x22011012, 0x22011111, 0x22011210, 0x22021112, 0x20011021, 0x20011122, 0x20011221, 0x20021121,
    0x21001021, 0x21001120, 0x21001221, 0x21001222, 0x21011020, 0x21011121, 0x21011221, 0x21011222,
    0x21021021, 0x21021122, 0x21021222, 0x22001121, 0x22011021, 0x22011222, 0x22021120, 0x20002000,
    0x20002002, 0x20002200, 0x20002202, 0x20012101, 0x20022000, 0x20022002, 0x20022200, 0x20022202,
    0x21002001, 0x21002101, 0x21012001, 0x21012100, 0x21012201, 0x21022101, 0x21022201, 0x22002000,
    0x22002002, 0x22002200, 0x22002202, 0x22012101, 0x22022000, 0x22022002, 0x22022200, 0x22022202,
    0x20002111, 0x20002112, 0x20012011, 0x20012110, 0x20012112, 0x20022111, 0x21002011, 0x21002110,
    0x21002112, 0x21002211, 0x21012010, 0x21012012, 0x21012111, 0x21012212, 0x21022011, 0x21022110,
    0x22002111, 0x22012112, 0x22012211, 0x22022111, 0x20002020, 0x20002022, 0x20002220, 0x20002222,
    0x20012121, 0x20022020, 0x20022022, 0x20022220, 0x20022222, 0x21002121, 0x21012021, 0x21012120,
    0x21012122, 0x22002020, 0x22002022, 0x22002220, 0x22002222, 0x22012121, 0x22022020, 0x22022022,
    0x22022220, 0x22022222, 0x20100101, 0x20110001, 0x20110102, 0x20110200, 0x20110201, 0x20120101,
    0x21100001, 0x21100102, 0x21100201, 0x21110101, 0x21110200, 0x21110202, 0x21120201, 0x21120202,
    0x22100101, 0x22110001, 0x22110100, 0x22110102, 0x22110201, 0x22120101, 0x20100011, 0x20100110,
    0x20100112, 0x20100211, 0x20110010, 0x20110111, 0x20110210, 0x20110212, 0x20120011, 0x20120110,
    0x20120112, 0x20120211, 0x21100010, 0x21100111, 0x21110010, 0x21110011, 0x21110110, 0x21110111,
    0x21110112, 0x21110211, 0x21120012, 0x21120111, 0x22100110, 0x22100112, 0x22110012, 0x22110111,
    0x22110210, 0x22120011, 0x22120110, 0x22120112, 0x22120211, 0x20100121, 0x20110021, 0x20110120,
    0x20110221, 0x20120121, 0x21100120, 0x21100122, 0x21100221, 0x21110020, 0x21110022, 0x21110121,
    0x21110220, 0x21120122, 0x21120221, 0x22100121, 0x22110120, 0x22110122, 0x22120221, 0x20101001,
    0x20101100, 0x20101102, 0x20111000, 0x20111101, 0x20111200, 0x20121102, 0x21101000, 0x21101202,
    0x21111001, 0x21111100, 0x21111101, 0x21111102, 0x21111200, 0x21111201, 0x21121000, 0x21121001,
    0x21121002, 0x21121101, 0x22101100, 0x22101102, 0x22111002, 0x22111100, 0x22111101, 0x22111200,
    0x22121001, 0x22121201, 0x20101010, 0x20101111, 0x20101210, 0x20101212, 0x20111010, 0x20111011,
    0x20111110, 0x20111111, 0x20111112, 0x20111211, 0x20121011, 0x20121111, 0x20121211, 0x20121212,
    0x21101011, 0x21101110, 0x21101111, 0x21101112, 0x21101211, 0x21111010, 0x21111011, 0x21111012,
    0x21111110, 0x21111111, 0x21111112, 0x21111210, 0x21111211, 0x21111212, 0x21121011, 0x21121110,
    0x21121111, 0x21121112, 0x21121211, 0x22101011, 0x22101111, 0x22101210, 0x22111011, 0x22111012,
    0x22111110, 0x22111111, 0x22111112, 0x22111211, 0x22111212, 0x22121010, 0x22121012, 0x22121111,
    0x22121210, 0x22121212, 0x20101021, 0x20101120, 0x20111020, 0x20111121, 0x20111221, 0x20121020,
    0x20121122, 0x20121221, 0x21101121, 0x21101220, 0x21101221, 0x21111021, 0x21111022, 0x21111121,
    0x21111122, 0x21111221, 0x21121121, 0x21121220, 0x22101022, 0x22101120, 0x22101221, 0x22101222,
    0x22111022, 0x22111120, 0x22111121, 0x22121120, 0x22121122, 0x22121221, 0x20102101, 0x20112102,
    0x20112201, 0x20122101, 0x21102001, 0x21102102, 0x21112000, 0x21112002, 0x21112101, 0x21112102,
    0x21112202, 0x21122100, 0x21122101, 0x22102101, 0x22112001, 0x22112102, 0x22112201, 0x22122101,
    0x20102110, 0x20102112, 0x20102211, 0x20112010, 0x20112012, 0x20112111, 0x20112210, 0x20112212,
    0x20122010, 0x20122011, 0x20122110, 0x20122112, 0x21102010, 0x21102012, 0x21102111, 0x21102210,
    0x21102212, 0x21112011, 0x21112110, 0x21112111, 0x21112112, 0x21112211, 0x21122012, 0x21122111,
    0x21122112, 0x21122212, 0x22102011, 0x22102110, 0x22112010, 0x22112012, 0x22112111, 0x22112212,
    0x22122011, 0x22122112, 0x20102121, 0x20112121, 0x20122121, 0x21102120, 0x21102122, 0x21102221,
    0x21112020, 0x21112121, 0x21112220, 0x21122021, 0x22102121, 0x22112021, 0x22112120, 0x22112121,
    0x22112122, 0x20200000, 0x20200002, 0x20200200, 0x20200202, 0x20210101, 0x20220000, 0x20220002,
    0x20220200, 0x20220202, 0x21200101, 0x21210001, 0x21210100, 0x21210102, 0x21210201, 0x22200000,
    0x22200002, 0x22200200, 0x22200202, 0x22210101, 0x22220000, 0x22220002, 0x22220200, 0x22220202,
    0x20200111, 0x20200211, 0x20210011, 0x20210110, 0x20210112, 0x20210211, 0x20210212, 0x21200112,
    0x21200211, 0x21210011, 0x21210111, 0x21210210, 0x21210212, 0x21220011, 0x21220110, 0x22200111,
    0x22210010, 0x22210012, 0x22210112, 0x22210211, 0x20200022, 0x20200220, 0x20200222, 0x20210020,
    0x20210221, 0x20220022, 0x20220220, 0x20220222, 0x21200121, 0x21210021, 0x21210122, 0x21210221,
    0x21220121, 0x22200020, 0x22200022, 0x22200220, 0x22200222, 0x22210121, 0x22220020, 0x22220022,
    0x22220220, 0x22220222, 0x20211201, 0x20221101, 0x21201001, 0x21201100, 0x21211000, 0x21211100,
    0x21211101, 0x21211200, 0x21211202, 0x21221001, 0x21221101, 0x21221102, 0x21221200, 0x21221201,
    0x22201101, 0x20201112, 0x20201211, 0x20211010, 0x20211012, 0x20211111, 0x20211210, 0x20221112,
    0x20221211, 0x21201012, 0x21201111, 0x21211011, 0x21211110, 0x21211111, 0x21211112, 0x21211211,
    0x21221111, 0x21221212, 0x22201011, 0x22201110, 0x22201111, 0x22201112, 0x22201211, 0x22211012,
    0x22211111, 0x22211210, 0x20201121, 0x20211021, 0x20211122, 0x20211222, 0x20221021, 0x20221121,
    0x21201120, 0x21201122, 0x21201222, 0x21211022, 0x21211121, 0x21211122, 0x21211220, 0x21221020,
    0x21221022, 0x22201122, 0x22211020, 0x22211121, 0x22211122, 0x22211221, 0x22221021, 0x22221120,
    0x22221122, 0x20202000, 0x20202002, 0x20202200, 0x20202202, 0x20222000, 0x20222002, 0x20222200,
    0x20222202, 0x21212001, 0x21212100, 0x21212102, 0x21212201, 0x22202000, 0x22202002, 0x22202200,
    0x22202202, 0x22212101, 0x22222000, 0x22222002, 0x22222200, 0x22222202, 0x20202111, 0x20212110,
    0x20212211, 0x20222011, 0x20222111, 0x21202011, 0x21212010, 0x21212111, 0x21212212, 0x21222011,
    0x21222112, 0x21222211, 0x22212010, 0x22212112, 0x20202020, 0x20202022, 0x20202220, 0x20202222,
    0x20222020, 0x20222022, 0x20222220, 0x20222222, 0x21212021, 0x21212120, 0x21212122, 0x22202020,
    0x22202022, 0x22202220, 0x22202222, 0x22212121, 0x22222020, 0x22222022, 0x22222220, 0x22222222,
};

inline float gguf_dot_f32_row(device const float* x_row, device const uchar* w_row, uint K, uint lane) {
    device const float* wp = reinterpret_cast<device const float*>(w_row);
    float acc = 0.0f;
    for (uint k = lane; k < K; k += 32u) acc += x_row[k] * wp[k];
    return simd_sum(acc);
}

inline float gguf_dot_bf16_row(device const float* x_row, device const uchar* w_row, uint K, uint lane) {
    device const ushort* wp = reinterpret_cast<device const ushort*>(w_row);
    float acc = 0.0f;
    for (uint k = lane; k < K; k += 32u) {
        float w = as_type<float>(uint(wp[k]) << 16);
        acc += x_row[k] * w;
    }
    return simd_sum(acc);
}

inline float gguf_dot_q5_k_row(device const float* x_row, device const uchar* w_row, uint K, uint lane) {
    uint nb = K >> 8;
    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        device const uchar* blk = w_row + b * 176u;
        float d = gguf_load_f16(blk);
        float dmin = gguf_load_f16(blk + 2u);
        device const uchar* scales = blk + 4u;
        device const uchar* qh = blk + 16u;
        device const uchar* qs = blk + 48u;
        device const float* xp = x_row + (b << 8);
        uchar h = qh[lane];
        uchar u1 = 1u;
        uchar u2 = 2u;
        for (uint sb = 0; sb < 4u; ++sb) {
            uchar sc0 = 0, m0 = 0, sc1 = 0, m1 = 0;
            gguf_get_scale_min_k4(sb * 2u, scales, sc0, m0);
            gguf_get_scale_min_k4(sb * 2u + 1u, scales, sc1, m1);
            float d1 = d * float(sc0);
            float min1 = dmin * float(m0);
            float d2 = d * float(sc1);
            float min2 = dmin * float(m1);
            uchar q = qs[sb * 32u + lane];
            float wlo = d1 * float((q & 0x0Fu) + ((h & u1) ? 16u : 0u)) - min1;
            float whi = d2 * float((q >> 4) + ((h & u2) ? 16u : 0u)) - min2;
            acc += xp[sb * 64u + lane] * wlo + xp[sb * 64u + 32u + lane] * whi;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    return simd_sum(acc);
}

inline float gguf_dot_iq2_xxs_row(device const float* x_row, device const uchar* w_row, uint K, uint lane) {
    uint nb32 = K >> 5;
    float acc = 0.0f;
    for (uint ib32 = lane; ib32 < nb32; ib32 += 32u) {
        uint ibl = ib32 >> 3;
        uint ib = ib32 & 7u;
        device const uchar* blk = w_row + ulong(ibl) * 66ul;
        float d = gguf_load_f16(blk);
        device const uchar* qp = blk + 2u + ib * 8u;
        uint aux_g = uint(qp[0]) | (uint(qp[1]) << 8) | (uint(qp[2]) << 16) | (uint(qp[3]) << 24);
        uint aux_s = uint(qp[4]) | (uint(qp[5]) << 8) | (uint(qp[6]) << 16) | (uint(qp[7]) << 24);
        float db = d * (0.5f + float(aux_s >> 28)) * 0.25f;
        device const float4* xp4 = reinterpret_cast<device const float4*>(x_row + (ib32 << 5));
        float s = 0.0f;
        for (uint l = 0; l < 4u; ++l) {
            uint g_idx = (aux_g >> (l * 8u)) & 0xFFu;
            ulong grid64 = kGgufIq2xxsGrid[g_idx];
            uchar signs = kGgufSignsIq2xs[(aux_s >> (7u * l)) & 127u];
            float4 v0 = xp4[l * 2u];
            float4 v1 = xp4[l * 2u + 1u];
            float4 g0 = float4(
                float((grid64 >> 0u) & 0xFFul) * ((signs & 1u) ? -1.0f : 1.0f),
                float((grid64 >> 8u) & 0xFFul) * ((signs & 2u) ? -1.0f : 1.0f),
                float((grid64 >> 16u) & 0xFFul) * ((signs & 4u) ? -1.0f : 1.0f),
                float((grid64 >> 24u) & 0xFFul) * ((signs & 8u) ? -1.0f : 1.0f));
            float4 g1 = float4(
                float((grid64 >> 32u) & 0xFFul) * ((signs & 16u) ? -1.0f : 1.0f),
                float((grid64 >> 40u) & 0xFFul) * ((signs & 32u) ? -1.0f : 1.0f),
                float((grid64 >> 48u) & 0xFFul) * ((signs & 64u) ? -1.0f : 1.0f),
                float((grid64 >> 56u) & 0xFFul) * ((signs & 128u) ? -1.0f : 1.0f));
            s += dot(v0, g0) + dot(v1, g1);
        }
        acc = fma(db, s, acc);
    }
    return simd_sum(acc);
}

inline float gguf_dot_iq1_s_row(device const float* x_row, device const uchar* w_row, uint K, uint lane) {
    uint nb32 = K >> 5;
    float acc = 0.0f;
    for (uint ib32 = lane; ib32 < nb32; ib32 += 32u) {
        uint ibl = ib32 >> 3;
        uint ib = ib32 & 7u;
        device const uchar* blk = w_row + ulong(ibl) * 50ul;
        float d = gguf_load_f16(blk);
        ushort qh = ushort(blk[34u + ib * 2u]) | (ushort(blk[35u + ib * 2u]) << 8);
        float dl = d * float(2u * ((uint(qh) >> 12) & 7u) + 1u);
        float delta = (qh & 0x8000u) ? (-1.0f - 0.125f) : (-1.0f + 0.125f);
        device const uchar* qs = blk + 2u + ib * 4u;
        device const float4* xp4 = reinterpret_cast<device const float4*>(x_row + (ib32 << 5));
        float s_g = 0.0f;
        float s_x = 0.0f;
        for (uint l = 0; l < 4u; ++l) {
            uint idx = uint(qs[l]) | (((uint(qh) >> (3u * l)) & 7u) << 8);
            uint packed = kGgufIq1sGridGpu[idx];
            float4 v0 = xp4[l * 2u];
            float4 v1 = xp4[l * 2u + 1u];
            uint b0 = packed & 0xFFu;
            uint b1 = (packed >> 8) & 0xFFu;
            uint b2 = (packed >> 16) & 0xFFu;
            uint b3 = (packed >> 24) & 0xFFu;
            float4 lo = float4(float(b0 & 0x0Fu), float(b1 & 0x0Fu), float(b2 & 0x0Fu), float(b3 & 0x0Fu));
            float4 hi = float4(float(b0 >> 4), float(b1 >> 4), float(b2 >> 4), float(b3 >> 4));
            s_g += dot(v0, lo) + dot(v1, hi);
            s_x += (v0.x + v0.y + v0.z + v0.w) + (v1.x + v1.y + v1.z + v1.w);
        }
        acc = fma(dl, fma(delta, s_x, s_g), acc);
    }
    return simd_sum(acc);
}

kernel void gguf_gemv_bf16(
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
    ulong row_bytes = ulong(p.K) * 2ul;
    float sum = gguf_dot_bf16_row(X + ulong(m) * p.K, W + ulong(n) * row_bytes, p.K, lane);
    if (lane == 0u) Y[ulong(m) * p.N + n] = sum;
}

kernel void gguf_gemv_q5_k(
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
    ulong row_bytes = ulong(p.K >> 8) * 176ul;
    float sum = gguf_dot_q5_k_row(X + ulong(m) * p.K, W + ulong(n) * row_bytes, p.K, lane);
    if (lane == 0u) Y[ulong(m) * p.N + n] = sum;
}

kernel void gguf_gemv_iq2_xxs(
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
    ulong row_bytes = ulong(p.K >> 8) * 66ul;
    float sum = gguf_dot_iq2_xxs_row(X + ulong(m) * p.K, W + ulong(n) * row_bytes, p.K, lane);
    if (lane == 0u) Y[ulong(m) * p.N + n] = sum;
}

kernel void gguf_gemv_iq1_s(
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
    ulong row_bytes = ulong(p.K >> 8) * 50ul;
    float sum = gguf_dot_iq1_s_row(X + ulong(m) * p.K, W + ulong(n) * row_bytes, p.K, lane);
    if (lane == 0u) Y[ulong(m) * p.N + n] = sum;
}

kernel void gguf_moe_indexed_f32(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const int* topk_ids [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant GgufMoeParams& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint tr = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (tr >= p.T * p.top_k || n >= p.N) return;
    uint t = tr / p.top_k;
    int e_id = topk_ids[tr];
    float sum = 0.0f;
    if (e_id >= 0) {
        ulong row_bytes = ulong(p.K) * 4ul;
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        sum = gguf_dot_f32_row(X + ulong(t) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(tr) * p.N + n] = sum;
}

kernel void gguf_moe_down_accum_f32(
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
    ulong row_bytes = ulong(p.K) * 4ul;
    float total = 0.0f;
    for (uint r = 0; r < p.top_k; ++r) {
        uint tr = t * p.top_k + r;
        int e_id = topk_ids[tr];
        if (e_id < 0) continue;
        float gw = topk_w[tr];
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        total += gw * gguf_dot_f32_row(X + ulong(tr) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(t) * p.N + n] += total;
}

kernel void gguf_moe_indexed_bf16(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const int* topk_ids [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant GgufMoeParams& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint tr = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (tr >= p.T * p.top_k || n >= p.N) return;
    uint t = tr / p.top_k;
    int e_id = topk_ids[tr];
    float sum = 0.0f;
    if (e_id >= 0) {
        ulong row_bytes = ulong(p.K) * 2ul;
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        sum = gguf_dot_bf16_row(X + ulong(t) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(tr) * p.N + n] = sum;
}

kernel void gguf_moe_down_accum_bf16(
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
    ulong row_bytes = ulong(p.K) * 2ul;
    float total = 0.0f;
    for (uint r = 0; r < p.top_k; ++r) {
        uint tr = t * p.top_k + r;
        int e_id = topk_ids[tr];
        if (e_id < 0) continue;
        float gw = topk_w[tr];
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        total += gw * gguf_dot_bf16_row(X + ulong(tr) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(t) * p.N + n] += total;
}

kernel void gguf_moe_indexed_q8_0(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const int* topk_ids [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant GgufMoeParams& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint tr = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (tr >= p.T * p.top_k || n >= p.N) return;
    uint t = tr / p.top_k;
    int e_id = topk_ids[tr];
    float sum = 0.0f;
    if (e_id >= 0) {
        ulong row_bytes = ulong(p.K >> 5) * 34ul;
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        sum = gguf_dot_q8_0_row(X + ulong(t) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(tr) * p.N + n] = sum;
}

kernel void gguf_moe_down_accum_q8_0(
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
    float total = 0.0f;
    for (uint r = 0; r < p.top_k; ++r) {
        uint tr = t * p.top_k + r;
        int e_id = topk_ids[tr];
        if (e_id < 0) continue;
        float gw = topk_w[tr];
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        total += gw * gguf_dot_q8_0_row(X + ulong(tr) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(t) * p.N + n] += total;
}

kernel void gguf_moe_indexed_iq4_nl(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const int* topk_ids [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant GgufMoeParams& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint tr = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (tr >= p.T * p.top_k || n >= p.N) return;
    uint t = tr / p.top_k;
    int e_id = topk_ids[tr];
    float sum = 0.0f;
    if (e_id >= 0) {
        ulong row_bytes = ulong(p.K >> 5) * 18ul;
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        sum = gguf_dot_iq4_nl_row(X + ulong(t) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(tr) * p.N + n] = sum;
}

kernel void gguf_moe_down_accum_iq4_nl(
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
    ulong row_bytes = ulong(p.K >> 5) * 18ul;
    float total = 0.0f;
    for (uint r = 0; r < p.top_k; ++r) {
        uint tr = t * p.top_k + r;
        int e_id = topk_ids[tr];
        if (e_id < 0) continue;
        float gw = topk_w[tr];
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        total += gw * gguf_dot_iq4_nl_row(X + ulong(tr) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(t) * p.N + n] += total;
}

kernel void gguf_moe_indexed_mxfp4(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const int* topk_ids [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant GgufMoeParams& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint tr = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (tr >= p.T * p.top_k || n >= p.N) return;
    uint t = tr / p.top_k;
    int e_id = topk_ids[tr];
    float sum = 0.0f;
    if (e_id >= 0) {
        ulong row_bytes = ulong(p.K >> 5) * 17ul;
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        sum = gguf_dot_mxfp4_row(X + ulong(t) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(tr) * p.N + n] = sum;
}

kernel void gguf_moe_down_accum_mxfp4(
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
    float total = 0.0f;
    for (uint r = 0; r < p.top_k; ++r) {
        uint tr = t * p.top_k + r;
        int e_id = topk_ids[tr];
        if (e_id < 0) continue;
        float gw = topk_w[tr];
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        total += gw * gguf_dot_mxfp4_row(X + ulong(tr) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(t) * p.N + n] += total;
}

kernel void gguf_moe_indexed_q4_k(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const int* topk_ids [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant GgufMoeParams& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint tr = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (tr >= p.T * p.top_k || n >= p.N) return;
    uint t = tr / p.top_k;
    int e_id = topk_ids[tr];
    float sum = 0.0f;
    if (e_id >= 0) {
        ulong row_bytes = ulong(p.K >> 8) * 144ul;
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        sum = gguf_dot_q4_k_row(X + ulong(t) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(tr) * p.N + n] = sum;
}

kernel void gguf_moe_down_accum_q4_k(
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
    float total = 0.0f;
    for (uint r = 0; r < p.top_k; ++r) {
        uint tr = t * p.top_k + r;
        int e_id = topk_ids[tr];
        if (e_id < 0) continue;
        float gw = topk_w[tr];
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        total += gw * gguf_dot_q4_k_row(X + ulong(tr) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(t) * p.N + n] += total;
}

kernel void gguf_moe_indexed_q5_k(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const int* topk_ids [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant GgufMoeParams& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint tr = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (tr >= p.T * p.top_k || n >= p.N) return;
    uint t = tr / p.top_k;
    int e_id = topk_ids[tr];
    float sum = 0.0f;
    if (e_id >= 0) {
        ulong row_bytes = ulong(p.K >> 8) * 176ul;
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        sum = gguf_dot_q5_k_row(X + ulong(t) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(tr) * p.N + n] = sum;
}

kernel void gguf_moe_down_accum_q5_k(
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
    ulong row_bytes = ulong(p.K >> 8) * 176ul;
    float total = 0.0f;
    for (uint r = 0; r < p.top_k; ++r) {
        uint tr = t * p.top_k + r;
        int e_id = topk_ids[tr];
        if (e_id < 0) continue;
        float gw = topk_w[tr];
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        total += gw * gguf_dot_q5_k_row(X + ulong(tr) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(t) * p.N + n] += total;
}

kernel void gguf_moe_indexed_q6_k(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const int* topk_ids [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant GgufMoeParams& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint tr = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (tr >= p.T * p.top_k || n >= p.N) return;
    uint t = tr / p.top_k;
    int e_id = topk_ids[tr];
    float sum = 0.0f;
    if (e_id >= 0) {
        ulong row_bytes = ulong(p.K >> 8) * 210ul;
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        sum = gguf_dot_q6_k_row(X + ulong(t) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(tr) * p.N + n] = sum;
}

kernel void gguf_moe_down_accum_q6_k(
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
    ulong row_bytes = ulong(p.K >> 8) * 210ul;
    float total = 0.0f;
    for (uint r = 0; r < p.top_k; ++r) {
        uint tr = t * p.top_k + r;
        int e_id = topk_ids[tr];
        if (e_id < 0) continue;
        float gw = topk_w[tr];
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        total += gw * gguf_dot_q6_k_row(X + ulong(tr) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(t) * p.N + n] += total;
}

kernel void gguf_moe_indexed_q2_k(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const int* topk_ids [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant GgufMoeParams& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint tr = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (tr >= p.T * p.top_k || n >= p.N) return;
    uint t = tr / p.top_k;
    int e_id = topk_ids[tr];
    float sum = 0.0f;
    if (e_id >= 0) {
        ulong row_bytes = ulong(p.K >> 8) * 84ul;
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        sum = gguf_dot_q2_k_row(X + ulong(t) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(tr) * p.N + n] = sum;
}

kernel void gguf_moe_down_accum_q2_k(
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
    float total = 0.0f;
    for (uint r = 0; r < p.top_k; ++r) {
        uint tr = t * p.top_k + r;
        int e_id = topk_ids[tr];
        if (e_id < 0) continue;
        float gw = topk_w[tr];
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        total += gw * gguf_dot_q2_k_row(X + ulong(tr) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(t) * p.N + n] += total;
}

kernel void gguf_moe_indexed_iq2_xxs(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const int* topk_ids [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant GgufMoeParams& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint tr = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (tr >= p.T * p.top_k || n >= p.N) return;
    uint t = tr / p.top_k;
    int e_id = topk_ids[tr];
    float sum = 0.0f;
    if (e_id >= 0) {
        ulong row_bytes = ulong(p.K >> 8) * 66ul;
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        sum = gguf_dot_iq2_xxs_row(X + ulong(t) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(tr) * p.N + n] = sum;
}

kernel void gguf_moe_down_accum_iq2_xxs(
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
    ulong row_bytes = ulong(p.K >> 8) * 66ul;
    float total = 0.0f;
    for (uint r = 0; r < p.top_k; ++r) {
        uint tr = t * p.top_k + r;
        int e_id = topk_ids[tr];
        if (e_id < 0) continue;
        float gw = topk_w[tr];
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        total += gw * gguf_dot_iq2_xxs_row(X + ulong(tr) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(t) * p.N + n] += total;
}

kernel void gguf_moe_indexed_iq1_s(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const int* topk_ids [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant GgufMoeParams& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint tr = tgid.y;
    uint n = tgid.x * 4u + (tid >> 5);
    uint lane = tid & 31u;
    if (tr >= p.T * p.top_k || n >= p.N) return;
    uint t = tr / p.top_k;
    int e_id = topk_ids[tr];
    float sum = 0.0f;
    if (e_id >= 0) {
        ulong row_bytes = ulong(p.K >> 8) * 50ul;
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        sum = gguf_dot_iq1_s_row(X + ulong(t) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(tr) * p.N + n] = sum;
}

kernel void gguf_moe_down_accum_iq1_s(
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
    ulong row_bytes = ulong(p.K >> 8) * 50ul;
    float total = 0.0f;
    for (uint r = 0; r < p.top_k; ++r) {
        uint tr = t * p.top_k + r;
        int e_id = topk_ids[tr];
        if (e_id < 0) continue;
        float gw = topk_w[tr];
        device const uchar* w_row = W + ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n) * row_bytes;
        total += gw * gguf_dot_iq1_s_row(X + ulong(tr) * p.K, w_row, p.K, lane);
    }
    if (lane == 0u) Y[ulong(t) * p.N + n] += total;
}

inline float gguf_iq1s_sub32(device const uchar* blk, uint ib, float4 v0, float4 v1, float4 v2, float4 v3, float4 v4, float4 v5, float4 v6, float4 v7, float sumy) {
    float d = gguf_load_f16(blk);
    ushort qh = ushort(blk[34u + ib * 2u]) | (ushort(blk[35u + ib * 2u]) << 8);
    float dl = d * float(2u * ((uint(qh) >> 12) & 7u) + 1u);
    float delta = (qh & 0x8000u) ? (-1.0f - 0.125f) : (-1.0f + 0.125f);
    device const uchar* qs = blk + 2u + ib * 4u;
    float s = 0.0f;
    #define IQ1S_G(L, VA, VB) do { \
        uint idx = uint(qs[(L)]) | (((uint(qh) >> (3u * (L))) & 7u) << 8); \
        uint pk = kGgufIq1sGridGpu[idx]; \
        uint b0 = pk & 0xFFu, b1 = (pk >> 8) & 0xFFu, b2 = (pk >> 16) & 0xFFu, b3 = (pk >> 24) & 0xFFu; \
        s += dot((VA), float4(float(b0 & 0x0Fu), float(b1 & 0x0Fu), float(b2 & 0x0Fu), float(b3 & 0x0Fu))) \
           + dot((VB), float4(float(b0 >> 4), float(b1 >> 4), float(b2 >> 4), float(b3 >> 4))); \
    } while (0)
    IQ1S_G(0u, v0, v1);
    IQ1S_G(1u, v2, v3);
    IQ1S_G(2u, v4, v5);
    IQ1S_G(3u, v6, v7);
    #undef IQ1S_G
    return dl * fma(delta, sumy, s);
}

inline float gguf_iq2xxs_sub32(device const uchar* blk, uint ib, float4 v0, float4 v1, float4 v2, float4 v3, float4 v4, float4 v5, float4 v6, float4 v7) {
    float d = gguf_load_f16(blk);
    device const uchar* qp = blk + 2u + ib * 8u;
    uint aux_g = uint(qp[0]) | (uint(qp[1]) << 8) | (uint(qp[2]) << 16) | (uint(qp[3]) << 24);
    uint aux_s = uint(qp[4]) | (uint(qp[5]) << 8) | (uint(qp[6]) << 16) | (uint(qp[7]) << 24);
    float db = d * (0.5f + float(aux_s >> 28)) * 0.25f;
    float s = 0.0f;
    #define IQ2XXS_G(L, VA, VB) do { \
        ulong g64 = kGgufIq2xxsGrid[(aux_g >> ((L) * 8u)) & 0xFFu]; \
        uchar sg = kGgufSignsIq2xs[(aux_s >> (7u * (L))) & 127u]; \
        float4 g0 = float4( \
            float((g64 >> 0u) & 0xFFul) * ((sg & 1u) ? -1.0f : 1.0f), \
            float((g64 >> 8u) & 0xFFul) * ((sg & 2u) ? -1.0f : 1.0f), \
            float((g64 >> 16u) & 0xFFul) * ((sg & 4u) ? -1.0f : 1.0f), \
            float((g64 >> 24u) & 0xFFul) * ((sg & 8u) ? -1.0f : 1.0f)); \
        float4 g1 = float4( \
            float((g64 >> 32u) & 0xFFul) * ((sg & 16u) ? -1.0f : 1.0f), \
            float((g64 >> 40u) & 0xFFul) * ((sg & 32u) ? -1.0f : 1.0f), \
            float((g64 >> 48u) & 0xFFul) * ((sg & 64u) ? -1.0f : 1.0f), \
            float((g64 >> 56u) & 0xFFul) * ((sg & 128u) ? -1.0f : 1.0f)); \
        s += dot((VA), g0) + dot((VB), g1); \
    } while (0)
    IQ2XXS_G(0u, v0, v1);
    IQ2XXS_G(1u, v2, v3);
    IQ2XXS_G(2u, v4, v5);
    IQ2XXS_G(3u, v6, v7);
    #undef IQ2XXS_G
    return db * s;
}

inline float gguf_iq1s_sub32_h(device const uchar* blk, uint ib, half4 v0, half4 v1, half4 v2, half4 v3, half4 v4, half4 v5, half4 v6, half4 v7, float sumy) {
    float d = gguf_load_f16(blk);
    uchar2 qhb = uchar2(*reinterpret_cast<device const packed_uchar2*>(blk + 34u + ib * 2u));
    ushort qh = ushort(qhb.x) | (ushort(qhb.y) << 8);
    float dl = d * float(2u * ((uint(qh) >> 12) & 7u) + 1u);
    float delta = (qh & 0x8000u) ? (-1.0f - 0.125f) : (-1.0f + 0.125f);
    uchar4 qs4 = uchar4(*reinterpret_cast<device const packed_uchar4*>(blk + 2u + ib * 4u));
    half s = half(0.0h);
    #define IQ1S_GH(L, QBYTE, VA, VB) do { \
        uint idx = uint(QBYTE) | (((uint(qh) >> (3u * (L))) & 7u) << 8); \
        uint pk = kGgufIq1sGridGpu[idx]; \
        uint b0 = pk & 0xFFu, b1 = (pk >> 8) & 0xFFu, b2 = (pk >> 16) & 0xFFu, b3 = (pk >> 24) & 0xFFu; \
        s += dot((VA), half4(half(b0 & 0x0Fu), half(b1 & 0x0Fu), half(b2 & 0x0Fu), half(b3 & 0x0Fu))) \
           + dot((VB), half4(half(b0 >> 4), half(b1 >> 4), half(b2 >> 4), half(b3 >> 4))); \
    } while (0)
    IQ1S_GH(0u, qs4.x, v0, v1);
    IQ1S_GH(1u, qs4.y, v2, v3);
    IQ1S_GH(2u, qs4.z, v4, v5);
    IQ1S_GH(3u, qs4.w, v6, v7);
    #undef IQ1S_GH
    return dl * fma(delta, sumy, float(s));
}

kernel void gguf_moe_gate_up_swiglu_iq1_s(
    device const float* X [[buffer(0)]],
    device const uchar* Wg [[buffer(1)]],
    device const uchar* Wu [[buffer(2)]],
    device const int* topk_ids [[buffer(3)]],
    device float* Y [[buffer(4)]],
    constant GgufMoeSwigluParams& p [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint tr = tgid.y;
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint n0 = tgid.x * 4u + sg * 2u;
    if (tr >= p.T * p.top_k || n0 >= p.N) return;
    bool has_r1 = (n0 + 1u < p.N);
    int e_id = topk_ids[tr];
    if (e_id < 0) {
        if (lane == 0u) {
            Y[ulong(tr) * p.N + n0] = 0.0f;
            if (has_r1) Y[ulong(tr) * p.N + n0 + 1u] = 0.0f;
        }
        return;
    }
    uint t = tr / p.top_k;
    uint nb32 = p.K >> 5;
    ulong row_bytes = ulong(p.K >> 8) * 50ul;
    ulong e_off = ulong(uint(e_id)) * ulong(p.expert_stride_bytes);
    device const float* x_row = X + ulong(t) * p.K;
    float ga0 = 0.0f, ga1 = 0.0f;
    float ua0 = 0.0f, ua1 = 0.0f;
    for (uint ib32 = lane; ib32 < nb32; ib32 += 32u) {
        device const float4* xp4 = reinterpret_cast<device const float4*>(x_row + (ib32 << 5));
        float4 f0 = xp4[0], f1 = xp4[1], f2 = xp4[2], f3 = xp4[3];
        float4 f4 = xp4[4], f5 = xp4[5], f6 = xp4[6], f7 = xp4[7];
        float4 sv = (f0 + f1) + (f2 + f3) + (f4 + f5) + (f6 + f7);
        float sumy = sv.x + sv.y + sv.z + sv.w;
        half4 v0 = half4(f0), v1 = half4(f1), v2 = half4(f2), v3 = half4(f3);
        half4 v4 = half4(f4), v5 = half4(f5), v6 = half4(f6), v7 = half4(f7);
        uint ibl = ib32 >> 3;
        uint ib = ib32 & 7u;
        ulong boff = e_off + ulong(n0) * row_bytes + ulong(ibl) * 50ul;
        ga0 += gguf_iq1s_sub32_h(Wg + boff, ib, v0, v1, v2, v3, v4, v5, v6, v7, sumy);
        ua0 += gguf_iq1s_sub32_h(Wu + boff, ib, v0, v1, v2, v3, v4, v5, v6, v7, sumy);
        if (has_r1) {
            ga1 += gguf_iq1s_sub32_h(Wg + boff + row_bytes, ib, v0, v1, v2, v3, v4, v5, v6, v7, sumy);
            ua1 += gguf_iq1s_sub32_h(Wu + boff + row_bytes, ib, v0, v1, v2, v3, v4, v5, v6, v7, sumy);
        }
    }
    float gv0 = simd_sum(ga0), uv0 = simd_sum(ua0);
    float gv1 = simd_sum(ga1), uv1 = simd_sum(ua1);
    if (lane == 0u) {
        if (p.clamp_val > 0.0f) {
            gv0 = clamp(gv0, -p.clamp_val, p.clamp_val);
            uv0 = clamp(uv0, -p.clamp_val, p.clamp_val);
            gv1 = clamp(gv1, -p.clamp_val, p.clamp_val);
            uv1 = clamp(uv1, -p.clamp_val, p.clamp_val);
        }
        Y[ulong(tr) * p.N + n0] = (gv0 / (1.0f + exp(-gv0))) * uv0;
        if (has_r1) Y[ulong(tr) * p.N + n0 + 1u] = (gv1 / (1.0f + exp(-gv1))) * uv1;
    }
}

kernel void gguf_moe_down_accum_iq1_s_4r(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const int* topk_ids [[buffer(2)]],
    device const float* topk_w [[buffer(3)]],
    device float* Y [[buffer(4)]],
    constant GgufMoeParams& p [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint t = tgid.y;
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint n0 = tgid.x * 8u + sg * 4u;
    if (t >= p.T || n0 >= p.N) return;
    uint nb32 = p.K >> 5;
    ulong row_bytes = ulong(p.K >> 8) * 50ul;
    uint cnt = min(4u, p.N - n0);
    float tot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint r_exp = 0; r_exp < p.top_k; ++r_exp) {
        uint tr = t * p.top_k + r_exp;
        int e_id = topk_ids[tr];
        if (e_id < 0) continue;
        float gw = topk_w[tr];
        ulong e_off = ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n0) * row_bytes;
        device const float* x_row = X + ulong(tr) * p.K;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint ib32 = lane; ib32 < nb32; ib32 += 32u) {
            device const float4* xp4 = reinterpret_cast<device const float4*>(x_row + (ib32 << 5));
            float4 f0 = xp4[0], f1 = xp4[1], f2 = xp4[2], f3 = xp4[3];
            float4 f4 = xp4[4], f5 = xp4[5], f6 = xp4[6], f7 = xp4[7];
            float4 sv = (f0 + f1) + (f2 + f3) + (f4 + f5) + (f6 + f7);
            float sumy = sv.x + sv.y + sv.z + sv.w;
            half4 v0 = half4(f0), v1 = half4(f1), v2 = half4(f2), v3 = half4(f3);
            half4 v4 = half4(f4), v5 = half4(f5), v6 = half4(f6), v7 = half4(f7);
            uint ibl = ib32 >> 3;
            uint ib = ib32 & 7u;
            ulong boff = e_off + ulong(ibl) * 50ul;
            for (uint r = 0; r < cnt; ++r) {
                acc[r] += gguf_iq1s_sub32_h(W + boff + ulong(r) * row_bytes, ib, v0, v1, v2, v3, v4, v5, v6, v7, sumy);
            }
        }
        for (uint r = 0; r < cnt; ++r) tot[r] = fma(gw, acc[r], tot[r]);
    }
    for (uint r = 0; r < cnt; ++r) {
        float s = simd_sum(tot[r]);
        if (lane == 0u) Y[ulong(t) * p.N + n0 + r] += s;
    }
}

kernel void gguf_moe_gate_up_swiglu_iq2_xxs(
    device const float* X [[buffer(0)]],
    device const uchar* Wg [[buffer(1)]],
    device const uchar* Wu [[buffer(2)]],
    device const int* topk_ids [[buffer(3)]],
    device float* Y [[buffer(4)]],
    constant GgufMoeSwigluParams& p [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint tr = tgid.y;
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint n0 = tgid.x * 8u + sg * 4u;
    if (tr >= p.T * p.top_k || n0 >= p.N) return;
    int e_id = topk_ids[tr];
    if (e_id < 0) {
        if (lane == 0u) {
            for (uint r = 0; r < 4u && n0 + r < p.N; ++r) Y[ulong(tr) * p.N + n0 + r] = 0.0f;
        }
        return;
    }
    uint t = tr / p.top_k;
    uint nb32 = p.K >> 5;
    ulong row_bytes = ulong(p.K >> 8) * 66ul;
    ulong e_off = ulong(uint(e_id)) * ulong(p.expert_stride_bytes);
    device const float* x_row = X + ulong(t) * p.K;
    uint cnt = min(4u, p.N - n0);
    float ga[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float ua[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint ib32 = lane; ib32 < nb32; ib32 += 32u) {
        device const float4* xp4 = reinterpret_cast<device const float4*>(x_row + (ib32 << 5));
        float4 v0 = xp4[0], v1 = xp4[1], v2 = xp4[2], v3 = xp4[3];
        float4 v4 = xp4[4], v5 = xp4[5], v6 = xp4[6], v7 = xp4[7];
        uint ibl = ib32 >> 3;
        uint ib = ib32 & 7u;
        ulong boff = e_off + ulong(n0) * row_bytes + ulong(ibl) * 66ul;
        for (uint r = 0; r < cnt; ++r) {
            ga[r] += gguf_iq2xxs_sub32(Wg + boff + ulong(r) * row_bytes, ib, v0, v1, v2, v3, v4, v5, v6, v7);
            ua[r] += gguf_iq2xxs_sub32(Wu + boff + ulong(r) * row_bytes, ib, v0, v1, v2, v3, v4, v5, v6, v7);
        }
    }
    for (uint r = 0; r < cnt; ++r) {
        float gv = simd_sum(ga[r]);
        float uv = simd_sum(ua[r]);
        if (lane == 0u) {
            if (p.clamp_val > 0.0f) {
                gv = clamp(gv, -p.clamp_val, p.clamp_val);
                uv = clamp(uv, -p.clamp_val, p.clamp_val);
            }
            Y[ulong(tr) * p.N + n0 + r] = (gv / (1.0f + exp(-gv))) * uv;
        }
    }
}

kernel void gguf_moe_down_accum_iq2_xxs_4r(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const int* topk_ids [[buffer(2)]],
    device const float* topk_w [[buffer(3)]],
    device float* Y [[buffer(4)]],
    constant GgufMoeParams& p [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint t = tgid.y;
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint n0 = tgid.x * 8u + sg * 4u;
    if (t >= p.T || n0 >= p.N) return;
    uint nb32 = p.K >> 5;
    ulong row_bytes = ulong(p.K >> 8) * 66ul;
    uint cnt = min(4u, p.N - n0);
    float tot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint r_exp = 0; r_exp < p.top_k; ++r_exp) {
        uint tr = t * p.top_k + r_exp;
        int e_id = topk_ids[tr];
        if (e_id < 0) continue;
        float gw = topk_w[tr];
        ulong e_off = ulong(uint(e_id)) * ulong(p.expert_stride_bytes) + ulong(n0) * row_bytes;
        device const float* x_row = X + ulong(tr) * p.K;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint ib32 = lane; ib32 < nb32; ib32 += 32u) {
            device const float4* xp4 = reinterpret_cast<device const float4*>(x_row + (ib32 << 5));
            float4 v0 = xp4[0], v1 = xp4[1], v2 = xp4[2], v3 = xp4[3];
            float4 v4 = xp4[4], v5 = xp4[5], v6 = xp4[6], v7 = xp4[7];
            uint ibl = ib32 >> 3;
            uint ib = ib32 & 7u;
            ulong boff = e_off + ulong(ibl) * 66ul;
            for (uint r = 0; r < cnt; ++r) {
                acc[r] += gguf_iq2xxs_sub32(W + boff + ulong(r) * row_bytes, ib, v0, v1, v2, v3, v4, v5, v6, v7);
            }
        }
        for (uint r = 0; r < cnt; ++r) tot[r] = fma(gw, acc[r], tot[r]);
    }
    for (uint r = 0; r < cnt; ++r) {
        float s = simd_sum(tot[r]);
        if (lane == 0u) Y[ulong(t) * p.N + n0 + r] += s;
    }
}

kernel void gguf_gemv_q8_0_fast(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant GgufGemvParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint m = tgid.y;
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint n0 = tgid.x * 4u + sg * 2u;
    if (m >= p.M || n0 >= p.N) return;
    uint nb = p.K >> 5;
    ulong row_bytes = ulong(nb) * 34ul;
    bool has_r1 = (n0 + 1u < p.N);
    device const float* x_row = X + ulong(m) * p.K;
    device const uchar* w0 = W + ulong(n0) * row_bytes;
    device const uchar* w1 = has_r1 ? (w0 + row_bytes) : w0;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        device const float4* xp4 = reinterpret_cast<device const float4*>(x_row + (b << 5));
        float4 v0 = xp4[0], v1 = xp4[1], v2 = xp4[2], v3 = xp4[3];
        float4 v4 = xp4[4], v5 = xp4[5], v6 = xp4[6], v7 = xp4[7];
        ulong boff = ulong(b) * 34ul;
        {
            device const uchar* blk = w0 + boff;
            float d = gguf_load_f16(blk);
            device const char* q = reinterpret_cast<device const char*>(blk + 2u);
            float s = dot(v0, float4(q[0], q[1], q[2], q[3]))
                    + dot(v1, float4(q[4], q[5], q[6], q[7]))
                    + dot(v2, float4(q[8], q[9], q[10], q[11]))
                    + dot(v3, float4(q[12], q[13], q[14], q[15]))
                    + dot(v4, float4(q[16], q[17], q[18], q[19]))
                    + dot(v5, float4(q[20], q[21], q[22], q[23]))
                    + dot(v6, float4(q[24], q[25], q[26], q[27]))
                    + dot(v7, float4(q[28], q[29], q[30], q[31]));
            acc0 = fma(d, s, acc0);
        }
        if (has_r1) {
            device const uchar* blk = w1 + boff;
            float d = gguf_load_f16(blk);
            device const char* q = reinterpret_cast<device const char*>(blk + 2u);
            float s = dot(v0, float4(q[0], q[1], q[2], q[3]))
                    + dot(v1, float4(q[4], q[5], q[6], q[7]))
                    + dot(v2, float4(q[8], q[9], q[10], q[11]))
                    + dot(v3, float4(q[12], q[13], q[14], q[15]))
                    + dot(v4, float4(q[16], q[17], q[18], q[19]))
                    + dot(v5, float4(q[20], q[21], q[22], q[23]))
                    + dot(v6, float4(q[24], q[25], q[26], q[27]))
                    + dot(v7, float4(q[28], q[29], q[30], q[31]));
            acc1 = fma(d, s, acc1);
        }
    }
    float out0 = simd_sum(acc0);
    float out1 = simd_sum(acc1);
    if (lane == 0u) {
        ulong yb = ulong(m) * p.N;
        Y[yb + n0] = out0;
        if (has_r1) Y[yb + n0 + 1u] = out1;
    }
}

kernel void gguf_gemv_q5_k_fast(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant GgufGemvParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint m = tgid.y;
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint n0 = tgid.x * 4u + sg * 2u;
    if (m >= p.M || n0 >= p.N) return;
    uint nb = p.K >> 8;
    ulong row_bytes = ulong(nb) * 176ul;
    bool has_r1 = (n0 + 1u < p.N);
    device const float* x_row = X + ulong(m) * p.K;
    device const uchar* w0 = W + ulong(n0) * row_bytes;
    device const uchar* w1 = has_r1 ? (w0 + row_bytes) : w0;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        device const float* xp = x_row + (b << 8);
        ulong boff = ulong(b) * 176ul;
        device const uchar* blk0 = w0 + boff;
        float d0 = gguf_load_f16(blk0);
        float dm0 = gguf_load_f16(blk0 + 2u);
        device const uchar* sc0_p = blk0 + 4u;
        uchar h0 = blk0[16u + lane];
        device const uchar* qs0 = blk0 + 48u;

        device const uchar* blk1 = w1 + boff;
        float d1_b = has_r1 ? gguf_load_f16(blk1) : 0.0f;
        float dm1_b = has_r1 ? gguf_load_f16(blk1 + 2u) : 0.0f;
        device const uchar* sc1_p = blk1 + 4u;
        uchar h1 = has_r1 ? blk1[16u + lane] : 0u;
        device const uchar* qs1 = blk1 + 48u;

        uchar u1 = 1u;
        uchar u2 = 2u;
        for (uint sb = 0; sb < 4u; ++sb) {
            float xl = xp[sb * 64u + lane];
            float xh = xp[sb * 64u + 32u + lane];
            {
                uchar s0 = 0, m0 = 0, s1 = 0, m1 = 0;
                gguf_get_scale_min_k4(sb * 2u, sc0_p, s0, m0);
                gguf_get_scale_min_k4(sb * 2u + 1u, sc0_p, s1, m1);
                uchar q = qs0[sb * 32u + lane];
                float wlo = d0 * float(s0) * float((q & 0x0Fu) + ((h0 & u1) ? 16u : 0u)) - dm0 * float(m0);
                float whi = d0 * float(s1) * float((q >> 4) + ((h0 & u2) ? 16u : 0u)) - dm0 * float(m1);
                acc0 = fma(xl, wlo, fma(xh, whi, acc0));
            }
            if (has_r1) {
                uchar s0 = 0, m0 = 0, s1 = 0, m1 = 0;
                gguf_get_scale_min_k4(sb * 2u, sc1_p, s0, m0);
                gguf_get_scale_min_k4(sb * 2u + 1u, sc1_p, s1, m1);
                uchar q = qs1[sb * 32u + lane];
                float wlo = d1_b * float(s0) * float((q & 0x0Fu) + ((h1 & u1) ? 16u : 0u)) - dm1_b * float(m0);
                float whi = d1_b * float(s1) * float((q >> 4) + ((h1 & u2) ? 16u : 0u)) - dm1_b * float(m1);
                acc1 = fma(xl, wlo, fma(xh, whi, acc1));
            }
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    float out0 = simd_sum(acc0);
    float out1 = simd_sum(acc1);
    if (lane == 0u) {
        ulong yb = ulong(m) * p.N;
        Y[yb + n0] = out0;
        if (has_r1) Y[yb + n0 + 1u] = out1;
    }
}

kernel void gguf_gemv_bf16_fast(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant GgufGemvParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint m = tgid.y;
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint n0 = tgid.x * 4u + sg * 2u;
    if (m >= p.M || n0 >= p.N) return;
    uint k4 = p.K >> 2;
    ulong row_bytes = ulong(p.K) * 2ul;
    bool has_r1 = (n0 + 1u < p.N);
    device const float4* xp4 = reinterpret_cast<device const float4*>(X + ulong(m) * p.K);
    device const ushort4* w0 = reinterpret_cast<device const ushort4*>(W + ulong(n0) * row_bytes);
    device const ushort4* w1 = has_r1 ? reinterpret_cast<device const ushort4*>(W + ulong(n0 + 1u) * row_bytes) : w0;
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    for (uint k = lane; k < k4; k += 32u) {
        float4 xv = xp4[k];
        ushort4 q0 = w0[k];
        float4 f0 = float4(as_type<float>(uint(q0.x) << 16), as_type<float>(uint(q0.y) << 16),
                           as_type<float>(uint(q0.z) << 16), as_type<float>(uint(q0.w) << 16));
        acc0 = fma(xv, f0, acc0);
        if (has_r1) {
            ushort4 q1 = w1[k];
            float4 f1 = float4(as_type<float>(uint(q1.x) << 16), as_type<float>(uint(q1.y) << 16),
                               as_type<float>(uint(q1.z) << 16), as_type<float>(uint(q1.w) << 16));
            acc1 = fma(xv, f1, acc1);
        }
    }
    float out0 = simd_sum(acc0.x + acc0.y + acc0.z + acc0.w);
    float out1 = simd_sum(acc1.x + acc1.y + acc1.z + acc1.w);
    if (lane == 0u) {
        ulong yb = ulong(m) * p.N;
        Y[yb + n0] = out0;
        if (has_r1) Y[yb + n0 + 1u] = out1;
    }
}

kernel void gguf_gemm_q6_k_tiled(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant GgufGemvParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint m0 = tgid.y * 32u;
    uint n0 = tgid.x * 32u;
    if (m0 >= p.M || n0 >= p.N) return;
    ulong row_bytes = ulong(p.K >> 8) * 210ul;

    threadgroup half smX[32 * 32];
    threadgroup half smW[32 * 32];
    threadgroup float smOut[32 * 32];

    uint sg_r = (sg_id >> 1) * 16u;
    uint sg_c = (sg_id & 1u) * 16u;

    simdgroup_matrix<float, 8, 8> acc[2][2];
    for (uint i = 0; i < 2u; ++i)
        for (uint j = 0; j < 2u; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < p.K; k0 += 32u) {
        uint b_idx = k0 >> 8;
        uint e_base = k0 & 255u;
        uint half_idx = e_base >> 7;
        uint quad = (e_base & 127u) >> 5;
        uint ql_quad = (half_idx << 6) + ((quad & 1u) << 5);
        uint qh_half = 128u + (half_idx << 5);
        uint sc_quad = 192u + (half_idx << 3) + (quad << 1);
        uint qh_shift = quad << 1;

        for (uint idx = tid; idx < 1024u; idx += 128u) {
            uint r = idx >> 5;
            uint l = idx & 31u;
            uint gm = m0 + r;
            uint gn = n0 + r;
            uint gk = k0 + l;
            smX[idx] = (gm < p.M && gk < p.K) ? half(X[ulong(gm) * p.K + gk]) : half(0.0h);
            half wv = half(0.0h);
            if (gn < p.N && gk < p.K) {
                device const uchar* blk = W + ulong(gn) * row_bytes + ulong(b_idx) * 210ul;
                uchar ql_b = blk[ql_quad + l];
                uchar ql_v = (quad < 2u) ? (ql_b & 0x0Fu) : (ql_b >> 4);
                uchar qh_v = (blk[qh_half + l] >> qh_shift) & 3u;
                int q = int(ql_v | (qh_v << 4)) - 32;
                char sc = reinterpret_cast<device const char*>(blk)[sc_quad + (l >> 4)];
                wv = half(gguf_load_f16(blk + 208u) * float(sc) * float(q));
            }
            smW[l * 32u + r] = wv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_matrix<half, 8, 8> ma0, ma1, mb0, mb1;
            simdgroup_load(ma0, smX + (sg_r + 0u) * 32u + kk, 32u);
            simdgroup_load(ma1, smX + (sg_r + 8u) * 32u + kk, 32u);
            simdgroup_load(mb0, smW + kk * 32u + (sg_c + 0u), 32u);
            simdgroup_load(mb1, smW + kk * 32u + (sg_c + 8u), 32u);
            simdgroup_multiply_accumulate(acc[0][0], ma0, mb0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], ma0, mb1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], ma1, mb0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], ma1, mb1, acc[1][1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < 2u; ++i)
        for (uint j = 0; j < 2u; ++j)
            simdgroup_store(acc[i][j], smOut + (sg_r + i * 8u) * 32u + (sg_c + j * 8u), 32u);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < 1024u; idx += 128u) {
        uint r = idx >> 5;
        uint c = idx & 31u;
        uint gm = m0 + r;
        uint gn = n0 + c;
        if (gm < p.M && gn < p.N) Y[ulong(gm) * p.N + gn] = smOut[idx];
    }
}

kernel void gguf_gemm_q5_k_tiled(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant GgufGemvParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint m0 = tgid.y * 32u;
    uint n0 = tgid.x * 32u;
    if (m0 >= p.M || n0 >= p.N) return;
    ulong row_bytes = ulong(p.K >> 8) * 176ul;

    threadgroup half smX[32 * 32];
    threadgroup half smW[32 * 32];
    threadgroup float smOut[32 * 32];

    uint sg_r = (sg_id >> 1) * 16u;
    uint sg_c = (sg_id & 1u) * 16u;

    simdgroup_matrix<float, 8, 8> acc[2][2];
    for (uint i = 0; i < 2u; ++i)
        for (uint j = 0; j < 2u; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < p.K; k0 += 32u) {
        uint b_idx = k0 >> 8;
        uint e_base = k0 & 255u;
        uint sb = e_base >> 6;
        uint hi = (e_base >> 5) & 1u;
        uint j_sc = (sb << 1) + hi;

        for (uint idx = tid; idx < 1024u; idx += 128u) {
            uint r = idx >> 5;
            uint l = idx & 31u;
            uint gm = m0 + r;
            uint gn = n0 + r;
            uint gk = k0 + l;
            smX[idx] = (gm < p.M && gk < p.K) ? half(X[ulong(gm) * p.K + gk]) : half(0.0h);
            half wv = half(0.0h);
            if (gn < p.N && gk < p.K) {
                device const uchar* blk = W + ulong(gn) * row_bytes + ulong(b_idx) * 176ul;
                float d = gguf_load_f16(blk);
                float dmin = gguf_load_f16(blk + 2u);
                uchar sc = 0, mn = 0;
                gguf_get_scale_min_k4(j_sc, blk + 4u, sc, mn);
                uchar qb = blk[48u + (sb << 5) + l];
                uint q4 = hi ? uint(qb >> 4) : uint(qb & 0x0Fu);
                uint hbit = (uint(blk[16u + l]) >> j_sc) & 1u;
                wv = half(d * float(sc) * float(q4 + (hbit << 4)) - dmin * float(mn));
            }
            smW[l * 32u + r] = wv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_matrix<half, 8, 8> ma0, ma1, mb0, mb1;
            simdgroup_load(ma0, smX + (sg_r + 0u) * 32u + kk, 32u);
            simdgroup_load(ma1, smX + (sg_r + 8u) * 32u + kk, 32u);
            simdgroup_load(mb0, smW + kk * 32u + (sg_c + 0u), 32u);
            simdgroup_load(mb1, smW + kk * 32u + (sg_c + 8u), 32u);
            simdgroup_multiply_accumulate(acc[0][0], ma0, mb0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], ma0, mb1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], ma1, mb0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], ma1, mb1, acc[1][1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < 2u; ++i)
        for (uint j = 0; j < 2u; ++j)
            simdgroup_store(acc[i][j], smOut + (sg_r + i * 8u) * 32u + (sg_c + j * 8u), 32u);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < 1024u; idx += 128u) {
        uint r = idx >> 5;
        uint c = idx & 31u;
        uint gm = m0 + r;
        uint gn = n0 + c;
        if (gm < p.M && gn < p.N) Y[ulong(gm) * p.N + gn] = smOut[idx];
    }
}

kernel void gguf_gemm_q8_0_tiled(
    device const float* X [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant GgufGemvParams& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint m0 = tgid.y * 32u;
    uint n0 = tgid.x * 32u;
    if (m0 >= p.M || n0 >= p.N) return;
    ulong row_bytes = ulong(p.K >> 5) * 34ul;

    threadgroup half smX[32 * 32];
    threadgroup half smW[32 * 32];
    threadgroup float smOut[32 * 32];

    uint sg_r = (sg_id >> 1) * 16u;
    uint sg_c = (sg_id & 1u) * 16u;

    simdgroup_matrix<float, 8, 8> acc[2][2];
    for (uint i = 0; i < 2u; ++i)
        for (uint j = 0; j < 2u; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < p.K; k0 += 32u) {
        uint b_idx = k0 >> 5;
        for (uint idx = tid; idx < 1024u; idx += 128u) {
            uint r = idx >> 5;
            uint l = idx & 31u;
            uint gm = m0 + r;
            uint gn = n0 + r;
            uint gk = k0 + l;
            smX[idx] = (gm < p.M && gk < p.K) ? half(X[ulong(gm) * p.K + gk]) : half(0.0h);
            half wv = half(0.0h);
            if (gn < p.N && gk < p.K) {
                device const uchar* blk = W + ulong(gn) * row_bytes + ulong(b_idx) * 34ul;
                char q = reinterpret_cast<device const char*>(blk + 2u)[l];
                wv = half(gguf_load_f16(blk) * float(q));
            }
            smW[l * 32u + r] = wv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_matrix<half, 8, 8> ma0, ma1, mb0, mb1;
            simdgroup_load(ma0, smX + (sg_r + 0u) * 32u + kk, 32u);
            simdgroup_load(ma1, smX + (sg_r + 8u) * 32u + kk, 32u);
            simdgroup_load(mb0, smW + kk * 32u + (sg_c + 0u), 32u);
            simdgroup_load(mb1, smW + kk * 32u + (sg_c + 8u), 32u);
            simdgroup_multiply_accumulate(acc[0][0], ma0, mb0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], ma0, mb1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], ma1, mb0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], ma1, mb1, acc[1][1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < 2u; ++i)
        for (uint j = 0; j < 2u; ++j)
            simdgroup_store(acc[i][j], smOut + (sg_r + i * 8u) * 32u + (sg_c + j * 8u), 32u);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < 1024u; idx += 128u) {
        uint r = idx >> 5;
        uint c = idx & 31u;
        uint gm = m0 + r;
        uint gn = n0 + c;
        if (gm < p.M && gn < p.N) Y[ulong(gm) * p.N + gn] = smOut[idx];
    }
}
