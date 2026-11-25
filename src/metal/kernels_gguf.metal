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
    device const float* qp = Q + (ulong(t) * p.n_heads + h) * p.head_dim;
    device float* op = O + (ulong(t) * p.n_heads + h) * p.head_dim;

    float m_max = -1e30f;
    for (uint s = 0; s < seq_end; ++s) {
        device const float* kp = K_cache + (ulong(s) * p.n_kv_heads + kvh) * p.head_dim;
        float dot = 0.0f;
        for (uint d = tid; d < p.head_dim; d += 32u) dot += qp[d] * kp[d];
        float sc = simd_sum(dot) * p.scale;
        m_max = max(m_max, sc);
    }

    float l_sum = 0.0f;
    for (uint d = tid; d < p.head_dim; d += 32u) op[d] = 0.0f;
    for (uint s = 0; s < seq_end; ++s) {
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
