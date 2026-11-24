#include <metal_stdlib>
using namespace metal;

constant float kE2M1Lut[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

inline float safe_tanh(float x) {
    return tanh(clamp(x, -20.0f, 20.0f));
}

constant float kE4M3Lut[256] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, 0.0f,
    -0.0f, -0.001953125f, -0.00390625f, -0.005859375f, -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f, -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f, -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f, -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    -0.125f, -0.140625f, -0.15625f, -0.171875f, -0.1875f, -0.203125f, -0.21875f, -0.234375f,
    -0.25f, -0.28125f, -0.3125f, -0.34375f, -0.375f, -0.40625f, -0.4375f, -0.46875f,
    -0.5f, -0.5625f, -0.625f, -0.6875f, -0.75f, -0.8125f, -0.875f, -0.9375f,
    -1.0f, -1.125f, -1.25f, -1.375f, -1.5f, -1.625f, -1.75f, -1.875f,
    -2.0f, -2.25f, -2.5f, -2.75f, -3.0f, -3.25f, -3.5f, -3.75f,
    -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -6.5f, -7.0f, -7.5f,
    -8.0f, -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f,
    -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -28.0f, -30.0f,
    -32.0f, -36.0f, -40.0f, -44.0f, -48.0f, -52.0f, -56.0f, -60.0f,
    -64.0f, -72.0f, -80.0f, -88.0f, -96.0f, -104.0f, -112.0f, -120.0f,
    -128.0f, -144.0f, -160.0f, -176.0f, -192.0f, -208.0f, -224.0f, -240.0f,
    -256.0f, -288.0f, -320.0f, -352.0f, -384.0f, -416.0f, -448.0f, 0.0f
};

inline float e4m3_to_f32(uchar b) {
    return kE4M3Lut[b];
}

inline uchar f32_to_e4m3(float v) {
    uint sign = (v < 0.0f) ? 0x80u : 0u;
    float av = fabs(v);
    if (!(av > 0.0f)) return uchar(sign);
    if (av >= 448.0f) return uchar(sign | 0x7Eu);
    if (av < 0.015625f) {
        uint m = uint(rint(av * 512.0f));
        if (m >= 8u) return uchar(sign | 0x08u);
        return uchar(sign | m);
    }
    uint bits = as_type<uint>(av);
    int fexp = int((bits >> 23) & 0xFFu) - 127;
    uint fmant = bits & 0x7FFFFFu;
    int e4 = fexp + 7;
    uint m3 = (fmant + (1u << 19)) >> 20;
    if (m3 >= 8u) {
        m3 = 0u;
        e4 += 1;
    }
    if (e4 >= 15 && m3 >= 7u) return uchar(sign | 0x7Eu);
    if (e4 <= 0) return uchar(sign);
    return uchar(sign | (uint(e4) << 3) | (m3 & 0x7u));
}

inline float tg_reduce_max(float v, uint tid, uint tg_size, threadgroup float* sm) {
    uint lane = tid & 31u;
    uint wid = tid >> 5;
    v = simd_max(v);
    if (lane == 0u) sm[wid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nw = (tg_size + 31u) >> 5;
    float acc = (tid < nw) ? sm[lane] : 0.0f;
    if (wid == 0u) acc = simd_max(acc);
    if (tid == 0u) sm[0] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float res = sm[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return res;
}

inline float ue8m0_to_f32(uchar e) {
    if (e == 0u) return 5.877471754111438e-39f;
    uint bits = uint(e) << 23;
    return as_type<float>(bits);
}

inline float tg_reduce_sum(float v, uint tid, uint tg_size, threadgroup float* sm) {
    uint lane = tid & 31u;
    uint wid = tid >> 5;
    v = simd_sum(v);
    if (lane == 0u) sm[wid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nw = (tg_size + 31u) >> 5;
    float acc = (tid < nw) ? sm[lane] : 0.0f;
    if (wid == 0u) acc = simd_sum(acc);
    if (tid == 0u) sm[0] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float res = sm[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return res;
}

struct RmsParams {
    uint T;
    uint D;
    float eps;
};

kernel void embed_norm_kernel(
    device const int* token_ids   [[buffer(0)]],
    device const bfloat* embed_w  [[buffer(1)]],
    device const bfloat* norm_w   [[buffer(2)]],
    device bfloat* out            [[buffer(3)]],
    constant RmsParams& p         [[buffer(4)]],
    uint tgid                     [[threadgroup_position_in_grid]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]])
{
    if (tgid >= p.T) return;
    long tok = long(token_ids[tgid]);
    device const bfloat* row = embed_w + tok * long(p.D);
    device bfloat* dst = out + long(tgid) * long(p.D);

    threadgroup float sm[32];
    float sum_sq = 0.0f;
    for (uint d = tid; d < p.D; d += tg_size) {
        float v = float(row[d]);
        sum_sq += v * v;
    }
    sum_sq = tg_reduce_sum(sum_sq, tid, tg_size, sm);
    float rstd = rsqrt(sum_sq / float(p.D) + p.eps);
    for (uint d = tid; d < p.D; d += tg_size) {
        dst[d] = bfloat(float(row[d]) * rstd * float(norm_w[d]));
    }
}

kernel void rmsnorm_kernel(
    device const bfloat* x  [[buffer(0)]],
    device const bfloat* w  [[buffer(1)]],
    device bfloat* out      [[buffer(2)]],
    constant RmsParams& p   [[buffer(3)]],
    uint tgid               [[threadgroup_position_in_grid]],
    uint tid                [[thread_index_in_threadgroup]],
    uint tg_size            [[threads_per_threadgroup]])
{
    if (tgid >= p.T) return;
    device const bfloat* xr = x + long(tgid) * long(p.D);
    device bfloat* yr = out + long(tgid) * long(p.D);

    threadgroup float sm[32];
    float sum_sq = 0.0f;
    for (uint d = tid; d < p.D; d += tg_size) {
        float v = float(xr[d]);
        sum_sq += v * v;
    }
    sum_sq = tg_reduce_sum(sum_sq, tid, tg_size, sm);
    float rstd = rsqrt(sum_sq / float(p.D) + p.eps);
    for (uint d = tid; d < p.D; d += tg_size) {
        yr[d] = bfloat(float(xr[d]) * rstd * float(w[d]));
    }
}

kernel void add_bf16_kernel(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device bfloat* out     [[buffer(2)]],
    constant uint& N       [[buffer(3)]],
    uint gid               [[thread_position_in_grid]])
{
    if (gid < N) {
        out[gid] = bfloat(float(a[gid]) + float(b[gid]));
    }
}

kernel void copy_bf16_kernel(
    device const bfloat* src [[buffer(0)]],
    device bfloat* dst       [[buffer(1)]],
    constant uint& N         [[buffer(2)]],
    uint gid                 [[thread_position_in_grid]])
{
    if (gid < N) {
        dst[gid] = src[gid];
    }
}

struct SplitQrgsParams {
    uint T;
    uint q_dim;
    uint r_dim;
    uint n_heads;
};

kernel void split_qrgs_kernel(
    device const bfloat* qrgs [[buffer(0)]],
    device bfloat* Q          [[buffer(1)]],
    device bfloat* R          [[buffer(2)]],
    device bfloat* gate       [[buffer(3)]],
    device bfloat* sinks      [[buffer(4)]],
    constant SplitQrgsParams& p [[buffer(5)]],
    uint2 gid                 [[thread_position_in_grid]])
{
    uint stride = 2u * p.q_dim + p.r_dim + p.n_heads;
    uint c = gid.x;
    uint t = gid.y;
    if (t >= p.T || c >= stride) return;
    bfloat v = qrgs[long(t) * long(stride) + long(c)];
    if (c < p.q_dim) {
        Q[long(t) * long(p.q_dim) + long(c)] = v;
    } else if (c < p.q_dim + p.r_dim) {
        R[long(t) * long(p.r_dim) + long(c - p.q_dim)] = v;
    } else if (c < 2u * p.q_dim + p.r_dim) {
        gate[long(t) * long(p.q_dim) + long(c - p.q_dim - p.r_dim)] = v;
    } else {
        sinks[long(t) * long(p.n_heads) + long(c - 2u * p.q_dim - p.r_dim)] = v;
    }
}

struct SplitKvParams {
    uint T;
    uint kv_dim;
};

kernel void split_kv_kernel(
    device const bfloat* kv [[buffer(0)]],
    device bfloat* K        [[buffer(1)]],
    device bfloat* V        [[buffer(2)]],
    constant SplitKvParams& p [[buffer(3)]],
    uint2 gid               [[thread_position_in_grid]])
{
    uint c = gid.x;
    uint t = gid.y;
    if (t >= p.T || c >= 2u * p.kv_dim) return;
    bfloat v = kv[long(t) * long(2u * p.kv_dim) + long(c)];
    if (c < p.kv_dim) {
        K[long(t) * long(p.kv_dim) + long(c)] = v;
    } else {
        V[long(t) * long(p.kv_dim) + long(c - p.kv_dim)] = v;
    }
}

struct AttnResParams {
    uint num_stack;
    uint stack_stride;
    uint has_addend;
    uint has_norm_w;
    uint T;
    uint D;
    float eps;
};

kernel void attn_res_kernel(
    device const bfloat* stack   [[buffer(0)]],
    device const bfloat* last    [[buffer(1)]],
    device const bfloat* addend  [[buffer(2)]],
    device const float* queries  [[buffer(3)]],
    device const bfloat* norm_w  [[buffer(4)]],
    device bfloat* out_bf16      [[buffer(5)]],
    device float* out_fp32       [[buffer(6)]],
    constant AttnResParams& p    [[buffer(7)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint2 tg_size2               [[threads_per_threadgroup]])
{
    uint tg_size = tg_size2.x;
    uint q_idx = tgid.x;
    uint t = tgid.y;
    if (t >= p.T) return;

    device const float* q_ptr = queries + long(q_idx) * long(p.D);
    long row_off = long(t) * long(p.D);
    uint N = p.num_stack + 1u;

    threadgroup float sm[32];
    threadgroup float s_logits[16];
    threadgroup float s_inv_rms[16];
    threadgroup float s_weights[16];

    for (uint n = 0; n < N; ++n) {
        float sum_sq = 0.0f;
        float dot = 0.0f;
        for (uint d = tid; d < p.D; d += tg_size) {
            float x;
            if (n < p.num_stack) {
                x = float(stack[long(n) * long(p.stack_stride) + row_off + long(d)]);
            } else {
                x = float(last[row_off + long(d)]);
                if (p.has_addend != 0u) x += float(addend[row_off + long(d)]);
            }
            sum_sq += x * x;
            dot += q_ptr[d] * x;
        }
        sum_sq = tg_reduce_sum(sum_sq, tid, tg_size, sm);
        dot = tg_reduce_sum(dot, tid, tg_size, sm);
        if (tid == 0u) {
            float rstd = rsqrt(sum_sq / float(p.D) + 1e-6f);
            s_inv_rms[n] = rstd;
            s_logits[n] = dot * rstd;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float max_l = s_logits[0];
        for (uint n = 1; n < N; ++n) max_l = max(max_l, s_logits[n]);
        float sum_e = 0.0f;
        for (uint n = 0; n < N; ++n) {
            float e = exp(s_logits[n] - max_l);
            s_weights[n] = e;
            sum_e += e;
        }
        float inv_sum = 1.0f / sum_e;
        for (uint n = 0; n < N; ++n) {
            s_weights[n] = s_weights[n] * inv_sum * s_inv_rms[n];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float out_sum_sq = 0.0f;
    float acc_reg[12];
    uint k = 0u;
    for (uint d = tid; d < p.D; d += tg_size) {
        float acc = 0.0f;
        for (uint n = 0; n < N; ++n) {
            float x;
            if (n < p.num_stack) {
                x = float(stack[long(n) * long(p.stack_stride) + row_off + long(d)]);
            } else {
                x = float(last[row_off + long(d)]);
                if (p.has_addend != 0u) x += float(addend[row_off + long(d)]);
            }
            acc = fma(s_weights[n], x, acc);
        }
        acc_reg[k++] = acc;
        out_sum_sq += acc * acc;
    }

    if (p.has_norm_w != 0u) {
        out_sum_sq = tg_reduce_sum(out_sum_sq, tid, tg_size, sm);
        float rstd = rsqrt(out_sum_sq / float(p.D) + p.eps);
        device const bfloat* nw = norm_w + long(q_idx) * long(p.D);
        device bfloat* dst = out_bf16 + (long(q_idx) * long(p.T) + long(t)) * long(p.D);
        k = 0u;
        for (uint d = tid; d < p.D; d += tg_size) {
            dst[d] = bfloat(acc_reg[k++] * rstd * float(nw[d]));
        }
    } else {
        device float* dst = out_fp32 + (long(q_idx) * long(p.T) + long(t)) * long(p.D);
        k = 0u;
        for (uint d = tid; d < p.D; d += tg_size) {
            dst[d] = acc_reg[k++];
        }
    }
}

kernel void pair_rms_norm_kernel(
    device const float* dest    [[buffer(0)]],
    device const float* message [[buffer(1)]],
    device bfloat* pair_out     [[buffer(2)]],
    constant RmsParams& p       [[buffer(3)]],
    uint tgid                   [[threadgroup_position_in_grid]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint tg_size                [[threads_per_threadgroup]])
{
    if (tgid >= p.T) return;
    device const float* dp = dest + long(tgid) * long(p.D);
    device const float* mp = message + long(tgid) * long(p.D);
    device bfloat* op = pair_out + long(tgid) * long(2u * p.D);

    threadgroup float sm[32];
    float sd = 0.0f, sm_v = 0.0f;
    for (uint d = tid; d < p.D; d += tg_size) {
        float vd = dp[d], vm = mp[d];
        sd += vd * vd;
        sm_v += vm * vm;
    }
    sd = tg_reduce_sum(sd, tid, tg_size, sm);
    sm_v = tg_reduce_sum(sm_v, tid, tg_size, sm);
    float inv_d = rsqrt(sd / float(p.D) + p.eps);
    float inv_m = rsqrt(sm_v / float(p.D) + p.eps);
    for (uint d = tid; d < p.D; d += tg_size) {
        op[d] = bfloat(dp[d] * inv_d);
        op[p.D + d] = bfloat(mp[d] * inv_m);
    }
}

struct CtrlParams {
    uint T;
    uint D;
    uint rank;
    float eps;
};

kernel void fused_ctrl_blend_rmsnorm_kernel(
    device const bfloat* ctrl_h   [[buffer(0)]],
    device const bfloat* gain_w   [[buffer(1)]],
    device const bfloat* forget_w [[buffer(2)]],
    device const float* dest      [[buffer(3)]],
    device const float* message   [[buffer(4)]],
    device const bfloat* norm_w   [[buffer(5)]],
    device bfloat* out            [[buffer(6)]],
    constant CtrlParams& p        [[buffer(7)]],
    uint tgid                     [[threadgroup_position_in_grid]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]])
{
    if (tgid >= p.T) return;
    threadgroup float s_h[64];
    threadgroup float sm[32];
    if (tid < p.rank) {
        s_h[tid] = float(ctrl_h[long(tgid) * long(p.rank) + long(tid)]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    long off = long(tgid) * long(p.D);
    float reg[12];
    float sum_sq = 0.0f;
    uint k = 0u;
    for (uint d = tid; d < p.D; d += tg_size) {
        device const bfloat* gw = gain_w + long(d) * long(p.rank);
        device const bfloat* fw = forget_w + long(d) * long(p.rank);
        float g = 0.0f, f = 0.0f;
        for (uint r = 0; r < p.rank; ++r) {
            float hr = s_h[r];
            g = fma(float(gw[r]), hr, g);
            f = fma(float(fw[r]), hr, f);
        }
        float v = (1.0f - safe_tanh(f)) * dest[off + long(d)] + safe_tanh(g) * message[off + long(d)];
        reg[k++] = v;
        sum_sq += v * v;
    }
    sum_sq = tg_reduce_sum(sum_sq, tid, tg_size, sm);
    float rstd = rsqrt(sum_sq / float(p.D) + p.eps);
    k = 0u;
    for (uint d = tid; d < p.D; d += tg_size) {
        out[off + long(d)] = bfloat(reg[k++] * rstd * float(norm_w[d]));
    }
}

kernel void ctrl_gf_blend_small_t_kernel(
    device const bfloat* ctrl_h   [[buffer(0)]],
    device const bfloat* gain_w   [[buffer(1)]],
    device const bfloat* forget_w [[buffer(2)]],
    device const float* dest      [[buffer(3)]],
    device const float* message   [[buffer(4)]],
    device float* out_fp32        [[buffer(5)]],
    constant CtrlParams& p        [[buffer(6)]],
    uint tgid                     [[threadgroup_position_in_grid]],
    uint sg_id                    [[simdgroup_index_in_threadgroup]],
    uint lane                     [[thread_index_in_simdgroup]],
    uint tid                      [[thread_index_in_threadgroup]])
{
    uint d = tgid * 4u + sg_id;
    threadgroup float s_h[8][64];
    uint total_h = p.T * 64u;
    for (uint idx = tid; idx < total_h; idx += 128u) {
        s_h[idx >> 6][idx & 63u] = float(ctrl_h[idx]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d >= p.D) return;

    device const bfloat2* gw2 = reinterpret_cast<device const bfloat2*>(gain_w + long(d) * 64);
    device const bfloat2* fw2 = reinterpret_cast<device const bfloat2*>(forget_w + long(d) * 64);
    float2 gf = float2(gw2[lane]);
    float2 ff = float2(fw2[lane]);
    uint r0 = lane * 2u;

    for (uint t = 0; t < p.T; ++t) {
        float h0 = s_h[t][r0];
        float h1 = s_h[t][r0 + 1u];
        float g = simd_sum(fma(gf.x, h0, gf.y * h1));
        float f = simd_sum(fma(ff.x, h0, ff.y * h1));
        if (lane == 0u) {
            long off = long(t) * long(p.D) + long(d);
            out_fp32[off] = (1.0f - safe_tanh(f)) * dest[off] + safe_tanh(g) * message[off];
        }
    }
}

kernel void fp32_rmsnorm_to_bf16_kernel(
    device const float* x   [[buffer(0)]],
    device const bfloat* w  [[buffer(1)]],
    device bfloat* out      [[buffer(2)]],
    constant RmsParams& p   [[buffer(3)]],
    uint tgid               [[threadgroup_position_in_grid]],
    uint tid                [[thread_index_in_threadgroup]],
    uint tg_size            [[threads_per_threadgroup]])
{
    if (tgid >= p.T) return;
    device const float* xr = x + long(tgid) * long(p.D);
    device bfloat* yr = out + long(tgid) * long(p.D);
    threadgroup float sm[32];
    float sum_sq = 0.0f;
    for (uint d = tid; d < p.D; d += tg_size) {
        float v = xr[d];
        sum_sq += v * v;
    }
    sum_sq = tg_reduce_sum(sum_sq, tid, tg_size, sm);
    float rstd = rsqrt(sum_sq / float(p.D) + p.eps);
    for (uint d = tid; d < p.D; d += tg_size) {
        yr[d] = bfloat(xr[d] * rstd * float(w[d]));
    }
}

struct SconvStaticParams {
    uint T;
    uint C;
    uint has_snap;
};

kernel void sconv_static_kernel(
    device const bfloat* x      [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device bfloat* state        [[buffer(2)]],
    device bfloat* out          [[buffer(3)]],
    device bfloat* snap         [[buffer(4)]],
    constant SconvStaticParams& p [[buffer(5)]],
    uint c                      [[thread_position_in_grid]])
{
    if (c >= p.C) return;
    float w0 = float(weight[long(c) * 4 + 0]);
    float w1 = float(weight[long(c) * 4 + 1]);
    float w2 = float(weight[long(c) * 4 + 2]);
    float w3 = float(weight[long(c) * 4 + 3]);
    const float inv_sqrt5 = 0.44721359549995793f;

    float s0 = float(state[0 * long(p.C) + long(c)]);
    float s1 = float(state[1 * long(p.C) + long(c)]);
    float s2 = float(state[2 * long(p.C) + long(c)]);

    for (uint t = 0; t < p.T; ++t) {
        float xt = float(x[long(t) * long(p.C) + long(c)]);
        float y = (w0 * s0 + w1 * s1 + w2 * s2 + w3 * xt + xt) * inv_sqrt5;
        out[long(t) * long(p.C) + long(c)] = bfloat(y);
        s0 = s1;
        s1 = s2;
        s2 = xt;
        if (p.has_snap != 0u) {
            device bfloat* sp = snap + long(t) * 3 * long(p.C);
            sp[0 * long(p.C) + long(c)] = bfloat(s0);
            sp[1 * long(p.C) + long(c)] = bfloat(s1);
            sp[2 * long(p.C) + long(c)] = bfloat(s2);
        }
    }
    state[0 * long(p.C) + long(c)] = bfloat(s0);
    state[1 * long(p.C) + long(c)] = bfloat(s1);
    state[2 * long(p.C) + long(c)] = bfloat(s2);
}

struct SconvDynamicParams {
    uint T;
    uint num_kv_heads;
    uint head_dim;
    uint has_snap;
};

kernel void sconv_dynamic_kernel(
    device const bfloat* x        [[buffer(0)]],
    device const bfloat* static_w [[buffer(1)]],
    device const bfloat* dyn_w    [[buffer(2)]],
    device bfloat* state          [[buffer(3)]],
    device bfloat* out            [[buffer(4)]],
    device bfloat* snap           [[buffer(5)]],
    constant SconvDynamicParams& p [[buffer(6)]],
    uint c                        [[thread_position_in_grid]])
{
    uint C = p.num_kv_heads * p.head_dim;
    if (c >= C) return;
    uint h = c / p.head_dim;

    float sw0 = float(static_w[0 * long(C) + long(c)]);
    float sw1 = float(static_w[1 * long(C) + long(c)]);
    float sw2 = float(static_w[2 * long(C) + long(c)]);
    float sw3 = float(static_w[3 * long(C) + long(c)]);
    const float inv_3 = 1.0f / 3.0f;

    float s0 = float(state[0 * long(C) + long(c)]);
    float s1 = float(state[1 * long(C) + long(c)]);
    float s2 = float(state[2 * long(C) + long(c)]);

    uint dyn_stride = p.num_kv_heads * 4u;
    for (uint t = 0; t < p.T; ++t) {
        float xt = float(x[long(t) * long(C) + long(c)]);
        device const bfloat* dw = dyn_w + long(t) * long(dyn_stride) + long(h) * 4;
        float tap0 = sw0 + float(dw[0]);
        float tap1 = sw1 + float(dw[1]);
        float tap2 = sw2 + float(dw[2]);
        float tap3 = sw3 + float(dw[3]);
        float y = (tap0 * xt + tap1 * s2 + tap2 * s1 + tap3 * s0 + xt) * inv_3;
        out[long(t) * long(C) + long(c)] = bfloat(y);
        s0 = s1;
        s1 = s2;
        s2 = xt;
        if (p.has_snap != 0u) {
            device bfloat* sp = snap + long(t) * 3 * long(C);
            sp[0 * long(C) + long(c)] = bfloat(s0);
            sp[1 * long(C) + long(c)] = bfloat(s1);
            sp[2 * long(C) + long(c)] = bfloat(s2);
        }
    }
    state[0 * long(C) + long(c)] = bfloat(s0);
    state[1 * long(C) + long(c)] = bfloat(s1);
    state[2 * long(C) + long(c)] = bfloat(s2);
}

struct AttnPrepQRParams {
    uint start_pos;
    uint T;
    uint is_local;
    uint n_heads;
    uint head_dim;
    uint d_rel;
    float qk_clip;
    float rel_clip;
    float log_n_floor;
    float log_alpha;
};

kernel void attn_prep_q_r_kernel(
    device bfloat* Q            [[buffer(0)]],
    device bfloat* R            [[buffer(1)]],
    device const bfloat* q_sc   [[buffer(2)]],
    device const bfloat* r_sc   [[buffer(3)]],
    constant AttnPrepQRParams& p [[buffer(4)]],
    uint2 tgid                  [[threadgroup_position_in_grid]],
    uint lane                   [[thread_index_in_simdgroup]])
{
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.n_heads || t >= p.T) return;

    uint pos = p.start_pos + t;
    float tau = 1.0f;
    if (p.is_local == 0u) {
        float ratio = max(1.0f, float(pos + 1u) / p.log_n_floor);
        tau = 1.0f + p.log_alpha * log(ratio);
    }

    device bfloat* q_ptr = Q + (long(t) * long(p.n_heads) + long(h)) * long(p.head_dim);
    device const bfloat* qs_ptr = q_sc + long(h) * long(p.head_dim);
    float qv[4];
    float sum_sq = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        uint d = lane + i * 32u;
        float v = float(q_ptr[d]) * float(qs_ptr[d]);
        qv[i] = v;
        sum_sq += v * v;
    }
    float norm = sqrt(simd_sum(sum_sq));
    float scale = min(1.0f, p.qk_clip / (norm + 1e-6f)) * tau;
    for (uint i = 0; i < 4; ++i) {
        uint d = lane + i * 32u;
        q_ptr[d] = bfloat(qv[i] * scale);
    }

    device bfloat* r_ptr = R + (long(t) * long(p.n_heads) + long(h)) * long(p.d_rel);
    device const bfloat* rs_ptr = r_sc + long(h) * long(p.d_rel);
    float rv = (lane < p.d_rel) ? (float(r_ptr[lane]) * float(rs_ptr[lane])) : 0.0f;
    float r_norm = sqrt(simd_sum(rv * rv));
    float r_s = min(1.0f, p.rel_clip / (r_norm + 1e-6f)) * tau;
    if (lane < p.d_rel) {
        r_ptr[lane] = bfloat(rv * r_s);
    }
}

struct AttnPrepKVParams {
    uint start_pos;
    uint T;
    uint cache_cap;
    uint is_local;
    uint n_kv_heads;
    uint head_dim;
    float qk_clip;
    float eps;
};

kernel void attn_prep_kv_kernel(
    device const bfloat* K_in [[buffer(0)]],
    device const bfloat* V_in [[buffer(1)]],
    device bfloat* k_cache    [[buffer(2)]],
    device bfloat* v_cache    [[buffer(3)]],
    constant AttnPrepKVParams& p [[buffer(4)]],
    uint2 tgid                [[threadgroup_position_in_grid]],
    uint lane                 [[thread_index_in_simdgroup]])
{
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.n_kv_heads || t >= p.T) return;

    uint pos = p.start_pos + t;
    uint slot = (p.is_local != 0u) ? (pos % p.cache_cap) : pos;

    device const bfloat* kin = K_in + (long(t) * long(p.n_kv_heads) + long(h)) * long(p.head_dim);
    device const bfloat* vin = V_in + (long(t) * long(p.n_kv_heads) + long(h)) * long(p.head_dim);

    float kv[4], vv[4];
    float k_sq = 0.0f, v_sq = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        uint d = lane + i * 32u;
        kv[i] = float(kin[d]);
        vv[i] = float(vin[d]);
        k_sq += kv[i] * kv[i];
        v_sq += vv[i] * vv[i];
    }
    float k_norm = sqrt(simd_sum(k_sq));
    float k_sc = min(1.0f, p.qk_clip / (k_norm + 1e-6f));
    float v_rstd = rsqrt(simd_sum(v_sq) / float(p.head_dim) + p.eps);

    device bfloat* kdst = k_cache + (long(slot) * long(p.n_kv_heads) + long(h)) * long(p.head_dim);
    device bfloat* vdst = v_cache + (long(slot) * long(p.n_kv_heads) + long(h)) * long(p.head_dim);
    for (uint i = 0; i < 4; ++i) {
        uint d = lane + i * 32u;
        kdst[d] = bfloat(kv[i] * k_sc);
        vdst[d] = bfloat(vv[i] * v_rstd);
    }
}

kernel void attn_prep_kv_fp8_kernel(
    device const bfloat* K_in  [[buffer(0)]],
    device const bfloat* V_in  [[buffer(1)]],
    device uchar* k_cache_fp8  [[buffer(2)]],
    device uchar* v_cache_fp8  [[buffer(3)]],
    device float* k_scale_fp8  [[buffer(4)]],
    device float* v_scale_fp8  [[buffer(5)]],
    constant AttnPrepKVParams& p [[buffer(6)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint lane                  [[thread_index_in_simdgroup]])
{
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.n_kv_heads || t >= p.T) return;

    uint pos = p.start_pos + t;
    uint slot = (p.is_local != 0u) ? (pos % p.cache_cap) : pos;

    device const bfloat* kin = K_in + (long(t) * long(p.n_kv_heads) + long(h)) * long(p.head_dim);
    device const bfloat* vin = V_in + (long(t) * long(p.n_kv_heads) + long(h)) * long(p.head_dim);

    float kv[4], vv[4];
    float k_sq = 0.0f, v_sq = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        uint d = lane + i * 32u;
        kv[i] = float(kin[d]);
        vv[i] = float(vin[d]);
        k_sq += kv[i] * kv[i];
        v_sq += vv[i] * vv[i];
    }
    float k_norm = sqrt(simd_sum(k_sq));
    float k_sc = min(1.0f, p.qk_clip / (k_norm + 1e-6f));
    float v_rstd = rsqrt(simd_sum(v_sq) / float(p.head_dim) + p.eps);

    float km = 0.0f, vm = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        kv[i] *= k_sc;
        vv[i] *= v_rstd;
        km = max(km, fabs(kv[i]));
        vm = max(vm, fabs(vv[i]));
    }
    float k_amax = simd_max(km);
    float v_amax = simd_max(vm);
    float ks = (k_amax > 0.0f) ? (k_amax / 448.0f) : 1.0f;
    float vs = (v_amax > 0.0f) ? (v_amax / 448.0f) : 1.0f;
    float inv_ks = 1.0f / ks;
    float inv_vs = 1.0f / vs;

    long sc_idx = long(slot) * long(p.n_kv_heads) + long(h);
    if (lane == 0u) {
        k_scale_fp8[sc_idx] = ks;
        v_scale_fp8[sc_idx] = vs;
    }
    device uchar* kdst = k_cache_fp8 + sc_idx * long(p.head_dim);
    device uchar* vdst = v_cache_fp8 + sc_idx * long(p.head_dim);
    for (uint i = 0; i < 4; ++i) {
        uint d = lane + i * 32u;
        kdst[d] = f32_to_e4m3(kv[i] * inv_ks);
        vdst[d] = f32_to_e4m3(vv[i] * inv_vs);
    }
}

struct RelMatmulParams {
    uint M;
    uint d_rel;
    uint rel_extent;
};

kernel void rel_logits_matmul_kernel(
    device const bfloat* R        [[buffer(0)]],
    device const bfloat* rel_proj [[buffer(1)]],
    device bfloat* out            [[buffer(2)]],
    constant RelMatmulParams& p   [[buffer(3)]],
    uint2 gid                     [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint m = gid.y;
    if (m >= p.M || col >= p.rel_extent) return;
    device const bfloat* r_row = R + long(m) * long(p.d_rel);
    float acc = 0.0f;
    for (uint k = 0; k < p.d_rel; ++k) {
        acc = fma(float(r_row[k]), float(rel_proj[long(k) * long(p.rel_extent) + long(col)]), acc);
    }
    out[long(m) * long(p.rel_extent) + long(col)] = bfloat(acc);
}

struct AttnParams {
    uint start_pos;
    uint T;
    uint cache_cap;
    uint is_local;
    uint window_size;
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint rel_extent;
};

kernel void attention_kernel(
    device const bfloat* Q          [[buffer(0)]],
    device const bfloat* k_cache    [[buffer(1)]],
    device const bfloat* v_cache    [[buffer(2)]],
    device const bfloat* rel_logits [[buffer(3)]],
    device const bfloat* gate       [[buffer(4)]],
    device const bfloat* sinks      [[buffer(5)]],
    device bfloat* out              [[buffer(6)]],
    constant AttnParams& p          [[buffer(7)]],
    uint2 tgid                      [[threadgroup_position_in_grid]],
    uint lane                       [[thread_index_in_simdgroup]])
{
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.n_heads || t >= p.T) return;

    uint kv_h = h / (p.n_heads / p.n_kv_heads);
    uint q_pos = p.start_pos + t;
    uint k_start = (p.is_local != 0u && q_pos + 1u > p.window_size) ? (q_pos + 1u - p.window_size) : 0u;
    uint k_end = q_pos + 1u;

    device const bfloat* q_ptr = Q + (long(t) * long(p.n_heads) + long(h)) * long(p.head_dim);
    device const bfloat* r_ptr = rel_logits + (long(t) * long(p.n_heads) + long(h)) * long(p.rel_extent);
    float qv[4];
    for (uint i = 0; i < 4; ++i) {
        qv[i] = float(q_ptr[lane + i * 32u]);
    }
    float inv_sqrt_d = rsqrt(float(p.head_dim));

    float sink_logit = float(sinks[long(t) * long(p.n_heads) + long(h)]);
    float m_val = sink_logit;
    float l_val = 1.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    long kv_stride = long(p.n_kv_heads) * long(p.head_dim);
    for (uint kp = k_start; kp < k_end; ++kp) {
        uint slot = (p.is_local != 0u) ? (kp % p.cache_cap) : kp;
        device const bfloat* k_ptr = k_cache + long(slot) * kv_stride + long(kv_h) * long(p.head_dim);
        device const bfloat* v_ptr = v_cache + long(slot) * kv_stride + long(kv_h) * long(p.head_dim);
        float dot = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            dot = fma(qv[i], float(k_ptr[lane + i * 32u]), dot);
        }
        dot = simd_sum(dot) * inv_sqrt_d;
        uint dist = q_pos - kp;
        if (dist < p.rel_extent) {
            dot += float(r_ptr[dist]);
        }
        float m_new = max(m_val, dot);
        float alpha = exp(m_val - m_new);
        float pr = exp(dot - m_new);
        l_val = l_val * alpha + pr;
        for (uint i = 0; i < 4; ++i) {
            acc[i] = acc[i] * alpha + pr * float(v_ptr[lane + i * 32u]);
        }
        m_val = m_new;
    }

    float inv_l = 1.0f / l_val;
    device const bfloat* g_ptr = gate + (long(t) * long(p.n_heads) + long(h)) * long(p.head_dim);
    device bfloat* o_ptr = out + (long(t) * long(p.n_heads) + long(h)) * long(p.head_dim);
    for (uint i = 0; i < 4; ++i) {
        uint d = lane + i * 32u;
        float gv = float(g_ptr[d]);
        float sig = 1.0f / (1.0f + exp(-gv));
        o_ptr[d] = bfloat(acc[i] * inv_l * sig);
    }
}

kernel void attention_fp8_kernel(
    device const bfloat* Q          [[buffer(0)]],
    device const uchar* k_cache_fp8 [[buffer(1)]],
    device const uchar* v_cache_fp8 [[buffer(2)]],
    device const float* k_scale_fp8 [[buffer(3)]],
    device const float* v_scale_fp8 [[buffer(4)]],
    device const bfloat* rel_logits [[buffer(5)]],
    device const bfloat* gate       [[buffer(6)]],
    device const bfloat* sinks      [[buffer(7)]],
    device bfloat* out              [[buffer(8)]],
    constant AttnParams& p          [[buffer(9)]],
    uint2 tgid                      [[threadgroup_position_in_grid]],
    uint lane                       [[thread_index_in_simdgroup]])
{
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.n_heads || t >= p.T) return;

    uint kv_h = h / (p.n_heads / p.n_kv_heads);
    uint q_pos = p.start_pos + t;
    uint k_start = (p.is_local != 0u && q_pos + 1u > p.window_size) ? (q_pos + 1u - p.window_size) : 0u;
    uint k_end = q_pos + 1u;

    device const bfloat* q_ptr = Q + (long(t) * long(p.n_heads) + long(h)) * long(p.head_dim);
    device const bfloat* r_ptr = rel_logits + (long(t) * long(p.n_heads) + long(h)) * long(p.rel_extent);
    float qv[4];
    for (uint i = 0; i < 4; ++i) {
        qv[i] = float(q_ptr[lane + i * 32u]);
    }
    float inv_sqrt_d = rsqrt(float(p.head_dim));

    float sink_logit = float(sinks[long(t) * long(p.n_heads) + long(h)]);
    float m_val = sink_logit;
    float l_val = 1.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint kp = k_start; kp < k_end; ++kp) {
        uint slot = (p.is_local != 0u) ? (kp % p.cache_cap) : kp;
        long sc_idx = long(slot) * long(p.n_kv_heads) + long(kv_h);
        float ks = k_scale_fp8[sc_idx];
        float vs = v_scale_fp8[sc_idx];
        device const uchar* k_ptr = k_cache_fp8 + sc_idx * long(p.head_dim);
        device const uchar* v_ptr = v_cache_fp8 + sc_idx * long(p.head_dim);
        float dot = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            dot = fma(qv[i], e4m3_to_f32(k_ptr[lane + i * 32u]), dot);
        }
        dot = simd_sum(dot) * inv_sqrt_d * ks;
        uint dist = q_pos - kp;
        if (dist < p.rel_extent) {
            dot += float(r_ptr[dist]);
        }
        float m_new = max(m_val, dot);
        float alpha = exp(m_val - m_new);
        float pr = exp(dot - m_new);
        float pvs = pr * vs;
        l_val = l_val * alpha + pr;
        for (uint i = 0; i < 4; ++i) {
            acc[i] = acc[i] * alpha + pvs * e4m3_to_f32(v_ptr[lane + i * 32u]);
        }
        m_val = m_new;
    }

    float inv_l = 1.0f / l_val;
    device const bfloat* g_ptr = gate + (long(t) * long(p.n_heads) + long(h)) * long(p.head_dim);
    device bfloat* o_ptr = out + (long(t) * long(p.n_heads) + long(h)) * long(p.head_dim);
    for (uint i = 0; i < 4; ++i) {
        uint d = lane + i * 32u;
        float gv = float(g_ptr[d]);
        float sig = 1.0f / (1.0f + exp(-gv));
        o_ptr[d] = bfloat(acc[i] * inv_l * sig);
    }
}

struct AttnSplitKParams {
    uint start_pos;
    uint T;
    uint cache_cap;
    uint is_local;
    uint window_size;
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint rel_extent;
    uint num_splits;
};

kernel void attention_fp8_splitk_kernel(
    device const bfloat* Q          [[buffer(0)]],
    device const uchar* k_cache_fp8 [[buffer(1)]],
    device const uchar* v_cache_fp8 [[buffer(2)]],
    device const float* k_scale_fp8 [[buffer(3)]],
    device const float* v_scale_fp8 [[buffer(4)]],
    device const bfloat* rel_logits [[buffer(5)]],
    device float2* part_ml          [[buffer(6)]],
    device float* part_acc          [[buffer(7)]],
    constant AttnSplitKParams& p    [[buffer(8)]],
    uint3 tgid                      [[threadgroup_position_in_grid]],
    uint lane                       [[thread_index_in_simdgroup]])
{
    uint h = tgid.x;
    uint t = tgid.y;
    uint sp = tgid.z;
    if (h >= p.n_heads || t >= p.T || sp >= p.num_splits) return;

    uint q_pos = p.start_pos + t;
    uint k_end = q_pos + 1u;
    uint chunk = (k_end + p.num_splits - 1u) / p.num_splits;
    uint ks_pos = sp * chunk;
    uint ke_pos = min(k_end, ks_pos + chunk);

    long base_sp = (long(t) * long(p.n_heads) + long(h)) * long(p.num_splits) + long(sp);
    if (ks_pos >= ke_pos) {
        if (lane == 0u) {
            part_ml[base_sp] = float2(-INFINITY, 0.0f);
        }
        return;
    }

    uint kv_h = h / (p.n_heads / p.n_kv_heads);
    device const bfloat* q_ptr = Q + (long(t) * long(p.n_heads) + long(h)) * long(p.head_dim);
    device const bfloat* r_ptr = rel_logits + (long(t) * long(p.n_heads) + long(h)) * long(p.rel_extent);
    float qv0 = float(q_ptr[lane]);
    float qv1 = float(q_ptr[lane + 32u]);
    float qv2 = float(q_ptr[lane + 64u]);
    float qv3 = float(q_ptr[lane + 96u]);
    float inv_sqrt_d = rsqrt(float(p.head_dim));

    float m_val = -INFINITY;
    float l_val = 0.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    for (uint kp = ks_pos; kp < ke_pos; ++kp) {
        long sc_idx = long(kp) * long(p.n_kv_heads) + long(kv_h);
        float ks = k_scale_fp8[sc_idx];
        float vs = v_scale_fp8[sc_idx];
        device const uchar* k_ptr = k_cache_fp8 + sc_idx * long(p.head_dim);
        device const uchar* v_ptr = v_cache_fp8 + sc_idx * long(p.head_dim);
        float dot = fma(qv0, e4m3_to_f32(k_ptr[lane]), 0.0f);
        dot = fma(qv1, e4m3_to_f32(k_ptr[lane + 32u]), dot);
        dot = fma(qv2, e4m3_to_f32(k_ptr[lane + 64u]), dot);
        dot = fma(qv3, e4m3_to_f32(k_ptr[lane + 96u]), dot);
        dot = simd_sum(dot) * inv_sqrt_d * ks;
        uint dist = q_pos - kp;
        if (dist < p.rel_extent) {
            dot += float(r_ptr[dist]);
        }
        float m_new = max(m_val, dot);
        float alpha = (l_val > 0.0f) ? exp(m_val - m_new) : 0.0f;
        float pr = exp(dot - m_new);
        float pvs = pr * vs;
        l_val = l_val * alpha + pr;
        a0 = a0 * alpha + pvs * e4m3_to_f32(v_ptr[lane]);
        a1 = a1 * alpha + pvs * e4m3_to_f32(v_ptr[lane + 32u]);
        a2 = a2 * alpha + pvs * e4m3_to_f32(v_ptr[lane + 64u]);
        a3 = a3 * alpha + pvs * e4m3_to_f32(v_ptr[lane + 96u]);
        m_val = m_new;
    }

    if (lane == 0u) {
        part_ml[base_sp] = float2(m_val, l_val);
    }
    device float* pa = part_acc + base_sp * long(p.head_dim);
    pa[lane] = a0;
    pa[lane + 32u] = a1;
    pa[lane + 64u] = a2;
    pa[lane + 96u] = a3;
}

kernel void attention_fp8_reduce_kernel(
    device const float2* part_ml    [[buffer(0)]],
    device const float* part_acc    [[buffer(1)]],
    device const bfloat* gate       [[buffer(2)]],
    device const bfloat* sinks      [[buffer(3)]],
    device bfloat* out              [[buffer(4)]],
    constant AttnSplitKParams& p    [[buffer(5)]],
    uint2 tgid                      [[threadgroup_position_in_grid]],
    uint lane                       [[thread_index_in_simdgroup]])
{
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.n_heads || t >= p.T) return;

    float m_val = float(sinks[long(t) * long(p.n_heads) + long(h)]);
    float l_val = 1.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    long base = (long(t) * long(p.n_heads) + long(h)) * long(p.num_splits);
    for (uint sp = 0; sp < p.num_splits; ++sp) {
        float2 ml = part_ml[base + long(sp)];
        if (ml.y <= 0.0f) continue;
        float m_new = max(m_val, ml.x);
        float c1 = exp(m_val - m_new);
        float c2 = exp(ml.x - m_new);
        device const float* pa = part_acc + (base + long(sp)) * long(p.head_dim);
        l_val = l_val * c1 + ml.y * c2;
        a0 = a0 * c1 + pa[lane] * c2;
        a1 = a1 * c1 + pa[lane + 32u] * c2;
        a2 = a2 * c1 + pa[lane + 64u] * c2;
        a3 = a3 * c1 + pa[lane + 96u] * c2;
        m_val = m_new;
    }

    float inv_l = 1.0f / l_val;
    device const bfloat* g_ptr = gate + (long(t) * long(p.n_heads) + long(h)) * long(p.head_dim);
    device bfloat* o_ptr = out + (long(t) * long(p.n_heads) + long(h)) * long(p.head_dim);
    o_ptr[lane]       = bfloat(a0 * inv_l / (1.0f + exp(-float(g_ptr[lane]))));
    o_ptr[lane + 32u] = bfloat(a1 * inv_l / (1.0f + exp(-float(g_ptr[lane + 32u]))));
    o_ptr[lane + 64u] = bfloat(a2 * inv_l / (1.0f + exp(-float(g_ptr[lane + 64u]))));
    o_ptr[lane + 96u] = bfloat(a3 * inv_l / (1.0f + exp(-float(g_ptr[lane + 96u]))));
}

kernel void attention_fp8_prefill4_kernel(
    device const bfloat* Q          [[buffer(0)]],
    device const uchar* k_cache_fp8 [[buffer(1)]],
    device const uchar* v_cache_fp8 [[buffer(2)]],
    device const float* k_scale_fp8 [[buffer(3)]],
    device const float* v_scale_fp8 [[buffer(4)]],
    device const bfloat* rel_logits [[buffer(5)]],
    device const bfloat* gate       [[buffer(6)]],
    device const bfloat* sinks      [[buffer(7)]],
    device bfloat* out              [[buffer(8)]],
    constant AttnParams& p          [[buffer(9)]],
    uint2 tgid                      [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]])
{
    uint h = tgid.x;
    uint tg_t_base = tgid.y * 16u;
    if (h >= p.n_heads || tg_t_base >= p.T) return;
    uint nt = min(16u, p.T - tg_t_base);
    uint kv_h = h / (p.n_heads / p.n_kv_heads);
    float inv_sqrt_d = rsqrt(float(p.head_dim));
    uint k_end_max = p.start_pos + tg_t_base + nt;

    threadgroup bfloat sQ[16 * 128];
    threadgroup bfloat sK[128 * 16];
    threadgroup bfloat sV[16 * 128];
    threadgroup float sS[16 * 16];
    threadgroup bfloat sP[16 * 16];
    threadgroup float sO[16 * 128];
    threadgroup float sM[16];
    threadgroup float sL[16];
    threadgroup float sAlpha[16];

    uint r_load = tid >> 3;
    uint c_load = (tid & 7u) * 16u;

    if (r_load < nt) {
        device const bfloat* qp = Q + (long(tg_t_base + r_load) * long(p.n_heads) + long(h)) * long(p.head_dim) + long(c_load);
        for (uint i = 0u; i < 16u; ++i) {
            sQ[r_load * 128u + c_load + i] = qp[i];
            sO[r_load * 128u + c_load + i] = 0.0f;
        }
    } else {
        for (uint i = 0u; i < 16u; ++i) {
            sQ[r_load * 128u + c_load + i] = bfloat(0.0f);
            sO[r_load * 128u + c_load + i] = 0.0f;
        }
    }
    if (tid < 16u) {
        sM[tid] = (tid < nt) ? float(sinks[long(tg_t_base + tid) * long(p.n_heads) + long(h)]) : -1e30f;
        sL[tid] = 1.0f;
        sAlpha[tid] = 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint qr_s = (sg_id >> 1) * 8u;
    uint kc_s = (sg_id & 1u) * 8u;
    uint dc_v = sg_id * 32u;

    for (uint kp0 = 0u; kp0 < k_end_max; kp0 += 16u) {
        uint n_kv = min(16u, k_end_max - kp0);
        if (r_load < n_kv) {
            long sc_idx = long(kp0 + r_load) * long(p.n_kv_heads) + long(kv_h);
            float ks = k_scale_fp8[sc_idx] * inv_sqrt_d;
            float vs = v_scale_fp8[sc_idx];
            device const uchar* k_ptr = k_cache_fp8 + sc_idx * long(p.head_dim) + long(c_load);
            device const uchar* v_ptr = v_cache_fp8 + sc_idx * long(p.head_dim) + long(c_load);
            for (uint i = 0u; i < 16u; ++i) {
                sK[(c_load + i) * 16u + r_load] = bfloat(e4m3_to_f32(k_ptr[i]) * ks);
                sV[r_load * 128u + c_load + i] = bfloat(e4m3_to_f32(v_ptr[i]) * vs);
            }
        } else {
            for (uint i = 0u; i < 16u; ++i) {
                sK[(c_load + i) * 16u + r_load] = bfloat(0.0f);
                sV[r_load * 128u + c_load + i] = bfloat(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> accS(0.0f);
        for (uint dd = 0u; dd < 128u; dd += 8u) {
            simdgroup_matrix<bfloat, 8, 8> ma, mb;
            simdgroup_load(ma, sQ + qr_s * 128u + dd, 128u);
            simdgroup_load(mb, sK + dd * 16u + kc_s, 16u);
            simdgroup_multiply_accumulate(accS, ma, mb, accS);
        }
        simdgroup_store(accS, sS + qr_s * 16u + kc_s, 16u);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < nt) {
            uint qp = p.start_pos + tg_t_base + tid;
            device const bfloat* rp = rel_logits + (long(tg_t_base + tid) * long(p.n_heads) + long(h)) * long(p.rel_extent);
            float m_old = sM[tid];
            float m_new = m_old;
            for (uint j = 0u; j < n_kv; ++j) {
                uint kp = kp0 + j;
                if (kp <= qp) {
                    float val = sS[tid * 16u + j];
                    uint dist = qp - kp;
                    if (dist < p.rel_extent) val += float(rp[dist]);
                    sS[tid * 16u + j] = val;
                    m_new = max(m_new, val);
                }
            }
            float al = exp(m_old - m_new);
            sAlpha[tid] = al;
            sM[tid] = m_new;
            float l_new = sL[tid] * al;
            for (uint j = 0u; j < 16u; ++j) {
                uint kp = kp0 + j;
                if (j < n_kv && kp <= qp) {
                    float pr = exp(sS[tid * 16u + j] - m_new);
                    l_new += pr;
                    sP[tid * 16u + j] = bfloat(pr);
                } else {
                    sP[tid * 16u + j] = bfloat(0.0f);
                }
            }
            sL[tid] = l_new;
        } else if (tid < 16u) {
            sAlpha[tid] = 1.0f;
            for (uint j = 0u; j < 16u; ++j) {
                sP[tid * 16u + j] = bfloat(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float al_r = sAlpha[r_load];
        for (uint i = 0u; i < 16u; ++i) {
            sO[r_load * 128u + c_load + i] *= al_r;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint qr = 0u; qr < 16u; qr += 8u) {
            for (uint cc = 0u; cc < 32u; cc += 8u) {
                simdgroup_matrix<float, 8, 8> accO;
                simdgroup_load(accO, sO + qr * 128u + dc_v + cc, 128u);
                for (uint jj = 0u; jj < 16u; jj += 8u) {
                    simdgroup_matrix<bfloat, 8, 8> mp, mv;
                    simdgroup_load(mp, sP + qr * 16u + jj, 16u);
                    simdgroup_load(mv, sV + jj * 128u + dc_v + cc, 128u);
                    simdgroup_multiply_accumulate(accO, mp, mv, accO);
                }
                simdgroup_store(accO, sO + qr * 128u + dc_v + cc, 128u);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (r_load < nt) {
        uint tt = tg_t_base + r_load;
        float inv_l = 1.0f / sL[r_load];
        device const bfloat* gp = gate + (long(tt) * long(p.n_heads) + long(h)) * long(p.head_dim) + long(c_load);
        device bfloat* op = out + (long(tt) * long(p.n_heads) + long(h)) * long(p.head_dim) + long(c_load);
        for (uint i = 0u; i < 16u; ++i) {
            float gv = float(gp[i]);
            op[i] = bfloat(sO[r_load * 128u + c_load + i] * inv_l / (1.0f + exp(-gv)));
        }
    }
}

struct RouterParams {
    uint T;
    uint D;
    uint E;
    uint K;
    float route_scale;
};

kernel void moe_router_logits_kernel(
    device const bfloat* x      [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float* logits        [[buffer(2)]],
    constant RouterParams& p    [[buffer(3)]],
    uint2 tgid                  [[threadgroup_position_in_grid]],
    uint lane                   [[thread_index_in_simdgroup]])
{
    uint t = tgid.y;
    uint e = tgid.x;
    if (e >= p.E || t >= p.T) return;

    device const bfloat4* xr4 = reinterpret_cast<device const bfloat4*>(x + long(t) * long(p.D));
    device const bfloat4* wr4 = reinterpret_cast<device const bfloat4*>(weight + long(e) * long(p.D));
    uint d4 = p.D >> 2;
    float acc = 0.0f;
    for (uint i = lane; i < d4; i += 32u) {
        float4 xv = float4(xr4[i]);
        float4 wv = float4(wr4[i]);
        acc = fma(wv.x, xv.x, fma(wv.y, xv.y, fma(wv.z, xv.z, fma(wv.w, xv.w, acc))));
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        logits[long(t) * long(p.E) + long(e)] = acc;
    }
}

kernel void moe_router_topk_kernel(
    device const float* logits [[buffer(0)]],
    device const float* bias   [[buffer(1)]],
    device int* topk_ids       [[buffer(2)]],
    device float* topk_weights [[buffer(3)]],
    constant RouterParams& p   [[buffer(4)]],
    uint tgid                  [[threadgroup_position_in_grid]],
    uint lane                  [[thread_index_in_simdgroup]])
{
    uint t = tgid;
    if (t >= p.T) return;

    float scores[4];
    float sel[4];
    for (uint j = 0; j < 4u; ++j) {
        uint e = lane + j * 32u;
        if (e < p.E) {
            float sig = 1.0f / (1.0f + exp(-logits[long(t) * long(p.E) + long(e)]));
            scores[j] = sig;
            sel[j] = sig + bias[e];
        } else {
            scores[j] = 0.0f;
            sel[j] = -1e30f;
        }
    }

    int ids[8];
    float wts[8];
    float wsum = 0.0f;
    for (uint k = 0; k < p.K; ++k) {
        float lv = sel[0];
        uint le = lane;
        for (uint j = 1u; j < 4u; ++j) {
            if (sel[j] > lv) {
                lv = sel[j];
                le = lane + j * 32u;
            }
        }
        float gv = simd_max(lv);
        uint cand = (lv == gv) ? le : 0xFFFFFFFFu;
        uint best_e = simd_min(cand);
        float best_sig = 0.0f;
        for (uint j = 0; j < 4u; ++j) {
            if (lane + j * 32u == best_e) {
                best_sig = scores[j];
                sel[j] = -1e30f;
            }
        }
        best_sig = simd_sum(best_sig);
        ids[k] = int(best_e);
        wts[k] = best_sig;
        wsum += best_sig;
    }
    if (lane == 0u) {
        float inv_sum = p.route_scale / (wsum + 1e-20f);
        for (uint k = 0; k < p.K; ++k) {
            topk_ids[long(t) * long(p.K) + long(k)] = ids[k];
            topk_weights[long(t) * long(p.K) + long(k)] = wts[k] * inv_sum;
        }
    }
}

struct MoeParams {
    uint T;
    uint D;
    uint inter;
    uint K;
    float gate_cap;
    float up_cap;
};

kernel void moe_nvfp4_w13_kernel(
    device const bfloat* x         [[buffer(0)]],
    device const int* topk_ids     [[buffer(1)]],
    device const float* topk_wts   [[buffer(2)]],
    device const uchar* w13_u8     [[buffer(3)]],
    device const uchar* w13_scale  [[buffer(4)]],
    device const float* w13_amax   [[buffer(5)]],
    device const float* w13_denom  [[buffer(6)]],
    device float* act_out          [[buffer(7)]],
    constant MoeParams& p          [[buffer(8)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]])
{
    uint tk = tgid.y;
    uint t = tk / p.K;
    int expert = topk_ids[tk];
    float gamma = topk_wts[tk];

    uint j = tgid.x * 4u + sg_id;
    if (j >= p.inter) return;

    float gscale = w13_amax[expert] / w13_denom[0];
    long row_g = long(2u * j);
    long row_u = long(2u * j + 1u);
    long half_D = long(p.D / 2u);
    long scale_D = long(p.D / 16u);
    long exp_base = long(expert) * long(2u * p.inter);

    device const uchar* wg = w13_u8 + (exp_base + row_g) * half_D;
    device const uchar* wu = w13_u8 + (exp_base + row_u) * half_D;
    device const uchar* sg = w13_scale + (exp_base + row_g) * scale_D;
    device const uchar* su = w13_scale + (exp_base + row_u) * scale_D;
    device const bfloat4* xr4 = reinterpret_cast<device const bfloat4*>(x + long(t) * long(p.D));
    device const uint2* wg2 = reinterpret_cast<device const uint2*>(wg);
    device const uint2* wu2 = reinterpret_cast<device const uint2*>(wu);

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (uint blk = lane; blk < uint(scale_D); blk += 32u) {
        float bsg = e4m3_to_f32(sg[blk]);
        float bsu = e4m3_to_f32(su[blk]);
        uint2 bg = wg2[blk];
        uint2 bu = wu2[blk];
        float lg = 0.0f, lu = 0.0f;
        uint b0 = bg.x, u0 = bu.x;
        for (uint q = 0; q < 2u; ++q) {
            float4 xv = float4(xr4[blk * 4u + q]);
            uint bg0 = (b0 >> (q * 16u)) & 0xFFu;
            uint bg1 = (b0 >> (q * 16u + 8u)) & 0xFFu;
            uint bu0 = (u0 >> (q * 16u)) & 0xFFu;
            uint bu1 = (u0 >> (q * 16u + 8u)) & 0xFFu;
            lg = fma(kE2M1Lut[bg0 & 0xFu], xv.x, fma(kE2M1Lut[bg0 >> 4], xv.y, fma(kE2M1Lut[bg1 & 0xFu], xv.z, fma(kE2M1Lut[bg1 >> 4], xv.w, lg))));
            lu = fma(kE2M1Lut[bu0 & 0xFu], xv.x, fma(kE2M1Lut[bu0 >> 4], xv.y, fma(kE2M1Lut[bu1 & 0xFu], xv.z, fma(kE2M1Lut[bu1 >> 4], xv.w, lu))));
        }
        uint b1 = bg.y, u1 = bu.y;
        for (uint q = 0; q < 2u; ++q) {
            float4 xv = float4(xr4[blk * 4u + 2u + q]);
            uint bg0 = (b1 >> (q * 16u)) & 0xFFu;
            uint bg1 = (b1 >> (q * 16u + 8u)) & 0xFFu;
            uint bu0 = (u1 >> (q * 16u)) & 0xFFu;
            uint bu1 = (u1 >> (q * 16u + 8u)) & 0xFFu;
            lg = fma(kE2M1Lut[bg0 & 0xFu], xv.x, fma(kE2M1Lut[bg0 >> 4], xv.y, fma(kE2M1Lut[bg1 & 0xFu], xv.z, fma(kE2M1Lut[bg1 >> 4], xv.w, lg))));
            lu = fma(kE2M1Lut[bu0 & 0xFu], xv.x, fma(kE2M1Lut[bu0 >> 4], xv.y, fma(kE2M1Lut[bu1 & 0xFu], xv.z, fma(kE2M1Lut[bu1 >> 4], xv.w, lu))));
        }
        acc_g = fma(lg, bsg, acc_g);
        acc_u = fma(lu, bsu, acc_u);
    }
    acc_g = simd_sum(acc_g) * gscale;
    acc_u = simd_sum(acc_u) * gscale;

    if (lane == 0u) {
        float sig = 1.0f / (1.0f + exp(-acc_g));
        float g_cap = safe_tanh(acc_g / p.gate_cap) * p.gate_cap;
        float u_cap = safe_tanh(acc_u / p.up_cap) * p.up_cap;
        float y = (g_cap * sig * u_cap) * gamma;
        act_out[long(tk) * long(p.inter) + long(j)] = float(bfloat(y));
    }
}

kernel void moe_nvfp4_w2_kernel(
    device const float* act_in     [[buffer(0)]],
    device const int* topk_ids     [[buffer(1)]],
    device const uchar* w2_u8      [[buffer(2)]],
    device const uchar* w2_scale   [[buffer(3)]],
    device const float* w2_amax    [[buffer(4)]],
    device const float* w2_denom   [[buffer(5)]],
    device bfloat* out             [[buffer(6)]],
    constant MoeParams& p          [[buffer(7)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]])
{
    uint t = tgid.y;
    uint d = tgid.x * 4u + sg_id;
    if (d >= p.D) return;

    float inv_denom = 1.0f / w2_denom[0];
    long half_I = long(p.inter / 2u);
    long scale_I = long(p.inter / 16u);

    float total = 0.0f;
    for (uint k = 0; k < p.K; ++k) {
        uint tk = t * p.K + k;
        int expert = topk_ids[tk];
        float gscale = w2_amax[expert] * inv_denom;
        long row_idx = long(expert) * long(p.D) + long(d);
        device const uint2* w2_2 = reinterpret_cast<device const uint2*>(w2_u8 + row_idx * half_I);
        device const uchar* s2r = w2_scale + row_idx * scale_I;
        device const float4* ar4 = reinterpret_cast<device const float4*>(act_in + long(tk) * long(p.inter));

        float acc = 0.0f;
        for (uint blk = lane; blk < uint(scale_I); blk += 32u) {
            float bs = e4m3_to_f32(s2r[blk]);
            uint2 bw = w2_2[blk];
            float4 a0 = ar4[blk * 4u + 0u];
            float4 a1 = ar4[blk * 4u + 1u];
            float4 a2 = ar4[blk * 4u + 2u];
            float4 a3 = ar4[blk * 4u + 3u];
            uint bx = bw.x, by = bw.y;
            float loc = 0.0f;
            loc = fma(kE2M1Lut[bx & 0xFu], a0.x, fma(kE2M1Lut[(bx >> 4) & 0xFu], a0.y, fma(kE2M1Lut[(bx >> 8) & 0xFu], a0.z, fma(kE2M1Lut[(bx >> 12) & 0xFu], a0.w, loc))));
            loc = fma(kE2M1Lut[(bx >> 16) & 0xFu], a1.x, fma(kE2M1Lut[(bx >> 20) & 0xFu], a1.y, fma(kE2M1Lut[(bx >> 24) & 0xFu], a1.z, fma(kE2M1Lut[(bx >> 28) & 0xFu], a1.w, loc))));
            loc = fma(kE2M1Lut[by & 0xFu], a2.x, fma(kE2M1Lut[(by >> 4) & 0xFu], a2.y, fma(kE2M1Lut[(by >> 8) & 0xFu], a2.z, fma(kE2M1Lut[(by >> 12) & 0xFu], a2.w, loc))));
            loc = fma(kE2M1Lut[(by >> 16) & 0xFu], a3.x, fma(kE2M1Lut[(by >> 20) & 0xFu], a3.y, fma(kE2M1Lut[(by >> 24) & 0xFu], a3.z, fma(kE2M1Lut[(by >> 28) & 0xFu], a3.w, loc))));
            acc = fma(loc, bs, acc);
        }
        acc = simd_sum(acc) * gscale;
        if (lane == 0u) total += acc;
    }
    if (lane == 0u) {
        out[long(t) * long(p.D) + long(d)] = bfloat(total);
    }
}

struct MoeGroupParams {
    uint total_tk;
    uint E;
};

kernel void moe_group_tokens_kernel(
    device const int* topk_ids [[buffer(0)]],
    device uint* offsets       [[buffer(1)]],
    device uint* sorted_tk     [[buffer(2)]],
    constant MoeGroupParams& p [[buffer(3)]],
    uint tid                   [[thread_index_in_threadgroup]])
{
    threadgroup uint counts[128];
    threadgroup uint starts[128];
    uint cnt = 0u;
    if (tid < p.E) {
        for (uint i = 0u; i < p.total_tk; ++i) {
            if (topk_ids[i] == int(tid)) cnt += 1u;
        }
        counts[tid] = cnt;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint acc = 0u;
        for (uint e = 0u; e < p.E; ++e) {
            starts[e] = acc;
            offsets[e] = acc;
            acc += counts[e];
        }
        offsets[p.E] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < p.E) {
        uint pos = starts[tid];
        for (uint i = 0u; i < p.total_tk; ++i) {
            if (topk_ids[i] == int(tid)) {
                sorted_tk[pos++] = i;
            }
        }
    }
}

kernel void moe_nvfp4_w13_grouped_kernel(
    device const bfloat* x         [[buffer(0)]],
    device const float* topk_wts   [[buffer(1)]],
    device const uint* offsets     [[buffer(2)]],
    device const uint* sorted_tk   [[buffer(3)]],
    device const uchar* w13_u8     [[buffer(4)]],
    device const uchar* w13_scale  [[buffer(5)]],
    device const float* w13_amax   [[buffer(6)]],
    device const float* w13_denom  [[buffer(7)]],
    device float* act_out          [[buffer(8)]],
    constant MoeParams& p          [[buffer(9)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]])
{
    uint expert = tgid.y;
    uint j0 = tgid.x * 16u;
    if (j0 >= p.inter) return;
    uint s0 = offsets[expert];
    uint s1 = offsets[expert + 1u];
    if (s0 >= s1) return;

    float gscale = w13_amax[expert] / w13_denom[0];
    long half_D = long(p.D / 2u);
    long scale_D = long(p.D / 16u);
    long exp_base = long(expert) * long(2u * p.inter) + long(2u * j0);
    uint n_rows = min(32u, 2u * (p.inter - j0));

    threadgroup bfloat smA[16 * 32];
    threadgroup bfloat smW[32 * 32];
    threadgroup float smOut[16 * 32];

    uint sg_r = (sg_id >> 1) * 8u;
    uint sg_c = (sg_id & 1u) * 16u;

    for (uint idx = s0; idx < s1; idx += 16u) {
        uint n = min(16u, s1 - idx);
        simdgroup_matrix<float, 8, 8> c0(0.0f);
        simdgroup_matrix<float, 8, 8> c1(0.0f);

        for (uint k0 = 0u; k0 < p.D; k0 += 32u) {
            uint am = tid >> 3;
            uint ak = (tid & 7u) * 4u;
            if (am < n) {
                uint tk = sorted_tk[idx + am];
                device const bfloat4* xp = reinterpret_cast<device const bfloat4*>(x + long(tk / p.K) * long(p.D) + long(k0 + ak));
                bfloat4 v = xp[0];
                smA[am * 32u + ak + 0u] = v.x;
                smA[am * 32u + ak + 1u] = v.y;
                smA[am * 32u + ak + 2u] = v.z;
                smA[am * 32u + ak + 3u] = v.w;
            } else {
                smA[am * 32u + ak + 0u] = bfloat(0.0f);
                smA[am * 32u + ak + 1u] = bfloat(0.0f);
                smA[am * 32u + ak + 2u] = bfloat(0.0f);
                smA[am * 32u + ak + 3u] = bfloat(0.0f);
            }

            uint wr = tid >> 2;
            uint wq = tid & 3u;
            uint k_loc = wq * 8u;
            if (wr < n_rows) {
                uint blk = (k0 >> 4) + (wq >> 1);
                float bs = e4m3_to_f32(w13_scale[(exp_base + long(wr)) * scale_D + long(blk)]);
                device const uint* wp = reinterpret_cast<device const uint*>(w13_u8 + (exp_base + long(wr)) * half_D);
                uint w32 = wp[blk * 2u + (wq & 1u)];
                for (uint b = 0u; b < 8u; ++b) {
                    smW[(k_loc + b) * 32u + wr] = bfloat(kE2M1Lut[(w32 >> (b * 4u)) & 0xFu] * bs);
                }
            } else {
                for (uint b = 0u; b < 8u; ++b) {
                    smW[(k_loc + b) * 32u + wr] = bfloat(0.0f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kk = 0u; kk < 32u; kk += 8u) {
                simdgroup_matrix<bfloat, 8, 8> ma, mb0, mb1;
                simdgroup_load(ma, smA + sg_r * 32u + kk, 32u);
                simdgroup_load(mb0, smW + kk * 32u + sg_c, 32u);
                simdgroup_load(mb1, smW + kk * 32u + sg_c + 8u, 32u);
                simdgroup_multiply_accumulate(c0, ma, mb0, c0);
                simdgroup_multiply_accumulate(c1, ma, mb1, c1);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        simdgroup_store(c0, smOut + sg_r * 32u + sg_c, 32u);
        simdgroup_store(c1, smOut + sg_r * 32u + sg_c + 8u, 32u);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint o = tid; o < 256u; o += 128u) {
            uint m = o >> 4;
            uint jl = o & 15u;
            if (m < n && (j0 + jl) < p.inter) {
                uint tk = sorted_tk[idx + m];
                float g = smOut[m * 32u + 2u * jl] * gscale;
                float u = smOut[m * 32u + 2u * jl + 1u] * gscale;
                float sig = 1.0f / (1.0f + exp(-g));
                float gc = safe_tanh(g / p.gate_cap) * p.gate_cap;
                float uc = safe_tanh(u / p.up_cap) * p.up_cap;
                float y = (gc * sig * uc) * topk_wts[tk];
                act_out[long(tk) * long(p.inter) + long(j0 + jl)] = float(bfloat(y));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void moe_nvfp4_w2_grouped_kernel(
    device const float* act_in     [[buffer(0)]],
    device const uint* offsets     [[buffer(1)]],
    device const uint* sorted_tk   [[buffer(2)]],
    device const uchar* w2_u8      [[buffer(3)]],
    device const uchar* w2_scale   [[buffer(4)]],
    device const float* w2_amax    [[buffer(5)]],
    device const float* w2_denom   [[buffer(6)]],
    device bfloat* w2_out          [[buffer(7)]],
    constant MoeParams& p          [[buffer(8)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]])
{
    uint expert = tgid.y;
    uint d0 = tgid.x * 32u;
    if (d0 >= p.D) return;
    uint s0 = offsets[expert];
    uint s1 = offsets[expert + 1u];
    if (s0 >= s1) return;

    float gscale = w2_amax[expert] / w2_denom[0];
    long half_I = long(p.inter / 2u);
    long scale_I = long(p.inter / 16u);
    long exp_base = long(expert) * long(p.D) + long(d0);
    uint n_rows = min(32u, p.D - d0);

    threadgroup bfloat smA[16 * 32];
    threadgroup bfloat smW[32 * 32];
    threadgroup float smOut[16 * 32];

    uint sg_r = (sg_id >> 1) * 8u;
    uint sg_c = (sg_id & 1u) * 16u;

    for (uint idx = s0; idx < s1; idx += 16u) {
        uint n = min(16u, s1 - idx);
        simdgroup_matrix<float, 8, 8> c0(0.0f);
        simdgroup_matrix<float, 8, 8> c1(0.0f);

        for (uint k0 = 0u; k0 < p.inter; k0 += 32u) {
            uint am = tid >> 3;
            uint ak = (tid & 7u) * 4u;
            if (am < n) {
                uint tk = sorted_tk[idx + am];
                device const float4* ap = reinterpret_cast<device const float4*>(act_in + long(tk) * long(p.inter) + long(k0 + ak));
                float4 v = ap[0];
                smA[am * 32u + ak + 0u] = bfloat(v.x);
                smA[am * 32u + ak + 1u] = bfloat(v.y);
                smA[am * 32u + ak + 2u] = bfloat(v.z);
                smA[am * 32u + ak + 3u] = bfloat(v.w);
            } else {
                smA[am * 32u + ak + 0u] = bfloat(0.0f);
                smA[am * 32u + ak + 1u] = bfloat(0.0f);
                smA[am * 32u + ak + 2u] = bfloat(0.0f);
                smA[am * 32u + ak + 3u] = bfloat(0.0f);
            }

            uint wr = tid >> 2;
            uint wq = tid & 3u;
            uint k_loc = wq * 8u;
            if (wr < n_rows) {
                uint blk = (k0 >> 4) + (wq >> 1);
                float bs = e4m3_to_f32(w2_scale[(exp_base + long(wr)) * scale_I + long(blk)]);
                device const uint* wp = reinterpret_cast<device const uint*>(w2_u8 + (exp_base + long(wr)) * half_I);
                uint w32 = wp[blk * 2u + (wq & 1u)];
                for (uint b = 0u; b < 8u; ++b) {
                    smW[(k_loc + b) * 32u + wr] = bfloat(kE2M1Lut[(w32 >> (b * 4u)) & 0xFu] * bs);
                }
            } else {
                for (uint b = 0u; b < 8u; ++b) {
                    smW[(k_loc + b) * 32u + wr] = bfloat(0.0f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kk = 0u; kk < 32u; kk += 8u) {
                simdgroup_matrix<bfloat, 8, 8> ma, mb0, mb1;
                simdgroup_load(ma, smA + sg_r * 32u + kk, 32u);
                simdgroup_load(mb0, smW + kk * 32u + sg_c, 32u);
                simdgroup_load(mb1, smW + kk * 32u + sg_c + 8u, 32u);
                simdgroup_multiply_accumulate(c0, ma, mb0, c0);
                simdgroup_multiply_accumulate(c1, ma, mb1, c1);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        simdgroup_store(c0, smOut + sg_r * 32u + sg_c, 32u);
        simdgroup_store(c1, smOut + sg_r * 32u + sg_c + 8u, 32u);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint o = tid; o < 512u; o += 128u) {
            uint m = o >> 5;
            uint dl = o & 31u;
            if (m < n && (d0 + dl) < p.D) {
                uint tk = sorted_tk[idx + m];
                w2_out[long(tk) * long(p.D) + long(d0 + dl)] = bfloat(smOut[m * 32u + dl] * gscale);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void moe_reduce_k_kernel(
    device const bfloat* w2_out [[buffer(0)]],
    device bfloat* out          [[buffer(1)]],
    constant MoeParams& p       [[buffer(2)]],
    uint2 gid                   [[thread_position_in_grid]])
{
    uint d = gid.x;
    uint t = gid.y;
    if (d >= p.D || t >= p.T) return;
    float sum = 0.0f;
    long base = long(t) * long(p.K);
    for (uint k = 0u; k < p.K; ++k) {
        sum += float(w2_out[(base + long(k)) * long(p.D) + long(d)]);
    }
    out[long(t) * long(p.D) + long(d)] = bfloat(sum);
}

kernel void moe_mxfp8_w13_grouped_kernel(
    device const bfloat* x         [[buffer(0)]],
    device const float* topk_wts   [[buffer(1)]],
    device const uint* offsets     [[buffer(2)]],
    device const uint* sorted_tk   [[buffer(3)]],
    device const uchar* w13_e4m3   [[buffer(4)]],
    device const uchar* w13_scale  [[buffer(5)]],
    device float* act_out          [[buffer(6)]],
    constant MoeParams& p          [[buffer(7)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]])
{
    uint expert = tgid.y;
    uint j0 = tgid.x * 16u;
    if (j0 >= p.inter) return;
    uint s0 = offsets[expert];
    uint s1 = offsets[expert + 1u];
    if (s0 >= s1) return;

    long scale_D = long(p.D / 32u);
    long exp_base = long(expert) * long(2u * p.inter) + long(2u * j0);
    uint n_rows = min(32u, 2u * (p.inter - j0));

    threadgroup bfloat smA[16 * 32];
    threadgroup bfloat smW[32 * 32];
    threadgroup float smOut[16 * 32];

    uint sg_r = (sg_id >> 1) * 8u;
    uint sg_c = (sg_id & 1u) * 16u;

    for (uint idx = s0; idx < s1; idx += 16u) {
        uint n = min(16u, s1 - idx);
        simdgroup_matrix<float, 8, 8> c0(0.0f);
        simdgroup_matrix<float, 8, 8> c1(0.0f);

        for (uint k0 = 0u; k0 < p.D; k0 += 32u) {
            uint am = tid >> 3;
            uint ak = (tid & 7u) * 4u;
            if (am < n) {
                uint tk = sorted_tk[idx + am];
                device const bfloat4* xp = reinterpret_cast<device const bfloat4*>(x + long(tk / p.K) * long(p.D) + long(k0 + ak));
                bfloat4 v = xp[0];
                smA[am * 32u + ak + 0u] = v.x;
                smA[am * 32u + ak + 1u] = v.y;
                smA[am * 32u + ak + 2u] = v.z;
                smA[am * 32u + ak + 3u] = v.w;
            } else {
                smA[am * 32u + ak + 0u] = bfloat(0.0f);
                smA[am * 32u + ak + 1u] = bfloat(0.0f);
                smA[am * 32u + ak + 2u] = bfloat(0.0f);
                smA[am * 32u + ak + 3u] = bfloat(0.0f);
            }

            uint wr = tid >> 2;
            uint wq = tid & 3u;
            uint k_loc = wq * 8u;
            if (wr < n_rows) {
                uint blk = k0 >> 5;
                float bs = ue8m0_to_f32(w13_scale[(exp_base + long(wr)) * scale_D + long(blk)]);
                device const uchar* wp = w13_e4m3 + (exp_base + long(wr)) * long(p.D) + long(k0 + k_loc);
                for (uint b = 0u; b < 8u; ++b) {
                    smW[(k_loc + b) * 32u + wr] = bfloat(e4m3_to_f32(wp[b]) * bs);
                }
            } else {
                for (uint b = 0u; b < 8u; ++b) {
                    smW[(k_loc + b) * 32u + wr] = bfloat(0.0f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kk = 0u; kk < 32u; kk += 8u) {
                simdgroup_matrix<bfloat, 8, 8> ma, mb0, mb1;
                simdgroup_load(ma, smA + sg_r * 32u + kk, 32u);
                simdgroup_load(mb0, smW + kk * 32u + sg_c, 32u);
                simdgroup_load(mb1, smW + kk * 32u + sg_c + 8u, 32u);
                simdgroup_multiply_accumulate(c0, ma, mb0, c0);
                simdgroup_multiply_accumulate(c1, ma, mb1, c1);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        simdgroup_store(c0, smOut + sg_r * 32u + sg_c, 32u);
        simdgroup_store(c1, smOut + sg_r * 32u + sg_c + 8u, 32u);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint o = tid; o < 256u; o += 128u) {
            uint m = o >> 4;
            uint jl = o & 15u;
            if (m < n && (j0 + jl) < p.inter) {
                uint tk = sorted_tk[idx + m];
                float g = smOut[m * 32u + 2u * jl];
                float u = smOut[m * 32u + 2u * jl + 1u];
                float sig = 1.0f / (1.0f + exp(-g));
                float gc = safe_tanh(g / p.gate_cap) * p.gate_cap;
                float uc = safe_tanh(u / p.up_cap) * p.up_cap;
                float y = (gc * sig * uc) * topk_wts[tk];
                act_out[long(tk) * long(p.inter) + long(j0 + jl)] = float(bfloat(y));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void moe_mxfp8_w2_grouped_kernel(
    device const float* act_in     [[buffer(0)]],
    device const uint* offsets     [[buffer(1)]],
    device const uint* sorted_tk   [[buffer(2)]],
    device const uchar* w2_e4m3    [[buffer(3)]],
    device const uchar* w2_scale   [[buffer(4)]],
    device bfloat* w2_out          [[buffer(5)]],
    constant MoeParams& p          [[buffer(6)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]])
{
    uint expert = tgid.y;
    uint d0 = tgid.x * 32u;
    if (d0 >= p.D) return;
    uint s0 = offsets[expert];
    uint s1 = offsets[expert + 1u];
    if (s0 >= s1) return;

    long scale_I = long(p.inter / 32u);
    long exp_base = long(expert) * long(p.D) + long(d0);
    uint n_rows = min(32u, p.D - d0);

    threadgroup bfloat smA[16 * 32];
    threadgroup bfloat smW[32 * 32];
    threadgroup float smOut[16 * 32];

    uint sg_r = (sg_id >> 1) * 8u;
    uint sg_c = (sg_id & 1u) * 16u;

    for (uint idx = s0; idx < s1; idx += 16u) {
        uint n = min(16u, s1 - idx);
        simdgroup_matrix<float, 8, 8> c0(0.0f);
        simdgroup_matrix<float, 8, 8> c1(0.0f);

        for (uint k0 = 0u; k0 < p.inter; k0 += 32u) {
            uint am = tid >> 3;
            uint ak = (tid & 7u) * 4u;
            if (am < n) {
                uint tk = sorted_tk[idx + am];
                device const float4* ap = reinterpret_cast<device const float4*>(act_in + long(tk) * long(p.inter) + long(k0 + ak));
                float4 v = ap[0];
                smA[am * 32u + ak + 0u] = bfloat(v.x);
                smA[am * 32u + ak + 1u] = bfloat(v.y);
                smA[am * 32u + ak + 2u] = bfloat(v.z);
                smA[am * 32u + ak + 3u] = bfloat(v.w);
            } else {
                smA[am * 32u + ak + 0u] = bfloat(0.0f);
                smA[am * 32u + ak + 1u] = bfloat(0.0f);
                smA[am * 32u + ak + 2u] = bfloat(0.0f);
                smA[am * 32u + ak + 3u] = bfloat(0.0f);
            }

            uint wr = tid >> 2;
            uint wq = tid & 3u;
            uint k_loc = wq * 8u;
            if (wr < n_rows) {
                uint blk = k0 >> 5;
                float bs = ue8m0_to_f32(w2_scale[(exp_base + long(wr)) * scale_I + long(blk)]);
                device const uchar* wp = w2_e4m3 + (exp_base + long(wr)) * long(p.inter) + long(k0 + k_loc);
                for (uint b = 0u; b < 8u; ++b) {
                    smW[(k_loc + b) * 32u + wr] = bfloat(e4m3_to_f32(wp[b]) * bs);
                }
            } else {
                for (uint b = 0u; b < 8u; ++b) {
                    smW[(k_loc + b) * 32u + wr] = bfloat(0.0f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kk = 0u; kk < 32u; kk += 8u) {
                simdgroup_matrix<bfloat, 8, 8> ma, mb0, mb1;
                simdgroup_load(ma, smA + sg_r * 32u + kk, 32u);
                simdgroup_load(mb0, smW + kk * 32u + sg_c, 32u);
                simdgroup_load(mb1, smW + kk * 32u + sg_c + 8u, 32u);
                simdgroup_multiply_accumulate(c0, ma, mb0, c0);
                simdgroup_multiply_accumulate(c1, ma, mb1, c1);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        simdgroup_store(c0, smOut + sg_r * 32u + sg_c, 32u);
        simdgroup_store(c1, smOut + sg_r * 32u + sg_c + 8u, 32u);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint o = tid; o < 512u; o += 128u) {
            uint m = o >> 5;
            uint dl = o & 31u;
            if (m < n && (d0 + dl) < p.D) {
                uint tk = sorted_tk[idx + m];
                w2_out[long(tk) * long(p.D) + long(d0 + dl)] = bfloat(smOut[m * 32u + dl]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void moe_mxfp8_w13_kernel(
    device const bfloat* x         [[buffer(0)]],
    device const int* topk_ids     [[buffer(1)]],
    device const float* topk_wts   [[buffer(2)]],
    device const uchar* w13_e4m3   [[buffer(3)]],
    device const uchar* w13_scale  [[buffer(4)]],
    device float* act_out          [[buffer(5)]],
    constant MoeParams& p          [[buffer(6)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]])
{
    uint tk = tgid.y;
    uint t = tk / p.K;
    int expert = topk_ids[tk];
    float gamma = topk_wts[tk];

    uint j = tgid.x * 4u + sg_id;
    if (j >= p.inter) return;

    long row_g = long(2u * j);
    long row_u = long(2u * j + 1u);
    long scale_D = long(p.D / 32u);
    long exp_base = long(expert) * long(2u * p.inter);

    device const uchar* wg = w13_e4m3 + (exp_base + row_g) * long(p.D);
    device const uchar* wu = w13_e4m3 + (exp_base + row_u) * long(p.D);
    device const uchar* sg = w13_scale + (exp_base + row_g) * scale_D;
    device const uchar* su = w13_scale + (exp_base + row_u) * scale_D;
    device const bfloat* xr = x + long(t) * long(p.D);

    float acc_g = 0.0f, acc_u = 0.0f;
    for (uint blk = lane; blk < uint(scale_D); blk += 32u) {
        float bsg = ue8m0_to_f32(sg[blk]);
        float bsu = ue8m0_to_f32(su[blk]);
        device const uchar* wgb = wg + long(blk) * 32;
        device const uchar* wub = wu + long(blk) * 32;
        device const bfloat* xblk = xr + long(blk) * 32;
        float lg = 0.0f, lu = 0.0f;
        for (uint n = 0; n < 32u; ++n) {
            float xf = float(xblk[n]);
            lg = fma(e4m3_to_f32(wgb[n]), xf, lg);
            lu = fma(e4m3_to_f32(wub[n]), xf, lu);
        }
        acc_g = fma(lg, bsg, acc_g);
        acc_u = fma(lu, bsu, acc_u);
    }
    acc_g = simd_sum(acc_g);
    acc_u = simd_sum(acc_u);
    if (lane == 0u) {
        float sig = 1.0f / (1.0f + exp(-acc_g));
        float g_cap = safe_tanh(acc_g / p.gate_cap) * p.gate_cap;
        float u_cap = safe_tanh(acc_u / p.up_cap) * p.up_cap;
        float y = (g_cap * sig * u_cap) * gamma;
        act_out[long(tk) * long(p.inter) + long(j)] = float(bfloat(y));
    }
}

kernel void moe_mxfp8_w2_kernel(
    device const float* act_in     [[buffer(0)]],
    device const int* topk_ids     [[buffer(1)]],
    device const uchar* w2_e4m3    [[buffer(2)]],
    device const uchar* w2_scale   [[buffer(3)]],
    device bfloat* out             [[buffer(4)]],
    constant MoeParams& p          [[buffer(5)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]])
{
    uint t = tgid.y;
    uint d = tgid.x * 4u + sg_id;
    if (d >= p.D) return;

    long scale_I = long(p.inter / 32u);
    float total = 0.0f;
    for (uint k = 0; k < p.K; ++k) {
        uint tk = t * p.K + k;
        int expert = topk_ids[tk];
        long row_idx = long(expert) * long(p.D) + long(d);
        device const uchar* w2r = w2_e4m3 + row_idx * long(p.inter);
        device const uchar* s2r = w2_scale + row_idx * scale_I;
        device const float* ar = act_in + long(tk) * long(p.inter);

        float acc = 0.0f;
        for (uint blk = lane; blk < uint(scale_I); blk += 32u) {
            float bs = ue8m0_to_f32(s2r[blk]);
            device const uchar* wb = w2r + long(blk) * 32;
            device const float* ab = ar + long(blk) * 32;
            float loc = 0.0f;
            for (uint n = 0; n < 32u; ++n) {
                loc = fma(e4m3_to_f32(wb[n]), ab[n], loc);
            }
            acc = fma(loc, bs, acc);
        }
        acc = simd_sum(acc);
        if (lane == 0u) total += acc;
    }
    if (lane == 0u) {
        out[long(t) * long(p.D) + long(d)] = bfloat(total);
    }
}

struct GemvParams {
    uint M;
    uint N;
    uint K;
    uint out_fp32;
};

kernel void bf16_gemv_kernel(
    device const bfloat* X   [[buffer(0)]],
    device const bfloat* W   [[buffer(1)]],
    device bfloat* Y_bf16    [[buffer(2)]],
    device float* Y_fp32     [[buffer(3)]],
    constant GemvParams& p   [[buffer(4)]],
    uint2 tgid               [[threadgroup_position_in_grid]],
    uint sg_id               [[simdgroup_index_in_threadgroup]],
    uint lane                [[thread_index_in_simdgroup]])
{
    uint m = tgid.y;
    uint n = tgid.x * 4u + sg_id;
    if (m >= p.M || n >= p.N) return;

    device const bfloat* xr = X + long(m) * long(p.K);
    device const bfloat* wr = W + long(n) * long(p.K);
    device const bfloat4* xr4 = reinterpret_cast<device const bfloat4*>(xr);
    device const bfloat4* wr4 = reinterpret_cast<device const bfloat4*>(wr);
    float acc = 0.0f;
    uint k4 = p.K >> 2;
    for (uint i = lane; i < k4; i += 32u) {
        float4 xv = float4(xr4[i]);
        float4 wv = float4(wr4[i]);
        acc = fma(wv.x, xv.x, fma(wv.y, xv.y, fma(wv.z, xv.z, fma(wv.w, xv.w, acc))));
    }
    for (uint k = (k4 << 2) + lane; k < p.K; k += 32u) {
        acc = fma(float(wr[k]), float(xr[k]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        long idx = long(m) * long(p.N) + long(n);
        if (p.out_fp32 != 0u) Y_fp32[idx] = acc;
        else Y_bf16[idx] = bfloat(acc);
    }
}

kernel void bf16_gemv_1sg_kernel(
    device const bfloat* X   [[buffer(0)]],
    device const bfloat* W   [[buffer(1)]],
    device bfloat* Y_bf16    [[buffer(2)]],
    device float* Y_fp32     [[buffer(3)]],
    constant GemvParams& p   [[buffer(4)]],
    uint2 tgid               [[threadgroup_position_in_grid]],
    uint lane                [[thread_index_in_simdgroup]])
{
    uint m = tgid.y;
    uint n = tgid.x;
    if (m >= p.M || n >= p.N) return;

    device const bfloat4* xr4 = reinterpret_cast<device const bfloat4*>(X + long(m) * long(p.K));
    device const bfloat4* wr4 = reinterpret_cast<device const bfloat4*>(W + long(n) * long(p.K));
    float acc = 0.0f;
    uint k4 = p.K >> 2;
    for (uint i = lane; i < k4; i += 32u) {
        float4 xv = float4(xr4[i]);
        float4 wv = float4(wr4[i]);
        acc = fma(wv.x, xv.x, fma(wv.y, xv.y, fma(wv.z, xv.z, fma(wv.w, xv.w, acc))));
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        long idx = long(m) * long(p.N) + long(n);
        if (p.out_fp32 != 0u) Y_fp32[idx] = acc;
        else Y_bf16[idx] = bfloat(acc);
    }
}

struct QuantParams {
    uint N;
    uint K;
};

kernel void quantize_bf16_to_fp8_rowwise_kernel(
    device const bfloat* W_bf16 [[buffer(0)]],
    device uchar* W_fp8         [[buffer(1)]],
    device float* W_scale       [[buffer(2)]],
    constant QuantParams& p     [[buffer(3)]],
    uint tgid                   [[threadgroup_position_in_grid]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint tg_size                [[threads_per_threadgroup]])
{
    uint n = tgid;
    if (n >= p.N) return;
    device const bfloat* src = W_bf16 + long(n) * long(p.K);
    device uchar* dst = W_fp8 + long(n) * long(p.K);

    threadgroup float sm[32];
    float amax = 0.0f;
    for (uint k = tid; k < p.K; k += tg_size) {
        amax = max(amax, fabs(float(src[k])));
    }
    amax = tg_reduce_max(amax, tid, tg_size, sm);
    float sc = (amax > 0.0f) ? (amax / 448.0f) : 1.0f;
    float inv_sc = 1.0f / sc;
    if (tid == 0u) W_scale[n] = sc;
    for (uint k = tid; k < p.K; k += tg_size) {
        dst[k] = f32_to_e4m3(float(src[k]) * inv_sc);
    }
}

kernel void fp8_gemv_kernel(
    device const bfloat* X   [[buffer(0)]],
    device const uchar* W8   [[buffer(1)]],
    device const float* Wsc  [[buffer(2)]],
    device bfloat* Y_bf16    [[buffer(3)]],
    device float* Y_fp32     [[buffer(4)]],
    constant GemvParams& p   [[buffer(5)]],
    uint2 tgid               [[threadgroup_position_in_grid]],
    uint sg_id               [[simdgroup_index_in_threadgroup]],
    uint lane                [[thread_index_in_simdgroup]])
{
    uint m = tgid.y;
    uint n = tgid.x * 4u + sg_id;
    if (m >= p.M || n >= p.N) return;

    device const bfloat4* xr4 = reinterpret_cast<device const bfloat4*>(X + long(m) * long(p.K));
    device const uchar4* wr4 = reinterpret_cast<device const uchar4*>(W8 + long(n) * long(p.K));
    float acc = 0.0f;
    uint k4 = p.K >> 2;
    for (uint i = lane; i < k4; i += 32u) {
        float4 xv = float4(xr4[i]);
        uchar4 wv = wr4[i];
        acc = fma(e4m3_to_f32(wv.x), xv.x,
              fma(e4m3_to_f32(wv.y), xv.y,
              fma(e4m3_to_f32(wv.z), xv.z,
              fma(e4m3_to_f32(wv.w), xv.w, acc))));
    }
    acc = simd_sum(acc) * Wsc[n];
    if (lane == 0u) {
        long idx = long(m) * long(p.N) + long(n);
        if (p.out_fp32 != 0u) Y_fp32[idx] = acc;
        else Y_bf16[idx] = bfloat(acc);
    }
}

kernel void bf16_gemm_tiled_kernel(
    device const bfloat* X  [[buffer(0)]],
    device const bfloat* W  [[buffer(1)]],
    device bfloat* Y        [[buffer(2)]],
    constant GemvParams& p  [[buffer(3)]],
    uint2 tgid              [[threadgroup_position_in_grid]],
    uint sg_id              [[simdgroup_index_in_threadgroup]],
    uint tid                [[thread_index_in_threadgroup]])
{
    uint m0 = tgid.y * 32u;
    uint n0 = tgid.x * 32u;
    if (m0 >= p.M || n0 >= p.N) return;

    threadgroup bfloat smX[32 * 32];
    threadgroup bfloat smW[32 * 32];
    threadgroup float smOut[32 * 32];

    uint sg_r = (sg_id >> 1) * 16u;
    uint sg_c = (sg_id & 1u) * 16u;

    simdgroup_matrix<float, 8, 8> acc[2][2];
    for (uint i = 0; i < 2u; ++i)
        for (uint j = 0; j < 2u; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < p.K; k0 += 32u) {
        for (uint idx = tid; idx < 1024u; idx += 128u) {
            uint r = idx >> 5;
            uint c = idx & 31u;
            uint gm = m0 + r;
            uint gn = n0 + r;
            uint gk = k0 + c;
            smX[idx] = (gm < p.M && gk < p.K) ? X[long(gm) * long(p.K) + long(gk)] : bfloat(0.0f);
            smW[c * 32u + r] = (gn < p.N && gk < p.K) ? W[long(gn) * long(p.K) + long(gk)] : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_matrix<bfloat, 8, 8> ma0, ma1, mb0, mb1;
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

    for (uint i = 0; i < 2u; ++i) {
        for (uint j = 0; j < 2u; ++j) {
            simdgroup_store(acc[i][j], smOut + (sg_r + i * 8u) * 32u + (sg_c + j * 8u), 32u);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < 1024u; idx += 128u) {
        uint r = idx >> 5;
        uint c = idx & 31u;
        uint gm = m0 + r;
        uint gn = n0 + c;
        if (gm < p.M && gn < p.N) {
            Y[long(gm) * long(p.N) + long(gn)] = bfloat(smOut[idx]);
        }
    }
}

struct ArgmaxParams {
    uint vocab_size;
    float softcap;
};

kernel void softcap_argmax_kernel(
    device float* logits      [[buffer(0)]],
    device int* out_token     [[buffer(1)]],
    constant ArgmaxParams& p  [[buffer(2)]],
    uint tid                  [[thread_index_in_threadgroup]],
    uint tg_size              [[threads_per_threadgroup]])
{
    float best_v = -1e30f;
    int best_i = 0;
    float inv_cap = 1.0f / p.softcap;
    for (uint i = tid; i < p.vocab_size; i += tg_size) {
        float v = safe_tanh(logits[i] * inv_cap) * p.softcap;
        logits[i] = v;
        if (v > best_v) {
            best_v = v;
            best_i = int(i);
        }
    }
    threadgroup float sm_v[32];
    threadgroup int sm_i[32];
    uint lane = tid & 31u;
    uint wid = tid >> 5;
    for (uint off = 16u; off > 0u; off >>= 1) {
        float ov = simd_shuffle_down(best_v, off);
        int oi = simd_shuffle_down(best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }
    if (lane == 0u) {
        sm_v[wid] = best_v;
        sm_i[wid] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (wid == 0u) {
        uint nw = (tg_size + 31u) >> 5;
        best_v = (lane < nw) ? sm_v[lane] : -1e30f;
        best_i = (lane < nw) ? sm_i[lane] : 0;
        for (uint off = 16u; off > 0u; off >>= 1) {
            float ov = simd_shuffle_down(best_v, off);
            int oi = simd_shuffle_down(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (lane == 0u) out_token[0] = best_i;
    }
}

kernel void fp8_small_gemm_kernel(
    device const bfloat* X   [[buffer(0)]],
    device const uchar* W8   [[buffer(1)]],
    device const float* Wsc  [[buffer(2)]],
    device bfloat* Y_bf16    [[buffer(3)]],
    device float* Y_fp32     [[buffer(4)]],
    constant GemvParams& p   [[buffer(5)]],
    uint tgid                [[threadgroup_position_in_grid]],
    uint sg_id               [[simdgroup_index_in_threadgroup]],
    uint lane                [[thread_index_in_simdgroup]])
{
    uint n0 = tgid * 8u + sg_id * 2u;
    if (n0 >= p.N) return;
    uint n1 = n0 + 1u;
    bool has_n1 = (n1 < p.N);

    device const uchar4* wrA = reinterpret_cast<device const uchar4*>(W8 + long(n0) * long(p.K));
    device const uchar4* wrB = has_n1 ? reinterpret_cast<device const uchar4*>(W8 + long(n1) * long(p.K)) : wrA;
    uint k4 = p.K >> 2;
    device const bfloat4* x0 = reinterpret_cast<device const bfloat4*>(X + 0L * long(p.K));
    device const bfloat4* x1 = reinterpret_cast<device const bfloat4*>(X + 1L * long(p.K));
    device const bfloat4* x2 = reinterpret_cast<device const bfloat4*>(X + 2L * long(p.K));
    device const bfloat4* x3 = reinterpret_cast<device const bfloat4*>(X + 3L * long(p.K));
    device const bfloat4* x4 = reinterpret_cast<device const bfloat4*>(X + 4L * long(p.K));
    device const bfloat4* x5 = reinterpret_cast<device const bfloat4*>(X + 5L * long(p.K));
    device const bfloat4* x6 = reinterpret_cast<device const bfloat4*>(X + 6L * long(p.K));
    device const bfloat4* x7 = reinterpret_cast<device const bfloat4*>(X + 7L * long(p.K));

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
    float b4 = 0.0f, b5 = 0.0f, b6 = 0.0f, b7 = 0.0f;
    uint M = p.M;

    if (M == 8u) {
        for (uint i = lane; i < k4; i += 32u) {
            uchar4 wva = wrA[i];
            uchar4 wvb = wrB[i];
            float4 fa = float4(kE4M3Lut[wva.x], kE4M3Lut[wva.y], kE4M3Lut[wva.z], kE4M3Lut[wva.w]);
            float4 fb = float4(kE4M3Lut[wvb.x], kE4M3Lut[wvb.y], kE4M3Lut[wvb.z], kE4M3Lut[wvb.w]);
            float4 v0 = float4(x0[i]); a0 += dot(fa, v0); b0 += dot(fb, v0);
            float4 v1 = float4(x1[i]); a1 += dot(fa, v1); b1 += dot(fb, v1);
            float4 v2 = float4(x2[i]); a2 += dot(fa, v2); b2 += dot(fb, v2);
            float4 v3 = float4(x3[i]); a3 += dot(fa, v3); b3 += dot(fb, v3);
            float4 v4 = float4(x4[i]); a4 += dot(fa, v4); b4 += dot(fb, v4);
            float4 v5 = float4(x5[i]); a5 += dot(fa, v5); b5 += dot(fb, v5);
            float4 v6 = float4(x6[i]); a6 += dot(fa, v6); b6 += dot(fb, v6);
            float4 v7 = float4(x7[i]); a7 += dot(fa, v7); b7 += dot(fb, v7);
        }
    } else if (M == 7u) {
        for (uint i = lane; i < k4; i += 32u) {
            uchar4 wva = wrA[i];
            uchar4 wvb = wrB[i];
            float4 fa = float4(kE4M3Lut[wva.x], kE4M3Lut[wva.y], kE4M3Lut[wva.z], kE4M3Lut[wva.w]);
            float4 fb = float4(kE4M3Lut[wvb.x], kE4M3Lut[wvb.y], kE4M3Lut[wvb.z], kE4M3Lut[wvb.w]);
            float4 v0 = float4(x0[i]); a0 += dot(fa, v0); b0 += dot(fb, v0);
            float4 v1 = float4(x1[i]); a1 += dot(fa, v1); b1 += dot(fb, v1);
            float4 v2 = float4(x2[i]); a2 += dot(fa, v2); b2 += dot(fb, v2);
            float4 v3 = float4(x3[i]); a3 += dot(fa, v3); b3 += dot(fb, v3);
            float4 v4 = float4(x4[i]); a4 += dot(fa, v4); b4 += dot(fb, v4);
            float4 v5 = float4(x5[i]); a5 += dot(fa, v5); b5 += dot(fb, v5);
            float4 v6 = float4(x6[i]); a6 += dot(fa, v6); b6 += dot(fb, v6);
        }
    } else if (M == 6u) {
        for (uint i = lane; i < k4; i += 32u) {
            uchar4 wva = wrA[i];
            uchar4 wvb = wrB[i];
            float4 fa = float4(kE4M3Lut[wva.x], kE4M3Lut[wva.y], kE4M3Lut[wva.z], kE4M3Lut[wva.w]);
            float4 fb = float4(kE4M3Lut[wvb.x], kE4M3Lut[wvb.y], kE4M3Lut[wvb.z], kE4M3Lut[wvb.w]);
            float4 v0 = float4(x0[i]); a0 += dot(fa, v0); b0 += dot(fb, v0);
            float4 v1 = float4(x1[i]); a1 += dot(fa, v1); b1 += dot(fb, v1);
            float4 v2 = float4(x2[i]); a2 += dot(fa, v2); b2 += dot(fb, v2);
            float4 v3 = float4(x3[i]); a3 += dot(fa, v3); b3 += dot(fb, v3);
            float4 v4 = float4(x4[i]); a4 += dot(fa, v4); b4 += dot(fb, v4);
            float4 v5 = float4(x5[i]); a5 += dot(fa, v5); b5 += dot(fb, v5);
        }
    } else {
        for (uint i = lane; i < k4; i += 32u) {
            uchar4 wva = wrA[i];
            uchar4 wvb = wrB[i];
            float4 fa = float4(kE4M3Lut[wva.x], kE4M3Lut[wva.y], kE4M3Lut[wva.z], kE4M3Lut[wva.w]);
            float4 fb = float4(kE4M3Lut[wvb.x], kE4M3Lut[wvb.y], kE4M3Lut[wvb.z], kE4M3Lut[wvb.w]);
            { float4 v0 = float4(x0[i]); a0 += dot(fa, v0); b0 += dot(fb, v0); }
            if (M > 1u) { float4 v1 = float4(x1[i]); a1 += dot(fa, v1); b1 += dot(fb, v1); }
            if (M > 2u) { float4 v2 = float4(x2[i]); a2 += dot(fa, v2); b2 += dot(fb, v2); }
            if (M > 3u) { float4 v3 = float4(x3[i]); a3 += dot(fa, v3); b3 += dot(fb, v3); }
            if (M > 4u) { float4 v4 = float4(x4[i]); a4 += dot(fa, v4); b4 += dot(fb, v4); }
            if (M > 5u) { float4 v5 = float4(x5[i]); a5 += dot(fa, v5); b5 += dot(fb, v5); }
            if (M > 6u) { float4 v6 = float4(x6[i]); a6 += dot(fa, v6); b6 += dot(fb, v6); }
            if (M > 7u) { float4 v7 = float4(x7[i]); a7 += dot(fa, v7); b7 += dot(fb, v7); }
        }
    }

    float scA = Wsc[n0];
    float va0 = simd_sum(a0) * scA;
    float va1 = (M > 1u) ? (simd_sum(a1) * scA) : 0.0f;
    float va2 = (M > 2u) ? (simd_sum(a2) * scA) : 0.0f;
    float va3 = (M > 3u) ? (simd_sum(a3) * scA) : 0.0f;
    float va4 = (M > 4u) ? (simd_sum(a4) * scA) : 0.0f;
    float va5 = (M > 5u) ? (simd_sum(a5) * scA) : 0.0f;
    float va6 = (M > 6u) ? (simd_sum(a6) * scA) : 0.0f;
    float va7 = (M > 7u) ? (simd_sum(a7) * scA) : 0.0f;

    float scB = has_n1 ? Wsc[n1] : 0.0f;
    float vb0 = has_n1 ? (simd_sum(b0) * scB) : 0.0f;
    float vb1 = (has_n1 && M > 1u) ? (simd_sum(b1) * scB) : 0.0f;
    float vb2 = (has_n1 && M > 2u) ? (simd_sum(b2) * scB) : 0.0f;
    float vb3 = (has_n1 && M > 3u) ? (simd_sum(b3) * scB) : 0.0f;
    float vb4 = (has_n1 && M > 4u) ? (simd_sum(b4) * scB) : 0.0f;
    float vb5 = (has_n1 && M > 5u) ? (simd_sum(b5) * scB) : 0.0f;
    float vb6 = (has_n1 && M > 6u) ? (simd_sum(b6) * scB) : 0.0f;
    float vb7 = (has_n1 && M > 7u) ? (simd_sum(b7) * scB) : 0.0f;

    if (lane == 0u) {
        if (p.out_fp32 != 0u) {
            Y_fp32[0L * long(p.N) + long(n0)] = va0;
            if (M > 1u) Y_fp32[1L * long(p.N) + long(n0)] = va1;
            if (M > 2u) Y_fp32[2L * long(p.N) + long(n0)] = va2;
            if (M > 3u) Y_fp32[3L * long(p.N) + long(n0)] = va3;
            if (M > 4u) Y_fp32[4L * long(p.N) + long(n0)] = va4;
            if (M > 5u) Y_fp32[5L * long(p.N) + long(n0)] = va5;
            if (M > 6u) Y_fp32[6L * long(p.N) + long(n0)] = va6;
            if (M > 7u) Y_fp32[7L * long(p.N) + long(n0)] = va7;
            if (has_n1) {
                Y_fp32[0L * long(p.N) + long(n1)] = vb0;
                if (M > 1u) Y_fp32[1L * long(p.N) + long(n1)] = vb1;
                if (M > 2u) Y_fp32[2L * long(p.N) + long(n1)] = vb2;
                if (M > 3u) Y_fp32[3L * long(p.N) + long(n1)] = vb3;
                if (M > 4u) Y_fp32[4L * long(p.N) + long(n1)] = vb4;
                if (M > 5u) Y_fp32[5L * long(p.N) + long(n1)] = vb5;
                if (M > 6u) Y_fp32[6L * long(p.N) + long(n1)] = vb6;
                if (M > 7u) Y_fp32[7L * long(p.N) + long(n1)] = vb7;
            }
        } else {
            Y_bf16[0L * long(p.N) + long(n0)] = bfloat(va0);
            if (M > 1u) Y_bf16[1L * long(p.N) + long(n0)] = bfloat(va1);
            if (M > 2u) Y_bf16[2L * long(p.N) + long(n0)] = bfloat(va2);
            if (M > 3u) Y_bf16[3L * long(p.N) + long(n0)] = bfloat(va3);
            if (M > 4u) Y_bf16[4L * long(p.N) + long(n0)] = bfloat(va4);
            if (M > 5u) Y_bf16[5L * long(p.N) + long(n0)] = bfloat(va5);
            if (M > 6u) Y_bf16[6L * long(p.N) + long(n0)] = bfloat(va6);
            if (M > 7u) Y_bf16[7L * long(p.N) + long(n0)] = bfloat(va7);
            if (has_n1) {
                Y_bf16[0L * long(p.N) + long(n1)] = bfloat(vb0);
                if (M > 1u) Y_bf16[1L * long(p.N) + long(n1)] = bfloat(vb1);
                if (M > 2u) Y_bf16[2L * long(p.N) + long(n1)] = bfloat(vb2);
                if (M > 3u) Y_bf16[3L * long(p.N) + long(n1)] = bfloat(vb3);
                if (M > 4u) Y_bf16[4L * long(p.N) + long(n1)] = bfloat(vb4);
                if (M > 5u) Y_bf16[5L * long(p.N) + long(n1)] = bfloat(vb5);
                if (M > 6u) Y_bf16[6L * long(p.N) + long(n1)] = bfloat(vb6);
                if (M > 7u) Y_bf16[7L * long(p.N) + long(n1)] = bfloat(vb7);
            }
        }
    }
}

struct BatchArgParams {
    uint M;
    uint vocab_size;
};

kernel void batched_argmax_kernel(
    device const float* logits [[buffer(0)]],
    device int* out_tokens     [[buffer(1)]],
    constant BatchArgParams& p [[buffer(2)]],
    uint tgid                  [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]])
{
    uint m = tgid;
    if (m >= p.M) return;
    device const float* row = logits + long(m) * long(p.vocab_size);
    float best_v = -1e30f;
    int best_i = 0;
    for (uint i = tid; i < p.vocab_size; i += tg_size) {
        float v = row[i];
        if (v > best_v) {
            best_v = v;
            best_i = int(i);
        }
    }
    threadgroup float sm_v[32];
    threadgroup int sm_i[32];
    uint lane = tid & 31u;
    uint wid = tid >> 5;
    for (uint off = 16u; off > 0u; off >>= 1) {
        float ov = simd_shuffle_down(best_v, off);
        int oi = simd_shuffle_down(best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }
    if (lane == 0u) {
        sm_v[wid] = best_v;
        sm_i[wid] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (wid == 0u) {
        uint nw = (tg_size + 31u) >> 5;
        best_v = (lane < nw) ? sm_v[lane] : -1e30f;
        best_i = (lane < nw) ? sm_i[lane] : 0;
        for (uint off = 16u; off > 0u; off >>= 1) {
            float ov = simd_shuffle_down(best_v, off);
            int oi = simd_shuffle_down(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (lane == 0u) out_tokens[m] = best_i;
    }
}

struct DflareMixParams {
    uint aux_stride;
    uint T;
    uint D;
    float eps;
};

kernel void dflare_mix_rmsnorm_kernel(
    device const bfloat* aux   [[buffer(0)]],
    device const float* alpha  [[buffer(1)]],
    device const bfloat* w     [[buffer(2)]],
    device bfloat* out         [[buffer(3)]],
    constant DflareMixParams& p [[buffer(4)]],
    uint tgid                  [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]])
{
    uint t = tgid;
    if (t >= p.T) return;
    float a0 = alpha[0], a1 = alpha[1], a2 = alpha[2], a3 = alpha[3];
    float reg[12];
    float sum_sq = 0.0f;
    uint k = 0u;
    for (uint d = tid; d < p.D; d += tg_size) {
        long off = long(t) * long(p.D) + long(d);
        float v = a0 * float(aux[0L * long(p.aux_stride) + off])
                + a1 * float(aux[1L * long(p.aux_stride) + off])
                + a2 * float(aux[2L * long(p.aux_stride) + off])
                + a3 * float(aux[3L * long(p.aux_stride) + off]);
        reg[k++] = v;
        sum_sq += v * v;
    }
    threadgroup float sm[32];
    sum_sq = tg_reduce_sum(sum_sq, tid, tg_size, sm);
    float rstd = rsqrt(sum_sq / float(p.D) + p.eps);
    k = 0u;
    for (uint d = tid; d < p.D; d += tg_size) {
        out[long(t) * long(p.D) + long(d)] = bfloat(reg[k++] * rstd * float(w[d]));
    }
}

struct QkRopeParams {
    uint start_pos;
    uint T;
    uint n_heads;
    uint head_dim;
    float rope_theta;
    float eps;
};

kernel void qk_norm_rope_kernel(
    device bfloat* x           [[buffer(0)]],
    device const bfloat* nw    [[buffer(1)]],
    constant QkRopeParams& p   [[buffer(2)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]])
{
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.n_heads || t >= p.T) return;
    uint half_d = p.head_dim >> 1;
    uint i = tid;
    if (i >= half_d) return;

    device bfloat* row = x + (long(t) * long(p.n_heads) + long(h)) * long(p.head_dim);
    float x1 = float(row[i]);
    float x2 = float(row[half_d + i]);

    threadgroup float sm[32];
    float sq = tg_reduce_sum(x1 * x1 + x2 * x2, tid, half_d, sm);
    float inv = rsqrt(sq / float(p.head_dim) + p.eps);
    x1 = float(bfloat(x1 * inv * float(nw[i])));
    x2 = float(bfloat(x2 * inv * float(nw[half_d + i])));

    float pos = float(p.start_pos + t);
    float inv_freq = pow(p.rope_theta, -2.0f * float(i) / float(p.head_dim));
    float ang = pos * inv_freq;
    float c = cos(ang), s = sin(ang);
    row[i] = bfloat(x1 * c - x2 * s);
    row[half_d + i] = bfloat(x2 * c + x1 * s);
}

struct DraftWriteKvParams {
    uint start_pos;
    uint T;
    uint cap;
    uint is_local;
    uint kv_dim;
};

kernel void draft_write_kv_kernel(
    device const bfloat* k_in  [[buffer(0)]],
    device const bfloat* v_in  [[buffer(1)]],
    device bfloat* k_cache     [[buffer(2)]],
    device bfloat* v_cache     [[buffer(3)]],
    constant DraftWriteKvParams& p [[buffer(4)]],
    uint2 gid                  [[thread_position_in_grid]])
{
    uint d = gid.x;
    uint t = gid.y;
    if (t >= p.T || d >= p.kv_dim) return;
    uint pos = p.start_pos + t;
    uint slot = (p.is_local != 0u) ? (pos % p.cap) : pos;
    k_cache[long(slot) * long(p.kv_dim) + long(d)] = k_in[long(t) * long(p.kv_dim) + long(d)];
    v_cache[long(slot) * long(p.kv_dim) + long(d)] = v_in[long(t) * long(p.kv_dim) + long(d)];
}

struct DraftAttnParams {
    uint anchor_pos;
    uint B;
    uint cap;
    uint is_local;
    uint window_size;
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
};

kernel void draft_block_attn_kernel(
    device const bfloat* Q       [[buffer(0)]],
    device const bfloat* k_cache [[buffer(1)]],
    device const bfloat* v_cache [[buffer(2)]],
    device bfloat* out           [[buffer(3)]],
    constant DraftAttnParams& p  [[buffer(4)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint sg_id                   [[simdgroup_index_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]])
{
    uint h = tgid.x;
    uint b = tgid.y;
    if (h >= p.n_heads || b >= p.B) return;
    uint kv_h = h / (p.n_heads / p.n_kv_heads);

    uint k_end = p.anchor_pos + p.B;
    uint k_start = (p.is_local != 0u && p.anchor_pos > p.window_size) ? (p.anchor_pos - p.window_size) : 0u;

    device const bfloat* q_ptr = Q + (long(b) * long(p.n_heads) + long(h)) * long(p.head_dim);
    float qv[4];
    for (uint i = 0; i < 4u; ++i) qv[i] = float(q_ptr[lane + i * 32u]);
    float inv_sqrt_d = rsqrt(float(p.head_dim));

    float m_val = -1e30f;
    float l_val = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    long kv_stride = long(p.n_kv_heads) * long(p.head_dim);

    for (uint kp = k_start + sg_id; kp < k_end; kp += 4u) {
        uint slot = (p.is_local != 0u) ? (kp % p.cap) : kp;
        device const bfloat* k_ptr = k_cache + long(slot) * kv_stride + long(kv_h) * long(p.head_dim);
        device const bfloat* v_ptr = v_cache + long(slot) * kv_stride + long(kv_h) * long(p.head_dim);
        float dot = 0.0f;
        for (uint i = 0; i < 4u; ++i) dot = fma(qv[i], float(k_ptr[lane + i * 32u]), dot);
        dot = simd_sum(dot) * inv_sqrt_d;
        float m_new = max(m_val, dot);
        float alpha = exp(m_val - m_new);
        float pr = exp(dot - m_new);
        l_val = l_val * alpha + pr;
        for (uint i = 0; i < 4u; ++i) acc[i] = acc[i] * alpha + pr * float(v_ptr[lane + i * 32u]);
        m_val = m_new;
    }

    threadgroup float sm_m[4];
    threadgroup float sm_l[4];
    threadgroup float sm_acc[4][128];
    if (lane == 0u) { sm_m[sg_id] = m_val; sm_l[sg_id] = l_val; }
    for (uint i = 0; i < 4u; ++i) sm_acc[sg_id][lane + i * 32u] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        float gm = -1e30f;
        for (uint w = 0; w < 4u; ++w) if (sm_l[w] > 0.0f) gm = max(gm, sm_m[w]);
        float gl = 0.0f;
        float gacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint w = 0; w < 4u; ++w) {
            if (sm_l[w] > 0.0f) {
                float sc = exp(sm_m[w] - gm);
                gl += sm_l[w] * sc;
                for (uint i = 0; i < 4u; ++i) gacc[i] += sm_acc[w][lane + i * 32u] * sc;
            }
        }
        float inv_l = 1.0f / gl;
        device bfloat* o_ptr = out + (long(b) * long(p.n_heads) + long(h)) * long(p.head_dim);
        for (uint i = 0; i < 4u; ++i) o_ptr[lane + i * 32u] = bfloat(gacc[i] * inv_l);
    }
}

kernel void add_rmsnorm_kernel(
    device const bfloat* x [[buffer(0)]],
    device bfloat* res     [[buffer(1)]],
    device const bfloat* w [[buffer(2)]],
    device bfloat* out     [[buffer(3)]],
    constant RmsParams& p  [[buffer(4)]],
    uint tgid              [[threadgroup_position_in_grid]],
    uint tid               [[thread_index_in_threadgroup]],
    uint tg_size           [[threads_per_threadgroup]])
{
    uint t = tgid;
    if (t >= p.T) return;
    long off = long(t) * long(p.D);
    float reg[12];
    float sum_sq = 0.0f;
    uint k = 0u;
    for (uint d = tid; d < p.D; d += tg_size) {
        float v = float(res[off + long(d)]) + float(x[off + long(d)]);
        res[off + long(d)] = bfloat(v);
        reg[k++] = v;
        sum_sq += v * v;
    }
    threadgroup float sm[32];
    sum_sq = tg_reduce_sum(sum_sq, tid, tg_size, sm);
    float rstd = rsqrt(sum_sq / float(p.D) + p.eps);
    k = 0u;
    for (uint d = tid; d < p.D; d += tg_size) {
        out[off + long(d)] = bfloat(reg[k++] * rstd * float(w[d]));
    }
}

kernel void silu_mul_kernel(
    device bfloat* g       [[buffer(0)]],
    device const bfloat* u [[buffer(1)]],
    constant uint& N       [[buffer(2)]],
    uint gid               [[thread_position_in_grid]])
{
    if (gid < N) {
        float gv = float(g[gid]);
        float uv = float(u[gid]);
        g[gid] = bfloat((gv / (1.0f + exp(-gv))) * uv);
    }
}

kernel void embed_draft_block_kernel(
    device const int* anchor_tok  [[buffer(0)]],
    device const bfloat* embed_w  [[buffer(1)]],
    device const bfloat* norm_w   [[buffer(2)]],
    device const bfloat* mask_emb [[buffer(3)]],
    device bfloat* out            [[buffer(4)]],
    constant RmsParams& p         [[buffer(5)]],
    uint tgid                     [[threadgroup_position_in_grid]],
    uint tid                      [[thread_index_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]])
{
    uint b = tgid;
    if (b >= p.T) return;
    device bfloat* dst = out + long(b) * long(p.D);
    if (b > 0u) {
        for (uint d = tid; d < p.D; d += tg_size) dst[d] = mask_emb[d];
        return;
    }
    long tok = long(anchor_tok[0]);
    device const bfloat* row = embed_w + tok * long(p.D);
    threadgroup float sm[32];
    float sum_sq = 0.0f;
    for (uint d = tid; d < p.D; d += tg_size) {
        float v = float(row[d]);
        sum_sq += v * v;
    }
    sum_sq = tg_reduce_sum(sum_sq, tid, tg_size, sm);
    float rstd = rsqrt(sum_sq / float(p.D) + p.eps);
    for (uint d = tid; d < p.D; d += tg_size) {
        dst[d] = bfloat(float(row[d]) * rstd * float(norm_w[d]));
    }
}

struct MarkovInParams {
    uint rank;
    uint D;
};

kernel void assemble_markov_in_dev_kernel(
    device const bfloat* state     [[buffer(0)]],
    device const bfloat* markov_w1 [[buffer(1)]],
    device const int* prev_tok_ptr [[buffer(2)]],
    device const bfloat* draft_h   [[buffer(3)]],
    device const bfloat* f_rnn     [[buffer(4)]],
    device bfloat* out             [[buffer(5)]],
    constant MarkovInParams& p     [[buffer(6)]],
    uint i                         [[thread_position_in_grid]])
{
    uint total = 2u * p.rank + 2u * p.D;
    if (i >= total) return;
    long prev_tok = long(prev_tok_ptr[0]);
    if (i < p.rank) out[i] = state[i];
    else if (i < 2u * p.rank) out[i] = markov_w1[prev_tok * long(p.rank) + long(i - p.rank)];
    else if (i < 2u * p.rank + p.D) out[i] = draft_h[i - 2u * p.rank];
    else out[i] = f_rnn[i - 2u * p.rank - p.D];
}

kernel void markov_gate_kernel(
    device const float* raw  [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device bfloat* state     [[buffer(2)]],
    device bfloat* latent    [[buffer(3)]],
    constant uint& rank      [[buffer(4)]],
    uint c                   [[thread_position_in_grid]])
{
    if (c >= rank) return;
    float g = float(bfloat(raw[c] + bias[c]));
    float cand = float(bfloat(raw[rank + c] + bias[rank + c]));
    float o = float(bfloat(raw[2u * rank + c] + bias[2u * rank + c]));
    float st = float(state[c]);
    g = 1.0f / (1.0f + exp(-g));
    cand = safe_tanh(cand);
    state[c] = bfloat(g * st + (1.0f - g) * cand);
    latent[c] = bfloat(safe_tanh(o));
}

struct MarkovArgParams {
    uint V;
    uint has_map;
};

kernel void markov_argmax_kernel(
    device const bfloat* base_logits [[buffer(0)]],
    device const float* fold_b       [[buffer(1)]],
    device const float* markov_bias  [[buffer(2)]],
    device const int* draft_to_tgt   [[buffer(3)]],
    device int* out_tok              [[buffer(4)]],
    constant MarkovArgParams& p      [[buffer(5)]],
    uint tid                         [[thread_index_in_threadgroup]],
    uint tg_size                     [[threads_per_threadgroup]])
{
    float best_v = -1e30f;
    int best_i = 0;
    for (uint i = tid; i < p.V; i += tg_size) {
        float v = float(base_logits[i]) + fold_b[i] + markov_bias[i];
        if (v > best_v) {
            best_v = v;
            best_i = int(i);
        }
    }
    threadgroup float sm_v[32];
    threadgroup int sm_i[32];
    uint lane = tid & 31u;
    uint wid = tid >> 5;
    for (uint off = 16u; off > 0u; off >>= 1) {
        float ov = simd_shuffle_down(best_v, off);
        int oi = simd_shuffle_down(best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }
    if (lane == 0u) {
        sm_v[wid] = best_v;
        sm_i[wid] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (wid == 0u) {
        uint nw = (tg_size + 31u) >> 5;
        best_v = (lane < nw) ? sm_v[lane] : -1e30f;
        best_i = (lane < nw) ? sm_i[lane] : 0;
        for (uint off = 16u; off > 0u; off >>= 1) {
            float ov = simd_shuffle_down(best_v, off);
            int oi = simd_shuffle_down(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (lane == 0u) out_tok[0] = (p.has_map != 0u) ? draft_to_tgt[best_i] : best_i;
    }
}

struct GatherParams {
    uint M;
    uint D;
};

kernel void gather_bf16_rows_kernel(
    device const bfloat* src [[buffer(0)]],
    device const int* idx    [[buffer(1)]],
    device bfloat* dst       [[buffer(2)]],
    constant GatherParams& p [[buffer(3)]],
    uint2 gid                [[thread_position_in_grid]])
{
    uint d = gid.x;
    uint m = gid.y;
    if (m >= p.M || d >= p.D) return;
    long row = long(idx[m]);
    dst[long(m) * long(p.D) + long(d)] = src[row * long(p.D) + long(d)];
}

struct FoldParams {
    uint V;
    uint D;
    uint rank;
};

kernel void fold_unembed_ho_kernel(
    device const bfloat* unembed [[buffer(0)]],
    device const bfloat* ho_w    [[buffer(1)]],
    device bfloat* fold_w        [[buffer(2)]],
    constant FoldParams& p       [[buffer(3)]],
    uint2 gid                    [[thread_position_in_grid]])
{
    uint r = gid.x;
    uint v = gid.y;
    if (v >= p.V || r >= p.rank) return;
    device const bfloat* urow = unembed + long(v) * long(p.D);
    float acc = 0.0f;
    for (uint d = 0; d < p.D; ++d) {
        acc = fma(float(urow[d]), float(ho_w[long(d) * long(p.rank) + long(r)]), acc);
    }
    fold_w[long(v) * long(p.rank) + long(r)] = bfloat(acc);
}

kernel void zero_bf16_kernel(
    device bfloat* dst [[buffer(0)]],
    constant uint& N   [[buffer(1)]],
    uint gid           [[thread_position_in_grid]])
{
    if (gid < N) dst[gid] = bfloat(0.0f);
}

struct Df2ConcatParams {
    uint T;
    uint D;
    uint stack_stride;
};

kernel void dflash2_concat_taps_kernel(
    device const bfloat* aux_states [[buffer(0)]],
    device bfloat* out              [[buffer(1)]],
    constant Df2ConcatParams& p     [[buffer(2)]],
    uint2 gid                       [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint t = gid.y;
    if (t >= p.T || col >= 4u * p.D) return;
    uint tap = col / p.D;
    uint d = col - tap * p.D;
    out[long(t) * long(4u * p.D) + long(col)] = aux_states[long(tap) * long(p.stack_stride) + long(t) * long(p.D) + long(d)];
}

struct Df2ConvParams {
    uint T;
    uint D;
    uint group_size;
    uint half_idx;
};

kernel void dflash2_grouped_dyn_conv_kernel(
    device const bfloat* x      [[buffer(0)]],
    device const bfloat* base_k [[buffer(1)]],
    device const bfloat* dyn_k  [[buffer(2)]],
    device bfloat* out          [[buffer(3)]],
    constant Df2ConvParams& p   [[buffer(4)]],
    uint2 gid                   [[thread_position_in_grid]])
{
    uint d = gid.x;
    uint t = gid.y;
    if (t >= p.T || d >= p.D) return;
    uint nG = p.D / p.group_size;
    uint g = d / p.group_size;
    long d_base = long(t) * long(4u * nG) + long(p.half_idx * 2u * nG) + long(g);
    float w0 = float(base_k[long(d)]) + float(dyn_k[d_base]);
    float w1 = float(base_k[long(p.D) + long(d)]) + float(dyn_k[d_base + long(nG)]);
    float x0 = float(x[long(t) * long(p.D) + long(d)]);
    float x1 = (t > 0u) ? float(x[long(t - 1u) * long(p.D) + long(d)]) : 0.0f;
    out[long(t) * long(p.D) + long(d)] = bfloat(fma(w0, x0, w1 * x1));
}

kernel void dflash2_block_attn_kernel(
    device const bfloat* q_blk   [[buffer(0)]],
    device const bfloat* k_blk   [[buffer(1)]],
    device const bfloat* v_blk   [[buffer(2)]],
    device const bfloat* k_cache [[buffer(3)]],
    device const bfloat* v_cache [[buffer(4)]],
    device bfloat* out           [[buffer(5)]],
    constant DraftAttnParams& p  [[buffer(6)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint lane                    [[thread_index_in_simdgroup]])
{
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.n_heads || t >= p.B) return;

    uint kv_h = h / (p.n_heads / p.n_kv_heads);
    uint q_pos = p.anchor_pos + t;
    uint c_start = (q_pos + 1u > p.window_size) ? (q_pos + 1u - p.window_size) : 0u;
    uint c_end = p.anchor_pos;
    float inv_sqrt_d = rsqrt(float(p.head_dim));

    device const bfloat* qp = q_blk + (long(t) * long(p.n_heads) + long(h)) * long(p.head_dim);
    float q0 = float(qp[lane]);
    float q1 = float(qp[lane + 32u]);
    float q2 = float(qp[lane + 64u]);
    float q3 = float(qp[lane + 96u]);

    float m_val = -INFINITY;
    float l_val = 0.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    long kv_stride = long(p.n_kv_heads) * long(p.head_dim);

    for (uint kp = c_start; kp < c_end; ++kp) {
        uint slot = kp % p.cap;
        device const bfloat* kp_ptr = k_cache + long(slot) * kv_stride + long(kv_h) * long(p.head_dim);
        device const bfloat* vp_ptr = v_cache + long(slot) * kv_stride + long(kv_h) * long(p.head_dim);
        float dot = fma(q0, float(kp_ptr[lane]), 0.0f);
        dot = fma(q1, float(kp_ptr[lane + 32u]), dot);
        dot = fma(q2, float(kp_ptr[lane + 64u]), dot);
        dot = fma(q3, float(kp_ptr[lane + 96u]), dot);
        dot = simd_sum(dot) * inv_sqrt_d;
        float m_new = max(m_val, dot);
        float al = (l_val > 0.0f) ? exp(m_val - m_new) : 0.0f;
        float pr = exp(dot - m_new);
        l_val = l_val * al + pr;
        a0 = a0 * al + pr * float(vp_ptr[lane]);
        a1 = a1 * al + pr * float(vp_ptr[lane + 32u]);
        a2 = a2 * al + pr * float(vp_ptr[lane + 64u]);
        a3 = a3 * al + pr * float(vp_ptr[lane + 96u]);
        m_val = m_new;
    }

    for (uint j = 0u; j < p.B; ++j) {
        device const bfloat* kp_ptr = k_blk + long(j) * kv_stride + long(kv_h) * long(p.head_dim);
        device const bfloat* vp_ptr = v_blk + long(j) * kv_stride + long(kv_h) * long(p.head_dim);
        float dot = fma(q0, float(kp_ptr[lane]), 0.0f);
        dot = fma(q1, float(kp_ptr[lane + 32u]), dot);
        dot = fma(q2, float(kp_ptr[lane + 64u]), dot);
        dot = fma(q3, float(kp_ptr[lane + 96u]), dot);
        dot = simd_sum(dot) * inv_sqrt_d;
        float m_new = max(m_val, dot);
        float al = (l_val > 0.0f) ? exp(m_val - m_new) : 0.0f;
        float pr = exp(dot - m_new);
        l_val = l_val * al + pr;
        a0 = a0 * al + pr * float(vp_ptr[lane]);
        a1 = a1 * al + pr * float(vp_ptr[lane + 32u]);
        a2 = a2 * al + pr * float(vp_ptr[lane + 64u]);
        a3 = a3 * al + pr * float(vp_ptr[lane + 96u]);
        m_val = m_new;
    }

    float inv_l = 1.0f / l_val;
    device bfloat* op = out + (long(t) * long(p.n_heads) + long(h)) * long(p.head_dim);
    op[lane]       = bfloat(a0 * inv_l);
    op[lane + 32u] = bfloat(a1 * inv_l);
    op[lane + 64u] = bfloat(a2 * inv_l);
    op[lane + 96u] = bfloat(a3 * inv_l);
}

struct Df2TopKParams {
    uint M;
    uint V;
};

kernel void dflash2_topk16_kernel(
    device float* logits       [[buffer(0)]],
    device float* top_vals     [[buffer(1)]],
    device int* top_ids        [[buffer(2)]],
    constant Df2TopKParams& p  [[buffer(3)]],
    uint tgid                  [[threadgroup_position_in_grid]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]])
{
    uint m = tgid;
    if (m >= p.M) return;
    device float* row = logits + long(m) * long(p.V);
    threadgroup float sm_v[32];
    threadgroup int sm_i[32];
    uint lane = tid & 31u;
    uint wid = tid >> 5;
    uint nw = (tg_size + 31u) >> 5;

    for (uint k = 0u; k < 16u; ++k) {
        float best_v = -1e30f;
        int best_i = 0;
        for (uint i = tid; i < p.V; i += tg_size) {
            float v = row[i];
            if (v > best_v) {
                best_v = v;
                best_i = int(i);
            }
        }
        for (uint off = 16u; off > 0u; off >>= 1) {
            float ov = simd_shuffle_down(best_v, off);
            int oi = simd_shuffle_down(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (lane == 0u) {
            sm_v[wid] = best_v;
            sm_i[wid] = best_i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (wid == 0u) {
            best_v = (lane < nw) ? sm_v[lane] : -1e30f;
            best_i = (lane < nw) ? sm_i[lane] : 0;
            for (uint off = 16u; off > 0u; off >>= 1) {
                float ov = simd_shuffle_down(best_v, off);
                int oi = simd_shuffle_down(best_i, off);
                if (ov > best_v || (ov == best_v && oi < best_i)) {
                    best_v = ov;
                    best_i = oi;
                }
            }
            if (lane == 0u) {
                top_vals[long(m) * 16L + long(k)] = best_v;
                top_ids[long(m) * 16L + long(k)] = best_i;
                row[best_i] = -1e30f;
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}

kernel void moe_nvfp4_w13_dedup_kernel(
    device const bfloat* x         [[buffer(0)]],
    device const float* topk_wts   [[buffer(1)]],
    device const uint* offsets     [[buffer(2)]],
    device const uint* sorted_tk   [[buffer(3)]],
    device const uchar* w13_u8     [[buffer(4)]],
    device const uchar* w13_scale  [[buffer(5)]],
    device const float* w13_amax   [[buffer(6)]],
    device const float* w13_denom  [[buffer(7)]],
    device float* act_out          [[buffer(8)]],
    constant MoeParams& p          [[buffer(9)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]])
{
    uint expert = tgid.y;
    uint j = tgid.x * 4u + sg_id;
    if (j >= p.inter) return;
    uint s0 = offsets[expert];
    uint s1 = offsets[expert + 1u];
    if (s0 >= s1) return;

    float gscale = w13_amax[expert] / w13_denom[0];
    long row_g = long(2u * j);
    long row_u = long(2u * j + 1u);
    long half_D = long(p.D / 2u);
    long scale_D = long(p.D / 16u);
    long exp_base = long(expert) * long(2u * p.inter);

    device const uint2* wg2 = reinterpret_cast<device const uint2*>(w13_u8 + (exp_base + row_g) * half_D);
    device const uint2* wu2 = reinterpret_cast<device const uint2*>(w13_u8 + (exp_base + row_u) * half_D);
    device const uchar* sg = w13_scale + (exp_base + row_g) * scale_D;
    device const uchar* su = w13_scale + (exp_base + row_u) * scale_D;

    for (uint idx = s0; idx < s1; idx += 4u) {
        uint n = min(4u, s1 - idx);
        uint tk0 = sorted_tk[idx];
        device const bfloat4* x0 = reinterpret_cast<device const bfloat4*>(x + long(tk0 / p.K) * long(p.D));
        if (n == 1u) {
            float ag0 = 0.0f, au0 = 0.0f;
            for (uint blk = lane; blk < uint(scale_D); blk += 32u) {
                float bsg = kE4M3Lut[sg[blk]];
                float bsu = kE4M3Lut[su[blk]];
                uint2 bg = wg2[blk];
                uint2 bu = wu2[blk];
                float lg0 = 0.0f, lu0 = 0.0f;
                for (uint h = 0u; h < 2u; ++h) {
                    uint bw_g = (h == 0u) ? bg.x : bg.y;
                    uint bw_u = (h == 0u) ? bu.x : bu.y;
                    for (uint q = 0u; q < 2u; ++q) {
                        uint bg0 = (bw_g >> (q * 16u)) & 0xFFu;
                        uint bg1 = (bw_g >> (q * 16u + 8u)) & 0xFFu;
                        uint bu0 = (bw_u >> (q * 16u)) & 0xFFu;
                        uint bu1 = (bw_u >> (q * 16u + 8u)) & 0xFFu;
                        float4 fg = float4(kE2M1Lut[bg0 & 0xFu], kE2M1Lut[bg0 >> 4], kE2M1Lut[bg1 & 0xFu], kE2M1Lut[bg1 >> 4]);
                        float4 fu = float4(kE2M1Lut[bu0 & 0xFu], kE2M1Lut[bu0 >> 4], kE2M1Lut[bu1 & 0xFu], kE2M1Lut[bu1 >> 4]);
                        float4 v0 = float4(x0[blk * 4u + h * 2u + q]);
                        lg0 += dot(fg, v0);
                        lu0 += dot(fu, v0);
                    }
                }
                ag0 = fma(lg0, bsg, ag0);
                au0 = fma(lu0, bsu, au0);
            }
            ag0 = simd_sum(ag0) * gscale;
            au0 = simd_sum(au0) * gscale;
            if (lane == 0u) {
                float sig = 1.0f / (1.0f + exp(-ag0));
                float gc = safe_tanh(ag0 / p.gate_cap) * p.gate_cap;
                float uc = safe_tanh(au0 / p.up_cap) * p.up_cap;
                act_out[long(tk0) * long(p.inter) + long(j)] = float(bfloat((gc * sig * uc) * topk_wts[tk0]));
            }
            continue;
        }
        uint tk1 = sorted_tk[idx + 1u];
        uint tk2 = (n > 2u) ? sorted_tk[idx + 2u] : tk0;
        uint tk3 = (n > 3u) ? sorted_tk[idx + 3u] : tk0;

        device const bfloat4* x1 = reinterpret_cast<device const bfloat4*>(x + long(tk1 / p.K) * long(p.D));
        device const bfloat4* x2 = reinterpret_cast<device const bfloat4*>(x + long(tk2 / p.K) * long(p.D));
        device const bfloat4* x3 = reinterpret_cast<device const bfloat4*>(x + long(tk3 / p.K) * long(p.D));

        float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
        float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;

        for (uint blk = lane; blk < uint(scale_D); blk += 32u) {
            float bsg = e4m3_to_f32(sg[blk]);
            float bsu = e4m3_to_f32(su[blk]);
            uint2 bg = wg2[blk];
            uint2 bu = wu2[blk];
            float lg0 = 0.0f, lg1 = 0.0f, lg2 = 0.0f, lg3 = 0.0f;
            float lu0 = 0.0f, lu1 = 0.0f, lu2 = 0.0f, lu3 = 0.0f;
            for (uint h = 0u; h < 2u; ++h) {
                uint bw_g = (h == 0u) ? bg.x : bg.y;
                uint bw_u = (h == 0u) ? bu.x : bu.y;
                for (uint q = 0u; q < 2u; ++q) {
                    uint bg0 = (bw_g >> (q * 16u)) & 0xFFu;
                    uint bg1 = (bw_g >> (q * 16u + 8u)) & 0xFFu;
                    uint bu0 = (bw_u >> (q * 16u)) & 0xFFu;
                    uint bu1 = (bw_u >> (q * 16u + 8u)) & 0xFFu;
                    float4 fg = float4(kE2M1Lut[bg0 & 0xFu], kE2M1Lut[bg0 >> 4], kE2M1Lut[bg1 & 0xFu], kE2M1Lut[bg1 >> 4]);
                    float4 fu = float4(kE2M1Lut[bu0 & 0xFu], kE2M1Lut[bu0 >> 4], kE2M1Lut[bu1 & 0xFu], kE2M1Lut[bu1 >> 4]);
                    uint off = blk * 4u + h * 2u + q;
                    float4 v0 = float4(x0[off]);
                    lg0 += dot(fg, v0); lu0 += dot(fu, v0);
                    if (n > 1u) { float4 v1 = float4(x1[off]); lg1 += dot(fg, v1); lu1 += dot(fu, v1); }
                    if (n > 2u) { float4 v2 = float4(x2[off]); lg2 += dot(fg, v2); lu2 += dot(fu, v2); }
                    if (n > 3u) { float4 v3 = float4(x3[off]); lg3 += dot(fg, v3); lu3 += dot(fu, v3); }
                }
            }
            ag0 = fma(lg0, bsg, ag0); au0 = fma(lu0, bsu, au0);
            if (n > 1u) { ag1 = fma(lg1, bsg, ag1); au1 = fma(lu1, bsu, au1); }
            if (n > 2u) { ag2 = fma(lg2, bsg, ag2); au2 = fma(lu2, bsu, au2); }
            if (n > 3u) { ag3 = fma(lg3, bsg, ag3); au3 = fma(lu3, bsu, au3); }
        }

        ag0 = simd_sum(ag0) * gscale; au0 = simd_sum(au0) * gscale;
        if (n > 1u) { ag1 = simd_sum(ag1) * gscale; au1 = simd_sum(au1) * gscale; }
        if (n > 2u) { ag2 = simd_sum(ag2) * gscale; au2 = simd_sum(au2) * gscale; }
        if (n > 3u) { ag3 = simd_sum(ag3) * gscale; au3 = simd_sum(au3) * gscale; }

        if (lane == 0u) {
            uint tks[4] = {tk0, tk1, tk2, tk3};
            float gs[4] = {ag0, ag1, ag2, ag3};
            float us[4] = {au0, au1, au2, au3};
            for (uint qi = 0u; qi < n; ++qi) {
                float sig = 1.0f / (1.0f + exp(-gs[qi]));
                float gc = safe_tanh(gs[qi] / p.gate_cap) * p.gate_cap;
                float uc = safe_tanh(us[qi] / p.up_cap) * p.up_cap;
                float y = (gc * sig * uc) * topk_wts[tks[qi]];
                act_out[long(tks[qi]) * long(p.inter) + long(j)] = float(bfloat(y));
            }
        }
    }
}

kernel void moe_nvfp4_w2_dedup_kernel(
    device const float* act_in     [[buffer(0)]],
    device const uint* offsets     [[buffer(1)]],
    device const uint* sorted_tk   [[buffer(2)]],
    device const uchar* w2_u8      [[buffer(3)]],
    device const uchar* w2_scale   [[buffer(4)]],
    device const float* w2_amax    [[buffer(5)]],
    device const float* w2_denom   [[buffer(6)]],
    device bfloat* w2_out          [[buffer(7)]],
    constant MoeParams& p          [[buffer(8)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]])
{
    uint expert = tgid.y;
    uint d = tgid.x * 4u + sg_id;
    if (d >= p.D) return;
    uint s0 = offsets[expert];
    uint s1 = offsets[expert + 1u];
    if (s0 >= s1) return;

    float gscale = w2_amax[expert] / w2_denom[0];
    long half_I = long(p.inter / 2u);
    long scale_I = long(p.inter / 16u);
    long row_idx = long(expert) * long(p.D) + long(d);
    device const uint2* w2_2 = reinterpret_cast<device const uint2*>(w2_u8 + row_idx * half_I);
    device const uchar* s2r = w2_scale + row_idx * scale_I;

    for (uint idx = s0; idx < s1; idx += 4u) {
        uint n = min(4u, s1 - idx);
        uint tk0 = sorted_tk[idx];
        device const float4* a0 = reinterpret_cast<device const float4*>(act_in + long(tk0) * long(p.inter));
        if (n == 1u) {
            float r0 = 0.0f;
            for (uint blk = lane; blk < uint(scale_I); blk += 32u) {
                float bs = kE4M3Lut[s2r[blk]];
                uint2 bw = w2_2[blk];
                float l0 = 0.0f;
                for (uint h = 0u; h < 2u; ++h) {
                    uint w32 = (h == 0u) ? bw.x : bw.y;
                    for (uint q = 0u; q < 2u; ++q) {
                        uint b0 = (w32 >> (q * 16u)) & 0xFFu;
                        uint b1 = (w32 >> (q * 16u + 8u)) & 0xFFu;
                        float4 fw = float4(kE2M1Lut[b0 & 0xFu], kE2M1Lut[b0 >> 4], kE2M1Lut[b1 & 0xFu], kE2M1Lut[b1 >> 4]);
                        l0 += dot(fw, a0[blk * 4u + h * 2u + q]);
                    }
                }
                r0 = fma(l0, bs, r0);
            }
            r0 = simd_sum(r0) * gscale;
            if (lane == 0u) w2_out[long(tk0) * long(p.D) + long(d)] = bfloat(r0);
            continue;
        }
        uint tk1 = sorted_tk[idx + 1u];
        uint tk2 = (n > 2u) ? sorted_tk[idx + 2u] : tk0;
        uint tk3 = (n > 3u) ? sorted_tk[idx + 3u] : tk0;

        device const float4* a1 = reinterpret_cast<device const float4*>(act_in + long(tk1) * long(p.inter));
        device const float4* a2 = reinterpret_cast<device const float4*>(act_in + long(tk2) * long(p.inter));
        device const float4* a3 = reinterpret_cast<device const float4*>(act_in + long(tk3) * long(p.inter));

        float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
        for (uint blk = lane; blk < uint(scale_I); blk += 32u) {
            float bs = e4m3_to_f32(s2r[blk]);
            uint2 bw = w2_2[blk];
            float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;
            for (uint h = 0u; h < 2u; ++h) {
                uint w32 = (h == 0u) ? bw.x : bw.y;
                for (uint q = 0u; q < 2u; ++q) {
                    uint b0 = (w32 >> (q * 16u)) & 0xFFu;
                    uint b1 = (w32 >> (q * 16u + 8u)) & 0xFFu;
                    float4 fw = float4(kE2M1Lut[b0 & 0xFu], kE2M1Lut[b0 >> 4], kE2M1Lut[b1 & 0xFu], kE2M1Lut[b1 >> 4]);
                    uint off = blk * 4u + h * 2u + q;
                    l0 += dot(fw, a0[off]);
                    if (n > 1u) l1 += dot(fw, a1[off]);
                    if (n > 2u) l2 += dot(fw, a2[off]);
                    if (n > 3u) l3 += dot(fw, a3[off]);
                }
            }
            r0 = fma(l0, bs, r0);
            if (n > 1u) r1 = fma(l1, bs, r1);
            if (n > 2u) r2 = fma(l2, bs, r2);
            if (n > 3u) r3 = fma(l3, bs, r3);
        }
        r0 = simd_sum(r0) * gscale;
        if (n > 1u) r1 = simd_sum(r1) * gscale;
        if (n > 2u) r2 = simd_sum(r2) * gscale;
        if (n > 3u) r3 = simd_sum(r3) * gscale;
        if (lane == 0u) {
            w2_out[long(tk0) * long(p.D) + long(d)] = bfloat(r0);
            if (n > 1u) w2_out[long(tk1) * long(p.D) + long(d)] = bfloat(r1);
            if (n > 2u) w2_out[long(tk2) * long(p.D) + long(d)] = bfloat(r2);
            if (n > 3u) w2_out[long(tk3) * long(p.D) + long(d)] = bfloat(r3);
        }
    }
}

kernel void moe_mxfp8_w13_dedup_kernel(
    device const bfloat* x         [[buffer(0)]],
    device const float* topk_wts   [[buffer(1)]],
    device const uint* offsets     [[buffer(2)]],
    device const uint* sorted_tk   [[buffer(3)]],
    device const uchar* w13_e4m3   [[buffer(4)]],
    device const uchar* w13_scale  [[buffer(5)]],
    device float* act_out          [[buffer(6)]],
    constant MoeParams& p          [[buffer(7)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]])
{
    uint expert = tgid.y;
    uint j = tgid.x * 4u + sg_id;
    if (j >= p.inter) return;
    uint s0 = offsets[expert];
    uint s1 = offsets[expert + 1u];
    if (s0 >= s1) return;

    long row_g = long(2u * j);
    long row_u = long(2u * j + 1u);
    long scale_D = long(p.D / 32u);
    long exp_base = long(expert) * long(2u * p.inter);

    device const uchar4* wg4 = reinterpret_cast<device const uchar4*>(w13_e4m3 + (exp_base + row_g) * long(p.D));
    device const uchar4* wu4 = reinterpret_cast<device const uchar4*>(w13_e4m3 + (exp_base + row_u) * long(p.D));
    device const uchar* sg = w13_scale + (exp_base + row_g) * scale_D;
    device const uchar* su = w13_scale + (exp_base + row_u) * scale_D;

    for (uint idx = s0; idx < s1; idx += 4u) {
        uint n = min(4u, s1 - idx);
        uint tk0 = sorted_tk[idx];
        uint tk1 = (n > 1u) ? sorted_tk[idx + 1u] : tk0;
        uint tk2 = (n > 2u) ? sorted_tk[idx + 2u] : tk0;
        uint tk3 = (n > 3u) ? sorted_tk[idx + 3u] : tk0;
        device const bfloat4* x0 = reinterpret_cast<device const bfloat4*>(x + long(tk0 / p.K) * long(p.D));
        device const bfloat4* x1 = reinterpret_cast<device const bfloat4*>(x + long(tk1 / p.K) * long(p.D));
        device const bfloat4* x2 = reinterpret_cast<device const bfloat4*>(x + long(tk2 / p.K) * long(p.D));
        device const bfloat4* x3 = reinterpret_cast<device const bfloat4*>(x + long(tk3 / p.K) * long(p.D));

        float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
        float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;
        for (uint blk = lane; blk < uint(scale_D); blk += 32u) {
            float bsg = ue8m0_to_f32(sg[blk]);
            float bsu = ue8m0_to_f32(su[blk]);
            float lg0 = 0.0f, lg1 = 0.0f, lg2 = 0.0f, lg3 = 0.0f;
            float lu0 = 0.0f, lu1 = 0.0f, lu2 = 0.0f, lu3 = 0.0f;
            for (uint q = 0u; q < 8u; ++q) {
                uint off = blk * 8u + q;
                uchar4 bg = wg4[off];
                uchar4 bu = wu4[off];
                float4 fg = float4(kE4M3Lut[bg.x], kE4M3Lut[bg.y], kE4M3Lut[bg.z], kE4M3Lut[bg.w]);
                float4 fu = float4(kE4M3Lut[bu.x], kE4M3Lut[bu.y], kE4M3Lut[bu.z], kE4M3Lut[bu.w]);
                float4 v0 = float4(x0[off]);
                lg0 += dot(fg, v0); lu0 += dot(fu, v0);
                if (n > 1u) { float4 v1 = float4(x1[off]); lg1 += dot(fg, v1); lu1 += dot(fu, v1); }
                if (n > 2u) { float4 v2 = float4(x2[off]); lg2 += dot(fg, v2); lu2 += dot(fu, v2); }
                if (n > 3u) { float4 v3 = float4(x3[off]); lg3 += dot(fg, v3); lu3 += dot(fu, v3); }
            }
            ag0 = fma(lg0, bsg, ag0); au0 = fma(lu0, bsu, au0);
            if (n > 1u) { ag1 = fma(lg1, bsg, ag1); au1 = fma(lu1, bsu, au1); }
            if (n > 2u) { ag2 = fma(lg2, bsg, ag2); au2 = fma(lu2, bsu, au2); }
            if (n > 3u) { ag3 = fma(lg3, bsg, ag3); au3 = fma(lu3, bsu, au3); }
        }
        ag0 = simd_sum(ag0); au0 = simd_sum(au0);
        if (n > 1u) { ag1 = simd_sum(ag1); au1 = simd_sum(au1); }
        if (n > 2u) { ag2 = simd_sum(ag2); au2 = simd_sum(au2); }
        if (n > 3u) { ag3 = simd_sum(ag3); au3 = simd_sum(au3); }
        if (lane == 0u) {
            uint tks[4] = {tk0, tk1, tk2, tk3};
            float gs[4] = {ag0, ag1, ag2, ag3};
            float us[4] = {au0, au1, au2, au3};
            for (uint qi = 0u; qi < n; ++qi) {
                float sig = 1.0f / (1.0f + exp(-gs[qi]));
                float gc = safe_tanh(gs[qi] / p.gate_cap) * p.gate_cap;
                float uc = safe_tanh(us[qi] / p.up_cap) * p.up_cap;
                act_out[long(tks[qi]) * long(p.inter) + long(j)] = float(bfloat((gc * sig * uc) * topk_wts[tks[qi]]));
            }
        }
    }
}

kernel void moe_mxfp8_w2_dedup_kernel(
    device const float* act_in     [[buffer(0)]],
    device const uint* offsets     [[buffer(1)]],
    device const uint* sorted_tk   [[buffer(2)]],
    device const uchar* w2_e4m3    [[buffer(3)]],
    device const uchar* w2_scale   [[buffer(4)]],
    device bfloat* w2_out          [[buffer(5)]],
    constant MoeParams& p          [[buffer(6)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]])
{
    uint expert = tgid.y;
    uint d = tgid.x * 4u + sg_id;
    if (d >= p.D) return;
    uint s0 = offsets[expert];
    uint s1 = offsets[expert + 1u];
    if (s0 >= s1) return;

    long scale_I = long(p.inter / 32u);
    long row_idx = long(expert) * long(p.D) + long(d);
    device const uchar4* w2_4 = reinterpret_cast<device const uchar4*>(w2_e4m3 + row_idx * long(p.inter));
    device const uchar* s2r = w2_scale + row_idx * scale_I;

    for (uint idx = s0; idx < s1; idx += 4u) {
        uint n = min(4u, s1 - idx);
        uint tk0 = sorted_tk[idx];
        uint tk1 = (n > 1u) ? sorted_tk[idx + 1u] : tk0;
        uint tk2 = (n > 2u) ? sorted_tk[idx + 2u] : tk0;
        uint tk3 = (n > 3u) ? sorted_tk[idx + 3u] : tk0;
        device const float4* a0 = reinterpret_cast<device const float4*>(act_in + long(tk0) * long(p.inter));
        device const float4* a1 = reinterpret_cast<device const float4*>(act_in + long(tk1) * long(p.inter));
        device const float4* a2 = reinterpret_cast<device const float4*>(act_in + long(tk2) * long(p.inter));
        device const float4* a3 = reinterpret_cast<device const float4*>(act_in + long(tk3) * long(p.inter));

        float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
        for (uint blk = lane; blk < uint(scale_I); blk += 32u) {
            float bs = ue8m0_to_f32(s2r[blk]);
            float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;
            for (uint q = 0u; q < 8u; ++q) {
                uint off = blk * 8u + q;
                uchar4 bw = w2_4[off];
                float4 fw = float4(kE4M3Lut[bw.x], kE4M3Lut[bw.y], kE4M3Lut[bw.z], kE4M3Lut[bw.w]);
                l0 += dot(fw, a0[off]);
                if (n > 1u) l1 += dot(fw, a1[off]);
                if (n > 2u) l2 += dot(fw, a2[off]);
                if (n > 3u) l3 += dot(fw, a3[off]);
            }
            r0 = fma(l0, bs, r0);
            if (n > 1u) r1 = fma(l1, bs, r1);
            if (n > 2u) r2 = fma(l2, bs, r2);
            if (n > 3u) r3 = fma(l3, bs, r3);
        }
        r0 = simd_sum(r0);
        if (n > 1u) r1 = simd_sum(r1);
        if (n > 2u) r2 = simd_sum(r2);
        if (n > 3u) r3 = simd_sum(r3);
        if (lane == 0u) {
            w2_out[long(tk0) * long(p.D) + long(d)] = bfloat(r0);
            if (n > 1u) w2_out[long(tk1) * long(p.D) + long(d)] = bfloat(r1);
            if (n > 2u) w2_out[long(tk2) * long(p.D) + long(d)] = bfloat(r2);
            if (n > 3u) w2_out[long(tk3) * long(p.D) + long(d)] = bfloat(r3);
        }
    }
}

kernel void moe_q2_w13_kernel(
    device const bfloat* x         [[buffer(0)]],
    device const int* topk_ids     [[buffer(1)]],
    device const float* topk_wts   [[buffer(2)]],
    device const uchar* w13_u8     [[buffer(3)]],
    device const half* w13_scale   [[buffer(4)]],
    device const half* w13_min     [[buffer(5)]],
    device float* act_out          [[buffer(6)]],
    constant MoeParams& p          [[buffer(7)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]])
{
    uint tk = tgid.y;
    uint t = tk / p.K;
    int expert = topk_ids[tk];
    float gamma = topk_wts[tk];

    uint j = tgid.x * 4u + sg_id;
    if (j >= p.inter) return;

    long q_D = long(p.D / 4u);
    long g_D = long(p.D / 64u);
    uint n_hg = p.D / 32u;
    long exp_base = long(expert) * long(2u * p.inter);
    long row_g = exp_base + long(2u * j);
    long row_u = row_g + 1L;

    device const uint2* wg2 = reinterpret_cast<device const uint2*>(w13_u8 + row_g * q_D);
    device const uint2* wu2 = reinterpret_cast<device const uint2*>(w13_u8 + row_u * q_D);
    device const half* sg = w13_scale + row_g * g_D;
    device const half* mg = w13_min + row_g * g_D;
    device const half* su = w13_scale + row_u * g_D;
    device const half* mu = w13_min + row_u * g_D;
    device const bfloat4* xr4 = reinterpret_cast<device const bfloat4*>(x + long(t) * long(p.D));

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (uint hg = lane; hg < n_hg; hg += 32u) {
        uint grp = hg >> 1;
        float scg = float(sg[grp]), mng = float(mg[grp]);
        float scu = float(su[grp]), mnu = float(mu[grp]);
        uint2 bg = wg2[hg];
        uint2 bu = wu2[hg];
        float qg_x = 0.0f, qu_x = 0.0f, sx = 0.0f;
        for (uint h = 0u; h < 2u; ++h) {
            uint pg = (h == 0u) ? bg.x : bg.y;
            uint pu = (h == 0u) ? bu.x : bu.y;
            for (uint q = 0u; q < 4u; ++q) {
                float4 xv = float4(xr4[hg * 8u + h * 4u + q]);
                uint bgb = (pg >> (q * 8u)) & 0xFFu;
                uint bub = (pu >> (q * 8u)) & 0xFFu;
                float4 fg = float4(float(bgb & 3u), float((bgb >> 2) & 3u), float((bgb >> 4) & 3u), float((bgb >> 6) & 3u));
                float4 fu = float4(float(bub & 3u), float((bub >> 2) & 3u), float((bub >> 4) & 3u), float((bub >> 6) & 3u));
                qg_x += dot(fg, xv);
                qu_x += dot(fu, xv);
                sx += (xv.x + xv.y + xv.z + xv.w);
            }
        }
        acc_g = fma(qg_x, scg, fma(sx, mng, acc_g));
        acc_u = fma(qu_x, scu, fma(sx, mnu, acc_u));
    }
    acc_g = simd_sum(acc_g);
    acc_u = simd_sum(acc_u);
    if (lane == 0u) {
        float sig = 1.0f / (1.0f + exp(-acc_g));
        float gc = safe_tanh(acc_g / p.gate_cap) * p.gate_cap;
        float uc = safe_tanh(acc_u / p.up_cap) * p.up_cap;
        act_out[long(tk) * long(p.inter) + long(j)] = float(bfloat((gc * sig * uc) * gamma));
    }
}

kernel void moe_q2_w2_kernel(
    device const float* act_in     [[buffer(0)]],
    device const int* topk_ids     [[buffer(1)]],
    device const uchar* w2_u8      [[buffer(2)]],
    device const half* w2_scale    [[buffer(3)]],
    device const half* w2_min      [[buffer(4)]],
    device bfloat* out             [[buffer(5)]],
    constant MoeParams& p          [[buffer(6)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]])
{
    uint t = tgid.y;
    uint d = tgid.x * 4u + sg_id;
    if (d >= p.D) return;

    long q_I = long(p.inter / 4u);
    long g_I = long(p.inter / 64u);
    uint n_hg = p.inter / 32u;

    float total = 0.0f;
    for (uint k = 0u; k < p.K; ++k) {
        uint tk = t * p.K + k;
        int expert = topk_ids[tk];
        long row_idx = long(expert) * long(p.D) + long(d);
        device const uint2* w2_2 = reinterpret_cast<device const uint2*>(w2_u8 + row_idx * q_I);
        device const half* s2r = w2_scale + row_idx * g_I;
        device const half* m2r = w2_min + row_idx * g_I;
        device const float4* ar4 = reinterpret_cast<device const float4*>(act_in + long(tk) * long(p.inter));

        float acc = 0.0f;
        for (uint hg = lane; hg < n_hg; hg += 32u) {
            uint grp = hg >> 1;
            float sc = float(s2r[grp]), mn = float(m2r[grp]);
            uint2 bw = w2_2[hg];
            float qa = 0.0f, sa = 0.0f;
            for (uint h = 0u; h < 2u; ++h) {
                uint pw = (h == 0u) ? bw.x : bw.y;
                for (uint q = 0u; q < 4u; ++q) {
                    float4 av = ar4[hg * 8u + h * 4u + q];
                    uint b = (pw >> (q * 8u)) & 0xFFu;
                    float4 fw = float4(float(b & 3u), float((b >> 2) & 3u), float((b >> 4) & 3u), float((b >> 6) & 3u));
                    qa += dot(fw, av);
                    sa += (av.x + av.y + av.z + av.w);
                }
            }
            acc = fma(qa, sc, fma(sa, mn, acc));
        }
        acc = simd_sum(acc);
        if (lane == 0u) total += acc;
    }
    if (lane == 0u) out[long(t) * long(p.D) + long(d)] = bfloat(total);
}

kernel void moe_q2_w13_dedup_kernel(
    device const bfloat* x         [[buffer(0)]],
    device const float* topk_wts   [[buffer(1)]],
    device const uint* offsets     [[buffer(2)]],
    device const uint* sorted_tk   [[buffer(3)]],
    device const uchar* w13_u8     [[buffer(4)]],
    device const half* w13_scale   [[buffer(5)]],
    device const half* w13_min     [[buffer(6)]],
    device float* act_out          [[buffer(7)]],
    constant MoeParams& p          [[buffer(8)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]])
{
    uint expert = tgid.y;
    uint j = tgid.x * 4u + sg_id;
    if (j >= p.inter) return;
    uint s0 = offsets[expert];
    uint s1 = offsets[expert + 1u];
    if (s0 >= s1) return;

    long q_D = long(p.D / 4u);
    long g_D = long(p.D / 64u);
    uint n_hg = p.D / 32u;
    long exp_base = long(expert) * long(2u * p.inter);
    long row_g = exp_base + long(2u * j);
    long row_u = row_g + 1L;

    device const uint2* wg2 = reinterpret_cast<device const uint2*>(w13_u8 + row_g * q_D);
    device const uint2* wu2 = reinterpret_cast<device const uint2*>(w13_u8 + row_u * q_D);
    device const half* sg = w13_scale + row_g * g_D;
    device const half* mg = w13_min + row_g * g_D;
    device const half* su = w13_scale + row_u * g_D;
    device const half* mu = w13_min + row_u * g_D;

    for (uint idx = s0; idx < s1; idx += 4u) {
        uint n = min(4u, s1 - idx);
        uint tk0 = sorted_tk[idx];
        uint tk1 = (n > 1u) ? sorted_tk[idx + 1u] : tk0;
        uint tk2 = (n > 2u) ? sorted_tk[idx + 2u] : tk0;
        uint tk3 = (n > 3u) ? sorted_tk[idx + 3u] : tk0;
        device const bfloat4* x0 = reinterpret_cast<device const bfloat4*>(x + long(tk0 / p.K) * long(p.D));
        device const bfloat4* x1 = reinterpret_cast<device const bfloat4*>(x + long(tk1 / p.K) * long(p.D));
        device const bfloat4* x2 = reinterpret_cast<device const bfloat4*>(x + long(tk2 / p.K) * long(p.D));
        device const bfloat4* x3 = reinterpret_cast<device const bfloat4*>(x + long(tk3 / p.K) * long(p.D));

        float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
        float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;
        for (uint hg = lane; hg < n_hg; hg += 32u) {
            uint grp = hg >> 1;
            float scg = float(sg[grp]), mng = float(mg[grp]);
            float scu = float(su[grp]), mnu = float(mu[grp]);
            uint2 bg = wg2[hg];
            uint2 bu = wu2[hg];
            float qg0 = 0.0f, qg1 = 0.0f, qg2 = 0.0f, qg3 = 0.0f;
            float qu0 = 0.0f, qu1 = 0.0f, qu2 = 0.0f, qu3 = 0.0f;
            float sx0 = 0.0f, sx1 = 0.0f, sx2 = 0.0f, sx3 = 0.0f;
            for (uint h = 0u; h < 2u; ++h) {
                uint pg = (h == 0u) ? bg.x : bg.y;
                uint pu = (h == 0u) ? bu.x : bu.y;
                for (uint q = 0u; q < 4u; ++q) {
                    uint off = hg * 8u + h * 4u + q;
                    uint bgb = (pg >> (q * 8u)) & 0xFFu;
                    uint bub = (pu >> (q * 8u)) & 0xFFu;
                    float4 fg = float4(float(bgb & 3u), float((bgb >> 2) & 3u), float((bgb >> 4) & 3u), float((bgb >> 6) & 3u));
                    float4 fu = float4(float(bub & 3u), float((bub >> 2) & 3u), float((bub >> 4) & 3u), float((bub >> 6) & 3u));
                    float4 v0 = float4(x0[off]);
                    qg0 += dot(fg, v0); qu0 += dot(fu, v0); sx0 += (v0.x + v0.y + v0.z + v0.w);
                    if (n > 1u) { float4 v1 = float4(x1[off]); qg1 += dot(fg, v1); qu1 += dot(fu, v1); sx1 += (v1.x + v1.y + v1.z + v1.w); }
                    if (n > 2u) { float4 v2 = float4(x2[off]); qg2 += dot(fg, v2); qu2 += dot(fu, v2); sx2 += (v2.x + v2.y + v2.z + v2.w); }
                    if (n > 3u) { float4 v3 = float4(x3[off]); qg3 += dot(fg, v3); qu3 += dot(fu, v3); sx3 += (v3.x + v3.y + v3.z + v3.w); }
                }
            }
            ag0 = fma(qg0, scg, fma(sx0, mng, ag0)); au0 = fma(qu0, scu, fma(sx0, mnu, au0));
            if (n > 1u) { ag1 = fma(qg1, scg, fma(sx1, mng, ag1)); au1 = fma(qu1, scu, fma(sx1, mnu, au1)); }
            if (n > 2u) { ag2 = fma(qg2, scg, fma(sx2, mng, ag2)); au2 = fma(qu2, scu, fma(sx2, mnu, au2)); }
            if (n > 3u) { ag3 = fma(qg3, scg, fma(sx3, mng, ag3)); au3 = fma(qu3, scu, fma(sx3, mnu, au3)); }
        }
        ag0 = simd_sum(ag0); au0 = simd_sum(au0);
        if (n > 1u) { ag1 = simd_sum(ag1); au1 = simd_sum(au1); }
        if (n > 2u) { ag2 = simd_sum(ag2); au2 = simd_sum(au2); }
        if (n > 3u) { ag3 = simd_sum(ag3); au3 = simd_sum(au3); }
        if (lane == 0u) {
            uint tks[4] = {tk0, tk1, tk2, tk3};
            float gs[4] = {ag0, ag1, ag2, ag3};
            float us[4] = {au0, au1, au2, au3};
            for (uint qi = 0u; qi < n; ++qi) {
                float sig = 1.0f / (1.0f + exp(-gs[qi]));
                float gc = safe_tanh(gs[qi] / p.gate_cap) * p.gate_cap;
                float uc = safe_tanh(us[qi] / p.up_cap) * p.up_cap;
                act_out[long(tks[qi]) * long(p.inter) + long(j)] = float(bfloat((gc * sig * uc) * topk_wts[tks[qi]]));
            }
        }
    }
}

kernel void moe_q2_w2_dedup_kernel(
    device const float* act_in     [[buffer(0)]],
    device const uint* offsets     [[buffer(1)]],
    device const uint* sorted_tk   [[buffer(2)]],
    device const uchar* w2_u8      [[buffer(3)]],
    device const half* w2_scale    [[buffer(4)]],
    device const half* w2_min      [[buffer(5)]],
    device bfloat* w2_out          [[buffer(6)]],
    constant MoeParams& p          [[buffer(7)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]])
{
    uint expert = tgid.y;
    uint d = tgid.x * 4u + sg_id;
    if (d >= p.D) return;
    uint s0 = offsets[expert];
    uint s1 = offsets[expert + 1u];
    if (s0 >= s1) return;

    long q_I = long(p.inter / 4u);
    long g_I = long(p.inter / 64u);
    uint n_hg = p.inter / 32u;
    long row_idx = long(expert) * long(p.D) + long(d);
    device const uint2* w2_2 = reinterpret_cast<device const uint2*>(w2_u8 + row_idx * q_I);
    device const half* s2r = w2_scale + row_idx * g_I;
    device const half* m2r = w2_min + row_idx * g_I;

    for (uint idx = s0; idx < s1; idx += 4u) {
        uint n = min(4u, s1 - idx);
        uint tk0 = sorted_tk[idx];
        uint tk1 = (n > 1u) ? sorted_tk[idx + 1u] : tk0;
        uint tk2 = (n > 2u) ? sorted_tk[idx + 2u] : tk0;
        uint tk3 = (n > 3u) ? sorted_tk[idx + 3u] : tk0;
        device const float4* a0 = reinterpret_cast<device const float4*>(act_in + long(tk0) * long(p.inter));
        device const float4* a1 = reinterpret_cast<device const float4*>(act_in + long(tk1) * long(p.inter));
        device const float4* a2 = reinterpret_cast<device const float4*>(act_in + long(tk2) * long(p.inter));
        device const float4* a3 = reinterpret_cast<device const float4*>(act_in + long(tk3) * long(p.inter));

        float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
        for (uint hg = lane; hg < n_hg; hg += 32u) {
            uint grp = hg >> 1;
            float sc = float(s2r[grp]), mn = float(m2r[grp]);
            uint2 bw = w2_2[hg];
            float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
            float s0v = 0.0f, s1v = 0.0f, s2v = 0.0f, s3v = 0.0f;
            for (uint h = 0u; h < 2u; ++h) {
                uint pw = (h == 0u) ? bw.x : bw.y;
                for (uint q = 0u; q < 4u; ++q) {
                    uint off = hg * 8u + h * 4u + q;
                    uint b = (pw >> (q * 8u)) & 0xFFu;
                    float4 fw = float4(float(b & 3u), float((b >> 2) & 3u), float((b >> 4) & 3u), float((b >> 6) & 3u));
                    float4 v0 = a0[off]; q0 += dot(fw, v0); s0v += (v0.x + v0.y + v0.z + v0.w);
                    if (n > 1u) { float4 v1 = a1[off]; q1 += dot(fw, v1); s1v += (v1.x + v1.y + v1.z + v1.w); }
                    if (n > 2u) { float4 v2 = a2[off]; q2 += dot(fw, v2); s2v += (v2.x + v2.y + v2.z + v2.w); }
                    if (n > 3u) { float4 v3 = a3[off]; q3 += dot(fw, v3); s3v += (v3.x + v3.y + v3.z + v3.w); }
                }
            }
            r0 = fma(q0, sc, fma(s0v, mn, r0));
            if (n > 1u) r1 = fma(q1, sc, fma(s1v, mn, r1));
            if (n > 2u) r2 = fma(q2, sc, fma(s2v, mn, r2));
            if (n > 3u) r3 = fma(q3, sc, fma(s3v, mn, r3));
        }
        r0 = simd_sum(r0);
        if (n > 1u) r1 = simd_sum(r1);
        if (n > 2u) r2 = simd_sum(r2);
        if (n > 3u) r3 = simd_sum(r3);
        if (lane == 0u) {
            w2_out[long(tk0) * long(p.D) + long(d)] = bfloat(r0);
            if (n > 1u) w2_out[long(tk1) * long(p.D) + long(d)] = bfloat(r1);
            if (n > 2u) w2_out[long(tk2) * long(p.D) + long(d)] = bfloat(r2);
            if (n > 3u) w2_out[long(tk3) * long(p.D) + long(d)] = bfloat(r3);
        }
    }
}

kernel void moe_q3_w13_kernel(
    device const bfloat* x         [[buffer(0)]],
    device const int* topk_ids     [[buffer(1)]],
    device const float* topk_wts   [[buffer(2)]],
    device const uchar* w13_lo     [[buffer(3)]],
    device const uchar* w13_hi     [[buffer(4)]],
    device const half* w13_scale   [[buffer(5)]],
    device const half* w13_min     [[buffer(6)]],
    device float* act_out          [[buffer(7)]],
    constant MoeParams& p          [[buffer(8)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]])
{
    uint tk = tgid.y;
    uint t = tk / p.K;
    int expert = topk_ids[tk];
    float gamma = topk_wts[tk];

    uint j = tgid.x * 4u + sg_id;
    if (j >= p.inter) return;

    long lo_D = long(p.D / 4u);
    long hi_D = long(p.D / 8u);
    long g_D = long(p.D / 64u);
    uint n_hg = p.D / 32u;
    long exp_base = long(expert) * long(2u * p.inter);
    long row_g = exp_base + long(2u * j);
    long row_u = row_g + 1L;

    device const uint2* wg_lo = reinterpret_cast<device const uint2*>(w13_lo + row_g * lo_D);
    device const uint2* wu_lo = reinterpret_cast<device const uint2*>(w13_lo + row_u * lo_D);
    device const uint* wg_hi = reinterpret_cast<device const uint*>(w13_hi + row_g * hi_D);
    device const uint* wu_hi = reinterpret_cast<device const uint*>(w13_hi + row_u * hi_D);
    device const half* sg = w13_scale + row_g * g_D;
    device const half* mg = w13_min + row_g * g_D;
    device const half* su = w13_scale + row_u * g_D;
    device const half* mu = w13_min + row_u * g_D;
    device const bfloat4* xr4 = reinterpret_cast<device const bfloat4*>(x + long(t) * long(p.D));

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (uint hg = lane; hg < n_hg; hg += 32u) {
        uint grp = hg >> 1;
        float scg = float(sg[grp]), mng = float(mg[grp]);
        float scu = float(su[grp]), mnu = float(mu[grp]);
        uint2 bg = wg_lo[hg];
        uint2 bu = wu_lo[hg];
        uint hg_bits = wg_hi[hg];
        uint hu_bits = wu_hi[hg];
        float qg_x = 0.0f, qu_x = 0.0f, sx = 0.0f;
        for (uint h = 0u; h < 2u; ++h) {
            uint pg = (h == 0u) ? bg.x : bg.y;
            uint pu = (h == 0u) ? bu.x : bu.y;
            for (uint q = 0u; q < 4u; ++q) {
                float4 xv = float4(xr4[hg * 8u + h * 4u + q]);
                uint bgb = (pg >> (q * 8u)) & 0xFFu;
                uint bub = (pu >> (q * 8u)) & 0xFFu;
                uint hgb = (hg_bits >> (h * 16u + q * 4u)) & 0xFu;
                uint hub = (hu_bits >> (h * 16u + q * 4u)) & 0xFu;
                float4 fg = float4(
                    float((bgb & 3u) | ((hgb & 1u) << 2)),
                    float(((bgb >> 2) & 3u) | ((hgb & 2u) << 1)),
                    float(((bgb >> 4) & 3u) | (hgb & 4u)),
                    float(((bgb >> 6) & 3u) | ((hgb & 8u) >> 1)));
                float4 fu = float4(
                    float((bub & 3u) | ((hub & 1u) << 2)),
                    float(((bub >> 2) & 3u) | ((hub & 2u) << 1)),
                    float(((bub >> 4) & 3u) | (hub & 4u)),
                    float(((bub >> 6) & 3u) | ((hub & 8u) >> 1)));
                qg_x += dot(fg, xv);
                qu_x += dot(fu, xv);
                sx += (xv.x + xv.y + xv.z + xv.w);
            }
        }
        acc_g = fma(qg_x, scg, fma(sx, mng, acc_g));
        acc_u = fma(qu_x, scu, fma(sx, mnu, acc_u));
    }
    acc_g = simd_sum(acc_g);
    acc_u = simd_sum(acc_u);
    if (lane == 0u) {
        float sig = 1.0f / (1.0f + exp(-acc_g));
        float gc = safe_tanh(acc_g / p.gate_cap) * p.gate_cap;
        float uc = safe_tanh(acc_u / p.up_cap) * p.up_cap;
        act_out[long(tk) * long(p.inter) + long(j)] = float(bfloat((gc * sig * uc) * gamma));
    }
}

kernel void moe_q3_w2_kernel(
    device const float* act_in     [[buffer(0)]],
    device const int* topk_ids     [[buffer(1)]],
    device const uchar* w2_lo      [[buffer(2)]],
    device const uchar* w2_hi      [[buffer(3)]],
    device const half* w2_scale    [[buffer(4)]],
    device const half* w2_min      [[buffer(5)]],
    device bfloat* out             [[buffer(6)]],
    constant MoeParams& p          [[buffer(7)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]])
{
    uint t = tgid.y;
    uint d = tgid.x * 4u + sg_id;
    if (d >= p.D) return;

    long lo_I = long(p.inter / 4u);
    long hi_I = long(p.inter / 8u);
    long g_I = long(p.inter / 64u);
    uint n_hg = p.inter / 32u;

    float total = 0.0f;
    for (uint k = 0u; k < p.K; ++k) {
        uint tk = t * p.K + k;
        int expert = topk_ids[tk];
        long row_idx = long(expert) * long(p.D) + long(d);
        device const uint2* w2_l = reinterpret_cast<device const uint2*>(w2_lo + row_idx * lo_I);
        device const uint* w2_h = reinterpret_cast<device const uint*>(w2_hi + row_idx * hi_I);
        device const half* s2r = w2_scale + row_idx * g_I;
        device const half* m2r = w2_min + row_idx * g_I;
        device const float4* ar4 = reinterpret_cast<device const float4*>(act_in + long(tk) * long(p.inter));

        float acc = 0.0f;
        for (uint hg = lane; hg < n_hg; hg += 32u) {
            uint grp = hg >> 1;
            float sc = float(s2r[grp]), mn = float(m2r[grp]);
            uint2 bw = w2_l[hg];
            uint hw = w2_h[hg];
            float qa = 0.0f, sa = 0.0f;
            for (uint h = 0u; h < 2u; ++h) {
                uint pw = (h == 0u) ? bw.x : bw.y;
                for (uint q = 0u; q < 4u; ++q) {
                    float4 av = ar4[hg * 8u + h * 4u + q];
                    uint b = (pw >> (q * 8u)) & 0xFFu;
                    uint hb = (hw >> (h * 16u + q * 4u)) & 0xFu;
                    float4 fw = float4(
                        float((b & 3u) | ((hb & 1u) << 2)),
                        float(((b >> 2) & 3u) | ((hb & 2u) << 1)),
                        float(((b >> 4) & 3u) | (hb & 4u)),
                        float(((b >> 6) & 3u) | ((hb & 8u) >> 1)));
                    qa += dot(fw, av);
                    sa += (av.x + av.y + av.z + av.w);
                }
            }
            acc = fma(qa, sc, fma(sa, mn, acc));
        }
        acc = simd_sum(acc);
        if (lane == 0u) total += acc;
    }
    if (lane == 0u) out[long(t) * long(p.D) + long(d)] = bfloat(total);
}

kernel void moe_q3_w13_dedup_kernel(
    device const bfloat* x         [[buffer(0)]],
    device const float* topk_wts   [[buffer(1)]],
    device const uint* offsets     [[buffer(2)]],
    device const uint* sorted_tk   [[buffer(3)]],
    device const uchar* w13_lo     [[buffer(4)]],
    device const uchar* w13_hi     [[buffer(5)]],
    device const half* w13_scale   [[buffer(6)]],
    device const half* w13_min     [[buffer(7)]],
    device float* act_out          [[buffer(8)]],
    constant MoeParams& p          [[buffer(9)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]])
{
    uint expert = tgid.y;
    uint j = tgid.x * 4u + sg_id;
    if (j >= p.inter) return;
    uint s0 = offsets[expert];
    uint s1 = offsets[expert + 1u];
    if (s0 >= s1) return;

    long lo_D = long(p.D / 4u);
    long hi_D = long(p.D / 8u);
    long g_D = long(p.D / 64u);
    uint n_hg = p.D / 32u;
    long exp_base = long(expert) * long(2u * p.inter);
    long row_g = exp_base + long(2u * j);
    long row_u = row_g + 1L;

    device const uint2* wg_lo = reinterpret_cast<device const uint2*>(w13_lo + row_g * lo_D);
    device const uint2* wu_lo = reinterpret_cast<device const uint2*>(w13_lo + row_u * lo_D);
    device const uint* wg_hi = reinterpret_cast<device const uint*>(w13_hi + row_g * hi_D);
    device const uint* wu_hi = reinterpret_cast<device const uint*>(w13_hi + row_u * hi_D);
    device const half* sg = w13_scale + row_g * g_D;
    device const half* mg = w13_min + row_g * g_D;
    device const half* su = w13_scale + row_u * g_D;
    device const half* mu = w13_min + row_u * g_D;

    for (uint idx = s0; idx < s1; idx += 4u) {
        uint n = min(4u, s1 - idx);
        uint tk0 = sorted_tk[idx];
        uint tk1 = (n > 1u) ? sorted_tk[idx + 1u] : tk0;
        uint tk2 = (n > 2u) ? sorted_tk[idx + 2u] : tk0;
        uint tk3 = (n > 3u) ? sorted_tk[idx + 3u] : tk0;
        device const bfloat4* x0 = reinterpret_cast<device const bfloat4*>(x + long(tk0 / p.K) * long(p.D));
        device const bfloat4* x1 = reinterpret_cast<device const bfloat4*>(x + long(tk1 / p.K) * long(p.D));
        device const bfloat4* x2 = reinterpret_cast<device const bfloat4*>(x + long(tk2 / p.K) * long(p.D));
        device const bfloat4* x3 = reinterpret_cast<device const bfloat4*>(x + long(tk3 / p.K) * long(p.D));

        float ag0 = 0.0f, ag1 = 0.0f, ag2 = 0.0f, ag3 = 0.0f;
        float au0 = 0.0f, au1 = 0.0f, au2 = 0.0f, au3 = 0.0f;
        for (uint hg = lane; hg < n_hg; hg += 32u) {
            uint grp = hg >> 1;
            float scg = float(sg[grp]), mng = float(mg[grp]);
            float scu = float(su[grp]), mnu = float(mu[grp]);
            uint2 bg = wg_lo[hg];
            uint2 bu = wu_lo[hg];
            uint hg_bits = wg_hi[hg];
            uint hu_bits = wu_hi[hg];
            float qg0 = 0.0f, qg1 = 0.0f, qg2 = 0.0f, qg3 = 0.0f;
            float qu0 = 0.0f, qu1 = 0.0f, qu2 = 0.0f, qu3 = 0.0f;
            float sx0 = 0.0f, sx1 = 0.0f, sx2 = 0.0f, sx3 = 0.0f;
            for (uint h = 0u; h < 2u; ++h) {
                uint pg = (h == 0u) ? bg.x : bg.y;
                uint pu = (h == 0u) ? bu.x : bu.y;
                for (uint q = 0u; q < 4u; ++q) {
                    uint off = hg * 8u + h * 4u + q;
                    uint bgb = (pg >> (q * 8u)) & 0xFFu;
                    uint bub = (pu >> (q * 8u)) & 0xFFu;
                    uint hgb = (hg_bits >> (h * 16u + q * 4u)) & 0xFu;
                    uint hub = (hu_bits >> (h * 16u + q * 4u)) & 0xFu;
                    float4 fg = float4(
                        float((bgb & 3u) | ((hgb & 1u) << 2)),
                        float(((bgb >> 2) & 3u) | ((hgb & 2u) << 1)),
                        float(((bgb >> 4) & 3u) | (hgb & 4u)),
                        float(((bgb >> 6) & 3u) | ((hgb & 8u) >> 1)));
                    float4 fu = float4(
                        float((bub & 3u) | ((hub & 1u) << 2)),
                        float(((bub >> 2) & 3u) | ((hub & 2u) << 1)),
                        float(((bub >> 4) & 3u) | (hub & 4u)),
                        float(((bub >> 6) & 3u) | ((hub & 8u) >> 1)));
                    float4 v0 = float4(x0[off]);
                    qg0 += dot(fg, v0); qu0 += dot(fu, v0); sx0 += (v0.x + v0.y + v0.z + v0.w);
                    if (n > 1u) { float4 v1 = float4(x1[off]); qg1 += dot(fg, v1); qu1 += dot(fu, v1); sx1 += (v1.x + v1.y + v1.z + v1.w); }
                    if (n > 2u) { float4 v2 = float4(x2[off]); qg2 += dot(fg, v2); qu2 += dot(fu, v2); sx2 += (v2.x + v2.y + v2.z + v2.w); }
                    if (n > 3u) { float4 v3 = float4(x3[off]); qg3 += dot(fg, v3); qu3 += dot(fu, v3); sx3 += (v3.x + v3.y + v3.z + v3.w); }
                }
            }
            ag0 = fma(qg0, scg, fma(sx0, mng, ag0)); au0 = fma(qu0, scu, fma(sx0, mnu, au0));
            if (n > 1u) { ag1 = fma(qg1, scg, fma(sx1, mng, ag1)); au1 = fma(qu1, scu, fma(sx1, mnu, au1)); }
            if (n > 2u) { ag2 = fma(qg2, scg, fma(sx2, mng, ag2)); au2 = fma(qu2, scu, fma(sx2, mnu, au2)); }
            if (n > 3u) { ag3 = fma(qg3, scg, fma(sx3, mng, ag3)); au3 = fma(qu3, scu, fma(sx3, mnu, au3)); }
        }
        ag0 = simd_sum(ag0); au0 = simd_sum(au0);
        if (n > 1u) { ag1 = simd_sum(ag1); au1 = simd_sum(au1); }
        if (n > 2u) { ag2 = simd_sum(ag2); au2 = simd_sum(au2); }
        if (n > 3u) { ag3 = simd_sum(ag3); au3 = simd_sum(au3); }
        if (lane == 0u) {
            uint tks[4] = {tk0, tk1, tk2, tk3};
            float gs[4] = {ag0, ag1, ag2, ag3};
            float us[4] = {au0, au1, au2, au3};
            for (uint qi = 0u; qi < n; ++qi) {
                float sig = 1.0f / (1.0f + exp(-gs[qi]));
                float gc = safe_tanh(gs[qi] / p.gate_cap) * p.gate_cap;
                float uc = safe_tanh(us[qi] / p.up_cap) * p.up_cap;
                act_out[long(tks[qi]) * long(p.inter) + long(j)] = float(bfloat((gc * sig * uc) * topk_wts[tks[qi]]));
            }
        }
    }
}

kernel void moe_q3_w2_dedup_kernel(
    device const float* act_in     [[buffer(0)]],
    device const uint* offsets     [[buffer(1)]],
    device const uint* sorted_tk   [[buffer(2)]],
    device const uchar* w2_lo      [[buffer(3)]],
    device const uchar* w2_hi      [[buffer(4)]],
    device const half* w2_scale    [[buffer(5)]],
    device const half* w2_min      [[buffer(6)]],
    device bfloat* w2_out          [[buffer(7)]],
    constant MoeParams& p          [[buffer(8)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]])
{
    uint expert = tgid.y;
    uint d = tgid.x * 4u + sg_id;
    if (d >= p.D) return;
    uint s0 = offsets[expert];
    uint s1 = offsets[expert + 1u];
    if (s0 >= s1) return;

    long lo_I = long(p.inter / 4u);
    long hi_I = long(p.inter / 8u);
    long g_I = long(p.inter / 64u);
    uint n_hg = p.inter / 32u;
    long row_idx = long(expert) * long(p.D) + long(d);
    device const uint2* w2_l = reinterpret_cast<device const uint2*>(w2_lo + row_idx * lo_I);
    device const uint* w2_h = reinterpret_cast<device const uint*>(w2_hi + row_idx * hi_I);
    device const half* s2r = w2_scale + row_idx * g_I;
    device const half* m2r = w2_min + row_idx * g_I;

    for (uint idx = s0; idx < s1; idx += 4u) {
        uint n = min(4u, s1 - idx);
        uint tk0 = sorted_tk[idx];
        uint tk1 = (n > 1u) ? sorted_tk[idx + 1u] : tk0;
        uint tk2 = (n > 2u) ? sorted_tk[idx + 2u] : tk0;
        uint tk3 = (n > 3u) ? sorted_tk[idx + 3u] : tk0;
        device const float4* a0 = reinterpret_cast<device const float4*>(act_in + long(tk0) * long(p.inter));
        device const float4* a1 = reinterpret_cast<device const float4*>(act_in + long(tk1) * long(p.inter));
        device const float4* a2 = reinterpret_cast<device const float4*>(act_in + long(tk2) * long(p.inter));
        device const float4* a3 = reinterpret_cast<device const float4*>(act_in + long(tk3) * long(p.inter));

        float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
        for (uint hg = lane; hg < n_hg; hg += 32u) {
            uint grp = hg >> 1;
            float sc = float(s2r[grp]), mn = float(m2r[grp]);
            uint2 bw = w2_l[hg];
            uint hw = w2_h[hg];
            float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
            float s0v = 0.0f, s1v = 0.0f, s2v = 0.0f, s3v = 0.0f;
            for (uint h = 0u; h < 2u; ++h) {
                uint pw = (h == 0u) ? bw.x : bw.y;
                for (uint q = 0u; q < 4u; ++q) {
                    uint off = hg * 8u + h * 4u + q;
                    uint b = (pw >> (q * 8u)) & 0xFFu;
                    uint hb = (hw >> (h * 16u + q * 4u)) & 0xFu;
                    float4 fw = float4(
                        float((b & 3u) | ((hb & 1u) << 2)),
                        float(((b >> 2) & 3u) | ((hb & 2u) << 1)),
                        float(((b >> 4) & 3u) | (hb & 4u)),
                        float(((b >> 6) & 3u) | ((hb & 8u) >> 1)));
                    float4 v0 = a0[off]; q0 += dot(fw, v0); s0v += (v0.x + v0.y + v0.z + v0.w);
                    if (n > 1u) { float4 v1 = a1[off]; q1 += dot(fw, v1); s1v += (v1.x + v1.y + v1.z + v1.w); }
                    if (n > 2u) { float4 v2 = a2[off]; q2 += dot(fw, v2); s2v += (v2.x + v2.y + v2.z + v2.w); }
                    if (n > 3u) { float4 v3 = a3[off]; q3 += dot(fw, v3); s3v += (v3.x + v3.y + v3.z + v3.w); }
                }
            }
            r0 = fma(q0, sc, fma(s0v, mn, r0));
            if (n > 1u) r1 = fma(q1, sc, fma(s1v, mn, r1));
            if (n > 2u) r2 = fma(q2, sc, fma(s2v, mn, r2));
            if (n > 3u) r3 = fma(q3, sc, fma(s3v, mn, r3));
        }
        r0 = simd_sum(r0);
        if (n > 1u) r1 = simd_sum(r1);
        if (n > 2u) r2 = simd_sum(r2);
        if (n > 3u) r3 = simd_sum(r3);
        if (lane == 0u) {
            w2_out[long(tk0) * long(p.D) + long(d)] = bfloat(r0);
            if (n > 1u) w2_out[long(tk1) * long(p.D) + long(d)] = bfloat(r1);
            if (n > 2u) w2_out[long(tk2) * long(p.D) + long(d)] = bfloat(r2);
            if (n > 3u) w2_out[long(tk3) * long(p.D) + long(d)] = bfloat(r3);
        }
    }
}

