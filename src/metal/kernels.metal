#include <metal_stdlib>
using namespace metal;

constant float kE2M1Lut[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

inline float e4m3_to_f32(uchar b) {
    uint sign = (b >> 7) & 1u;
    uint exp = (b >> 3) & 0xFu;
    uint mant = b & 0x7u;
    if (exp == 15u && mant == 7u) return 0.0f;
    float v;
    if (exp == 0u) {
        v = float(mant) * 0.001953125f;
    } else {
        uint fbits = ((exp + 120u) << 23) | (mant << 20);
        v = as_type<float>(fbits);
    }
    return sign ? -v : v;
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
        float v = (1.0f - tanh(f)) * dest[off + long(d)] + tanh(g) * message[off + long(d)];
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
            out_fp32[off] = (1.0f - tanh(f)) * dest[off] + tanh(g) * message[off];
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
        float g_cap = tanh(acc_g / p.gate_cap) * p.gate_cap;
        float u_cap = tanh(acc_u / p.up_cap) * p.up_cap;
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
        float g_cap = tanh(acc_g / p.gate_cap) * p.gate_cap;
        float u_cap = tanh(acc_u / p.up_cap) * p.up_cap;
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
        float v = tanh(logits[i] * inv_cap) * p.softcap;
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
    uint n = tgid * 4u + sg_id;
    if (n >= p.N) return;

    device const uchar4* wr4 = reinterpret_cast<device const uchar4*>(W8 + long(n) * long(p.K));
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
    uint M = p.M;

    for (uint i = lane; i < k4; i += 32u) {
        uchar4 wv = wr4[i];
        float w0 = e4m3_to_f32(wv.x);
        float w1 = e4m3_to_f32(wv.y);
        float w2 = e4m3_to_f32(wv.z);
        float w3 = e4m3_to_f32(wv.w);
        { float4 xv = float4(x0[i]); a0 = fma(w0, xv.x, fma(w1, xv.y, fma(w2, xv.z, fma(w3, xv.w, a0)))); }
        if (M > 1u) { float4 xv = float4(x1[i]); a1 = fma(w0, xv.x, fma(w1, xv.y, fma(w2, xv.z, fma(w3, xv.w, a1)))); }
        if (M > 2u) { float4 xv = float4(x2[i]); a2 = fma(w0, xv.x, fma(w1, xv.y, fma(w2, xv.z, fma(w3, xv.w, a2)))); }
        if (M > 3u) { float4 xv = float4(x3[i]); a3 = fma(w0, xv.x, fma(w1, xv.y, fma(w2, xv.z, fma(w3, xv.w, a3)))); }
        if (M > 4u) { float4 xv = float4(x4[i]); a4 = fma(w0, xv.x, fma(w1, xv.y, fma(w2, xv.z, fma(w3, xv.w, a4)))); }
        if (M > 5u) { float4 xv = float4(x5[i]); a5 = fma(w0, xv.x, fma(w1, xv.y, fma(w2, xv.z, fma(w3, xv.w, a5)))); }
        if (M > 6u) { float4 xv = float4(x6[i]); a6 = fma(w0, xv.x, fma(w1, xv.y, fma(w2, xv.z, fma(w3, xv.w, a6)))); }
        if (M > 7u) { float4 xv = float4(x7[i]); a7 = fma(w0, xv.x, fma(w1, xv.y, fma(w2, xv.z, fma(w3, xv.w, a7)))); }
    }

    float sc = Wsc[n];
    float v0 = simd_sum(a0) * sc;
    float v1 = (M > 1u) ? (simd_sum(a1) * sc) : 0.0f;
    float v2 = (M > 2u) ? (simd_sum(a2) * sc) : 0.0f;
    float v3 = (M > 3u) ? (simd_sum(a3) * sc) : 0.0f;
    float v4 = (M > 4u) ? (simd_sum(a4) * sc) : 0.0f;
    float v5 = (M > 5u) ? (simd_sum(a5) * sc) : 0.0f;
    float v6 = (M > 6u) ? (simd_sum(a6) * sc) : 0.0f;
    float v7 = (M > 7u) ? (simd_sum(a7) * sc) : 0.0f;

    if (lane == 0u) {
        if (p.out_fp32 != 0u) {
            Y_fp32[0L * long(p.N) + long(n)] = v0;
            if (M > 1u) Y_fp32[1L * long(p.N) + long(n)] = v1;
            if (M > 2u) Y_fp32[2L * long(p.N) + long(n)] = v2;
            if (M > 3u) Y_fp32[3L * long(p.N) + long(n)] = v3;
            if (M > 4u) Y_fp32[4L * long(p.N) + long(n)] = v4;
            if (M > 5u) Y_fp32[5L * long(p.N) + long(n)] = v5;
            if (M > 6u) Y_fp32[6L * long(p.N) + long(n)] = v6;
            if (M > 7u) Y_fp32[7L * long(p.N) + long(n)] = v7;
        } else {
            Y_bf16[0L * long(p.N) + long(n)] = bfloat(v0);
            if (M > 1u) Y_bf16[1L * long(p.N) + long(n)] = bfloat(v1);
            if (M > 2u) Y_bf16[2L * long(p.N) + long(n)] = bfloat(v2);
            if (M > 3u) Y_bf16[3L * long(p.N) + long(n)] = bfloat(v3);
            if (M > 4u) Y_bf16[4L * long(p.N) + long(n)] = bfloat(v4);
            if (M > 5u) Y_bf16[5L * long(p.N) + long(n)] = bfloat(v5);
            if (M > 6u) Y_bf16[6L * long(p.N) + long(n)] = bfloat(v6);
            if (M > 7u) Y_bf16[7L * long(p.N) + long(n)] = bfloat(v7);
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
    cand = tanh(cand);
    state[c] = bfloat(g * st + (1.0f - g) * cand);
    latent[c] = bfloat(tanh(o));
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

