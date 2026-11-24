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
    uint sg_id                  [[simdgroup_index_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]])
{
    uint t = tgid.y;
    uint e = tgid.x * 4u + sg_id;
    if (e >= p.E || t >= p.T) return;

    device const bfloat* xr = x + long(t) * long(p.D);
    device const bfloat* wr = weight + long(e) * long(p.D);
    float acc = 0.0f;
    for (uint d = lane; d < p.D; d += 32u) {
        acc = fma(float(wr[d]), float(xr[d]), acc);
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
    uint tid                   [[thread_index_in_threadgroup]])
{
    uint t = tgid;
    if (t >= p.T) return;
    threadgroup float s_scores[128];
    threadgroup float s_sel[128];
    if (tid < p.E) {
        float sig = 1.0f / (1.0f + exp(-logits[long(t) * long(p.E) + long(tid)]));
        s_scores[tid] = sig;
        s_sel[tid] = sig + bias[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        int ids[8];
        float wts[8];
        float wsum = 0.0f;
        for (uint k = 0; k < p.K; ++k) {
            int best_e = 0;
            float best_v = -1e30f;
            for (uint i = 0; i < p.E; ++i) {
                if (s_sel[i] > best_v) {
                    best_v = s_sel[i];
                    best_e = int(i);
                }
            }
            ids[k] = best_e;
            wts[k] = s_scores[best_e];
            wsum += wts[k];
            s_sel[best_e] = -1e30f;
        }
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
    device const bfloat* xr = x + long(t) * long(p.D);

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (uint blk = lane; blk < uint(scale_D); blk += 32u) {
        float bsg = e4m3_to_f32(sg[blk]);
        float bsu = e4m3_to_f32(su[blk]);
        device const uchar* wgb = wg + long(blk) * 8;
        device const uchar* wub = wu + long(blk) * 8;
        device const bfloat* xblk = xr + long(blk) * 16;
        float lg = 0.0f, lu = 0.0f;
        for (uint b = 0; b < 8u; ++b) {
            uchar bg = wgb[b];
            uchar bu = wub[b];
            float x0 = float(xblk[b * 2u + 0u]);
            float x1 = float(xblk[b * 2u + 1u]);
            lg = fma(kE2M1Lut[bg & 0xFu], x0, fma(kE2M1Lut[bg >> 4], x1, lg));
            lu = fma(kE2M1Lut[bu & 0xFu], x0, fma(kE2M1Lut[bu >> 4], x1, lu));
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
        device const uchar* w2r = w2_u8 + row_idx * half_I;
        device const uchar* s2r = w2_scale + row_idx * scale_I;
        device const float* ar = act_in + long(tk) * long(p.inter);

        float acc = 0.0f;
        for (uint blk = lane; blk < uint(scale_I); blk += 32u) {
            float bs = e4m3_to_f32(s2r[blk]);
            device const uchar* wb = w2r + long(blk) * 8;
            device const float* ab = ar + long(blk) * 16;
            float loc = 0.0f;
            for (uint b = 0; b < 8u; ++b) {
                uchar bw = wb[b];
                loc = fma(kE2M1Lut[bw & 0xFu], ab[b * 2u + 0u], fma(kE2M1Lut[bw >> 4], ab[b * 2u + 1u], loc));
            }
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
    float acc = 0.0f;
    uint k4 = p.K >> 2;
    for (uint i = lane; i < k4; i += 32u) {
        uint k = i << 2;
        acc = fma(float(wr[k + 0u]), float(xr[k + 0u]), acc);
        acc = fma(float(wr[k + 1u]), float(xr[k + 1u]), acc);
        acc = fma(float(wr[k + 2u]), float(xr[k + 2u]), acc);
        acc = fma(float(wr[k + 3u]), float(xr[k + 3u]), acc);
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
