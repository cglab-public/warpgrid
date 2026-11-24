#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#ifdef WARPGRID_METAL_PRECOMPILED
#include "../src/metal/kernels_metallib_embed.hpp"
#else
#include "../src/metal/kernels_source_embed.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_passed = 0;
static int g_failed = 0;

#define EXPECT_TRUE(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failed; \
        return; \
    } \
} while (0)

static uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return (uint16_t)(u >> 16);
}

static float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static float e4m3_ref(uint8_t b) {
    int sign = (b >> 7) & 1;
    int exp = (b >> 3) & 0xF;
    int mant = b & 0x7;
    if (exp == 15 && mant == 7) return 0.0f;
    float v = (exp == 0) ? (mant / 8.0f) * std::ldexp(1.0f, -6)
                         : (1.0f + mant / 8.0f) * std::ldexp(1.0f, exp - 7);
    return sign ? -v : v;
}

static float ue8m0_ref(uint8_t e) {
    if (e == 0) return 5.877471754111438e-39f;
    uint32_t bits = (uint32_t)e << 23;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

struct MetalCtx {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> q = nil;
    id<MTLLibrary> lib = nil;

    bool init() {
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) return false;
        q = [dev newCommandQueue];
        NSError* err = nil;
#ifdef WARPGRID_METAL_PRECOMPILED
        dispatch_data_t data = dispatch_data_create(
            kMetalLibBytes, kMetalLibLen, dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        lib = [dev newLibraryWithData:data error:&err];
#else
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion3_1;
        NSString* src = [[NSString alloc] initWithBytes:kMetalSourceBytes
                                                 length:kMetalSourceLen
                                               encoding:NSUTF8StringEncoding];
        lib = [dev newLibraryWithSource:src options:opts error:&err];
#endif
        if (!lib) {
            fprintf(stderr, "Metal library error: %s\n", err ? err.localizedDescription.UTF8String : "unknown");
            return false;
        }
        return true;
    }

    id<MTLComputePipelineState> pso(const char* name) {
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (!fn) {
            fprintf(stderr, "Missing Metal kernel: %s\n", name);
            return nil;
        }
        NSError* err = nil;
        id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!p) {
            fprintf(stderr, "PSO error (%s): %s\n", name, err ? err.localizedDescription.UTF8String : "unknown");
        }
        return p;
    }

    id<MTLBuffer> buf(size_t nbytes, const void* init_data = nullptr) {
        id<MTLBuffer> b = [dev newBufferWithLength:std::max<size_t>(16, nbytes) options:MTLResourceStorageModeShared];
        if (init_data && nbytes > 0) memcpy(b.contents, init_data, nbytes);
        else memset(b.contents, 0, std::max<size_t>(16, nbytes));
        return b;
    }

    bool run_cb(id<MTLCommandBuffer> cb) {
        [cb commit];
        [cb waitUntilCompleted];
        return cb.status == MTLCommandBufferStatusCompleted && cb.error == nil;
    }
};

static MetalCtx g_ctx;

static void run_test(const char* name, void (*fn)()) {
    int before = g_failed;
    fn();
    if (g_failed == before) {
        ++g_passed;
        printf("[PASS] %s\n", name);
    } else {
        printf("[FAIL] %s\n", name);
    }
}

static void test_rmsnorm_and_add() {
    auto p_rms = g_ctx.pso("rmsnorm_kernel");
    auto p_add = g_ctx.pso("add_bf16_kernel");
    EXPECT_TRUE(p_rms != nil && p_add != nil);

    const uint32_t T = 2, D = 256;
    const float eps = 1e-6f;
    std::vector<uint16_t> x(T * D), w(D);
    for (uint32_t i = 0; i < T * D; ++i) x[i] = f32_to_bf16(std::sin((float)i * 0.13f));
    for (uint32_t d = 0; d < D; ++d) w[d] = f32_to_bf16(0.5f + 0.5f * std::cos((float)d * 0.07f));

    auto bx = g_ctx.buf(x.size() * 2, x.data());
    auto bw = g_ctx.buf(w.size() * 2, w.data());
    auto by = g_ctx.buf(x.size() * 2);
    auto bsum = g_ctx.buf(x.size() * 2);

    struct RmsParams { uint32_t T; uint32_t D; float eps; } rp{T, D, eps};
    uint32_t N = T * D;

    id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p_rms];
    [enc setBuffer:bx offset:0 atIndex:0];
    [enc setBuffer:bw offset:0 atIndex:1];
    [enc setBuffer:by offset:0 atIndex:2];
    [enc setBytes:&rp length:sizeof(rp) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc setComputePipelineState:p_add];
    [enc setBuffer:bx offset:0 atIndex:0];
    [enc setBuffer:by offset:0 atIndex:1];
    [enc setBuffer:bsum offset:0 atIndex:2];
    [enc setBytes:&N length:sizeof(N) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    EXPECT_TRUE(g_ctx.run_cb(cb));

    const uint16_t* gy = (const uint16_t*)by.contents;
    const uint16_t* gs = (const uint16_t*)bsum.contents;
    float max_err = 0.0f;
    for (uint32_t t = 0; t < T; ++t) {
        double ss = 0.0;
        for (uint32_t d = 0; d < D; ++d) {
            float v = bf16_to_f32(x[t * D + d]);
            ss += (double)v * v;
        }
        float rstd = 1.0f / std::sqrt((float)(ss / D) + eps);
        for (uint32_t d = 0; d < D; ++d) {
            float gv = bf16_to_f32(gy[t * D + d]);
            float sv = bf16_to_f32(gs[t * D + d]);
            EXPECT_TRUE(std::isfinite(gv) && std::isfinite(sv));
            float ref = bf16_to_f32(x[t * D + d]) * rstd * bf16_to_f32(w[d]);
            max_err = std::max(max_err, std::abs(gv - ref));
            float sref = bf16_to_f32(x[t * D + d]) + gv;
            max_err = std::max(max_err, std::abs(sv - sref));
        }
    }
    EXPECT_TRUE(max_err < 2e-2f);
}

static void test_sconv_static_and_dynamic() {
    auto p_st = g_ctx.pso("sconv_static_kernel");
    auto p_dy = g_ctx.pso("sconv_dynamic_kernel");
    EXPECT_TRUE(p_st != nil && p_dy != nil);

    const uint32_t T = 3, C = 128;
    std::vector<uint16_t> x(T * C), w(C * 4), st(3 * C, 0);
    for (uint32_t i = 0; i < T * C; ++i) x[i] = f32_to_bf16(std::sin((float)i * 0.11f));
    for (uint32_t i = 0; i < C * 4; ++i) w[i] = f32_to_bf16(0.25f * std::cos((float)i * 0.17f));

    auto bx = g_ctx.buf(x.size() * 2, x.data());
    auto bw = g_ctx.buf(w.size() * 2, w.data());
    auto bst = g_ctx.buf(st.size() * 2, st.data());
    auto bout = g_ctx.buf(x.size() * 2);
    auto bsnap = g_ctx.buf(T * 3 * C * 2);

    struct StaticParams { uint32_t T; uint32_t C; uint32_t has_snap; } sp{T, C, 1};
    id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p_st];
    [enc setBuffer:bx offset:0 atIndex:0];
    [enc setBuffer:bw offset:0 atIndex:1];
    [enc setBuffer:bst offset:0 atIndex:2];
    [enc setBuffer:bout offset:0 atIndex:3];
    [enc setBuffer:bsnap offset:0 atIndex:4];
    [enc setBytes:&sp length:sizeof(sp) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(C, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    EXPECT_TRUE(g_ctx.run_cb(cb));

    const float inv_sqrt5 = 0.44721359549995793f;
    const uint16_t* go = (const uint16_t*)bout.contents;
    float max_err = 0.0f;
    for (uint32_t c = 0; c < C; ++c) {
        float w0 = bf16_to_f32(w[c * 4 + 0]);
        float w1 = bf16_to_f32(w[c * 4 + 1]);
        float w2 = bf16_to_f32(w[c * 4 + 2]);
        float w3 = bf16_to_f32(w[c * 4 + 3]);
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
        for (uint32_t t = 0; t < T; ++t) {
            float xt = bf16_to_f32(x[t * C + c]);
            float y = (w0 * s0 + w1 * s1 + w2 * s2 + w3 * xt + xt) * inv_sqrt5;
            float gv = bf16_to_f32(go[t * C + c]);
            EXPECT_TRUE(std::isfinite(gv));
            max_err = std::max(max_err, std::abs(gv - y));
            s0 = s1; s1 = s2; s2 = xt;
        }
    }
    EXPECT_TRUE(max_err < 2e-2f);
}

static void test_moe_nvfp4_and_mxfp8() {
    auto p_nv13 = g_ctx.pso("moe_nvfp4_w13_kernel");
    auto p_nv2  = g_ctx.pso("moe_nvfp4_w2_kernel");
    auto p_mx13 = g_ctx.pso("moe_mxfp8_w13_kernel");
    auto p_mx2  = g_ctx.pso("moe_mxfp8_w2_kernel");
    EXPECT_TRUE(p_nv13 != nil && p_nv2 != nil && p_mx13 != nil && p_mx2 != nil);

    const uint32_t T = 1, D = 64, inter = 32, E = 2, K = 1;
    std::vector<uint16_t> x(D);
    for (uint32_t d = 0; d < D; ++d) x[d] = f32_to_bf16(0.5f * std::sin((float)d * 0.2f));
    std::vector<int32_t> topk_ids = {1};
    std::vector<float> topk_wts = {1.0f};

    std::vector<uint8_t> w13_u8(E * 2 * inter * (D / 2), 0x22);
    std::vector<uint8_t> w13_sc(E * 2 * inter * (D / 16), 0x38);
    std::vector<float> w13_amax = {1.0f, 1.0f}, w13_denom = {1.0f};
    std::vector<uint8_t> w2_u8(E * D * (inter / 2), 0x22);
    std::vector<uint8_t> w2_sc(E * D * (inter / 16), 0x38);
    std::vector<float> w2_amax = {1.0f, 1.0f}, w2_denom = {1.0f};

    auto bx = g_ctx.buf(x.size() * 2, x.data());
    auto bids = g_ctx.buf(topk_ids.size() * 4, topk_ids.data());
    auto bwts = g_ctx.buf(topk_wts.size() * 4, topk_wts.data());
    auto bw13 = g_ctx.buf(w13_u8.size(), w13_u8.data());
    auto bs13 = g_ctx.buf(w13_sc.size(), w13_sc.data());
    auto ba13 = g_ctx.buf(w13_amax.size() * 4, w13_amax.data());
    auto bd13 = g_ctx.buf(4, w13_denom.data());
    auto bw2  = g_ctx.buf(w2_u8.size(), w2_u8.data());
    auto bs2  = g_ctx.buf(w2_sc.size(), w2_sc.data());
    auto ba2  = g_ctx.buf(w2_amax.size() * 4, w2_amax.data());
    auto bd2  = g_ctx.buf(4, w2_denom.data());
    auto bact = g_ctx.buf(T * K * inter * 4);
    auto bout = g_ctx.buf(T * D * 2);

    struct MoeParams { uint32_t T; uint32_t D; uint32_t inter; uint32_t K; float gate_cap; float up_cap; } mp{T, D, inter, K, 4.0f, 25.0f};
    id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p_nv13];
    [enc setBuffer:bx offset:0 atIndex:0];
    [enc setBuffer:bids offset:0 atIndex:1];
    [enc setBuffer:bwts offset:0 atIndex:2];
    [enc setBuffer:bw13 offset:0 atIndex:3];
    [enc setBuffer:bs13 offset:0 atIndex:4];
    [enc setBuffer:ba13 offset:0 atIndex:5];
    [enc setBuffer:bd13 offset:0 atIndex:6];
    [enc setBuffer:bact offset:0 atIndex:7];
    [enc setBytes:&mp length:sizeof(mp) atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake((inter + 3) / 4, T * K, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    [enc setComputePipelineState:p_nv2];
    [enc setBuffer:bact offset:0 atIndex:0];
    [enc setBuffer:bids offset:0 atIndex:1];
    [enc setBuffer:bw2 offset:0 atIndex:2];
    [enc setBuffer:bs2 offset:0 atIndex:3];
    [enc setBuffer:ba2 offset:0 atIndex:4];
    [enc setBuffer:bd2 offset:0 atIndex:5];
    [enc setBuffer:bout offset:0 atIndex:6];
    [enc setBytes:&mp length:sizeof(mp) atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake((D + 3) / 4, T, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    EXPECT_TRUE(g_ctx.run_cb(cb));

    float sum_x = 0.0f;
    for (uint32_t d = 0; d < D; ++d) sum_x += bf16_to_f32(x[d]);
    float g = sum_x;
    float sig = 1.0f / (1.0f + std::exp(-g));
    float act_ref = bf16_to_f32(f32_to_bf16(std::tanh(g / 4.0f) * 4.0f * sig * std::tanh(g / 25.0f) * 25.0f));
    float out_ref = act_ref * (float)inter;

    const uint16_t* go = (const uint16_t*)bout.contents;
    float max_err = 0.0f;
    for (uint32_t d = 0; d < D; ++d) {
        float gv = bf16_to_f32(go[d]);
        EXPECT_TRUE(std::isfinite(gv));
        max_err = std::max(max_err, std::abs(gv - out_ref));
    }
    EXPECT_TRUE(max_err < 5e-2f);
}

static void test_bf16_gemv_and_softcap_argmax() {
    auto p_gemv = g_ctx.pso("bf16_gemv_kernel");
    auto p_arg  = g_ctx.pso("softcap_argmax_kernel");
    EXPECT_TRUE(p_gemv != nil && p_arg != nil);

    const uint32_t M = 1, N = 64, K = 128;
    std::vector<uint16_t> x(K), W(N * K);
    for (uint32_t k = 0; k < K; ++k) x[k] = f32_to_bf16(0.1f * std::cos((float)k));
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t k = 0; k < K; ++k) {
            float v = (n == 37) ? 1.0f : -0.1f;
            W[n * K + k] = f32_to_bf16(v * bf16_to_f32(x[k]));
        }
    }

    auto bx = g_ctx.buf(x.size() * 2, x.data());
    auto bW = g_ctx.buf(W.size() * 2, W.data());
    auto bY_bf = g_ctx.buf(N * 2);
    auto bY_f32 = g_ctx.buf(N * 4);
    auto bout_tok = g_ctx.buf(4);

    struct GemvParams { uint32_t M; uint32_t N; uint32_t K; uint32_t out_fp32; } gp{M, N, K, 1};
    struct ArgParams { uint32_t vocab; float softcap; } ap{N, 30.0f};

    id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p_gemv];
    [enc setBuffer:bx offset:0 atIndex:0];
    [enc setBuffer:bW offset:0 atIndex:1];
    [enc setBuffer:bY_bf offset:0 atIndex:2];
    [enc setBuffer:bY_f32 offset:0 atIndex:3];
    [enc setBytes:&gp length:sizeof(gp) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((N + 3) / 4, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    [enc setComputePipelineState:p_arg];
    [enc setBuffer:bY_f32 offset:0 atIndex:0];
    [enc setBuffer:bout_tok offset:0 atIndex:1];
    [enc setBytes:&ap length:sizeof(ap) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    EXPECT_TRUE(g_ctx.run_cb(cb));

    int32_t best = *(const int32_t*)bout_tok.contents;
    EXPECT_TRUE(best == 37);

    std::vector<float> big_logits(N, 5000.0f);
    big_logits[51] = 9000.0f;
    auto bBig = g_ctx.buf(N * 4, big_logits.data());
    id<MTLCommandBuffer> cb2 = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc2 = [cb2 computeCommandEncoder];
    [enc2 setComputePipelineState:p_arg];
    [enc2 setBuffer:bBig offset:0 atIndex:0];
    [enc2 setBuffer:bout_tok offset:0 atIndex:1];
    [enc2 setBytes:&ap length:sizeof(ap) atIndex:2];
    [enc2 dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc2 endEncoding];
    EXPECT_TRUE(g_ctx.run_cb(cb2));
    const float* big_out = (const float*)bBig.contents;
    EXPECT_TRUE(std::isfinite(big_out[51]) && std::abs(big_out[51] - 30.0f) < 1e-3f);
}

static void test_w8a16_fp8_quant_and_gemv() {
    auto p_q = g_ctx.pso("quantize_bf16_to_fp8_rowwise_kernel");
    auto p_g = g_ctx.pso("fp8_gemv_kernel");
    EXPECT_TRUE(p_q != nil && p_g != nil);

    const uint32_t M = 2, N = 64, K = 128;
    std::vector<uint16_t> X(M * K), W(N * K);
    for (uint32_t i = 0; i < M * K; ++i) X[i] = f32_to_bf16(0.3f * std::sin((float)i * 0.17f));
    for (uint32_t i = 0; i < N * K; ++i) W[i] = f32_to_bf16(0.5f * std::cos((float)i * 0.09f));

    auto bX = g_ctx.buf(X.size() * 2, X.data());
    auto bW = g_ctx.buf(W.size() * 2, W.data());
    auto bW8 = g_ctx.buf(N * K);
    auto bSc = g_ctx.buf(N * 4);
    auto bY_bf = g_ctx.buf(M * N * 2);
    auto bY_f32 = g_ctx.buf(M * N * 4);

    struct QuantParams { uint32_t N; uint32_t K; } qp{N, K};
    struct GemvParams { uint32_t M; uint32_t N; uint32_t K; uint32_t out_fp32; } gp{M, N, K, 1};

    id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p_q];
    [enc setBuffer:bW offset:0 atIndex:0];
    [enc setBuffer:bW8 offset:0 atIndex:1];
    [enc setBuffer:bSc offset:0 atIndex:2];
    [enc setBytes:&qp length:sizeof(qp) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc setComputePipelineState:p_g];
    [enc setBuffer:bX offset:0 atIndex:0];
    [enc setBuffer:bW8 offset:0 atIndex:1];
    [enc setBuffer:bSc offset:0 atIndex:2];
    [enc setBuffer:bY_bf offset:0 atIndex:3];
    [enc setBuffer:bY_f32 offset:0 atIndex:4];
    [enc setBytes:&gp length:sizeof(gp) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((N + 3) / 4, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    EXPECT_TRUE(g_ctx.run_cb(cb));

    const float* gy = (const float*)bY_f32.contents;
    float max_err = 0.0f;
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float ref = 0.0f;
            for (uint32_t k = 0; k < K; ++k) ref += bf16_to_f32(X[m * K + k]) * bf16_to_f32(W[n * K + k]);
            float gv = gy[m * N + n];
            EXPECT_TRUE(std::isfinite(gv));
            max_err = std::max(max_err, std::abs(gv - ref));
        }
    }
    EXPECT_TRUE(max_err < 0.25f);
}

static void test_bf16_gemm_tiled() {
    auto p_mm = g_ctx.pso("bf16_gemm_tiled_kernel");
    EXPECT_TRUE(p_mm != nil);

    const uint32_t M = 32, N = 64, K = 128;
    std::vector<uint16_t> X(M * K), W(N * K);
    for (uint32_t i = 0; i < M * K; ++i) X[i] = f32_to_bf16(0.25f * std::sin((float)i * 0.13f));
    for (uint32_t i = 0; i < N * K; ++i) W[i] = f32_to_bf16(0.25f * std::cos((float)i * 0.11f));

    auto bX = g_ctx.buf(X.size() * 2, X.data());
    auto bW = g_ctx.buf(W.size() * 2, W.data());
    auto bY = g_ctx.buf(M * N * 2);

    struct GemvParams { uint32_t M; uint32_t N; uint32_t K; uint32_t out_fp32; } gp{M, N, K, 0};
    id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p_mm];
    [enc setBuffer:bX offset:0 atIndex:0];
    [enc setBuffer:bW offset:0 atIndex:1];
    [enc setBuffer:bY offset:0 atIndex:2];
    [enc setBytes:&gp length:sizeof(gp) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((N + 31) / 32, (M + 31) / 32, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    EXPECT_TRUE(g_ctx.run_cb(cb));

    const uint16_t* gy = (const uint16_t*)bY.contents;
    float max_err = 0.0f;
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float ref = 0.0f;
            for (uint32_t k = 0; k < K; ++k) ref += bf16_to_f32(X[m * K + k]) * bf16_to_f32(W[n * K + k]);
            float gv = bf16_to_f32(gy[m * N + n]);
            EXPECT_TRUE(std::isfinite(gv));
            max_err = std::max(max_err, std::abs(gv - ref));
        }
    }
    EXPECT_TRUE(max_err < 5e-2f);
}

static void test_dspark_and_verify_kernels() {
    auto p_sg = g_ctx.pso("fp8_small_gemm_kernel");
    auto p_ba = g_ctx.pso("batched_argmax_kernel");
    auto p_mg = g_ctx.pso("markov_gate_kernel");
    EXPECT_TRUE(p_sg != nil && p_ba != nil && p_mg != nil);

    const uint32_t M = 4, N = 64, K = 128;
    std::vector<uint16_t> X(M * K);
    std::vector<uint8_t> W8(N * K, 0x38);
    std::vector<float> Wsc(N, 1.0f);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t k = 0; k < K; ++k) {
            X[m * K + k] = f32_to_bf16(0.1f * float(m + 1));
        }
    }
    for (uint32_t n = 0; n < N; ++n) Wsc[n] = (n == 19) ? 2.0f : 0.5f;

    auto bX = g_ctx.buf(X.size() * 2, X.data());
    auto bW8 = g_ctx.buf(W8.size(), W8.data());
    auto bWsc = g_ctx.buf(Wsc.size() * 4, Wsc.data());
    auto bY_bf = g_ctx.buf(M * N * 2);
    auto bY_f32 = g_ctx.buf(M * N * 4);
    auto bOut = g_ctx.buf(M * 4);

    struct GemvParams { uint32_t M; uint32_t N; uint32_t K; uint32_t out_fp32; } gp{M, N, K, 1};
    struct BatchArgParams { uint32_t M; uint32_t vocab; } bap{M, N};

    id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p_sg];
    [enc setBuffer:bX offset:0 atIndex:0];
    [enc setBuffer:bW8 offset:0 atIndex:1];
    [enc setBuffer:bWsc offset:0 atIndex:2];
    [enc setBuffer:bY_bf offset:0 atIndex:3];
    [enc setBuffer:bY_f32 offset:0 atIndex:4];
    [enc setBytes:&gp length:sizeof(gp) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((N + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    [enc setComputePipelineState:p_ba];
    [enc setBuffer:bY_f32 offset:0 atIndex:0];
    [enc setBuffer:bOut offset:0 atIndex:1];
    [enc setBytes:&bap length:sizeof(bap) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    EXPECT_TRUE(g_ctx.run_cb(cb));

    const int32_t* gout = (const int32_t*)bOut.contents;
    for (uint32_t m = 0; m < M; ++m) {
        EXPECT_TRUE(gout[m] == 19);
    }
}

static void test_attention_fp8_splitk() {
    struct AttnPrepKVParams {
        uint32_t start_pos, T, cache_cap, is_local, n_kv_heads, head_dim;
        float clip_thresh, eps;
    };
    struct AttnParams {
        uint32_t start_pos, T, cache_cap, is_local, window_size, n_heads, n_kv_heads, head_dim, rel_extent;
    };
    struct AttnSplitKParams {
        uint32_t start_pos, T, cache_cap, is_local, window_size, n_heads, n_kv_heads, head_dim, rel_extent, num_splits;
    };

    const uint32_t S = 1024;
    const uint32_t T = 2;
    const uint32_t n_heads = 6;
    const uint32_t n_kv_heads = 2;
    const uint32_t head_dim = 128;
    const uint32_t rel_extent = 16;
    const uint32_t num_splits = 8;

    id<MTLBuffer> bKraw = g_ctx.buf(S * n_kv_heads * head_dim * 2);
    id<MTLBuffer> bVraw = g_ctx.buf(S * n_kv_heads * head_dim * 2);
    id<MTLBuffer> bKfp8 = g_ctx.buf(S * n_kv_heads * head_dim);
    id<MTLBuffer> bVfp8 = g_ctx.buf(S * n_kv_heads * head_dim);
    id<MTLBuffer> bKsc = g_ctx.buf(S * n_kv_heads * 4);
    id<MTLBuffer> bVsc = g_ctx.buf(S * n_kv_heads * 4);
    id<MTLBuffer> bQ = g_ctx.buf(T * n_heads * head_dim * 2);
    id<MTLBuffer> bRel = g_ctx.buf(T * n_heads * rel_extent * 2);
    id<MTLBuffer> bGate = g_ctx.buf(T * n_heads * head_dim * 2);
    id<MTLBuffer> bSinks = g_ctx.buf(T * n_heads * 2);
    id<MTLBuffer> bOutRef = g_ctx.buf(T * n_heads * head_dim * 2);
    id<MTLBuffer> bOutSplit = g_ctx.buf(T * n_heads * head_dim * 2);
    id<MTLBuffer> bPartML = g_ctx.buf(T * n_heads * num_splits * 2 * 4);
    id<MTLBuffer> bPartAcc = g_ctx.buf(T * n_heads * num_splits * head_dim * 4);

    uint16_t* kr = (uint16_t*)bKraw.contents;
    uint16_t* vr = (uint16_t*)bVraw.contents;
    for (uint32_t i = 0; i < S * n_kv_heads * head_dim; ++i) {
        kr[i] = f32_to_bf16(0.5f * sinf(0.013f * (float)i));
        vr[i] = f32_to_bf16(0.4f * cosf(0.017f * (float)i));
    }
    uint16_t* qp = (uint16_t*)bQ.contents;
    uint16_t* gp = (uint16_t*)bGate.contents;
    for (uint32_t i = 0; i < T * n_heads * head_dim; ++i) {
        qp[i] = f32_to_bf16(0.6f * sinf(0.029f * (float)i + 0.3f));
        gp[i] = f32_to_bf16(0.3f * cosf(0.011f * (float)i));
    }
    uint16_t* rp = (uint16_t*)bRel.contents;
    for (uint32_t i = 0; i < T * n_heads * rel_extent; ++i) {
        rp[i] = f32_to_bf16(0.1f * sinf(0.1f * (float)i));
    }
    uint16_t* sp = (uint16_t*)bSinks.contents;
    for (uint32_t i = 0; i < T * n_heads; ++i) {
        sp[i] = f32_to_bf16(-0.5f + 0.1f * (float)i);
    }

    id<MTLComputePipelineState> p_prep = g_ctx.pso("attn_prep_kv_fp8_kernel");
    id<MTLComputePipelineState> p_ref = g_ctx.pso("attention_fp8_kernel");
    id<MTLComputePipelineState> p_sk = g_ctx.pso("attention_fp8_splitk_kernel");
    id<MTLComputePipelineState> p_red = g_ctx.pso("attention_fp8_reduce_kernel");
    id<MTLComputePipelineState> p_fl = g_ctx.pso("attention_fp8_prefill4_kernel");
    EXPECT_TRUE(p_prep != nil && p_ref != nil && p_sk != nil && p_red != nil && p_fl != nil);

    id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    AttnPrepKVParams kvp{0u, S, S, 0u, n_kv_heads, head_dim, 100.0f, 1e-6f};
    [enc setComputePipelineState:p_prep];
    [enc setBuffer:bKraw offset:0 atIndex:0];
    [enc setBuffer:bVraw offset:0 atIndex:1];
    [enc setBuffer:bKfp8 offset:0 atIndex:2];
    [enc setBuffer:bVfp8 offset:0 atIndex:3];
    [enc setBuffer:bKsc offset:0 atIndex:4];
    [enc setBuffer:bVsc offset:0 atIndex:5];
    [enc setBytes:&kvp length:sizeof(kvp) atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(n_kv_heads, S, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    uint32_t start_pos = S - T;
    AttnParams ap{start_pos, T, S, 0u, 128u, n_heads, n_kv_heads, head_dim, rel_extent};
    [enc setComputePipelineState:p_ref];
    [enc setBuffer:bQ offset:0 atIndex:0];
    [enc setBuffer:bKfp8 offset:0 atIndex:1];
    [enc setBuffer:bVfp8 offset:0 atIndex:2];
    [enc setBuffer:bKsc offset:0 atIndex:3];
    [enc setBuffer:bVsc offset:0 atIndex:4];
    [enc setBuffer:bRel offset:0 atIndex:5];
    [enc setBuffer:bGate offset:0 atIndex:6];
    [enc setBuffer:bSinks offset:0 atIndex:7];
    [enc setBuffer:bOutRef offset:0 atIndex:8];
    [enc setBytes:&ap length:sizeof(ap) atIndex:9];
    [enc dispatchThreadgroups:MTLSizeMake(n_heads, T, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    AttnSplitKParams skp{start_pos, T, S, 0u, 128u, n_heads, n_kv_heads, head_dim, rel_extent, num_splits};
    [enc setComputePipelineState:p_sk];
    [enc setBuffer:bQ offset:0 atIndex:0];
    [enc setBuffer:bKfp8 offset:0 atIndex:1];
    [enc setBuffer:bVfp8 offset:0 atIndex:2];
    [enc setBuffer:bKsc offset:0 atIndex:3];
    [enc setBuffer:bVsc offset:0 atIndex:4];
    [enc setBuffer:bRel offset:0 atIndex:5];
    [enc setBuffer:bPartML offset:0 atIndex:6];
    [enc setBuffer:bPartAcc offset:0 atIndex:7];
    [enc setBytes:&skp length:sizeof(skp) atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake(n_heads, T, num_splits) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    [enc setComputePipelineState:p_red];
    [enc setBuffer:bPartML offset:0 atIndex:0];
    [enc setBuffer:bPartAcc offset:0 atIndex:1];
    [enc setBuffer:bGate offset:0 atIndex:2];
    [enc setBuffer:bSinks offset:0 atIndex:3];
    [enc setBuffer:bOutSplit offset:0 atIndex:4];
    [enc setBytes:&skp length:sizeof(skp) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(n_heads, T, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    [enc endEncoding];
    EXPECT_TRUE(g_ctx.run_cb(cb));

    const uint16_t* o_ref = (const uint16_t*)bOutRef.contents;
    const uint16_t* o_spl = (const uint16_t*)bOutSplit.contents;
    float max_err = 0.0f;
    for (uint32_t i = 0; i < T * n_heads * head_dim; ++i) {
        float vr0 = bf16_to_f32(o_ref[i]);
        float vs0 = bf16_to_f32(o_spl[i]);
        EXPECT_TRUE(std::isfinite(vr0) && std::isfinite(vs0));
        max_err = std::max(max_err, std::fabs(vr0 - vs0));
    }
    EXPECT_TRUE(max_err < 1e-2f);

    id<MTLBuffer> bOutFlash = g_ctx.buf(T * n_heads * head_dim * 2);
    id<MTLCommandBuffer> cb2 = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc2 = [cb2 computeCommandEncoder];
    [enc2 setComputePipelineState:p_fl];
    [enc2 setBuffer:bQ offset:0 atIndex:0];
    [enc2 setBuffer:bKfp8 offset:0 atIndex:1];
    [enc2 setBuffer:bVfp8 offset:0 atIndex:2];
    [enc2 setBuffer:bKsc offset:0 atIndex:3];
    [enc2 setBuffer:bVsc offset:0 atIndex:4];
    [enc2 setBuffer:bRel offset:0 atIndex:5];
    [enc2 setBuffer:bGate offset:0 atIndex:6];
    [enc2 setBuffer:bSinks offset:0 atIndex:7];
    [enc2 setBuffer:bOutFlash offset:0 atIndex:8];
    [enc2 setBytes:&ap length:sizeof(ap) atIndex:9];
    [enc2 dispatchThreadgroups:MTLSizeMake(n_heads, (T + 15u) / 16u, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc2 endEncoding];
    EXPECT_TRUE(g_ctx.run_cb(cb2));

    const uint16_t* o_fl = (const uint16_t*)bOutFlash.contents;
    float max_err_fl = 0.0f;
    for (uint32_t i = 0; i < T * n_heads * head_dim; ++i) {
        float vr0 = bf16_to_f32(o_ref[i]);
        float vf0 = bf16_to_f32(o_fl[i]);
        EXPECT_TRUE(std::isfinite(vr0) && std::isfinite(vf0));
        max_err_fl = std::max(max_err_fl, std::fabs(vr0 - vf0));
    }
    EXPECT_TRUE(max_err_fl < 2e-2f);
}

static void test_moe_router_and_grouped_parity() {
    auto p_rlog = g_ctx.pso("moe_router_logits_kernel");
    auto p_rtop = g_ctx.pso("moe_router_topk_kernel");
    auto p_grp  = g_ctx.pso("moe_group_tokens_kernel");
    auto p_nv13 = g_ctx.pso("moe_nvfp4_w13_kernel");
    auto p_nv2  = g_ctx.pso("moe_nvfp4_w2_kernel");
    auto p_g13  = g_ctx.pso("moe_nvfp4_w13_grouped_kernel");
    auto p_g2   = g_ctx.pso("moe_nvfp4_w2_grouped_kernel");
    auto p_red  = g_ctx.pso("moe_reduce_k_kernel");
    EXPECT_TRUE(p_rlog && p_rtop && p_grp && p_nv13 && p_nv2 && p_g13 && p_g2 && p_red);

    const uint32_t T = 16, D = 64, inter = 32, E = 16, K = 4;
    const float route_scale = 2.0f;
    std::vector<uint16_t> x(T * D), gw(E * D);
    std::vector<float> gb(E);
    for (uint32_t i = 0; i < T * D; ++i) x[i] = f32_to_bf16(0.3f * sinf(0.13f * (float)i));
    for (uint32_t i = 0; i < E * D; ++i) gw[i] = f32_to_bf16(0.25f * cosf(0.07f * (float)i));
    for (uint32_t e = 0; e < E; ++e) gb[e] = 0.05f * (float)((int)e - 8);

    std::vector<uint8_t> w13_u8(E * 2 * inter * (D / 2));
    std::vector<uint8_t> w13_sc(E * 2 * inter * (D / 16), 0x38);
    std::vector<float> w13_amax(E), w13_denom = {1.0f};
    std::vector<uint8_t> w2_u8(E * D * (inter / 2));
    std::vector<uint8_t> w2_sc(E * D * (inter / 16), 0x38);
    std::vector<float> w2_amax(E), w2_denom = {1.0f};
    for (uint32_t e = 0; e < E; ++e) {
        w13_amax[e] = 0.5f + 0.1f * (float)e;
        w2_amax[e] = 0.4f + 0.08f * (float)((e * 5u + 3u) % E);
    }
    size_t w13_e_stride = 2 * inter * (D / 2);
    size_t w2_e_stride = D * (inter / 2);
    for (size_t i = 0; i < w13_u8.size(); ++i) {
        size_t e = i / w13_e_stride;
        w13_u8[i] = (uint8_t)((((i * 3u + e * 5u + 1u) & 15u)) | ((((i * 5u + e * 3u + 2u) & 15u)) << 4));
    }
    for (size_t i = 0; i < w2_u8.size(); ++i) {
        size_t e = i / w2_e_stride;
        w2_u8[i] = (uint8_t)((((i * 7u + e * 3u + 1u) & 15u)) | ((((i * 3u + e * 7u + 2u) & 15u)) << 4));
    }

    auto bx = g_ctx.buf(x.size() * 2, x.data());
    auto bgw = g_ctx.buf(gw.size() * 2, gw.data());
    auto bgb = g_ctx.buf(gb.size() * 4, gb.data());
    auto brlog = g_ctx.buf(T * E * 4);
    auto bids = g_ctx.buf(T * K * 4);
    auto bwts = g_ctx.buf(T * K * 4);
    auto boffs = g_ctx.buf((E + 1) * 4);
    auto bsort = g_ctx.buf(T * K * 4);
    auto bw13 = g_ctx.buf(w13_u8.size(), w13_u8.data());
    auto bs13 = g_ctx.buf(w13_sc.size(), w13_sc.data());
    auto ba13 = g_ctx.buf(w13_amax.size() * 4, w13_amax.data());
    auto bd13 = g_ctx.buf(4, w13_denom.data());
    auto bw2 = g_ctx.buf(w2_u8.size(), w2_u8.data());
    auto bs2 = g_ctx.buf(w2_sc.size(), w2_sc.data());
    auto ba2 = g_ctx.buf(w2_amax.size() * 4, w2_amax.data());
    auto bd2 = g_ctx.buf(4, w2_denom.data());
    auto bact_ref = g_ctx.buf(T * K * inter * 4);
    auto bout_ref = g_ctx.buf(T * D * 2);
    auto bact_grp = g_ctx.buf(T * K * inter * 4);
    auto bw2_grp = g_ctx.buf(T * K * D * 2);
    auto bout_grp = g_ctx.buf(T * D * 2);

    struct RouterParams { uint32_t T; uint32_t D; uint32_t E; uint32_t K; float route_scale; } rp{T, D, E, K, route_scale};
    struct MoeGroupParams { uint32_t total_tk; uint32_t E; } gp{T * K, E};
    struct MoeParams { uint32_t T; uint32_t D; uint32_t inter; uint32_t K; float gate_cap; float up_cap; } mp{T, D, inter, K, 4.0f, 25.0f};

    id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    [enc setComputePipelineState:p_rlog];
    [enc setBuffer:bx offset:0 atIndex:0];
    [enc setBuffer:bgw offset:0 atIndex:1];
    [enc setBuffer:brlog offset:0 atIndex:2];
    [enc setBytes:&rp length:sizeof(rp) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(E, T, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    [enc setComputePipelineState:p_rtop];
    [enc setBuffer:brlog offset:0 atIndex:0];
    [enc setBuffer:bgb offset:0 atIndex:1];
    [enc setBuffer:bids offset:0 atIndex:2];
    [enc setBuffer:bwts offset:0 atIndex:3];
    [enc setBytes:&rp length:sizeof(rp) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    [enc setComputePipelineState:p_nv13];
    [enc setBuffer:bx offset:0 atIndex:0];
    [enc setBuffer:bids offset:0 atIndex:1];
    [enc setBuffer:bwts offset:0 atIndex:2];
    [enc setBuffer:bw13 offset:0 atIndex:3];
    [enc setBuffer:bs13 offset:0 atIndex:4];
    [enc setBuffer:ba13 offset:0 atIndex:5];
    [enc setBuffer:bd13 offset:0 atIndex:6];
    [enc setBuffer:bact_ref offset:0 atIndex:7];
    [enc setBytes:&mp length:sizeof(mp) atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake((inter + 3u) / 4u, T * K, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    [enc setComputePipelineState:p_nv2];
    [enc setBuffer:bact_ref offset:0 atIndex:0];
    [enc setBuffer:bids offset:0 atIndex:1];
    [enc setBuffer:bw2 offset:0 atIndex:2];
    [enc setBuffer:bs2 offset:0 atIndex:3];
    [enc setBuffer:ba2 offset:0 atIndex:4];
    [enc setBuffer:bd2 offset:0 atIndex:5];
    [enc setBuffer:bout_ref offset:0 atIndex:6];
    [enc setBytes:&mp length:sizeof(mp) atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake((D + 3u) / 4u, T, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    [enc setComputePipelineState:p_grp];
    [enc setBuffer:bids offset:0 atIndex:0];
    [enc setBuffer:boffs offset:0 atIndex:1];
    [enc setBuffer:bsort offset:0 atIndex:2];
    [enc setBytes:&gp length:sizeof(gp) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    [enc setComputePipelineState:p_g13];
    [enc setBuffer:bx offset:0 atIndex:0];
    [enc setBuffer:bwts offset:0 atIndex:1];
    [enc setBuffer:boffs offset:0 atIndex:2];
    [enc setBuffer:bsort offset:0 atIndex:3];
    [enc setBuffer:bw13 offset:0 atIndex:4];
    [enc setBuffer:bs13 offset:0 atIndex:5];
    [enc setBuffer:ba13 offset:0 atIndex:6];
    [enc setBuffer:bd13 offset:0 atIndex:7];
    [enc setBuffer:bact_grp offset:0 atIndex:8];
    [enc setBytes:&mp length:sizeof(mp) atIndex:9];
    [enc dispatchThreadgroups:MTLSizeMake((inter + 15u) / 16u, E, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    [enc setComputePipelineState:p_g2];
    [enc setBuffer:bact_grp offset:0 atIndex:0];
    [enc setBuffer:boffs offset:0 atIndex:1];
    [enc setBuffer:bsort offset:0 atIndex:2];
    [enc setBuffer:bw2 offset:0 atIndex:3];
    [enc setBuffer:bs2 offset:0 atIndex:4];
    [enc setBuffer:ba2 offset:0 atIndex:5];
    [enc setBuffer:bd2 offset:0 atIndex:6];
    [enc setBuffer:bw2_grp offset:0 atIndex:7];
    [enc setBytes:&mp length:sizeof(mp) atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake((D + 31u) / 32u, E, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    [enc setComputePipelineState:p_red];
    [enc setBuffer:bw2_grp offset:0 atIndex:0];
    [enc setBuffer:bout_grp offset:0 atIndex:1];
    [enc setBytes:&mp length:sizeof(mp) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(D, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc endEncoding];
    EXPECT_TRUE(g_ctx.run_cb(cb));

    const int32_t* gids = (const int32_t*)bids.contents;
    const float* gwts = (const float*)bwts.contents;
    for (uint32_t t = 0; t < T; ++t) {
        std::vector<float> sigs(E), sels(E);
        for (uint32_t e = 0; e < E; ++e) {
            float acc = 0.0f;
            for (uint32_t d = 0; d < D; ++d) {
                acc += bf16_to_f32(x[t * D + d]) * bf16_to_f32(gw[e * D + d]);
            }
            float s = 1.0f / (1.0f + std::exp(-acc));
            sigs[e] = s;
            sels[e] = s + gb[e];
        }
        std::vector<int32_t> ref_ids(K);
        std::vector<float> ref_wts(K);
        float ref_sum = 0.0f;
        for (uint32_t k = 0; k < K; ++k) {
            uint32_t best = 0;
            float bv = -1e30f;
            for (uint32_t e = 0; e < E; ++e) {
                if (sels[e] > bv) {
                    bv = sels[e];
                    best = e;
                }
            }
            ref_ids[k] = (int32_t)best;
            ref_wts[k] = sigs[best];
            ref_sum += sigs[best];
            sels[best] = -1e30f;
        }
        for (uint32_t k = 0; k < K; ++k) {
            float exp_w = ref_wts[k] * (route_scale / (ref_sum + 1e-20f));
            EXPECT_TRUE(gids[t * K + k] == ref_ids[k]);
            EXPECT_TRUE(std::fabs(gwts[t * K + k] - exp_w) < 2e-3f);
        }
    }

    const uint16_t* o_ref = (const uint16_t*)bout_ref.contents;
    const uint16_t* o_grp = (const uint16_t*)bout_grp.contents;
    float max_rel = 0.0f;
    for (uint32_t i = 0; i < T * D; ++i) {
        float vr = bf16_to_f32(o_ref[i]);
        float vg = bf16_to_f32(o_grp[i]);
        EXPECT_TRUE(std::isfinite(vr) && std::isfinite(vg));
        float rel = std::fabs(vr - vg) / std::max(1.0f, std::fabs(vr));
        max_rel = std::max(max_rel, rel);
    }
    EXPECT_TRUE(max_rel < 2e-2f);
}

int main() {
    @autoreleasepool {
        if (!g_ctx.init()) {
            fprintf(stderr, "Failed to initialize Metal context\n");
            return 1;
        }
        printf("Metal kernel tests on %s\n", g_ctx.dev.name.UTF8String);
        run_test("test_rmsnorm_and_add", test_rmsnorm_and_add);
        run_test("test_sconv_static_and_dynamic", test_sconv_static_and_dynamic);
        run_test("test_moe_nvfp4_and_mxfp8", test_moe_nvfp4_and_mxfp8);
        run_test("test_bf16_gemv_and_softcap_argmax", test_bf16_gemv_and_softcap_argmax);
        run_test("test_w8a16_fp8_quant_and_gemv", test_w8a16_fp8_quant_and_gemv);
        run_test("test_bf16_gemm_tiled", test_bf16_gemm_tiled);
        run_test("test_dspark_and_verify_kernels", test_dspark_and_verify_kernels);
        run_test("test_attention_fp8_splitk", test_attention_fp8_splitk);
        run_test("test_moe_router_and_grouped_parity", test_moe_router_and_grouped_parity);
        printf("Metal kernel tests: %d passed, %d failed\n", g_passed, g_failed);
        return g_failed == 0 ? 0 : 1;
    }
}
