#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#ifdef WARPGRID_METAL_PRECOMPILED
#include "../src/metal/kernels_gguf_metallib_embed.hpp"
#else
#include "../src/metal/kernels_gguf_source_embed.hpp"
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

static uint16_t f32_to_f16_bits(float v) {
    uint32_t u = 0;
    memcpy(&u, &v, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((u >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = (u >> 13) & 0x3FFu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t u = 0;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
            mant &= 0x3FFu;
            u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out = 0.0f;
    memcpy(&out, &u, 4);
    return out;
}

static float e8m0_to_f32(uint8_t e) {
    if (e == 0) return 5.877471754111438e-39f;
    uint32_t bits = (uint32_t)e << 23;
    float f = 0.0f;
    memcpy(&f, &bits, 4);
    return f;
}

struct GgufGemvParams {
    uint32_t M;
    uint32_t N;
    uint32_t K;
};

struct GgufMoeParams {
    uint32_t T;
    uint32_t N;
    uint32_t K;
    uint32_t top_k;
    uint32_t expert_stride_bytes;
};

struct DsMhcParams {
    uint32_t T;
    uint32_t D;
    uint32_t hc_mult;
    uint32_t sinkhorn_iters;
    float eps;
};

struct DsRouterParams {
    uint32_t T;
    uint32_t E;
    uint32_t top_k;
    uint32_t has_bias;
    float route_scale;
};

struct DsAttnParams {
    uint32_t T;
    uint32_t start_pos;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t max_seq;
    uint32_t compress_ratio;
    uint32_t local_window;
    float scale;
};

struct MetalGgufCtx {
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
            kGgufMetalLibBytes, kGgufMetalLibLen, dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        lib = [dev newLibraryWithData:data error:&err];
#else
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion3_1;
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeSafe;
            opts.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        }
        NSString* src = [[NSString alloc] initWithBytes:kGgufMetalSourceBytes
                                                 length:kGgufMetalSourceLen
                                               encoding:NSUTF8StringEncoding];
        lib = [dev newLibraryWithSource:src options:opts error:&err];
#endif
        if (!lib) {
            fprintf(stderr, "Failed to load GGUF Metal library: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "unknown");
            return false;
        }
        return true;
    }

    id<MTLComputePipelineState> pso(const char* name) {
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (!fn) {
            fprintf(stderr, "Missing GGUF Metal function: %s\n", name);
            return nil;
        }
        NSError* err = nil;
        id<MTLComputePipelineState> st = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!st) {
            fprintf(stderr, "Failed PSO %s: %s\n", name,
                    err ? [[err localizedDescription] UTF8String] : "unknown");
        }
        return st;
    }
};

static MetalGgufCtx g_ctx;

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

static void dispatch_gguf_gemv(
    const char* kernel_name,
    const std::vector<float>& x,
    const std::vector<uint8_t>& w,
    uint32_t M, uint32_t N, uint32_t K,
    std::vector<float>& out) {
    id<MTLComputePipelineState> p = g_ctx.pso(kernel_name);
    EXPECT_TRUE(p != nil);
    id<MTLBuffer> bx = [g_ctx.dev newBufferWithBytes:x.data() length:x.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bw = [g_ctx.dev newBufferWithBytes:w.data() length:w.size() options:MTLResourceStorageModeShared];
    id<MTLBuffer> by = [g_ctx.dev newBufferWithLength:(size_t)M * N * sizeof(float) options:MTLResourceStorageModeShared];
    GgufGemvParams params{M, N, K};

    id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    [enc setBuffer:bx offset:0 atIndex:0];
    [enc setBuffer:bw offset:0 atIndex:1];
    [enc setBuffer:by offset:0 atIndex:2];
    [enc setBytes:&params length:sizeof(params) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((N + 3) / 4, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    out.resize((size_t)M * N);
    memcpy(out.data(), by.contents, out.size() * sizeof(float));
}

static void test_gguf_gemv_q8_0() {
    const uint32_t M = 3, N = 16, K = 256;
    const uint32_t nb = K / 32;
    std::vector<float> x((size_t)M * K);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin((float)(i + 1) * 0.13f);

    std::vector<uint8_t> w((size_t)N * nb * 34, 0);
    std::vector<float> w_ref((size_t)N * K, 0.0f);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < nb; ++b) {
            uint8_t* blk = w.data() + ((size_t)n * nb + b) * 34;
            uint16_t db = f32_to_f16_bits(0.05f * (float)((n + b) % 7 + 1));
            memcpy(blk, &db, 2);
            float d = f16_bits_to_f32(db);
            for (int i = 0; i < 32; ++i) {
                int8_t q = (int8_t)(((int)(n * 11 + b * 5 + i * 3) % 31) - 15);
                blk[2 + i] = (uint8_t)q;
                w_ref[(size_t)n * K + b * 32 + i] = d * (float)q;
            }
        }
    }

    std::vector<float> y;
    dispatch_gguf_gemv("gguf_gemv_q8_0", x, w, M, N, K, y);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float ref = 0.0f;
            for (uint32_t k = 0; k < K; ++k) ref += x[(size_t)m * K + k] * w_ref[(size_t)n * K + k];
            EXPECT_TRUE(std::abs(y[(size_t)m * N + n] - ref) < 1e-3f);
        }
    }
}

static void test_gguf_gemv_iq4_nl() {
    static constexpr int8_t kIq4Nl[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113
    };
    const uint32_t M = 2, N = 12, K = 256;
    const uint32_t nb = K / 32;
    std::vector<float> x((size_t)M * K);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos((float)(i + 2) * 0.07f);

    std::vector<uint8_t> w((size_t)N * nb * 18, 0);
    std::vector<float> w_ref((size_t)N * K, 0.0f);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < nb; ++b) {
            uint8_t* blk = w.data() + ((size_t)n * nb + b) * 18;
            uint16_t db = f32_to_f16_bits(0.01f * (float)((n + b) % 5 + 1));
            memcpy(blk, &db, 2);
            float d = f16_bits_to_f32(db);
            for (int i = 0; i < 16; ++i) {
                uint8_t lo = (uint8_t)((n + b + i) & 15u);
                uint8_t hi = (uint8_t)((n * 3 + b + i * 2) & 15u);
                blk[2 + i] = (uint8_t)((hi << 4) | lo);
                w_ref[(size_t)n * K + b * 32 + i] = d * (float)kIq4Nl[lo];
                w_ref[(size_t)n * K + b * 32 + 16 + i] = d * (float)kIq4Nl[hi];
            }
        }
    }

    std::vector<float> y;
    dispatch_gguf_gemv("gguf_gemv_iq4_nl", x, w, M, N, K, y);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float ref = 0.0f;
            for (uint32_t k = 0; k < K; ++k) ref += x[(size_t)m * K + k] * w_ref[(size_t)n * K + k];
            EXPECT_TRUE(std::abs(y[(size_t)m * N + n] - ref) < 1e-2f);
        }
    }
}

static void test_gguf_gemv_mxfp4() {
    static constexpr float kE2M1[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
       -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
    };
    const uint32_t M = 2, N = 16, K = 256;
    const uint32_t nb = K / 32;
    std::vector<float> x((size_t)M * K);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin((float)(i + 5) * 0.11f);

    std::vector<uint8_t> w((size_t)N * nb * 17, 0);
    std::vector<float> w_ref((size_t)N * K, 0.0f);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < nb; ++b) {
            uint8_t* blk = w.data() + ((size_t)n * nb + b) * 17;
            uint8_t e = (uint8_t)(124 + ((n + b) % 5));
            blk[0] = e;
            float sc = e8m0_to_f32(e);
            for (int i = 0; i < 16; ++i) {
                uint8_t lo = (uint8_t)((n + i) & 15u);
                uint8_t hi = (uint8_t)((b + i * 5) & 15u);
                blk[1 + i] = (uint8_t)((hi << 4) | lo);
                w_ref[(size_t)n * K + b * 32 + i] = sc * kE2M1[lo];
                w_ref[(size_t)n * K + b * 32 + 16 + i] = sc * kE2M1[hi];
            }
        }
    }

    std::vector<float> y;
    dispatch_gguf_gemv("gguf_gemv_mxfp4", x, w, M, N, K, y);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float ref = 0.0f;
            for (uint32_t k = 0; k < K; ++k) ref += x[(size_t)m * K + k] * w_ref[(size_t)n * K + k];
            EXPECT_TRUE(std::abs(y[(size_t)m * N + n] - ref) < 1e-3f);
        }
    }
}

static void test_gguf_gemv_q4_k() {
    const uint32_t M = 2, N = 8, K = 512;
    const uint32_t nb = K / 256;
    std::vector<float> x((size_t)M * K);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin((float)(i + 3) * 0.09f);

    std::vector<uint8_t> w((size_t)N * nb * 144, 0);
    std::vector<float> w_ref((size_t)N * K, 0.0f);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < nb; ++b) {
            uint8_t* blk = w.data() + ((size_t)n * nb + b) * 144;
            uint16_t d_b = f32_to_f16_bits(0.125f);
            uint16_t dm_b = f32_to_f16_bits(0.0625f);
            memcpy(blk, &d_b, 2);
            memcpy(blk + 2, &dm_b, 2);
            float d = f16_bits_to_f32(d_b);
            float dmin = f16_bits_to_f32(dm_b);
            for (int i = 0; i < 12; ++i) blk[4 + i] = (uint8_t)((n * 7 + b * 3 + i * 5) & 0xFFu);
            for (int i = 0; i < 128; ++i) blk[16 + i] = (uint8_t)((n * 13 + b * 11 + i * 17) & 0xFFu);

            const uint8_t* scales = blk + 4;
            const uint8_t* qs = blk + 16;
            auto get_sm = [&](int j, uint8_t& sc, uint8_t& mn) {
                if (j < 4) {
                    sc = scales[j] & 63u;
                    mn = scales[j + 4] & 63u;
                } else {
                    sc = (scales[j + 4] & 0xFu) | ((scales[j - 4] >> 6) << 4);
                    mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
                }
            };
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc0 = 0, m0 = 0, sc1 = 0, m1 = 0;
                get_sm(is + 0, sc0, m0);
                get_sm(is + 1, sc1, m1);
                float d1 = d * (float)sc0, min1 = dmin * (float)m0;
                float d2 = d * (float)sc1, min2 = dmin * (float)m1;
                for (int l = 0; l < 32; ++l) {
                    w_ref[(size_t)n * K + b * 256 + j + l] = d1 * (float)(qs[l] & 0x0Fu) - min1;
                    w_ref[(size_t)n * K + b * 256 + j + 32 + l] = d2 * (float)(qs[l] >> 4) - min2;
                }
                qs += 32;
                is += 2;
            }
        }
    }

    std::vector<float> y;
    dispatch_gguf_gemv("gguf_gemv_q4_k", x, w, M, N, K, y);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float ref = 0.0f;
            for (uint32_t k = 0; k < K; ++k) ref += x[(size_t)m * K + k] * w_ref[(size_t)n * K + k];
            EXPECT_TRUE(std::abs(y[(size_t)m * N + n] - ref) < 1e-2f);
        }
    }
}

static void test_gguf_gemv_q6_k() {
    const uint32_t M = 2, N = 8, K = 512;
    const uint32_t nb = K / 256;
    std::vector<float> x((size_t)M * K);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos((float)(i + 1) * 0.05f);

    std::vector<uint8_t> w((size_t)N * nb * 210, 0);
    std::vector<float> w_ref((size_t)N * K, 0.0f);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < nb; ++b) {
            uint8_t* blk = w.data() + ((size_t)n * nb + b) * 210;
            for (int i = 0; i < 128; ++i) blk[i] = (uint8_t)((n * 9 + b * 7 + i * 3) & 0xFFu);
            for (int i = 0; i < 64; ++i) blk[128 + i] = (uint8_t)((n * 5 + b * 13 + i) & 0xFFu);
            for (int i = 0; i < 16; ++i) blk[192 + i] = (uint8_t)(int8_t)(((int)(n + b + i) % 15) - 7);
            uint16_t db = f32_to_f16_bits(0.05f);
            memcpy(blk + 208, &db, 2);
            float d = f16_bits_to_f32(db);

            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);
            float* yr = w_ref.data() + (size_t)n * K + b * 256;
            for (int n128 = 0; n128 < 256; n128 += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = l / 16;
                    int8_t q1 = (int8_t)((ql[l + 0] & 0xFu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
                    int8_t q2 = (int8_t)((ql[l + 32] & 0xFu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
                    int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32;
                    int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32;
                    yr[l + 0] = d * (float)sc[is + 0] * (float)q1;
                    yr[l + 32] = d * (float)sc[is + 2] * (float)q2;
                    yr[l + 64] = d * (float)sc[is + 4] * (float)q3;
                    yr[l + 96] = d * (float)sc[is + 6] * (float)q4;
                }
                yr += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
    }

    std::vector<float> y;
    dispatch_gguf_gemv("gguf_gemv_q6_k", x, w, M, N, K, y);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float ref = 0.0f;
            for (uint32_t k = 0; k < K; ++k) ref += x[(size_t)m * K + k] * w_ref[(size_t)n * K + k];
            EXPECT_TRUE(std::abs(y[(size_t)m * N + n] - ref) < 1e-2f);
        }
    }
}

static void test_gguf_gemv_q2_k() {
    const uint32_t M = 2, N = 8, K = 512;
    const uint32_t nb = K / 256;
    std::vector<float> x((size_t)M * K);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin((float)(i + 7) * 0.06f);

    std::vector<uint8_t> w((size_t)N * nb * 84, 0);
    std::vector<float> w_ref((size_t)N * K, 0.0f);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < nb; ++b) {
            uint8_t* blk = w.data() + ((size_t)n * nb + b) * 84;
            for (int i = 0; i < 16; ++i) blk[i] = (uint8_t)(((i + n) & 0xFu) | (((i + b + 1) & 0xFu) << 4));
            for (int i = 0; i < 64; ++i) blk[16 + i] = (uint8_t)((n * 19 + b * 7 + i * 11) & 0xFFu);
            uint16_t d_b = f32_to_f16_bits(0.1f);
            uint16_t dm_b = f32_to_f16_bits(0.05f);
            memcpy(blk + 80, &d_b, 2);
            memcpy(blk + 82, &dm_b, 2);
            float d = f16_bits_to_f32(d_b);
            float dmin = f16_bits_to_f32(dm_b);

            const uint8_t* scales = blk;
            const uint8_t* q = blk + 16;
            float* yr = w_ref.data() + (size_t)n * K + b * 256;
            int is = 0;
            for (int n128 = 0; n128 < 256; n128 += 128) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    uint8_t sc = scales[is++];
                    float dl = d * (float)(sc & 0xFu);
                    float ml = dmin * (float)(sc >> 4);
                    for (int l = 0; l < 16; ++l) {
                        *yr++ = dl * (float)((q[l] >> shift) & 3u) - ml;
                    }
                    sc = scales[is++];
                    dl = d * (float)(sc & 0xFu);
                    ml = dmin * (float)(sc >> 4);
                    for (int l = 0; l < 16; ++l) {
                        *yr++ = dl * (float)((q[l + 16] >> shift) & 3u) - ml;
                    }
                    shift += 2;
                }
                q += 32;
            }
        }
    }

    std::vector<float> y;
    dispatch_gguf_gemv("gguf_gemv_q2_k", x, w, M, N, K, y);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float ref = 0.0f;
            for (uint32_t k = 0; k < K; ++k) ref += x[(size_t)m * K + k] * w_ref[(size_t)n * K + k];
            EXPECT_TRUE(std::abs(y[(size_t)m * N + n] - ref) < 1e-2f);
        }
    }
}

static void test_gguf_moe_topk6_accumulate() {
    const uint32_t T = 2, E = 16, N = 8, K = 256, TOP_K = 6;
    const uint32_t nb = K / 32;
    const uint32_t expert_stride = N * nb * 34;

    std::vector<float> x((size_t)T * K);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin((float)(i + 1) * 0.17f);

    std::vector<uint8_t> w((size_t)E * expert_stride, 0);
    std::vector<float> w_ref((size_t)E * N * K, 0.0f);
    for (uint32_t e = 0; e < E; ++e) {
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t b = 0; b < nb; ++b) {
                uint8_t* blk = w.data() + (size_t)e * expert_stride + ((size_t)n * nb + b) * 34;
                uint16_t db = f32_to_f16_bits(0.02f * (float)((e + n + b) % 5 + 1));
                memcpy(blk, &db, 2);
                float d = f16_bits_to_f32(db);
                for (int i = 0; i < 32; ++i) {
                    int8_t q = (int8_t)(((int)(e * 7 + n * 5 + b * 3 + i) % 21) - 10);
                    blk[2 + i] = (uint8_t)q;
                    w_ref[((size_t)e * N + n) * K + b * 32 + i] = d * (float)q;
                }
            }
        }
    }

    std::vector<int32_t> topk_ids = {0, 3, 5, 8, 11, 15, 1, 2, 6, 9, 12, 14};
    std::vector<float> topk_w = {0.25f, 0.20f, 0.15f, 0.15f, 0.15f, 0.10f,
                                 0.30f, 0.20f, 0.20f, 0.10f, 0.10f, 0.10f};

    id<MTLComputePipelineState> p = g_ctx.pso("gguf_moe_gemv_q8_0");
    EXPECT_TRUE(p != nil);
    id<MTLBuffer> bx = [g_ctx.dev newBufferWithBytes:x.data() length:x.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bw = [g_ctx.dev newBufferWithBytes:w.data() length:w.size() options:MTLResourceStorageModeShared];
    id<MTLBuffer> bids = [g_ctx.dev newBufferWithBytes:topk_ids.data() length:topk_ids.size() * sizeof(int32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bws = [g_ctx.dev newBufferWithBytes:topk_w.data() length:topk_w.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> by = [g_ctx.dev newBufferWithLength:(size_t)T * N * sizeof(float) options:MTLResourceStorageModeShared];
    GgufMoeParams params{T, N, K, TOP_K, expert_stride};

    id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    [enc setBuffer:bx offset:0 atIndex:0];
    [enc setBuffer:bw offset:0 atIndex:1];
    [enc setBuffer:bids offset:0 atIndex:2];
    [enc setBuffer:bws offset:0 atIndex:3];
    [enc setBuffer:by offset:0 atIndex:4];
    [enc setBytes:&params length:sizeof(params) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((N + 3) / 4, T, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    const float* y = static_cast<const float*>(by.contents);
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t n = 0; n < N; ++n) {
            float ref = 0.0f;
            for (uint32_t r = 0; r < TOP_K; ++r) {
                uint32_t e = (uint32_t)topk_ids[t * TOP_K + r];
                float gw = topk_w[t * TOP_K + r];
                float dot = 0.0f;
                for (uint32_t k = 0; k < K; ++k) {
                    dot += x[(size_t)t * K + k] * w_ref[((size_t)e * N + n) * K + k];
                }
                ref += gw * dot;
            }
            EXPECT_TRUE(std::abs(y[(size_t)t * N + n] - ref) < 1e-3f);
        }
    }
}

static void test_ds_mhc_sinkhorn_doubly_stochastic() {
    const uint32_t T = 2, D = 32, M = 4, ITERS = 20;
    std::vector<float> streams((size_t)T * M * D);
    std::vector<float> sublayer((size_t)T * D);
    for (size_t i = 0; i < streams.size(); ++i) streams[i] = std::sin((float)(i + 1) * 0.19f);
    for (size_t i = 0; i < sublayer.size(); ++i) sublayer[i] = std::cos((float)(i + 2) * 0.11f);

    std::vector<float> h_logits = {
         0.5f, -0.2f,  0.1f,  0.3f,
        -0.1f,  0.6f, -0.3f,  0.2f,
         0.2f, -0.1f,  0.7f, -0.2f,
         0.0f,  0.3f, -0.1f,  0.8f
    };
    std::vector<float> h_post = {0.25f, 0.25f, 0.25f, 0.25f};

    std::vector<float> mat(M * M);
    for (size_t i = 0; i < M * M; ++i) mat[i] = std::exp(h_logits[i]);
    for (uint32_t it = 0; it < ITERS; ++it) {
        for (uint32_t r = 0; r < M; ++r) {
            float s = 1e-6f;
            for (uint32_t c = 0; c < M; ++c) s += mat[r * M + c];
            for (uint32_t c = 0; c < M; ++c) mat[r * M + c] /= s;
        }
        for (uint32_t c = 0; c < M; ++c) {
            float s = 1e-6f;
            for (uint32_t r = 0; r < M; ++r) s += mat[r * M + c];
            for (uint32_t r = 0; r < M; ++r) mat[r * M + c] /= s;
        }
    }

    id<MTLComputePipelineState> p = g_ctx.pso("ds_mhc_sinkhorn_mix");
    EXPECT_TRUE(p != nil);
    id<MTLBuffer> bs = [g_ctx.dev newBufferWithBytes:streams.data() length:streams.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bsub = [g_ctx.dev newBufferWithBytes:sublayer.data() length:sublayer.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bmat = [g_ctx.dev newBufferWithBytes:h_logits.data() length:h_logits.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bpost = [g_ctx.dev newBufferWithBytes:h_post.data() length:h_post.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bout = [g_ctx.dev newBufferWithLength:streams.size() * sizeof(float) options:MTLResourceStorageModeShared];
    DsMhcParams params{T, D, M, ITERS, 1e-6f};

    id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    [enc setBuffer:bs offset:0 atIndex:0];
    [enc setBuffer:bsub offset:0 atIndex:1];
    [enc setBuffer:bmat offset:0 atIndex:2];
    [enc setBuffer:bpost offset:0 atIndex:3];
    [enc setBuffer:bout offset:0 atIndex:4];
    [enc setBytes:&params length:sizeof(params) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((D + 63) / 64, T, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    const float* out = static_cast<const float*>(bout.contents);
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t r = 0; r < M; ++r) {
            for (uint32_t d = 0; d < D; ++d) {
                float ref = h_post[r] * sublayer[t * D + d];
                for (uint32_t c = 0; c < M; ++c) {
                    ref += mat[r * M + c] * streams[(t * M + c) * D + d];
                }
                EXPECT_TRUE(std::abs(out[(t * M + r) * D + d] - ref) < 1e-4f);
            }
        }
    }

    std::vector<float> h_pre = {1.0f, 0.0f, 0.0f, 0.0f};
    id<MTLComputePipelineState> pcol = g_ctx.pso("ds_mhc_collapse_f32");
    EXPECT_TRUE(pcol != nil);
    id<MTLBuffer> bpre = [g_ctx.dev newBufferWithBytes:h_pre.data() length:h_pre.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bhid = [g_ctx.dev newBufferWithLength:(size_t)T * D * sizeof(float) options:MTLResourceStorageModeShared];
    struct { uint32_t T; uint32_t D; uint32_t hc_mult; } scp{T, D, M};
    id<MTLCommandBuffer> cb2 = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc2 = [cb2 computeCommandEncoder];
    [enc2 setComputePipelineState:pcol];
    [enc2 setBuffer:bout offset:0 atIndex:0];
    [enc2 setBuffer:bpre offset:0 atIndex:1];
    [enc2 setBuffer:bhid offset:0 atIndex:2];
    [enc2 setBytes:&scp length:sizeof(scp) atIndex:3];
    [enc2 dispatchThreads:MTLSizeMake(D, T, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc2 endEncoding];
    [cb2 commit];
    [cb2 waitUntilCompleted];
    const float* hid = static_cast<const float*>(bhid.contents);
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t d = 0; d < D; ++d) {
            EXPECT_TRUE(std::abs(hid[t * D + d] - out[(t * M + 0) * D + d]) < 1e-5f);
        }
    }
}

static void test_ds_router_topk6_sigmoid_bias() {
    const uint32_t T = 2, E = 32, K = 6;
    std::vector<float> logits((size_t)T * E);
    std::vector<float> bias(E);
    for (size_t i = 0; i < logits.size(); ++i) logits[i] = std::sin((float)(i + 1) * 0.37f) * 2.0f;
    for (size_t i = 0; i < E; ++i) bias[i] = std::cos((float)(i + 3) * 0.23f) * 0.2f;

    id<MTLComputePipelineState> p = g_ctx.pso("ds_router_topk_f32");
    EXPECT_TRUE(p != nil);
    id<MTLBuffer> bl = [g_ctx.dev newBufferWithBytes:logits.data() length:logits.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [g_ctx.dev newBufferWithBytes:bias.data() length:bias.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bids = [g_ctx.dev newBufferWithLength:(size_t)T * K * sizeof(int32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bw = [g_ctx.dev newBufferWithLength:(size_t)T * K * sizeof(float) options:MTLResourceStorageModeShared];
    DsRouterParams params{T, E, K, 1, 1.0f};

    id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    [enc setBuffer:bl offset:0 atIndex:0];
    [enc setBuffer:bb offset:0 atIndex:1];
    [enc setBuffer:bids offset:0 atIndex:2];
    [enc setBuffer:bw offset:0 atIndex:3];
    [enc setBytes:&params length:sizeof(params) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    const int32_t* ids = static_cast<const int32_t*>(bids.contents);
    const float* weights = static_cast<const float*>(bw.contents);
    for (uint32_t t = 0; t < T; ++t) {
        std::vector<std::pair<float, int32_t>> scored(E);
        std::vector<float> sig(E);
        for (uint32_t e = 0; e < E; ++e) {
            sig[e] = 1.0f / (1.0f + std::exp(-logits[t * E + e]));
            scored[e] = {sig[e] + bias[e], (int32_t)e};
        }
        std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        float sum = 1e-20f;
        for (uint32_t k = 0; k < K; ++k) sum += sig[(size_t)scored[k].second];
        for (uint32_t k = 0; k < K; ++k) {
            EXPECT_TRUE(ids[t * K + k] == scored[k].second);
            float ref_w = sig[(size_t)scored[k].second] / sum;
            EXPECT_TRUE(std::abs(weights[t * K + k] - ref_w) < 1e-4f);
        }
    }
}

static void test_ds_csa_hca_causal_attention() {
    const uint32_t T = 3, H = 4, HKV = 2, HD = 16, MAX_SEQ = 8;
    std::vector<float> q((size_t)T * H * HD);
    std::vector<float> k_cache((size_t)MAX_SEQ * HKV * HD, 0.0f);
    std::vector<float> v_cache((size_t)MAX_SEQ * HKV * HD, 0.0f);
    for (size_t i = 0; i < q.size(); ++i) q[i] = std::sin((float)(i + 1) * 0.21f);
    for (uint32_t p = 0; p < T; ++p) {
        for (uint32_t j = 0; j < HKV * HD; ++j) {
            k_cache[(size_t)p * HKV * HD + j] = std::cos((float)(p * 17 + j + 1) * 0.13f);
            v_cache[(size_t)p * HKV * HD + j] = std::sin((float)(p * 11 + j + 2) * 0.17f);
        }
    }

    id<MTLComputePipelineState> pso = g_ctx.pso("ds_csa_hca_attn_f32");
    EXPECT_TRUE(pso != nil);
    id<MTLBuffer> bq = [g_ctx.dev newBufferWithBytes:q.data() length:q.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bk = [g_ctx.dev newBufferWithBytes:k_cache.data() length:k_cache.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bv = [g_ctx.dev newBufferWithBytes:v_cache.data() length:v_cache.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo = [g_ctx.dev newBufferWithLength:q.size() * sizeof(float) options:MTLResourceStorageModeShared];
    float scale = 1.0f / std::sqrt((float)HD);
    DsAttnParams params{T, 0, H, HKV, HD, MAX_SEQ, 1, 2048, scale};

    id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bq offset:0 atIndex:0];
    [enc setBuffer:bk offset:0 atIndex:1];
    [enc setBuffer:bv offset:0 atIndex:2];
    [enc setBuffer:bo offset:0 atIndex:3];
    [enc setBytes:&params length:sizeof(params) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(H, T, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    const float* out = static_cast<const float*>(bo.contents);
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t kvh = h / (H / HKV);
            std::vector<float> scores(t + 1);
            float mx = -1e30f;
            for (uint32_t s = 0; s <= t; ++s) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < HD; ++d) {
                    dot += q[(t * H + h) * HD + d] * k_cache[(s * HKV + kvh) * HD + d];
                }
                scores[s] = dot * scale;
                if (scores[s] > mx) mx = scores[s];
            }
            float denom = 0.0f;
            for (uint32_t s = 0; s <= t; ++s) { scores[s] = std::exp(scores[s] - mx); denom += scores[s]; }
            for (uint32_t d = 0; d < HD; ++d) {
                float ref = 0.0f;
                for (uint32_t s = 0; s <= t; ++s) {
                    ref += (scores[s] / denom) * v_cache[(s * HKV + kvh) * HD + d];
                }
                EXPECT_TRUE(std::abs(out[(t * H + h) * HD + d] - ref) < 1e-4f);
            }
        }
    }
}

int main() {
    @autoreleasepool {
        if (!g_ctx.init()) {
            fprintf(stderr, "No Metal device available\n");
            return 1;
        }
        printf("GGUF Metal kernel tests on %s\n", [[g_ctx.dev name] UTF8String]);
        run_test("test_gguf_gemv_q8_0", test_gguf_gemv_q8_0);
        run_test("test_gguf_gemv_iq4_nl", test_gguf_gemv_iq4_nl);
        run_test("test_gguf_gemv_mxfp4", test_gguf_gemv_mxfp4);
        run_test("test_gguf_gemv_q4_k", test_gguf_gemv_q4_k);
        run_test("test_gguf_gemv_q6_k", test_gguf_gemv_q6_k);
        run_test("test_gguf_gemv_q2_k", test_gguf_gemv_q2_k);
        run_test("test_gguf_moe_topk6_accumulate", test_gguf_moe_topk6_accumulate);
        run_test("test_ds_mhc_sinkhorn_doubly_stochastic", test_ds_mhc_sinkhorn_doubly_stochastic);
        run_test("test_ds_router_topk6_sigmoid_bias", test_ds_router_topk6_sigmoid_bias);
        run_test("test_ds_csa_hca_causal_attention", test_ds_csa_hca_causal_attention);
        printf("GGUF Metal kernel tests: %d passed, %d failed\n", g_passed, g_failed);
        return g_failed == 0 ? 0 : 1;
    }
}
