#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#ifdef WARPGRID_METAL_PRECOMPILED
#include "../src/metal/kernels_gguf_metallib_embed.hpp"
#else
#include "../src/metal/kernels_gguf_source_embed.hpp"
#endif
#include "../src/gguf_iq_tables.hpp"
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

static void test_gguf_gemv_bf16() {
    const uint32_t M = 2, N = 8, K = 64;
    std::vector<float> X((size_t)M * K);
    std::vector<uint16_t> W((size_t)N * K);
    std::vector<float> Wf((size_t)N * K);
    for (size_t i = 0; i < X.size(); ++i) X[i] = std::sin((float)(i + 1) * 0.13f);
    for (size_t i = 0; i < W.size(); ++i) {
        float v = std::cos((float)(i + 3) * 0.07f);
        uint32_t u = 0;
        memcpy(&u, &v, 4);
        W[i] = (uint16_t)(u >> 16);
        uint32_t u2 = (uint32_t)W[i] << 16;
        memcpy(&Wf[i], &u2, 4);
    }
    id<MTLComputePipelineState> pso = g_ctx.pso("gguf_gemv_bf16");
    EXPECT_TRUE(pso != nil);
    id<MTLBuffer> bx = [g_ctx.dev newBufferWithBytes:X.data() length:X.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bw = [g_ctx.dev newBufferWithBytes:W.data() length:W.size() * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> by = [g_ctx.dev newBufferWithLength:(size_t)M * N * sizeof(float) options:MTLResourceStorageModeShared];
    GgufGemvParams p{M, N, K};
    id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bx offset:0 atIndex:0];
    [enc setBuffer:bw offset:0 atIndex:1];
    [enc setBuffer:by offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((N + 3u) / 4u, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    const float* Y = static_cast<const float*>(by.contents);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float ref = 0.0f;
            for (uint32_t k = 0; k < K; ++k) ref += X[m * K + k] * Wf[n * K + k];
            EXPECT_TRUE(std::abs(Y[m * N + n] - ref) < 1e-4f);
        }
    }
}

static void get_scale_min_k4_host(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static void test_gguf_gemv_q5_k() {
    const uint32_t M = 2, N = 5, K = 256;
    const size_t blk_bytes = 176;
    std::vector<float> X((size_t)M * K);
    std::vector<uint8_t> W((size_t)N * blk_bytes);
    for (size_t i = 0; i < X.size(); ++i) X[i] = std::sin((float)(i + 1) * 0.09f);
    for (uint32_t n = 0; n < N; ++n) {
        uint8_t* b = W.data() + (size_t)n * blk_bytes;
        uint16_t d16 = f32_to_f16_bits(0.05f * (float)(n + 1));
        uint16_t m16 = f32_to_f16_bits(0.02f * (float)(n + 1));
        memcpy(b + 0, &d16, 2);
        memcpy(b + 2, &m16, 2);
        for (int i = 0; i < 12; ++i) b[4 + i] = (uint8_t)((n * 17 + i * 11 + 5) & 0xFF);
        for (int i = 0; i < 32; ++i) b[16 + i] = (uint8_t)((n * 23 + i * 7 + 3) & 0xFF);
        for (int i = 0; i < 128; ++i) b[48 + i] = (uint8_t)((n * 31 + i * 13 + 1) & 0xFF);
    }
    id<MTLComputePipelineState> pso = g_ctx.pso("gguf_gemv_q5_k");
    EXPECT_TRUE(pso != nil);
    id<MTLBuffer> bx = [g_ctx.dev newBufferWithBytes:X.data() length:X.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bw = [g_ctx.dev newBufferWithBytes:W.data() length:W.size() options:MTLResourceStorageModeShared];
    id<MTLBuffer> by = [g_ctx.dev newBufferWithLength:(size_t)M * N * sizeof(float) options:MTLResourceStorageModeShared];
    GgufGemvParams p{M, N, K};
    id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bx offset:0 atIndex:0];
    [enc setBuffer:bw offset:0 atIndex:1];
    [enc setBuffer:by offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((N + 3u) / 4u, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    const float* Y = static_cast<const float*>(by.contents);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            const uint8_t* b = W.data() + (size_t)n * blk_bytes;
            uint16_t d16 = 0, m16 = 0;
            memcpy(&d16, b, 2);
            memcpy(&m16, b + 2, 2);
            float d = f16_bits_to_f32(d16);
            float dmin = f16_bits_to_f32(m16);
            const uint8_t* scales = b + 4;
            const uint8_t* qh = b + 16;
            const uint8_t* ql = b + 48;
            float ref = 0.0f;
            uint8_t u1 = 1, u2 = 2;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc0 = 0, mm0 = 0, sc1 = 0, mm1 = 0;
                get_scale_min_k4_host(is + 0, scales, sc0, mm0);
                get_scale_min_k4_host(is + 1, scales, sc1, mm1);
                float d1 = d * sc0, min1 = dmin * mm0;
                float d2 = d * sc1, min2 = dmin * mm1;
                for (int l = 0; l < 32; ++l) {
                    float w1 = d1 * (float)((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - min1;
                    float w2 = d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - min2;
                    ref += X[m * K + j + l] * w1 + X[m * K + j + 32 + l] * w2;
                }
                ql += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
            EXPECT_TRUE(std::abs(Y[m * N + n] - ref) < 1e-3f);
        }
    }
}

static void dequant_iq2_xxs_block_ref(const uint8_t* blk, float* out) {
    uint16_t d16 = 0;
    memcpy(&d16, blk, 2);
    float d = f16_bits_to_f32(d16);
    const uint16_t* qs = reinterpret_cast<const uint16_t*>(blk + 2);
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        uint32_t aux32[2];
        memcpy(aux32, qs + 4 * ib32, 8);
        const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(aux32);
        float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t* grid = reinterpret_cast<const uint8_t*>(&ink::kIq2xxsGrid[aux8[l]]);
            uint8_t signs = ink::kSignsIq2xs[(aux32[1] >> (7 * l)) & 127u];
            for (int j = 0; j < 8; ++j) {
                out[ib32 * 32 + l * 8 + j] = db * (float)grid[j] * ((signs & ink::kMaskIq2xs[j]) ? -1.0f : 1.0f);
            }
        }
    }
}

static void dequant_iq1_s_block_ref(const uint8_t* blk, float* out) {
    uint16_t d16 = 0;
    memcpy(&d16, blk, 2);
    float d = f16_bits_to_f32(d16);
    const uint8_t* qs = blk + 2;
    const uint16_t* qh = reinterpret_cast<const uint16_t*>(blk + 34);
    for (int ib = 0; ib < 8; ++ib) {
        float dl = d * (float)(2 * ((qh[ib] >> 12) & 7) + 1);
        float ml = dl * ((qh[ib] & 0x8000u) ? (-1.0f - ink::kIq1sDelta) : (-1.0f + ink::kIq1sDelta));
        for (int l = 0; l < 4; ++l) {
            uint32_t idx = (uint32_t)qs[ib * 4 + l] | (((uint32_t)((qh[ib] >> (3 * l)) & 7u)) << 8);
            uint32_t packed = ink::kIq1sGridGpu[idx];
            const uint8_t* gp = reinterpret_cast<const uint8_t*>(&packed);
            for (int j = 0; j < 4; ++j) {
                out[ib * 32 + l * 8 + j] = dl * (float)(gp[j] & 0x0Fu) + ml;
                out[ib * 32 + l * 8 + 4 + j] = dl * (float)(gp[j] >> 4) + ml;
            }
        }
    }
}

static void test_gguf_gemv_iq2_xxs() {
    const uint32_t M = 2, N = 6, K = 256;
    const size_t blk_bytes = 66;
    std::vector<float> X((size_t)M * K);
    std::vector<uint8_t> W((size_t)N * blk_bytes);
    for (size_t i = 0; i < X.size(); ++i) X[i] = std::cos((float)(i + 2) * 0.11f);
    for (uint32_t n = 0; n < N; ++n) {
        uint8_t* b = W.data() + (size_t)n * blk_bytes;
        uint16_t d16 = f32_to_f16_bits(0.125f * (float)(n + 1));
        memcpy(b, &d16, 2);
        for (size_t i = 2; i < blk_bytes; ++i) b[i] = (uint8_t)((n * 41 + i * 19 + 7) & 0xFF);
    }
    id<MTLComputePipelineState> pso = g_ctx.pso("gguf_gemv_iq2_xxs");
    EXPECT_TRUE(pso != nil);
    id<MTLBuffer> bx = [g_ctx.dev newBufferWithBytes:X.data() length:X.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bw = [g_ctx.dev newBufferWithBytes:W.data() length:W.size() options:MTLResourceStorageModeShared];
    id<MTLBuffer> by = [g_ctx.dev newBufferWithLength:(size_t)M * N * sizeof(float) options:MTLResourceStorageModeShared];
    GgufGemvParams p{M, N, K};
    id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bx offset:0 atIndex:0];
    [enc setBuffer:bw offset:0 atIndex:1];
    [enc setBuffer:by offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((N + 3u) / 4u, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    const float* Y = static_cast<const float*>(by.contents);
    std::vector<float> row(K);
    for (uint32_t n = 0; n < N; ++n) {
        dequant_iq2_xxs_block_ref(W.data() + (size_t)n * blk_bytes, row.data());
        for (uint32_t m = 0; m < M; ++m) {
            float ref = 0.0f;
            for (uint32_t k = 0; k < K; ++k) ref += X[m * K + k] * row[k];
            EXPECT_TRUE(std::abs(Y[m * N + n] - ref) < 1e-3f);
        }
    }
}

static void test_gguf_gemv_iq1_s() {
    const uint32_t M = 2, N = 6, K = 256;
    const size_t blk_bytes = 50;
    std::vector<float> X((size_t)M * K);
    std::vector<uint8_t> W((size_t)N * blk_bytes);
    for (size_t i = 0; i < X.size(); ++i) X[i] = std::sin((float)(i + 5) * 0.07f);
    for (uint32_t n = 0; n < N; ++n) {
        uint8_t* b = W.data() + (size_t)n * blk_bytes;
        uint16_t d16 = f32_to_f16_bits(0.1f * (float)(n + 1));
        memcpy(b, &d16, 2);
        for (size_t i = 2; i < blk_bytes; ++i) b[i] = (uint8_t)((n * 53 + i * 29 + 11) & 0xFF);
    }
    id<MTLComputePipelineState> pso = g_ctx.pso("gguf_gemv_iq1_s");
    EXPECT_TRUE(pso != nil);
    id<MTLBuffer> bx = [g_ctx.dev newBufferWithBytes:X.data() length:X.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bw = [g_ctx.dev newBufferWithBytes:W.data() length:W.size() options:MTLResourceStorageModeShared];
    id<MTLBuffer> by = [g_ctx.dev newBufferWithLength:(size_t)M * N * sizeof(float) options:MTLResourceStorageModeShared];
    GgufGemvParams p{M, N, K};
    id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bx offset:0 atIndex:0];
    [enc setBuffer:bw offset:0 atIndex:1];
    [enc setBuffer:by offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((N + 3u) / 4u, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    const float* Y = static_cast<const float*>(by.contents);
    std::vector<float> row(K);
    for (uint32_t n = 0; n < N; ++n) {
        dequant_iq1_s_block_ref(W.data() + (size_t)n * blk_bytes, row.data());
        for (uint32_t m = 0; m < M; ++m) {
            float ref = 0.0f;
            for (uint32_t k = 0; k < K; ++k) ref += X[m * K + k] * row[k];
            EXPECT_TRUE(std::abs(Y[m * N + n] - ref) < 1e-3f);
        }
    }
}

static void test_gguf_moe_indexed_iq1_s() {
    const uint32_t T = 2, E = 4, N = 6, K = 256, TOPK = 2;
    const size_t blk_bytes = 50;
    const uint32_t stride = N * blk_bytes;
    std::vector<float> X((size_t)T * K);
    std::vector<uint8_t> W((size_t)E * stride);
    std::vector<int32_t> topk_ids = {2, 0, 3, 1};
    std::vector<float> topk_w = {0.7f, 0.3f, 0.4f, 0.6f};
    for (size_t i = 0; i < X.size(); ++i) X[i] = std::sin((float)(i + 1) * 0.05f);
    for (uint32_t e = 0; e < E; ++e) {
        for (uint32_t n = 0; n < N; ++n) {
            uint8_t* b = W.data() + (size_t)e * stride + (size_t)n * blk_bytes;
            uint16_t d16 = f32_to_f16_bits(0.08f * (float)(e + 1));
            memcpy(b, &d16, 2);
            for (size_t i = 2; i < blk_bytes; ++i) b[i] = (uint8_t)((e * 67 + n * 19 + i * 13 + 5) & 0xFF);
        }
    }
    id<MTLComputePipelineState> pso_idx = g_ctx.pso("gguf_moe_indexed_iq1_s");
    id<MTLComputePipelineState> pso_acc = g_ctx.pso("gguf_moe_down_accum_iq1_s");
    EXPECT_TRUE(pso_idx != nil);
    EXPECT_TRUE(pso_acc != nil);
    id<MTLBuffer> bx = [g_ctx.dev newBufferWithBytes:X.data() length:X.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bw = [g_ctx.dev newBufferWithBytes:W.data() length:W.size() options:MTLResourceStorageModeShared];
    id<MTLBuffer> bids = [g_ctx.dev newBufferWithBytes:topk_ids.data() length:topk_ids.size() * sizeof(int32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> btw = [g_ctx.dev newBufferWithBytes:topk_w.data() length:topk_w.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> by_idx = [g_ctx.dev newBufferWithLength:(size_t)T * TOPK * N * sizeof(float) options:MTLResourceStorageModeShared];
    std::vector<float> base_y((size_t)T * N, 1.25f);
    id<MTLBuffer> by_acc = [g_ctx.dev newBufferWithBytes:base_y.data() length:base_y.size() * sizeof(float) options:MTLResourceStorageModeShared];
    std::vector<float> X_per((size_t)T * TOPK * K);
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t r = 0; r < TOPK; ++r) {
            memcpy(&X_per[(t * TOPK + r) * K], &X[t * K], K * sizeof(float));
        }
    }
    id<MTLBuffer> bx_per = [g_ctx.dev newBufferWithBytes:X_per.data() length:X_per.size() * sizeof(float) options:MTLResourceStorageModeShared];

    GgufMoeParams mp{T, N, K, TOPK, stride};
    id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso_idx];
    [enc setBuffer:bx offset:0 atIndex:0];
    [enc setBuffer:bw offset:0 atIndex:1];
    [enc setBuffer:bids offset:0 atIndex:2];
    [enc setBuffer:by_idx offset:0 atIndex:3];
    [enc setBytes:&mp length:sizeof(mp) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((N + 3u) / 4u, T * TOPK, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    [enc setComputePipelineState:pso_acc];
    [enc setBuffer:bx_per offset:0 atIndex:0];
    [enc setBuffer:bw offset:0 atIndex:1];
    [enc setBuffer:bids offset:0 atIndex:2];
    [enc setBuffer:btw offset:0 atIndex:3];
    [enc setBuffer:by_acc offset:0 atIndex:4];
    [enc setBytes:&mp length:sizeof(mp) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((N + 3u) / 4u, T, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    const float* Y_idx = static_cast<const float*>(by_idx.contents);
    const float* Y_acc = static_cast<const float*>(by_acc.contents);
    std::vector<float> row(K);
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t n = 0; n < N; ++n) {
            float ref_acc = 1.25f;
            for (uint32_t r = 0; r < TOPK; ++r) {
                int e = topk_ids[t * TOPK + r];
                dequant_iq1_s_block_ref(W.data() + (size_t)e * stride + (size_t)n * blk_bytes, row.data());
                float dot = 0.0f;
                for (uint32_t k = 0; k < K; ++k) dot += X[t * K + k] * row[k];
                EXPECT_TRUE(std::abs(Y_idx[(t * TOPK + r) * N + n] - dot) < 1e-3f);
                ref_acc += topk_w[t * TOPK + r] * dot;
            }
            EXPECT_TRUE(std::abs(Y_acc[t * N + n] - ref_acc) < 1e-3f);
        }
    }
}

static void test_gguf_gemm_tiled_and_smallk_parity() {
    const uint32_t M = 37, N = 41, K = 512;
    std::vector<float> x((size_t)M * K);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin((float)(i + 3) * 0.031f) * 0.5f;

    auto run_tiled = [&](const char* kname, const std::vector<uint8_t>& w, std::vector<float>& y) {
        id<MTLBuffer> bx = [g_ctx.dev newBufferWithBytes:x.data() length:x.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bw = [g_ctx.dev newBufferWithBytes:w.data() length:w.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [g_ctx.dev newBufferWithLength:(size_t)M * N * sizeof(float) options:MTLResourceStorageModeShared];
        GgufGemvParams p{M, N, K};
        id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_ctx.pso(kname)];
        [enc setBuffer:bx offset:0 atIndex:0];
        [enc setBuffer:bw offset:0 atIndex:1];
        [enc setBuffer:by offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((N + 31u) / 32u, (M + 31u) / 32u, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        y.resize((size_t)M * N);
        memcpy(y.data(), by.contents, y.size() * sizeof(float));
    };

    {
        const uint32_t nb = K / 256;
        std::vector<uint8_t> w((size_t)N * nb * 210, 0);
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t b = 0; b < nb; ++b) {
                uint8_t* blk = w.data() + ((size_t)n * nb + b) * 210;
                for (int i = 0; i < 128; ++i) blk[i] = (uint8_t)((n * 17 + b * 23 + i * 7) & 0xFFu);
                for (int i = 0; i < 64; ++i) blk[128 + i] = (uint8_t)((n * 13 + b * 5 + i * 11) & 0xFFu);
                for (int i = 0; i < 16; ++i) ((int8_t*)(blk + 192))[i] = (int8_t)((int)((n + b + i) % 15) - 7);
                uint16_t d = f32_to_f16_bits(0.05f);
                memcpy(blk + 208, &d, 2);
            }
        }
        std::vector<float> y_ref, y_tiled, y_smallk;
        dispatch_gguf_gemv("gguf_gemv_q6_k", x, w, M, N, K, y_ref);
        run_tiled("gguf_gemm_q6_k_tiled", w, y_tiled);
        for (size_t i = 0; i < y_ref.size(); ++i) EXPECT_TRUE(std::abs(y_tiled[i] - y_ref[i]) <= 5e-2f + 2e-3f * std::abs(y_ref[i]));

        id<MTLBuffer> bx = [g_ctx.dev newBufferWithBytes:x.data() length:x.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bw = [g_ctx.dev newBufferWithBytes:w.data() length:w.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [g_ctx.dev newBufferWithLength:(size_t)M * N * sizeof(float) options:MTLResourceStorageModeShared];
        GgufGemvParams p{M, N, K};
        id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_ctx.pso("gguf_gemv_q6_k_smallk")];
        [enc setBuffer:bx offset:0 atIndex:0];
        [enc setBuffer:bw offset:0 atIndex:1];
        [enc setBuffer:by offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((N + 7u) / 8u, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        y_smallk.resize((size_t)M * N);
        memcpy(y_smallk.data(), by.contents, y_smallk.size() * sizeof(float));
        for (size_t i = 0; i < y_ref.size(); ++i) EXPECT_TRUE(std::abs(y_smallk[i] - y_ref[i]) < 1e-2f);
    }
    {
        const uint32_t nb = K / 256;
        std::vector<uint8_t> w((size_t)N * nb * 176, 0);
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t b = 0; b < nb; ++b) {
                uint8_t* blk = w.data() + ((size_t)n * nb + b) * 176;
                uint16_t d = f32_to_f16_bits(0.005f);
                uint16_t dm = f32_to_f16_bits(0.002f);
                memcpy(blk, &d, 2);
                memcpy(blk + 2, &dm, 2);
                for (int i = 0; i < 12; ++i) blk[4 + i] = (uint8_t)((n * 7 + b * 3 + i * 5) & 0x3Fu);
                for (int i = 0; i < 32; ++i) blk[16 + i] = (uint8_t)((n * 11 + b * 13 + i) & 0xFFu);
                for (int i = 0; i < 128; ++i) blk[48 + i] = (uint8_t)((n * 19 + b * 7 + i * 3) & 0xFFu);
            }
        }
        std::vector<float> y_ref, y_tiled;
        dispatch_gguf_gemv("gguf_gemv_q5_k", x, w, M, N, K, y_ref);
        run_tiled("gguf_gemm_q5_k_tiled", w, y_tiled);
        for (size_t i = 0; i < y_ref.size(); ++i) EXPECT_TRUE(std::abs(y_tiled[i] - y_ref[i]) <= 5e-2f + 2e-3f * std::abs(y_ref[i]));
    }
    {
        const uint32_t nb = K / 32;
        std::vector<uint8_t> w((size_t)N * nb * 34, 0);
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t b = 0; b < nb; ++b) {
                uint8_t* blk = w.data() + ((size_t)n * nb + b) * 34;
                uint16_t d = f32_to_f16_bits(0.03f);
                memcpy(blk, &d, 2);
                for (int i = 0; i < 32; ++i) ((int8_t*)(blk + 2))[i] = (int8_t)((int)((n * 5 + b * 7 + i * 3) % 63) - 31);
            }
        }
        std::vector<float> y_ref, y_tiled;
        dispatch_gguf_gemv("gguf_gemv_q8_0", x, w, M, N, K, y_ref);
        run_tiled("gguf_gemm_q8_0_tiled", w, y_tiled);
        for (size_t i = 0; i < y_ref.size(); ++i) EXPECT_TRUE(std::abs(y_tiled[i] - y_ref[i]) <= 5e-2f + 2e-3f * std::abs(y_ref[i]));
    }
}

static void test_ds_dspark_and_multi_gemv_parity() {
    const uint32_t M = 4, N = 37, K = 512;
    std::vector<float> x((size_t)M * K);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin((float)(i + 3) * 0.031f);

    auto run_multi = [&](const char* kname, const std::vector<uint8_t>& w, std::vector<float>& y_out) {
        id<MTLComputePipelineState> pso = g_ctx.pso(kname);
        EXPECT_TRUE(pso != nil);
        id<MTLBuffer> bx = [g_ctx.dev newBufferWithBytes:x.data() length:x.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bw = [g_ctx.dev newBufferWithBytes:w.data() length:w.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [g_ctx.dev newBufferWithLength:(size_t)M * N * sizeof(float) options:MTLResourceStorageModeShared];
        GgufGemvParams p{M, N, K};
        id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:0 atIndex:0];
        [enc setBuffer:bw offset:0 atIndex:1];
        [enc setBuffer:by offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((N + 1u) / 2u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        y_out.resize((size_t)M * N);
        memcpy(y_out.data(), by.contents, y_out.size() * sizeof(float));
    };

    {
        const uint32_t nb = K / 256;
        std::vector<uint8_t> w((size_t)N * nb * 210, 0);
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t b = 0; b < nb; ++b) {
                uint8_t* blk = w.data() + ((size_t)n * nb + b) * 210;
                for (int i = 0; i < 128; ++i) blk[i] = (uint8_t)((n * 13 + b * 7 + i * 5) & 0xFFu);
                for (int i = 0; i < 64; ++i) blk[128 + i] = (uint8_t)((n * 17 + b * 11 + i * 3) & 0xFFu);
                for (int i = 0; i < 16; ++i) ((int8_t*)(blk + 192))[i] = (int8_t)((int)((n + b + i) % 15) - 7);
                uint16_t d = f32_to_f16_bits(0.01f);
                memcpy(blk + 208, &d, 2);
            }
        }
        std::vector<float> y_ref, y_m;
        dispatch_gguf_gemv("gguf_gemv_q6_k", x, w, M, N, K, y_ref);
        run_multi("gguf_gemv_q6_k_multi", w, y_m);
        for (size_t i = 0; i < y_ref.size(); ++i) EXPECT_TRUE(std::abs(y_m[i] - y_ref[i]) < 1e-3f);
    }
    {
        const uint32_t nb = K / 256;
        std::vector<uint8_t> w((size_t)N * nb * 176, 0);
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t b = 0; b < nb; ++b) {
                uint8_t* blk = w.data() + ((size_t)n * nb + b) * 176;
                uint16_t d = f32_to_f16_bits(0.005f), dm = f32_to_f16_bits(0.002f);
                memcpy(blk, &d, 2);
                memcpy(blk + 2, &dm, 2);
                for (int i = 0; i < 12; ++i) blk[4 + i] = (uint8_t)((n * 7 + b * 3 + i * 5) & 0x3Fu);
                for (int i = 0; i < 32; ++i) blk[16 + i] = (uint8_t)((n * 11 + b * 13 + i) & 0xFFu);
                for (int i = 0; i < 128; ++i) blk[48 + i] = (uint8_t)((n * 19 + b * 7 + i * 3) & 0xFFu);
            }
        }
        std::vector<float> y_ref, y_m;
        dispatch_gguf_gemv("gguf_gemv_q5_k", x, w, M, N, K, y_ref);
        run_multi("gguf_gemv_q5_k_multi", w, y_m);
        for (size_t i = 0; i < y_ref.size(); ++i) EXPECT_TRUE(std::abs(y_m[i] - y_ref[i]) < 1e-3f);
    }
    {
        const uint32_t nb = K / 32;
        std::vector<uint8_t> w((size_t)N * nb * 34, 0);
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t b = 0; b < nb; ++b) {
                uint8_t* blk = w.data() + ((size_t)n * nb + b) * 34;
                uint16_t d = f32_to_f16_bits(0.03f);
                memcpy(blk, &d, 2);
                for (int i = 0; i < 32; ++i) ((int8_t*)(blk + 2))[i] = (int8_t)((int)((n * 5 + b * 7 + i * 3) % 63) - 31);
            }
        }
        std::vector<float> y_ref, y_m;
        dispatch_gguf_gemv("gguf_gemv_q8_0", x, w, M, N, K, y_ref);
        run_multi("gguf_gemv_q8_0_multi", w, y_m);
        for (size_t i = 0; i < y_ref.size(); ++i) EXPECT_TRUE(std::abs(y_m[i] - y_ref[i]) < 1e-3f);
    }
    {
        struct DsMarkovBiasParams { uint32_t V; uint32_t R; };
        const uint32_t V = 19, R = 16;
        std::vector<float> base(V);
        std::vector<uint16_t> w1((size_t)V * R), w2((size_t)V * R);
        auto f32_to_bf16 = [](float f) -> uint16_t { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); };
        auto bf16_to_f32 = [](uint16_t b) -> float { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; };
        for (uint32_t v = 0; v < V; ++v) {
            base[v] = std::cos((float)v * 0.2f);
            for (uint32_t r = 0; r < R; ++r) {
                w1[v * R + r] = f32_to_bf16(std::sin((float)(v * R + r) * 0.1f));
                w2[v * R + r] = f32_to_bf16(std::cos((float)(v * R + r) * 0.13f));
            }
        }
        int32_t prev = 7;
        id<MTLBuffer> bb = [g_ctx.dev newBufferWithBytes:base.data() length:V * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bw1 = [g_ctx.dev newBufferWithBytes:w1.data() length:w1.size() * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bw2 = [g_ctx.dev newBufferWithBytes:w2.data() length:w2.size() * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bp = [g_ctx.dev newBufferWithBytes:&prev length:sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bo = [g_ctx.dev newBufferWithLength:V * sizeof(float) options:MTLResourceStorageModeShared];
        DsMarkovBiasParams mp{V, R};
        id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_ctx.pso("ds_markov_bias_add_bf16")];
        [enc setBuffer:bb offset:0 atIndex:0];
        [enc setBuffer:bw1 offset:0 atIndex:1];
        [enc setBuffer:bw2 offset:0 atIndex:2];
        [enc setBuffer:bp offset:0 atIndex:3];
        [enc setBuffer:bo offset:0 atIndex:4];
        [enc setBytes:&mp length:sizeof(mp) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((V + 3u) / 4u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const float* out = static_cast<const float*>(bo.contents);
        for (uint32_t v = 0; v < V; ++v) {
            float ref = base[v];
            for (uint32_t r = 0; r < R; ++r) ref += bf16_to_f32(w1[(size_t)prev * R + r]) * bf16_to_f32(w2[(size_t)v * R + r]);
            EXPECT_TRUE(std::abs(out[v] - ref) < 1e-4f);
        }
    }
}

static void test_ds_grouped_moe_and_mxfp4_fused_parity() {
    {
        struct GgufGroupedGemvParams { uint32_t T; uint32_t G; uint32_t N; uint32_t K; uint32_t group_stride_bytes; };
        const uint32_t T = 3, G = 4, N = 14, K = 256;
        const uint32_t nb = K / 32;
        const uint32_t g_stride = N * nb * 34;
        std::vector<float> x((size_t)T * G * K);
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin((float)(i + 1) * 0.041f);
        std::vector<uint8_t> w((size_t)G * g_stride, 0);
        for (uint32_t g = 0; g < G; ++g) {
            for (uint32_t n = 0; n < N; ++n) {
                for (uint32_t b = 0; b < nb; ++b) {
                    uint8_t* blk = w.data() + (size_t)g * g_stride + ((size_t)n * nb + b) * 34;
                    uint16_t d = f32_to_f16_bits(0.025f);
                    memcpy(blk, &d, 2);
                    for (int i = 0; i < 32; ++i) ((int8_t*)(blk + 2))[i] = (int8_t)((int)((g * 11 + n * 5 + b * 7 + i * 3) % 63) - 31);
                }
            }
        }
        id<MTLComputePipelineState> pso = g_ctx.pso("gguf_grouped_gemv_q8_0_fast");
        EXPECT_TRUE(pso != nil);
        if (pso == nil) return;
        id<MTLBuffer> bx = [g_ctx.dev newBufferWithBytes:x.data() length:x.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bw = [g_ctx.dev newBufferWithBytes:w.data() length:w.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [g_ctx.dev newBufferWithLength:(size_t)T * G * N * sizeof(float) options:MTLResourceStorageModeShared];
        GgufGroupedGemvParams gp{T, G, N, K, g_stride};
        id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:0 atIndex:0];
        [enc setBuffer:bw offset:0 atIndex:1];
        [enc setBuffer:by offset:0 atIndex:2];
        [enc setBytes:&gp length:sizeof(gp) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((N + 3u) / 4u, T * G, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const float* Y = static_cast<const float*>(by.contents);
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t g = 0; g < G; ++g) {
                std::vector<float> xg(x.begin() + ((size_t)t * G + g) * K, x.begin() + ((size_t)t * G + g + 1) * K);
                std::vector<uint8_t> wg(w.begin() + (size_t)g * g_stride, w.begin() + (size_t)(g + 1) * g_stride);
                std::vector<float> yref;
                dispatch_gguf_gemv("gguf_gemv_q8_0", xg, wg, 1, N, K, yref);
                for (uint32_t n = 0; n < N; ++n) {
                    EXPECT_TRUE(std::abs(Y[((size_t)t * G + g) * N + n] - yref[n]) < 1e-3f);
                }
            }
        }
    }
    {
        struct GgufMoeParams { uint32_t T; uint32_t N; uint32_t K; uint32_t top_k; uint32_t expert_stride_bytes; };
        struct GgufMoeSwigluParams { uint32_t T; uint32_t N; uint32_t K; uint32_t top_k; uint32_t expert_stride_bytes; float clamp_val; };
        const uint32_t T = 3, E = 4, top_k = 2, D = 256, I = 128;
        const uint32_t gu_stride = I * (D / 32) * 17;
        const uint32_t d_stride = D * (I / 32) * 17;
        std::vector<float> x((size_t)T * D);
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos((float)(i + 2) * 0.037f);
        std::vector<uint8_t> wg((size_t)E * gu_stride), wu((size_t)E * gu_stride), wd((size_t)E * d_stride);
        for (size_t i = 0; i < wg.size(); ++i) { wg[i] = (uint8_t)((i * 37 + 13) & 0xFFu); wu[i] = (uint8_t)((i * 53 + 7) & 0xFFu); }
        for (size_t i = 0; i < wd.size(); ++i) wd[i] = (uint8_t)((i * 29 + 19) & 0xFFu);
        for (uint32_t e = 0; e < E; ++e) {
            for (uint32_t r = 0; r < I; ++r) {
                for (uint32_t b = 0; b < D / 32; ++b) {
                    wg[(size_t)e * gu_stride + ((size_t)r * (D / 32) + b) * 17] = 125;
                    wu[(size_t)e * gu_stride + ((size_t)r * (D / 32) + b) * 17] = 125;
                }
            }
            for (uint32_t r = 0; r < D; ++r) {
                for (uint32_t b = 0; b < I / 32; ++b) {
                    wd[(size_t)e * d_stride + ((size_t)r * (I / 32) + b) * 17] = 125;
                }
            }
        }
        std::vector<int32_t> tids = {0, 2, 2, 1, 0, 3};
        std::vector<float> tw = {0.6f, 0.4f, 0.7f, 0.3f, 0.55f, 0.45f};
        id<MTLComputePipelineState> p_gu = g_ctx.pso("gguf_moe_gate_up_swiglu_mxfp4");
        id<MTLComputePipelineState> p_dn = g_ctx.pso("gguf_moe_down_accum_mxfp4_4r");
        EXPECT_TRUE(p_gu != nil && p_dn != nil);
        if (p_gu == nil || p_dn == nil) return;
        id<MTLBuffer> bx = [g_ctx.dev newBufferWithBytes:x.data() length:x.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bwg = [g_ctx.dev newBufferWithBytes:wg.data() length:wg.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bwu = [g_ctx.dev newBufferWithBytes:wu.data() length:wu.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bwd = [g_ctx.dev newBufferWithBytes:wd.data() length:wd.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> btids = [g_ctx.dev newBufferWithBytes:tids.data() length:tids.size() * sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> btw = [g_ctx.dev newBufferWithBytes:tw.data() length:tw.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bact = [g_ctx.dev newBufferWithLength:(size_t)T * top_k * I * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> by_new = [g_ctx.dev newBufferWithLength:(size_t)T * D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> by_ref = [g_ctx.dev newBufferWithLength:(size_t)T * D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bg_ref = [g_ctx.dev newBufferWithLength:(size_t)T * top_k * I * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bu_ref = [g_ctx.dev newBufferWithLength:(size_t)T * top_k * I * sizeof(float) options:MTLResourceStorageModeShared];
        memset(by_new.contents, 0, by_new.length);
        memset(by_ref.contents, 0, by_ref.length);
        GgufMoeSwigluParams sp{T, I, D, top_k, gu_stride, 10.0f};
        GgufMoeParams mp_gu{T, I, D, top_k, gu_stride};
        GgufMoeParams mp_dn{T, D, I, top_k, d_stride};
        id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p_gu];
        [enc setBuffer:bx offset:0 atIndex:0];
        [enc setBuffer:bwg offset:0 atIndex:1];
        [enc setBuffer:bwu offset:0 atIndex:2];
        [enc setBuffer:btids offset:0 atIndex:3];
        [enc setBuffer:bact offset:0 atIndex:4];
        [enc setBytes:&sp length:sizeof(sp) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((I + 3u) / 4u, T * top_k, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc setComputePipelineState:p_dn];
        [enc setBuffer:bact offset:0 atIndex:0];
        [enc setBuffer:bwd offset:0 atIndex:1];
        [enc setBuffer:btids offset:0 atIndex:2];
        [enc setBuffer:btw offset:0 atIndex:3];
        [enc setBuffer:by_new offset:0 atIndex:4];
        [enc setBytes:&mp_dn length:sizeof(mp_dn) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((D + 7u) / 8u, T, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc setComputePipelineState:g_ctx.pso("gguf_moe_indexed_mxfp4")];
        [enc setBuffer:bx offset:0 atIndex:0];
        [enc setBuffer:bwg offset:0 atIndex:1];
        [enc setBuffer:btids offset:0 atIndex:2];
        [enc setBuffer:bg_ref offset:0 atIndex:3];
        [enc setBytes:&mp_gu length:sizeof(mp_gu) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((I + 3u) / 4u, T * top_k, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc setBuffer:bwu offset:0 atIndex:1];
        [enc setBuffer:bu_ref offset:0 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((I + 3u) / 4u, T * top_k, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        float* gr = static_cast<float*>(bg_ref.contents);
        const float* ur = static_cast<const float*>(bu_ref.contents);
        for (size_t i = 0; i < (size_t)T * top_k * I; ++i) {
            float gv = std::clamp(gr[i], -10.0f, 10.0f);
            float uv = std::clamp(ur[i], -10.0f, 10.0f);
            gr[i] = (gv / (1.0f + std::exp(-gv))) * uv;
        }
        cb = [g_ctx.q commandBuffer];
        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_ctx.pso("gguf_moe_down_accum_mxfp4")];
        [enc setBuffer:bg_ref offset:0 atIndex:0];
        [enc setBuffer:bwd offset:0 atIndex:1];
        [enc setBuffer:btids offset:0 atIndex:2];
        [enc setBuffer:btw offset:0 atIndex:3];
        [enc setBuffer:by_ref offset:0 atIndex:4];
        [enc setBytes:&mp_dn length:sizeof(mp_dn) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((D + 3u) / 4u, T, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const float* yn = static_cast<const float*>(by_new.contents);
        const float* yr = static_cast<const float*>(by_ref.contents);
        for (size_t i = 0; i < (size_t)T * D; ++i) EXPECT_TRUE(std::abs(yn[i] - yr[i]) <= 2e-3f + 5e-3f * std::abs(yr[i]));
    }
}

static void test_ds_attn_router_fast_kernels_parity() {
    {
        const uint32_t M = 4, N = 384, K = 512;
        std::vector<float> x((size_t)M * K), w((size_t)N * K);
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin((float)(i + 1) * 0.031f);
        for (size_t i = 0; i < w.size(); ++i) w[i] = std::cos((float)(i + 3) * 0.017f);
        id<MTLComputePipelineState> pso = g_ctx.pso("gguf_gemv_f32_fast");
        EXPECT_TRUE(pso != nil);
        if (pso == nil) return;
        id<MTLBuffer> bx = [g_ctx.dev newBufferWithBytes:x.data() length:x.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bw = [g_ctx.dev newBufferWithBytes:w.data() length:w.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [g_ctx.dev newBufferWithLength:(size_t)M * N * sizeof(float) options:MTLResourceStorageModeShared];
        GgufGemvParams p{M, N, K};
        id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:0 atIndex:0];
        [enc setBuffer:bw offset:0 atIndex:1];
        [enc setBuffer:by offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((N + 1u) / 2u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const float* Y = static_cast<const float*>(by.contents);
        for (uint32_t m = 0; m < M; ++m) {
            for (uint32_t n = 0; n < N; ++n) {
                float ref = 0.0f;
                for (uint32_t k = 0; k < K; ++k) ref += x[(size_t)m * K + k] * w[(size_t)n * K + k];
                EXPECT_TRUE(std::abs(Y[(size_t)m * N + n] - ref) < 1e-3f);
            }
        }
    }
    {
        struct GgufGroupedGemvParams { uint32_t T; uint32_t G; uint32_t N; uint32_t K; uint32_t group_stride_bytes; };
        const uint32_t T = 4, G = 4, N = 14, K = 256;
        const uint32_t nb = K / 256;
        const uint32_t g_stride = N * nb * 210;
        std::vector<float> x((size_t)T * G * K);
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin((float)(i + 2) * 0.023f);
        std::vector<uint8_t> w((size_t)G * g_stride, 0);
        for (uint32_t g = 0; g < G; ++g) {
            for (uint32_t n = 0; n < N; ++n) {
                uint8_t* blk = w.data() + (size_t)g * g_stride + (size_t)n * 210;
                for (int i = 0; i < 128; ++i) blk[i] = (uint8_t)((g * 19 + n * 13 + i * 5) & 0xFFu);
                for (int i = 0; i < 64; ++i) blk[128 + i] = (uint8_t)((g * 7 + n * 17 + i * 3) & 0xFFu);
                for (int i = 0; i < 16; ++i) ((int8_t*)(blk + 192))[i] = (int8_t)((int)((g + n + i) % 15) - 7);
                uint16_t d = f32_to_f16_bits(0.01f);
                memcpy(blk + 208, &d, 2);
            }
        }
        id<MTLComputePipelineState> p_m = g_ctx.pso("gguf_grouped_gemv_q6_k_multi");
        EXPECT_TRUE(p_m != nil);
        if (p_m == nil) return;
        id<MTLBuffer> bx = [g_ctx.dev newBufferWithBytes:x.data() length:x.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bw = [g_ctx.dev newBufferWithBytes:w.data() length:w.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> by_m = [g_ctx.dev newBufferWithLength:(size_t)T * G * N * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> by_r = [g_ctx.dev newBufferWithLength:(size_t)T * G * N * sizeof(float) options:MTLResourceStorageModeShared];
        GgufGroupedGemvParams gp{T, G, N, K, g_stride};
        id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p_m];
        [enc setBuffer:bx offset:0 atIndex:0];
        [enc setBuffer:bw offset:0 atIndex:1];
        [enc setBuffer:by_m offset:0 atIndex:2];
        [enc setBytes:&gp length:sizeof(gp) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((N + 1u) / 2u, G, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc setComputePipelineState:g_ctx.pso("gguf_grouped_gemv_q6_k_fast")];
        [enc setBuffer:bx offset:0 atIndex:0];
        [enc setBuffer:bw offset:0 atIndex:1];
        [enc setBuffer:by_r offset:0 atIndex:2];
        [enc setBytes:&gp length:sizeof(gp) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((N + 3u) / 4u, T * G, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const float* ym = static_cast<const float*>(by_m.contents);
        const float* yr = static_cast<const float*>(by_r.contents);
        for (size_t i = 0; i < (size_t)T * G * N; ++i) EXPECT_TRUE(std::abs(ym[i] - yr[i]) < 1e-3f);
    }
    {
        struct DsKvWriteParams { uint32_t T; uint32_t head_dim; uint32_t start_pos; uint32_t write_comp; };
        const uint32_t T = 3, HD = 64, start_pos = 5;
        std::vector<float> kv_in((size_t)T * HD), ckv_in((size_t)T * HD), csc_in((size_t)T * HD);
        for (size_t i = 0; i < kv_in.size(); ++i) { kv_in[i] = (float)i + 1.0f; ckv_in[i] = (float)i + 100.0f; csc_in[i] = (float)i + 200.0f; }
        id<MTLComputePipelineState> p_kw = g_ctx.pso("ds_kv_write_f32");
        EXPECT_TRUE(p_kw != nil);
        if (p_kw == nil) return;
        id<MTLBuffer> bki = [g_ctx.dev newBufferWithBytes:kv_in.data() length:kv_in.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bci = [g_ctx.dev newBufferWithBytes:ckv_in.data() length:ckv_in.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bsi = [g_ctx.dev newBufferWithBytes:csc_in.data() length:csc_in.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bko = [g_ctx.dev newBufferWithLength:16u * HD * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bco = [g_ctx.dev newBufferWithLength:16u * HD * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bso = [g_ctx.dev newBufferWithLength:16u * HD * sizeof(float) options:MTLResourceStorageModeShared];
        memset(bko.contents, 0, bko.length);
        memset(bco.contents, 0, bco.length);
        memset(bso.contents, 0, bso.length);
        DsKvWriteParams kp{T, HD, start_pos, 1u};
        id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p_kw];
        [enc setBuffer:bki offset:0 atIndex:0];
        [enc setBuffer:bko offset:0 atIndex:1];
        [enc setBuffer:bci offset:0 atIndex:2];
        [enc setBuffer:bco offset:0 atIndex:3];
        [enc setBuffer:bsi offset:0 atIndex:4];
        [enc setBuffer:bso offset:0 atIndex:5];
        [enc setBytes:&kp length:sizeof(kp) atIndex:6];
        [enc dispatchThreads:MTLSizeMake(HD, T, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const float* ko = static_cast<const float*>(bko.contents);
        const float* co = static_cast<const float*>(bco.contents);
        const float* so = static_cast<const float*>(bso.contents);
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t d = 0; d < HD; ++d) {
                EXPECT_TRUE(ko[(start_pos + t) * HD + d] == kv_in[t * HD + d]);
                EXPECT_TRUE(co[(start_pos + t) * HD + d] == ckv_in[t * HD + d]);
                EXPECT_TRUE(so[(start_pos + t) * HD + d] == csc_in[t * HD + d]);
            }
        }
    }
}

static void test_ds_dspark_conf_head_parity() {
    struct DsDsparkConfParams { uint32_t B; uint32_t D; uint32_t R; uint32_t V; };
    const uint32_t B = 4, D = 64, R = 16, V = 32;
    auto f32_to_bf16 = [](float f) -> uint16_t { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); };
    auto bf16_to_f32 = [](uint16_t b) -> float { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; };
    std::vector<float> h((size_t)B * D), proj((size_t)(D + R));
    std::vector<uint16_t> w1((size_t)V * R);
    std::vector<int32_t> toks = {3, 11, 0, 27};
    for (size_t i = 0; i < h.size(); ++i) h[i] = std::sin((float)(i + 1) * 0.09f);
    for (size_t i = 0; i < proj.size(); ++i) proj[i] = std::cos((float)(i + 2) * 0.07f) * 0.2f;
    for (size_t i = 0; i < w1.size(); ++i) w1[i] = f32_to_bf16(std::sin((float)(i + 5) * 0.11f));
    id<MTLComputePipelineState> pso = g_ctx.pso("ds_dspark_conf_f32");
    EXPECT_TRUE(pso != nil);
    if (pso == nil) return;
    id<MTLBuffer> bh = [g_ctx.dev newBufferWithBytes:h.data() length:h.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bw1 = [g_ctx.dev newBufferWithBytes:w1.data() length:w1.size() * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bproj = [g_ctx.dev newBufferWithBytes:proj.data() length:proj.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> btok = [g_ctx.dev newBufferWithBytes:toks.data() length:toks.size() * sizeof(int32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bout = [g_ctx.dev newBufferWithLength:B * sizeof(float) options:MTLResourceStorageModeShared];
    DsDsparkConfParams cp{B, D, R, V};
    id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bh offset:0 atIndex:0];
    [enc setBuffer:bw1 offset:0 atIndex:1];
    [enc setBuffer:bproj offset:0 atIndex:2];
    [enc setBuffer:btok offset:0 atIndex:3];
    [enc setBuffer:bout offset:0 atIndex:4];
    [enc setBytes:&cp length:sizeof(cp) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    const float* out = static_cast<const float*>(bout.contents);
    for (uint32_t b = 0; b < B; ++b) {
        float acc = 0.0f;
        for (uint32_t d = 0; d < D; ++d) acc += h[(size_t)b * D + d] * proj[d];
        for (uint32_t r = 0; r < R; ++r) acc += bf16_to_f32(w1[(size_t)toks[b] * R + r]) * proj[D + r];
        float ref = 1.0f / (1.0f + std::exp(-acc));
        EXPECT_TRUE(std::abs(out[b] - ref) < 1e-5f);
    }
}

static void test_ds_dspark_draft_fast_kernels_parity() {
    {
        struct GgufGroupedGemvParams { uint32_t T; uint32_t G; uint32_t N; uint32_t K; uint32_t group_stride_bytes; };
        const uint32_t T = 5, G = 4, N = 37, K = 512;
        const uint32_t nb = K / 32;
        const uint32_t g_stride = N * nb * 34u;
        std::vector<float> x((size_t)T * G * K);
        std::vector<uint8_t> w((size_t)G * g_stride, 0);
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin((float)(i + 5) * 0.037f);
        for (uint32_t g = 0; g < G; ++g) {
            for (uint32_t n = 0; n < N; ++n) {
                for (uint32_t b = 0; b < nb; ++b) {
                    uint8_t* blk = w.data() + (size_t)g * g_stride + ((size_t)n * nb + b) * 34;
                    uint16_t d = f32_to_f16_bits(0.025f);
                    memcpy(blk, &d, 2);
                    for (int i = 0; i < 32; ++i) {
                        ((int8_t*)(blk + 2))[i] = (int8_t)((int)((g * 13 + n * 7 + b * 5 + i * 3) % 63) - 31);
                    }
                }
            }
        }
        id<MTLComputePipelineState> p_m = g_ctx.pso("gguf_grouped_gemv_q8_0_multi");
        EXPECT_TRUE(p_m != nil);
        if (p_m == nil) return;
        id<MTLBuffer> bx = [g_ctx.dev newBufferWithBytes:x.data() length:x.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bw = [g_ctx.dev newBufferWithBytes:w.data() length:w.size() * sizeof(uint8_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> by_m = [g_ctx.dev newBufferWithLength:(size_t)T * G * N * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> by_r = [g_ctx.dev newBufferWithLength:(size_t)T * G * N * sizeof(float) options:MTLResourceStorageModeShared];
        GgufGroupedGemvParams gp{T, G, N, K, g_stride};
        id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p_m];
        [enc setBuffer:bx offset:0 atIndex:0];
        [enc setBuffer:bw offset:0 atIndex:1];
        [enc setBuffer:by_m offset:0 atIndex:2];
        [enc setBytes:&gp length:sizeof(gp) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((N + 1u) / 2u, G, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc setComputePipelineState:g_ctx.pso("gguf_grouped_gemv_q8_0_fast")];
        [enc setBuffer:bx offset:0 atIndex:0];
        [enc setBuffer:bw offset:0 atIndex:1];
        [enc setBuffer:by_r offset:0 atIndex:2];
        [enc setBytes:&gp length:sizeof(gp) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((N + 3u) / 4u, T * G, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const float* ym = static_cast<const float*>(by_m.contents);
        const float* yr = static_cast<const float*>(by_r.contents);
        for (size_t i = 0; i < (size_t)T * G * N; ++i) EXPECT_TRUE(std::abs(ym[i] - yr[i]) < 1e-3f);
    }
    {
        const uint32_t M = 5, N = 37, K = 512;
        const uint32_t nb = K / 32;
        std::vector<float> x((size_t)M * K);
        std::vector<uint8_t> w((size_t)N * nb * 34, 0);
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos((float)(i + 7) * 0.029f);
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t b = 0; b < nb; ++b) {
                uint8_t* blk = w.data() + ((size_t)n * nb + b) * 34;
                uint16_t d = f32_to_f16_bits(0.03f);
                memcpy(blk, &d, 2);
                for (int i = 0; i < 32; ++i) {
                    ((int8_t*)(blk + 2))[i] = (int8_t)((int)((n * 11 + b * 7 + i * 5) % 63) - 31);
                }
            }
        }
        id<MTLComputePipelineState> p_2r = g_ctx.pso("gguf_gemv_q8_0_multi_2r");
        EXPECT_TRUE(p_2r != nil);
        if (p_2r == nil) return;
        std::vector<float> y_ref;
        dispatch_gguf_gemv("gguf_gemv_q8_0", x, w, M, N, K, y_ref);
        id<MTLBuffer> bx = [g_ctx.dev newBufferWithBytes:x.data() length:x.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bw = [g_ctx.dev newBufferWithBytes:w.data() length:w.size() * sizeof(uint8_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [g_ctx.dev newBufferWithLength:(size_t)M * N * sizeof(float) options:MTLResourceStorageModeShared];
        GgufGemvParams p{M, N, K};
        id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p_2r];
        [enc setBuffer:bx offset:0 atIndex:0];
        [enc setBuffer:bw offset:0 atIndex:1];
        [enc setBuffer:by offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((N + 3u) / 4u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const float* y2r = static_cast<const float*>(by.contents);
        for (size_t i = 0; i < y_ref.size(); ++i) EXPECT_TRUE(std::abs(y2r[i] - y_ref[i]) < 1e-3f);
    }
    {
        struct DsMarkovBiasParams { uint32_t V; uint32_t R; };
        const uint32_t V = 277, R = 256;
        auto f32_to_bf16 = [](float f) -> uint16_t { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); };
        std::vector<float> base_logits(V);
        std::vector<uint16_t> w1((size_t)V * R), w2((size_t)V * R);
        for (uint32_t v = 0; v < V; ++v) base_logits[v] = std::sin((float)(v + 1) * 0.13f);
        for (size_t i = 0; i < w1.size(); ++i) {
            w1[i] = f32_to_bf16(std::sin((float)(i + 3) * 0.017f) * 0.15f);
            w2[i] = f32_to_bf16(std::cos((float)(i + 5) * 0.019f) * 0.15f);
        }
        int32_t prev_tok = 41;
        id<MTLComputePipelineState> p1 = g_ctx.pso("ds_markov_bias_argmax_bf16_p1");
        id<MTLComputePipelineState> p2 = g_ctx.pso("ds_markov_bias_argmax_p2");
        EXPECT_TRUE(p1 != nil);
        EXPECT_TRUE(p2 != nil);
        if (p1 == nil || p2 == nil) return;
        uint32_t num_tg = (V + 15u) / 16u;
        id<MTLBuffer> bbase = [g_ctx.dev newBufferWithBytes:base_logits.data() length:V * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bw1 = [g_ctx.dev newBufferWithBytes:w1.data() length:w1.size() * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bw2 = [g_ctx.dev newBufferWithBytes:w2.data() length:w2.size() * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bprev = [g_ctx.dev newBufferWithBytes:&prev_tok length:sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bpval = [g_ctx.dev newBufferWithLength:num_tg * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bpidx = [g_ctx.dev newBufferWithLength:num_tg * sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bout = [g_ctx.dev newBufferWithLength:sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bref_logits = [g_ctx.dev newBufferWithLength:V * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bref_tok = [g_ctx.dev newBufferWithLength:sizeof(int32_t) options:MTLResourceStorageModeShared];
        DsMarkovBiasParams mp{V, R};
        id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p1];
        [enc setBuffer:bbase offset:0 atIndex:0];
        [enc setBuffer:bw1 offset:0 atIndex:1];
        [enc setBuffer:bw2 offset:0 atIndex:2];
        [enc setBuffer:bprev offset:0 atIndex:3];
        [enc setBuffer:bpval offset:0 atIndex:4];
        [enc setBuffer:bpidx offset:0 atIndex:5];
        [enc setBytes:&mp length:sizeof(mp) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc setComputePipelineState:p2];
        [enc setBuffer:bpval offset:0 atIndex:0];
        [enc setBuffer:bpidx offset:0 atIndex:1];
        [enc setBuffer:bout offset:0 atIndex:2];
        [enc setBytes:&num_tg length:sizeof(num_tg) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc setComputePipelineState:g_ctx.pso("ds_markov_bias_add_bf16")];
        [enc setBuffer:bbase offset:0 atIndex:0];
        [enc setBuffer:bw1 offset:0 atIndex:1];
        [enc setBuffer:bw2 offset:0 atIndex:2];
        [enc setBuffer:bprev offset:0 atIndex:3];
        [enc setBuffer:bref_logits offset:0 atIndex:4];
        [enc setBytes:&mp length:sizeof(mp) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((V + 3u) / 4u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc setComputePipelineState:g_ctx.pso("ds_argmax_f32")];
        [enc setBuffer:bref_logits offset:0 atIndex:0];
        [enc setBuffer:bref_tok offset:0 atIndex:1];
        [enc setBytes:&V length:sizeof(V) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        int32_t got = *static_cast<const int32_t*>(bout.contents);
        int32_t ref = *static_cast<const int32_t*>(bref_tok.contents);
        EXPECT_TRUE(got == ref);

        std::vector<float> tie_base(V, 0.0f);
        tie_base[19] = 5.0f;
        tie_base[145] = 5.0f;
        std::vector<uint16_t> zero_w((size_t)V * R, 0);
        memcpy(bbase.contents, tie_base.data(), V * sizeof(float));
        memcpy(bw1.contents, zero_w.data(), zero_w.size() * sizeof(uint16_t));
        memcpy(bw2.contents, zero_w.data(), zero_w.size() * sizeof(uint16_t));
        id<MTLCommandBuffer> cb2 = [g_ctx.q commandBuffer];
        id<MTLComputeCommandEncoder> enc2 = [cb2 computeCommandEncoder];
        [enc2 setComputePipelineState:p1];
        [enc2 setBuffer:bbase offset:0 atIndex:0];
        [enc2 setBuffer:bw1 offset:0 atIndex:1];
        [enc2 setBuffer:bw2 offset:0 atIndex:2];
        [enc2 setBuffer:bprev offset:0 atIndex:3];
        [enc2 setBuffer:bpval offset:0 atIndex:4];
        [enc2 setBuffer:bpidx offset:0 atIndex:5];
        [enc2 setBytes:&mp length:sizeof(mp) atIndex:6];
        [enc2 dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc2 setComputePipelineState:p2];
        [enc2 setBuffer:bpval offset:0 atIndex:0];
        [enc2 setBuffer:bpidx offset:0 atIndex:1];
        [enc2 setBuffer:bout offset:0 atIndex:2];
        [enc2 setBytes:&num_tg length:sizeof(num_tg) atIndex:3];
        [enc2 dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc2 endEncoding];
        [cb2 commit];
        [cb2 waitUntilCompleted];
        EXPECT_TRUE(*static_cast<const int32_t*>(bout.contents) == 19);
    }
}

static void test_ds_q6_k_multi_fast_and_shexp_swiglu_parity() {
    struct GgufGemvSwigluParams { uint32_t M; uint32_t N; uint32_t K; float clamp_val; };
    const uint32_t N = 37, K = 512;
    const uint32_t nb = K / 256;
    std::vector<uint8_t> wg((size_t)N * nb * 210, 0);
    std::vector<uint8_t> wu((size_t)N * nb * 210, 0);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < nb; ++b) {
            uint8_t* bg = wg.data() + ((size_t)n * nb + b) * 210;
            uint8_t* bu = wu.data() + ((size_t)n * nb + b) * 210;
            for (int i = 0; i < 128; ++i) {
                bg[i] = (uint8_t)((n * 17 + b * 23 + i * 7) & 0xFFu);
                bu[i] = (uint8_t)((n * 29 + b * 11 + i * 13) & 0xFFu);
            }
            for (int i = 0; i < 64; ++i) {
                bg[128 + i] = (uint8_t)((n * 13 + b * 5 + i * 11) & 0xFFu);
                bu[128 + i] = (uint8_t)((n * 19 + b * 7 + i * 3) & 0xFFu);
            }
            for (int i = 0; i < 16; ++i) {
                ((int8_t*)(bg + 192))[i] = (int8_t)((int)((n + b + i) % 15) - 7);
                ((int8_t*)(bu + 192))[i] = (int8_t)((int)((n * 3 + b + i) % 15) - 7);
            }
            uint16_t dg = f32_to_f16_bits(0.05f);
            uint16_t du = f32_to_f16_bits(0.04f);
            memcpy(bg + 208, &dg, 2);
            memcpy(bu + 208, &du, 2);
        }
    }
    {
        const uint32_t M = 5;
        std::vector<float> x((size_t)M * K);
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin((float)(i + 1) * 0.031f);
        id<MTLComputePipelineState> p_mf = g_ctx.pso("gguf_gemv_q6_k_multi_fast");
        EXPECT_TRUE(p_mf != nil);
        if (p_mf == nil) return;
        std::vector<float> y_ref;
        dispatch_gguf_gemv("gguf_gemv_q6_k", x, wg, M, N, K, y_ref);
        id<MTLBuffer> bx = [g_ctx.dev newBufferWithBytes:x.data() length:x.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bw = [g_ctx.dev newBufferWithBytes:wg.data() length:wg.size() * sizeof(uint8_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [g_ctx.dev newBufferWithLength:(size_t)M * N * sizeof(float) options:MTLResourceStorageModeShared];
        GgufGemvParams p{M, N, K};
        id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p_mf];
        [enc setBuffer:bx offset:0 atIndex:0];
        [enc setBuffer:bw offset:0 atIndex:1];
        [enc setBuffer:by offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((N + 3u) / 4u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const float* ymf = static_cast<const float*>(by.contents);
        for (size_t i = 0; i < y_ref.size(); ++i) EXPECT_TRUE(std::abs(ymf[i] - y_ref[i]) < 1e-2f);
    }
    for (int tc = 0; tc < 2; ++tc) {
        const uint32_t M = (tc == 0) ? 1u : 5u;
        const float clamp_val = (tc == 0) ? 0.0f : 7.0f;
        const float scale_x = (tc == 0) ? 0.5f : 2.5f;
        std::vector<float> x((size_t)M * K);
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos((float)(i + 3) * 0.027f) * scale_x;
        id<MTLComputePipelineState> p_sw = g_ctx.pso("gguf_gemv_gate_up_swiglu_q6_k");
        EXPECT_TRUE(p_sw != nil);
        if (p_sw == nil) return;
        std::vector<float> yg_ref, yu_ref;
        dispatch_gguf_gemv("gguf_gemv_q6_k", x, wg, M, N, K, yg_ref);
        dispatch_gguf_gemv("gguf_gemv_q6_k", x, wu, M, N, K, yu_ref);
        id<MTLBuffer> bx = [g_ctx.dev newBufferWithBytes:x.data() length:x.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bwg = [g_ctx.dev newBufferWithBytes:wg.data() length:wg.size() * sizeof(uint8_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bwu = [g_ctx.dev newBufferWithBytes:wu.data() length:wu.size() * sizeof(uint8_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [g_ctx.dev newBufferWithLength:(size_t)M * N * sizeof(float) options:MTLResourceStorageModeShared];
        GgufGemvSwigluParams sp{M, N, K, clamp_val};
        id<MTLCommandBuffer> cb = [g_ctx.q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p_sw];
        [enc setBuffer:bx offset:0 atIndex:0];
        [enc setBuffer:bwg offset:0 atIndex:1];
        [enc setBuffer:bwu offset:0 atIndex:2];
        [enc setBuffer:by offset:0 atIndex:3];
        [enc setBytes:&sp length:sizeof(sp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((N + 1u) / 2u, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const float* ysw = static_cast<const float*>(by.contents);
        for (size_t i = 0; i < (size_t)M * N; ++i) {
            float gv = yg_ref[i];
            float uv = yu_ref[i];
            if (clamp_val > 0.0f) {
                gv = std::clamp(gv, -clamp_val, clamp_val);
                uv = std::clamp(uv, -clamp_val, clamp_val);
            }
            float ref = (gv / (1.0f + std::exp(-gv))) * uv;
            EXPECT_TRUE(std::abs(ysw[i] - ref) <= 2e-2f + 1e-3f * std::abs(ref));
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
        run_test("test_gguf_gemv_bf16", test_gguf_gemv_bf16);
        run_test("test_gguf_gemv_q5_k", test_gguf_gemv_q5_k);
        run_test("test_gguf_gemv_iq2_xxs", test_gguf_gemv_iq2_xxs);
        run_test("test_gguf_gemv_iq1_s", test_gguf_gemv_iq1_s);
        run_test("test_gguf_moe_topk6_accumulate", test_gguf_moe_topk6_accumulate);
        run_test("test_gguf_moe_indexed_iq1_s", test_gguf_moe_indexed_iq1_s);
        run_test("test_ds_mhc_sinkhorn_doubly_stochastic", test_ds_mhc_sinkhorn_doubly_stochastic);
        run_test("test_ds_router_topk6_sigmoid_bias", test_ds_router_topk6_sigmoid_bias);
        run_test("test_ds_csa_hca_causal_attention", test_ds_csa_hca_causal_attention);
        run_test("test_gguf_gemm_tiled_and_smallk_parity", test_gguf_gemm_tiled_and_smallk_parity);
        run_test("test_ds_dspark_and_multi_gemv_parity", test_ds_dspark_and_multi_gemv_parity);
        run_test("test_ds_grouped_moe_and_mxfp4_fused_parity", test_ds_grouped_moe_and_mxfp4_fused_parity);
        run_test("test_ds_attn_router_fast_kernels_parity", test_ds_attn_router_fast_kernels_parity);
        run_test("test_ds_dspark_conf_head_parity", test_ds_dspark_conf_head_parity);
        run_test("test_ds_dspark_draft_fast_kernels_parity", test_ds_dspark_draft_fast_kernels_parity);
        run_test("test_ds_q6_k_multi_fast_and_shexp_swiglu_parity", test_ds_q6_k_multi_fast_and_shexp_swiglu_parity);
        printf("GGUF Metal kernel tests: %d passed, %d failed\n", g_passed, g_failed);
        return g_failed == 0 ? 0 : 1;
    }
}
