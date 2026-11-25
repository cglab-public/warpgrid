#include "../src/cuda/kernels_gguf.cuh"
#include "../src/gguf_iq_tables.hpp"
#include <cuda_runtime.h>
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

static void test_cuda_gguf_gemv_q8_0() {
    const uint32_t M = 2, N = 8, K = 64;
    const size_t row_bytes = (K / 32) * 34;
    std::vector<float> X((size_t)M * K);
    std::vector<uint8_t> W((size_t)N * row_bytes);
    for (size_t i = 0; i < X.size(); ++i) X[i] = std::sin((float)(i + 1) * 0.13f);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < K / 32; ++b) {
            uint8_t* blk = W.data() + n * row_bytes + b * 34;
            uint16_t d16 = f32_to_f16_bits(0.05f * (float)(n + 1));
            memcpy(blk, &d16, 2);
            for (int i = 0; i < 32; ++i) blk[2 + i] = (uint8_t)(int8_t)(((n * 7 + b * 13 + i * 5) % 21) - 10);
        }
    }
    float* dx = nullptr;
    uint8_t* dw = nullptr;
    float* dy = nullptr;
    EXPECT_TRUE(cudaMalloc(&dx, X.size() * sizeof(float)) == cudaSuccess);
    EXPECT_TRUE(cudaMalloc(&dw, W.size()) == cudaSuccess);
    EXPECT_TRUE(cudaMalloc(&dy, (size_t)M * N * sizeof(float)) == cudaSuccess);
    cudaMemcpy(dx, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dw, W.data(), W.size(), cudaMemcpyHostToDevice);

    ink::cuda_gguf::launch_gemv(ink::GgmlType::Q8_0, dx, dw, dy, M, N, K, 0);
    EXPECT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<float> Y((size_t)M * N);
    cudaMemcpy(Y.data(), dy, Y.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dy);

    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float ref = 0.0f;
            for (uint32_t b = 0; b < K / 32; ++b) {
                const uint8_t* blk = W.data() + n * row_bytes + b * 34;
                uint16_t d16 = 0;
                memcpy(&d16, blk, 2);
                float d = f16_bits_to_f32(d16);
                for (int i = 0; i < 32; ++i) {
                    ref += X[m * K + b * 32 + i] * d * (float)(int8_t)blk[2 + i];
                }
            }
            EXPECT_TRUE(std::abs(Y[m * N + n] - ref) < 1e-3f);
        }
    }
}

static void test_cuda_gguf_gemv_iq1_s_and_iq2_xxs() {
    const uint32_t M = 2, N = 6, K = 256;
    std::vector<float> X((size_t)M * K);
    for (size_t i = 0; i < X.size(); ++i) X[i] = std::sin((float)(i + 3) * 0.07f);

    std::vector<uint8_t> W1((size_t)N * 50);
    std::vector<uint8_t> W2((size_t)N * 66);
    for (uint32_t n = 0; n < N; ++n) {
        uint16_t d16 = f32_to_f16_bits(0.1f * (float)(n + 1));
        memcpy(W1.data() + n * 50, &d16, 2);
        memcpy(W2.data() + n * 66, &d16, 2);
        for (size_t i = 2; i < 50; ++i) W1[n * 50 + i] = (uint8_t)((n * 53 + i * 29 + 11) & 0xFF);
        for (size_t i = 2; i < 66; ++i) W2[n * 66 + i] = (uint8_t)((n * 41 + i * 19 + 7) & 0xFF);
    }

    float* dx = nullptr;
    uint8_t* dw1 = nullptr;
    uint8_t* dw2 = nullptr;
    float* dy1 = nullptr;
    float* dy2 = nullptr;
    cudaMalloc(&dx, X.size() * sizeof(float));
    cudaMalloc(&dw1, W1.size());
    cudaMalloc(&dw2, W2.size());
    cudaMalloc(&dy1, (size_t)M * N * sizeof(float));
    cudaMalloc(&dy2, (size_t)M * N * sizeof(float));
    cudaMemcpy(dx, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dw1, W1.data(), W1.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dw2, W2.data(), W2.size(), cudaMemcpyHostToDevice);

    ink::cuda_gguf::launch_gemv(ink::GgmlType::IQ1_S, dx, dw1, dy1, M, N, K, 0);
    ink::cuda_gguf::launch_gemv(ink::GgmlType::IQ2_XXS, dx, dw2, dy2, M, N, K, 0);
    EXPECT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<float> Y1((size_t)M * N), Y2((size_t)M * N);
    cudaMemcpy(Y1.data(), dy1, Y1.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Y2.data(), dy2, Y2.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(dw1);
    cudaFree(dw2);
    cudaFree(dy1);
    cudaFree(dy2);

    std::vector<float> r1(K), r2(K);
    for (uint32_t n = 0; n < N; ++n) {
        dequant_iq1_s_block_ref(W1.data() + n * 50, r1.data());
        dequant_iq2_xxs_block_ref(W2.data() + n * 66, r2.data());
        for (uint32_t m = 0; m < M; ++m) {
            float ref1 = 0.0f, ref2 = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                ref1 += X[m * K + k] * r1[k];
                ref2 += X[m * K + k] * r2[k];
            }
            EXPECT_TRUE(std::abs(Y1[m * N + n] - ref1) < 1e-3f);
            EXPECT_TRUE(std::abs(Y2[m * N + n] - ref2) < 1e-3f);
        }
    }
}

static void test_cuda_moe_indexed_and_down_accum_iq1_s() {
    const uint32_t T = 2, E = 4, N = 6, K = 256, TOPK = 2;
    const uint32_t stride = N * 50;
    std::vector<float> X((size_t)T * K);
    std::vector<uint8_t> W((size_t)E * stride);
    std::vector<int32_t> topk_ids = {2, 0, 3, 1};
    std::vector<float> topk_w = {0.7f, 0.3f, 0.4f, 0.6f};
    std::vector<float> base_y((size_t)T * N, 1.25f);
    for (size_t i = 0; i < X.size(); ++i) X[i] = std::sin((float)(i + 1) * 0.05f);
    for (uint32_t e = 0; e < E; ++e) {
        for (uint32_t n = 0; n < N; ++n) {
            uint8_t* b = W.data() + (size_t)e * stride + (size_t)n * 50;
            uint16_t d16 = f32_to_f16_bits(0.08f * (float)(e + 1));
            memcpy(b, &d16, 2);
            for (size_t i = 2; i < 50; ++i) b[i] = (uint8_t)((e * 67 + n * 19 + i * 13 + 5) & 0xFF);
        }
    }
    std::vector<float> X_per((size_t)T * TOPK * K);
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t r = 0; r < TOPK; ++r) {
            memcpy(&X_per[(t * TOPK + r) * K], &X[t * K], K * sizeof(float));
        }
    }

    float *dx = nullptr, *dxp = nullptr, *dtw = nullptr, *dy_idx = nullptr, *dy_acc = nullptr;
    uint8_t* dw = nullptr;
    int32_t* dids = nullptr;
    cudaMalloc(&dx, X.size() * sizeof(float));
    cudaMalloc(&dxp, X_per.size() * sizeof(float));
    cudaMalloc(&dw, W.size());
    cudaMalloc(&dids, topk_ids.size() * sizeof(int32_t));
    cudaMalloc(&dtw, topk_w.size() * sizeof(float));
    cudaMalloc(&dy_idx, (size_t)T * TOPK * N * sizeof(float));
    cudaMalloc(&dy_acc, base_y.size() * sizeof(float));
    cudaMemcpy(dx, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dxp, X_per.data(), X_per.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dw, W.data(), W.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dids, topk_ids.data(), topk_ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dtw, topk_w.data(), topk_w.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy_acc, base_y.data(), base_y.size() * sizeof(float), cudaMemcpyHostToDevice);

    ink::cuda_gguf::launch_moe_indexed(ink::GgmlType::IQ1_S, dx, dw, dids, dy_idx, T, N, K, TOPK, stride, 0);
    ink::cuda_gguf::launch_moe_down_accum(ink::GgmlType::IQ1_S, dxp, dw, dids, dtw, dy_acc, T, N, K, TOPK, stride, 0);
    EXPECT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<float> Y_idx((size_t)T * TOPK * N), Y_acc((size_t)T * N);
    cudaMemcpy(Y_idx.data(), dy_idx, Y_idx.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Y_acc.data(), dy_acc, Y_acc.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(dxp);
    cudaFree(dw);
    cudaFree(dids);
    cudaFree(dtw);
    cudaFree(dy_idx);
    cudaFree(dy_acc);

    std::vector<float> row(K);
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t n = 0; n < N; ++n) {
            float ref_acc = 1.25f;
            for (uint32_t r = 0; r < TOPK; ++r) {
                int e = topk_ids[t * TOPK + r];
                dequant_iq1_s_block_ref(W.data() + (size_t)e * stride + (size_t)n * 50, row.data());
                float dot = 0.0f;
                for (uint32_t k = 0; k < K; ++k) dot += X[t * K + k] * row[k];
                EXPECT_TRUE(std::abs(Y_idx[(t * TOPK + r) * N + n] - dot) < 1e-3f);
                ref_acc += topk_w[t * TOPK + r] * dot;
            }
            EXPECT_TRUE(std::abs(Y_acc[t * N + n] - ref_acc) < 1e-3f);
        }
    }
}

int main() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        fprintf(stderr, "No CUDA device available\n");
        return 1;
    }
    printf("GGUF CUDA kernel tests on %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    run_test("test_cuda_gguf_gemv_q8_0", test_cuda_gguf_gemv_q8_0);
    run_test("test_cuda_gguf_gemv_iq1_s_and_iq2_xxs", test_cuda_gguf_gemv_iq1_s_and_iq2_xxs);
    run_test("test_cuda_moe_indexed_and_down_accum_iq1_s", test_cuda_moe_indexed_and_down_accum_iq1_s);
    printf("GGUF CUDA kernel tests: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
