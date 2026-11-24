#include "../src/dspark.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <numeric>
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

#define CHECK_CUDA_TEST(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA FAIL %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
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

static void test_dflash2_concat_taps() {
    const int T = 4;
    const int D = 64;
    const int stride = 8 * D;
    std::vector<__nv_bfloat16> h_aux(4 * stride);
    for (size_t i = 0; i < h_aux.size(); ++i) {
        h_aux[i] = __float2bfloat16(sinf((float)i * 0.013f));
    }

    __nv_bfloat16 *d_aux = nullptr, *d_out = nullptr;
    CHECK_CUDA_TEST(cudaMalloc(&d_aux, h_aux.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA_TEST(cudaMalloc(&d_out, (size_t)T * 4 * D * sizeof(__nv_bfloat16)));
    CHECK_CUDA_TEST(cudaMemcpy(d_aux, h_aux.data(), h_aux.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    ink::launch_dflash2_concat_taps(d_aux, stride, d_out, T, D, 0);
    CHECK_CUDA_TEST(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_out((size_t)T * 4 * D);
    CHECK_CUDA_TEST(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    cudaFree(d_aux);
    cudaFree(d_out);

    for (int t = 0; t < T; ++t) {
        for (int tap = 0; tap < 4; ++tap) {
            for (int d = 0; d < D; ++d) {
                float got = __bfloat162float(h_out[(size_t)t * 4 * D + tap * D + d]);
                float ref = __bfloat162float(h_aux[(size_t)tap * stride + t * D + d]);
                EXPECT_TRUE(fabsf(got - ref) < 1e-6f);
            }
        }
    }
}

static void test_dflash2_grouped_dyn_conv() {
    const int T = 5;
    const int D = 64;
    const int G = 16;
    const int nG = D / G;
    std::vector<__nv_bfloat16> h_x(T * D), h_base(4 * D), h_dyn(T * 4 * nG);
    for (int i = 0; i < T * D; ++i) h_x[i] = __float2bfloat16(cosf(i * 0.11f));
    for (int i = 0; i < 4 * D; ++i) h_base[i] = __float2bfloat16(0.25f * sinf(i * 0.07f));
    for (int i = 0; i < T * 4 * nG; ++i) h_dyn[i] = __float2bfloat16(0.15f * cosf(i * 0.19f));

    __nv_bfloat16 *d_x = nullptr, *d_base = nullptr, *d_dyn = nullptr, *d_out = nullptr;
    CHECK_CUDA_TEST(cudaMalloc(&d_x, h_x.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA_TEST(cudaMalloc(&d_base, h_base.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA_TEST(cudaMalloc(&d_dyn, h_dyn.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA_TEST(cudaMalloc(&d_out, h_x.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA_TEST(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA_TEST(cudaMemcpy(d_base, h_base.data(), h_base.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA_TEST(cudaMemcpy(d_dyn, h_dyn.data(), h_dyn.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    for (int half_idx = 0; half_idx < 2; ++half_idx) {
        ink::launch_dflash2_grouped_dyn_conv(d_x, d_base + half_idx * 2 * D, d_dyn, d_out, T, D, G, half_idx, 0);
        CHECK_CUDA_TEST(cudaDeviceSynchronize());
        std::vector<__nv_bfloat16> h_out(T * D);
        CHECK_CUDA_TEST(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < D; ++d) {
                int g = d / G;
                int db = t * 4 * nG + half_idx * 2 * nG + g;
                float w0 = __bfloat162float(h_base[half_idx * 2 * D + d]) + __bfloat162float(h_dyn[db]);
                float w1 = __bfloat162float(h_base[half_idx * 2 * D + D + d]) + __bfloat162float(h_dyn[db + nG]);
                float x0 = __bfloat162float(h_x[t * D + d]);
                float x1 = (t > 0) ? __bfloat162float(h_x[(t - 1) * D + d]) : 0.0f;
                float ref = __bfloat162float(__float2bfloat16(w0 * x0 + w1 * x1));
                float got = __bfloat162float(h_out[t * D + d]);
                EXPECT_TRUE(fabsf(got - ref) <= 2e-2f);
            }
        }
    }
    cudaFree(d_x);
    cudaFree(d_base);
    cudaFree(d_dyn);
    cudaFree(d_out);
}

static void test_dflash2_block_attn() {
    const int B = 4;
    const int n_heads = 8;
    const int n_kv = 2;
    const int hd = 128;
    const int cap = 32;
    const int win = 16;
    const int ap = 25;
    const int kv_dim = n_kv * hd;

    std::vector<__nv_bfloat16> h_q(B * n_heads * hd);
    std::vector<__nv_bfloat16> h_kb(B * kv_dim), h_vb(B * kv_dim);
    std::vector<__nv_bfloat16> h_kc(cap * kv_dim), h_vc(cap * kv_dim);
    for (size_t i = 0; i < h_q.size(); ++i) h_q[i] = __float2bfloat16(0.3f * sinf(i * 0.03f));
    for (size_t i = 0; i < h_kb.size(); ++i) {
        h_kb[i] = __float2bfloat16(0.3f * cosf(i * 0.05f));
        h_vb[i] = __float2bfloat16(0.5f * sinf(i * 0.07f));
    }
    for (size_t i = 0; i < h_kc.size(); ++i) {
        h_kc[i] = __float2bfloat16(0.3f * sinf(i * 0.02f + 1.0f));
        h_vc[i] = __float2bfloat16(0.5f * cosf(i * 0.04f + 0.5f));
    }

    __nv_bfloat16 *d_q = nullptr, *d_kb = nullptr, *d_vb = nullptr, *d_kc = nullptr, *d_vc = nullptr, *d_out = nullptr;
    int32_t* d_ap = nullptr;
    CHECK_CUDA_TEST(cudaMalloc(&d_q, h_q.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA_TEST(cudaMalloc(&d_kb, h_kb.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA_TEST(cudaMalloc(&d_vb, h_vb.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA_TEST(cudaMalloc(&d_kc, h_kc.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA_TEST(cudaMalloc(&d_vc, h_vc.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA_TEST(cudaMalloc(&d_out, h_q.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA_TEST(cudaMalloc(&d_ap, sizeof(int32_t)));
    CHECK_CUDA_TEST(cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA_TEST(cudaMemcpy(d_kb, h_kb.data(), h_kb.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA_TEST(cudaMemcpy(d_vb, h_vb.data(), h_vb.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA_TEST(cudaMemcpy(d_kc, h_kc.data(), h_kc.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA_TEST(cudaMemcpy(d_vc, h_vc.data(), h_vc.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA_TEST(cudaMemcpy(d_ap, &ap, sizeof(int32_t), cudaMemcpyHostToDevice));

    ink::launch_dflash2_block_attn(d_q, d_kb, d_vb, d_kc, d_vc, d_out, 0, d_ap, B, cap, win, n_heads, n_kv, hd, 0);
    CHECK_CUDA_TEST(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> h_out(h_q.size());
    CHECK_CUDA_TEST(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    cudaFree(d_q); cudaFree(d_kb); cudaFree(d_vb); cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_out); cudaFree(d_ap);

    float scale = 1.0f / sqrtf((float)hd);
    float max_err = 0.0f;
    for (int t = 0; t < B; ++t) {
        int q_pos = ap + t;
        int c_start = std::max(0, q_pos + 1 - win);
        int c_end = ap;
        for (int h = 0; h < n_heads; ++h) {
            int kvh = h / (n_heads / n_kv);
            std::vector<float> scores;
            std::vector<const __nv_bfloat16*> vptrs;
            const __nv_bfloat16* qp = &h_q[(t * n_heads + h) * hd];
            for (int kp = c_start; kp < c_end; ++kp) {
                int sl = kp % cap;
                const __nv_bfloat16* kptr = &h_kc[sl * kv_dim + kvh * hd];
                float dot = 0.0f;
                for (int d = 0; d < hd; ++d) dot += __bfloat162float(qp[d]) * __bfloat162float(kptr[d]);
                scores.push_back(dot * scale);
                vptrs.push_back(&h_vc[sl * kv_dim + kvh * hd]);
            }
            for (int j = 0; j < B; ++j) {
                const __nv_bfloat16* kptr = &h_kb[j * kv_dim + kvh * hd];
                float dot = 0.0f;
                for (int d = 0; d < hd; ++d) dot += __bfloat162float(qp[d]) * __bfloat162float(kptr[d]);
                scores.push_back(dot * scale);
                vptrs.push_back(&h_vb[j * kv_dim + kvh * hd]);
            }
            float m = *std::max_element(scores.begin(), scores.end());
            float denom = 0.0f;
            for (float& s : scores) { s = expf(s - m); denom += s; }
            for (int d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (size_t idx = 0; idx < scores.size(); ++idx) {
                    acc += (scores[idx] / denom) * __bfloat162float(vptrs[idx][d]);
                }
                float got = __bfloat162float(h_out[(t * n_heads + h) * hd + d]);
                max_err = std::max(max_err, fabsf(got - acc));
            }
        }
    }
    EXPECT_TRUE(max_err < 1.5e-2f);
}

static void test_dflash2_topk16_and_select_path() {
    const int M = 3;
    const int V = 2048;
    const int rank = 256;
    const int vocab_cb = 4096;
    std::vector<float> h_logits(M * V);
    std::vector<int32_t> h_map(V);
    for (int i = 0; i < V; ++i) h_map[i] = i * 2 + 1;
    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < V; ++i) {
            h_logits[m * V + i] = -5.0f + 0.001f * (float)((i * 37 + m * 101) % V);
        }
        h_logits[m * V + 17 + m] = 10.0f;
        h_logits[m * V + 500 + m] = 9.5f;
    }

    float *d_logits = nullptr, *d_topv = nullptr;
    int32_t *d_topi = nullptr, *d_map = nullptr;
    CHECK_CUDA_TEST(cudaMalloc(&d_logits, h_logits.size() * sizeof(float)));
    CHECK_CUDA_TEST(cudaMalloc(&d_topv, M * 16 * sizeof(float)));
    CHECK_CUDA_TEST(cudaMalloc(&d_topi, M * 16 * sizeof(int32_t)));
    CHECK_CUDA_TEST(cudaMalloc(&d_map, V * sizeof(int32_t)));
    CHECK_CUDA_TEST(cudaMemcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_TEST(cudaMemcpy(d_map, h_map.data(), V * sizeof(int32_t), cudaMemcpyHostToDevice));

    ink::launch_dflash2_topk16(d_logits, d_topv, d_topi, d_map, M, V, 0);
    CHECK_CUDA_TEST(cudaDeviceSynchronize());

    std::vector<float> h_topv(M * 16);
    std::vector<int32_t> h_topi(M * 16);
    CHECK_CUDA_TEST(cudaMemcpy(h_topv.data(), d_topv, h_topv.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_TEST(cudaMemcpy(h_topi.data(), d_topi, h_topi.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));

    for (int m = 0; m < M; ++m) {
        EXPECT_TRUE(h_topi[m * 16 + 0] == h_map[17 + m]);
        EXPECT_TRUE(h_topi[m * 16 + 1] == h_map[500 + m]);
        for (int k = 1; k < 16; ++k) {
            EXPECT_TRUE(h_topv[m * 16 + k - 1] >= h_topv[m * 16 + k]);
        }
    }

    std::vector<__nv_bfloat16> h_sel(M * rank);
    std::vector<__nv_bfloat16> h_pred_cb(vocab_cb * rank), h_succ_cb(vocab_cb * rank);
    for (size_t i = 0; i < h_sel.size(); ++i) h_sel[i] = __float2bfloat16(0.2f * sinf(i * 0.05f));
    for (size_t i = 0; i < h_pred_cb.size(); ++i) {
        h_pred_cb[i] = __float2bfloat16(0.2f * cosf(i * 0.01f));
        h_succ_cb[i] = __float2bfloat16(0.2f * sinf(i * 0.02f));
    }
    int32_t anchor_tok = 7;

    __nv_bfloat16 *d_hsel = nullptr, *d_pcb = nullptr, *d_scb = nullptr;
    int32_t *d_anchor = nullptr, *d_out_tok = nullptr;
    CHECK_CUDA_TEST(cudaMalloc(&d_hsel, h_sel.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA_TEST(cudaMalloc(&d_pcb, h_pred_cb.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA_TEST(cudaMalloc(&d_scb, h_succ_cb.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA_TEST(cudaMalloc(&d_anchor, sizeof(int32_t)));
    CHECK_CUDA_TEST(cudaMalloc(&d_out_tok, M * sizeof(int32_t)));
    CHECK_CUDA_TEST(cudaMemcpy(d_hsel, h_sel.data(), h_sel.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA_TEST(cudaMemcpy(d_pcb, h_pred_cb.data(), h_pred_cb.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA_TEST(cudaMemcpy(d_scb, h_succ_cb.data(), h_succ_cb.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA_TEST(cudaMemcpy(d_anchor, &anchor_tok, sizeof(int32_t), cudaMemcpyHostToDevice));

    ink::launch_dflash2_select_path(d_anchor, d_topv, d_topi, d_hsel, d_pcb, d_scb, d_out_tok, M, rank, 0);
    CHECK_CUDA_TEST(cudaDeviceSynchronize());

    std::vector<int32_t> h_out_tok(M);
    CHECK_CUDA_TEST(cudaMemcpy(h_out_tok.data(), d_out_tok, M * sizeof(int32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_logits); cudaFree(d_topv); cudaFree(d_topi); cudaFree(d_map);
    cudaFree(d_hsel); cudaFree(d_pcb); cudaFree(d_scb); cudaFree(d_anchor); cudaFree(d_out_tok);

    int32_t pred = anchor_tok;
    for (int pos = 0; pos < M; ++pos) {
        float best_s = -1e30f;
        int32_t best_tok = h_topi[pos * 16];
        for (int k = 0; k < 16; ++k) {
            int32_t cid = h_topi[pos * 16 + k];
            float corr = 0.0f;
            for (int r = 0; r < rank; ++r) {
                corr += __bfloat162float(h_pred_cb[pred * rank + r])
                      * __bfloat162float(h_sel[pos * rank + r])
                      * __bfloat162float(h_succ_cb[cid * rank + r]);
            }
            float sc = h_topv[pos * 16 + k] + corr;
            if (sc > best_s) {
                best_s = sc;
                best_tok = cid;
            }
        }
        EXPECT_TRUE(h_out_tok[pos] == best_tok);
        pred = best_tok;
    }
}

int main() {
    run_test("test_dflash2_concat_taps", test_dflash2_concat_taps);
    run_test("test_dflash2_grouped_dyn_conv", test_dflash2_grouped_dyn_conv);
    run_test("test_dflash2_block_attn", test_dflash2_block_attn);
    run_test("test_dflash2_topk16_and_select_path", test_dflash2_topk16_and_select_path);
    printf("CUDA DFlash2 tests: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
