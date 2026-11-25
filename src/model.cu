#include "model.hpp"
#include "kernels.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <chrono>

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    throw std::runtime_error(cudaGetErrorString(e)); } } while(0)

#define CHECK_CUBLAS(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
    throw std::runtime_error("cuBLAS error"); } } while(0)

namespace ink {

Inkling2Model::~Inkling2Model() {
    for (int b = 0; b <= 8; ++b) {
        if (batch_graph_execs[b]) cudaGraphExecDestroy(batch_graph_execs[b]);
        if (batch_graphs[b]) cudaGraphDestroy(batch_graphs[b]);
    }
    for (int b = 0; b <= 8; ++b) {
        for (int l = 0; l <= 5; ++l) {
            if (batch_verify_graph_execs[b][l]) cudaGraphExecDestroy(batch_verify_graph_execs[b][l]);
            if (batch_verify_graphs[b][l]) cudaGraphDestroy(batch_verify_graphs[b][l]);
        }
    }
    for (int t = 0; t <= 8; ++t) {
        if (verify_graph_execs[t]) cudaGraphExecDestroy(verify_graph_execs[t]);
        if (verify_graphs[t]) cudaGraphDestroy(verify_graphs[t]);
    }
    if (decode_graph_exec) cudaGraphExecDestroy(decode_graph_exec);
    if (decode_graph) cudaGraphDestroy(decode_graph);
    for (auto& kv : fp4_plan_cache) {
        if (kv.second.layoutA) cublasLtMatrixLayoutDestroy(kv.second.layoutA);
        if (kv.second.layoutB) cublasLtMatrixLayoutDestroy(kv.second.layoutB);
        if (kv.second.layoutD) cublasLtMatrixLayoutDestroy(kv.second.layoutD);
    }
    if (fp4_op_desc) cublasLtMatmulDescDestroy(fp4_op_desc);
    for (void* p : extra_allocs) cudaFree(p);
    if (cublas_lt) cublasLtDestroy(cublas_lt);
    if (cublas) cublasDestroy(cublas);
    if (stream) cudaStreamDestroy(stream);
}

void* Inkling2Model::alloc_gpu(size_t bytes) {
    void* p = nullptr;
    CHECK_CUDA(cudaMalloc(&p, bytes));
    extra_allocs.push_back(p);
    return p;
}

void Inkling2Model::load(const std::string& model_dir, int max_seq, int max_chunk, int n_slots) {
    auto t0 = std::chrono::steady_clock::now();
    max_seq_len = max_seq;
    max_batch_tokens = max_chunk;
    max_slots = std::max(1, std::min(8, n_slots));

    cfg = InklingConfig::load(model_dir + "/config.json");
    st.open(model_dir + "/model-00001-of-00001.safetensors");

    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasCreate(&cublas));
    CHECK_CUBLAS(cublasSetStream(cublas, stream));
    CHECK_CUBLAS(cublasLtCreate(&cublas_lt));
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&fp4_op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    {
        cublasOperation_t transA = CUBLAS_OP_T, transB = CUBLAS_OP_N;
        cublasLtMatmulMatrixScale_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(fp4_op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(fp4_op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(fp4_op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(fp4_op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm)));
    }
    d_lt_workspace = alloc_gpu(lt_workspace_bytes);

    std::vector<uint8_t> scratch;
    scratch.reserve(512 * 1024 * 1024);

    // Top-level weights
    embed_w = (__nv_bfloat16*)st.load_to_gpu("model.llm.embed.weight", scratch);
    embed_norm_w = (__nv_bfloat16*)st.load_to_gpu("model.llm.embed_norm.weight", scratch);
    final_norm_w = (__nv_bfloat16*)st.load_to_gpu("model.llm.norm.weight", scratch);
    unembed_w = (__nv_bfloat16*)st.load_to_gpu("model.llm.unembed.weight", scratch);
    unembed_fp8 = (uint8_t*)alloc_gpu((size_t)cfg.unpadded_vocab_size * cfg.hidden_size);
    unembed_sc  = (float*)alloc_gpu((size_t)cfg.unpadded_vocab_size * sizeof(float));
    launch_quantize_bf16_to_fp8_rowwise(unembed_w, unembed_fp8, unembed_sc, cfg.unpadded_vocab_size, cfg.hidden_size, stream);

    // Pre-scale AttnRes queries by 1/sqrt(D)
    float inv_sqrt_d = 1.0f / std::sqrt((float)cfg.hidden_size);
    auto load_scaled_query = [&](const std::string& name) -> float* {
        std::vector<uint8_t> raw;
        st.load_to_host(name, raw);
        size_t n = raw.size() / sizeof(float);
        float* fp = reinterpret_cast<float*>(raw.data());
        for (size_t i = 0; i < n; ++i) fp[i] *= inv_sqrt_d;
        float* d_p = (float*)alloc_gpu(raw.size());
        CHECK_CUDA(cudaMemcpy(d_p, raw.data(), raw.size(), cudaMemcpyHostToDevice));
        return d_p;
    };

    attn_queries.resize(cfg.num_hidden_layers - 1);
    for (int i = 0; i < cfg.num_hidden_layers - 1; ++i) {
        attn_queries[i] = load_scaled_query("model.llm.attn_res.attn_queries." + std::to_string(i));
    }
    mlp_queries.resize(cfg.num_hidden_layers);
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        mlp_queries[i] = load_scaled_query("model.llm.attn_res.mlp_queries." + std::to_string(i));
    }
    output_query = load_scaled_query("model.llm.attn_res.output_query");

    // Loop cross-pass message queries: stack [attn, k, v, mlp] per core layer, scaled by 1/sqrt(D)
    {
        int n_core = cfg.loop_core_end - cfg.loop_core_start;
        int D = cfg.hidden_size;
        std::vector<uint8_t> q_raw, kv_raw;
        st.load_to_host("model.llm.middle_cycle_cross_pass_queries", q_raw);
        st.load_to_host("model.llm.middle_cycle_cross_pass_kv_queries", kv_raw);
        const float* q_f = reinterpret_cast<const float*>(q_raw.data());
        const float* kv_f = reinterpret_cast<const float*>(kv_raw.data());
        std::vector<float> stacked((size_t)n_core * 4 * D);
        for (int c = 0; c < n_core; ++c) {
            const float* q_attn = q_f + (c * 2 + 0) * D;
            const float* q_mlp  = q_f + (c * 2 + 1) * D;
            const float* q_k    = kv_f + (c * 2 + 0) * D;
            const float* q_v    = kv_f + (c * 2 + 1) * D;
            float* dst = stacked.data() + (size_t)c * 4 * D;
            for (int d = 0; d < D; ++d) {
                dst[0 * D + d] = q_attn[d] * inv_sqrt_d;
                dst[1 * D + d] = q_k[d] * inv_sqrt_d;
                dst[2 * D + d] = q_v[d] * inv_sqrt_d;
                dst[3 * D + d] = q_mlp[d] * inv_sqrt_d;
            }
        }
        message_queries = (float*)alloc_gpu(stacked.size() * sizeof(float));
        CHECK_CUDA(cudaMemcpy(message_queries, stacked.data(), stacked.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Load 24 layers
    layers.resize(cfg.num_hidden_layers);
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        auto& lw = layers[i];
        std::string p = "model.llm.layers." + std::to_string(i) + ".";
        lw.is_local = cfg.is_local(i);
        lw.is_loop_core = cfg.is_loop_core(i);

        if (lw.is_local) {
            lw.attn_norm = (__nv_bfloat16*)st.load_to_gpu(p + "attn_norm.weight", scratch);
        } else {
            // Stack q_input_norm, k_input_norm, v_input_norm into contiguous [3, D]
            int D = cfg.hidden_size;
            lw.attn_norm = (__nv_bfloat16*)alloc_gpu(3 * D * sizeof(__nv_bfloat16));
            const char* nnames[3] = {"q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight"};
            for (int j = 0; j < 3; ++j) {
                std::vector<uint8_t> raw;
                st.load_to_host(p + nnames[j], raw);
                CHECK_CUDA(cudaMemcpy(lw.attn_norm + j * D, raw.data(), raw.size(), cudaMemcpyHostToDevice));
            }
        }
        lw.mlp_norm = (__nv_bfloat16*)st.load_to_gpu(p + "mlp_norm.weight", scratch);

        // Pack [wq(3072), wr(384), gate_attn(3072), sink_proj(24)] into one contiguous w_qrgs[6552, D]
        {
            int D = cfg.hidden_size;
            int q_dim = cfg.num_attention_heads * cfg.head_dim;
            int r_dim = cfg.num_attention_heads * cfg.d_rel;
            int kv_dim = cfg.num_key_value_heads * cfg.head_dim;
            int qrgs_dim = 2 * q_dim + r_dim + cfg.num_attention_heads;
            lw.w_qrgs = (__nv_bfloat16*)alloc_gpu((size_t)qrgs_dim * D * sizeof(__nv_bfloat16));
            lw.wq        = lw.w_qrgs;
            lw.wr        = lw.w_qrgs + (int64_t)q_dim * D;
            lw.gate_attn = lw.w_qrgs + (int64_t)(q_dim + r_dim) * D;
            lw.sink_proj = lw.w_qrgs + (int64_t)(2 * q_dim + r_dim) * D;
            auto copy_part = [&](const std::string& name, __nv_bfloat16* dst) {
                std::vector<uint8_t> raw;
                st.load_to_host(name, raw);
                CHECK_CUDA(cudaMemcpy(dst, raw.data(), raw.size(), cudaMemcpyHostToDevice));
            };
            copy_part(p + "attn.wq_du.weight", lw.wq);
            copy_part(p + "attn.wr_du.weight", lw.wr);
            copy_part(p + "attn.gate_attn.weight", lw.gate_attn);
            copy_part(p + "attn.attn_sink_proj.weight", lw.sink_proj);

            lw.w_kv = (__nv_bfloat16*)alloc_gpu((size_t)2 * kv_dim * D * sizeof(__nv_bfloat16));
            lw.wk = lw.w_kv;
            lw.wv = lw.w_kv + (int64_t)kv_dim * D;
            copy_part(p + "attn.wk_dv.weight", lw.wk);
            copy_part(p + "attn.wv_dv.weight", lw.wv);

            // Quantize dense attention projections to per-row FP8 E4M3 for 2x M=1..8 bandwidth!
            lw.w_qrgs_fp8 = (uint8_t*)alloc_gpu((size_t)qrgs_dim * D);
            lw.w_qrgs_sc  = (float*)alloc_gpu((size_t)qrgs_dim * sizeof(float));
            launch_quantize_bf16_to_fp8_rowwise(lw.w_qrgs, lw.w_qrgs_fp8, lw.w_qrgs_sc, qrgs_dim, D, stream);
            lw.w_kv_fp8   = (uint8_t*)alloc_gpu((size_t)2 * kv_dim * D);
            lw.w_kv_sc    = (float*)alloc_gpu((size_t)2 * kv_dim * sizeof(float));
            launch_quantize_bf16_to_fp8_rowwise(lw.w_kv, lw.w_kv_fp8, lw.w_kv_sc, 2 * kv_dim, D, stream);
        }
        lw.wo = (__nv_bfloat16*)st.load_to_gpu(p + "attn.wo_ud.weight", scratch);
        {
            int D = cfg.hidden_size;
            int q_dim = cfg.num_attention_heads * cfg.head_dim;
            lw.wo_fp8 = (uint8_t*)alloc_gpu((size_t)D * q_dim);
            lw.wo_sc  = (float*)alloc_gpu((size_t)D * sizeof(float));
            launch_quantize_bf16_to_fp8_rowwise(lw.wo, lw.wo_fp8, lw.wo_sc, D, q_dim, stream);
        }

        lw.q_clip_scale = (__nv_bfloat16*)st.load_to_gpu(p + "attn.q_l2_clip.scale", scratch);
        lw.rel_clip_scale = (__nv_bfloat16*)st.load_to_gpu(p + "attn.rel_l2_clip.scale", scratch);
        lw.rel_proj = (__nv_bfloat16*)st.load_to_gpu(p + "attn.rel_logits_proj.proj", scratch);

        if (lw.is_local) {
            lw.k_sconv_w = (__nv_bfloat16*)st.load_to_gpu(p + "attn.k_sconv.weight", scratch);
            lw.v_sconv_w = (__nv_bfloat16*)st.load_to_gpu(p + "attn.v_sconv.weight", scratch);
        } else {
            lw.k_sconv_w = (__nv_bfloat16*)st.load_to_gpu(p + "attn.k_sconv.static_w", scratch);
            lw.k_sconv_proj = (__nv_bfloat16*)st.load_to_gpu(p + "attn.k_sconv.weight_proj", scratch);
            lw.v_sconv_w = (__nv_bfloat16*)st.load_to_gpu(p + "attn.v_sconv.static_w", scratch);
            lw.v_sconv_proj = (__nv_bfloat16*)st.load_to_gpu(p + "attn.v_sconv.weight_proj", scratch);
        }
        lw.attn_sconv_w = (__nv_bfloat16*)st.load_to_gpu(p + "attn_sconv.weight", scratch);
        lw.mlp_sconv_w = (__nv_bfloat16*)st.load_to_gpu(p + "mlp_sconv.weight", scratch);

        if (lw.is_loop_core) {
            const char* snames[4] = {"attn", "k", "v", "mlp"};
            for (int s = 0; s < 4; ++s) {
                std::string cp = p + "controllers." + snames[s] + ".";
                lw.ctrl_down[s] = (__nv_bfloat16*)st.load_to_gpu(cp + "down_weight", scratch);
                lw.ctrl_gain[s] = (__nv_bfloat16*)st.load_to_gpu(cp + "gain_weight", scratch);
                lw.ctrl_forget[s] = (__nv_bfloat16*)st.load_to_gpu(cp + "forget_weight", scratch);
            }
        }

        lw.gate_weight = (__nv_bfloat16*)st.load_to_gpu(p + "mlp.gate.weight", scratch);
        lw.gate_bias = (float*)st.load_to_gpu(p + "mlp.gate.bias", scratch);

        lw.is_mxfp8 = (st.info(p + "mlp.experts.w13_weight").dtype == "F8_E4M3");
        lw.w13_weight = (uint8_t*)st.load_to_gpu(p + "mlp.experts.w13_weight", scratch);
        lw.w13_scale  = (uint8_t*)st.load_to_gpu(p + "mlp.experts.w13_weight_scale", scratch);
        lw.w2_weight  = (uint8_t*)st.load_to_gpu(p + "mlp.experts.w2_weight", scratch);
        lw.w2_scale   = (uint8_t*)st.load_to_gpu(p + "mlp.experts.w2_weight_scale", scratch);
        if (!lw.is_mxfp8) {
            lw.w13_amax  = (float*)st.load_to_gpu(p + "mlp.experts.w13_weight_amax", scratch);
            lw.w13_denom = (float*)st.load_to_gpu(p + "mlp.experts.w13_weight_outer_scale_denominator", scratch);
            lw.w2_amax   = (float*)st.load_to_gpu(p + "mlp.experts.w2_weight_amax", scratch);
            lw.w2_denom  = (float*)st.load_to_gpu(p + "mlp.experts.w2_weight_outer_scale_denominator", scratch);

            // Pre-swizzle NVFP4 scales into Blackwell 128x4 layout for native cuBLASLt FP4 Tensor Core GEMM!
            int E = cfg.n_routed_experts;
            int D = cfg.hidden_size;
            int inter = cfg.intermediate_size;
            size_t s13_bytes = (size_t)E * (2 * inter) * (D / 16);
            size_t s2_bytes  = (size_t)E * D * (inter / 16);
            lw.w13_scale_swiz = (uint8_t*)alloc_gpu(s13_bytes);
            lw.w2_scale_swiz  = (uint8_t*)alloc_gpu(s2_bytes);
            launch_swizzle_nvfp4_scales(lw.w13_scale, lw.w13_scale_swiz, E, 2 * inter, D / 16, stream);
            launch_swizzle_nvfp4_scales(lw.w2_scale,  lw.w2_scale_swiz,  E, D, inter / 16, stream);

            // Cache host amax/denom per expert for zero-sync cuBLASLt alpha
            std::vector<float> h_a13(E), h_a2(E);
            float h_d13 = 1.0f, h_d2 = 1.0f;
            CHECK_CUDA(cudaMemcpy(h_a13.data(), lw.w13_amax, E * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(&h_d13, lw.w13_denom, sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_a2.data(), lw.w2_amax, E * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(&h_d2, lw.w2_denom, sizeof(float), cudaMemcpyDeviceToHost));
            lw.h_w13_gscale.resize(E);
            lw.h_w2_gscale.resize(E);
            for (int e = 0; e < E; ++e) {
                lw.h_w13_gscale[e] = h_a13[e] / h_d13;
                lw.h_w2_gscale[e]  = h_a2[e]  / h_d2;
            }
        }
    }

    // Build visits schedule (48 visits)
    visits.clear();
    for (int i = 0; i < cfg.loop_core_start; ++i) visits.push_back({i, 0});
    for (int p = 0; p < cfg.loop_num_passes; ++p)
        for (int i = cfg.loop_core_start; i < cfg.loop_core_end; ++i) visits.push_back({i, p});
    for (int i = cfg.loop_core_end; i < cfg.num_hidden_layers; ++i) visits.push_back({i, 0});

    // Allocate KV caches & conv states for all 48 visits x max_slots
    max_blocks_per_seq = (max_seq_len + 63) / 64;
    num_kv_blocks = max_slots * max_blocks_per_seq;
    num_snapshots = (max_slots >= 4) ? std::max(max_slots, 12) : max_slots;

    caches.resize(visits.size());
    size_t kv_bytes_total = 0;
    for (size_t v = 0; v < visits.size(); ++v) {
        int lid = visits[v].first;
        auto& vc = caches[v];
        vc.is_local = cfg.is_local(lid);
        vc.cap = vc.is_local ? (cfg.sliding_window_size + max_batch_tokens) : (max_blocks_per_seq * 64);
        if (vc.is_local) {
            size_t kv_sz = (size_t)max_slots * vc.cap * cfg.num_key_value_heads * cfg.head_dim * sizeof(__nv_bfloat16);
            vc.k_cache = (__nv_bfloat16*)alloc_gpu(kv_sz);
            vc.v_cache = (__nv_bfloat16*)alloc_gpu(kv_sz);
            kv_bytes_total += 2 * kv_sz;
        } else {
            // Global layers use paged per-(pos, head) FP8 E4M3 KV cache (block_size=64)
            size_t total_pos = (size_t)num_kv_blocks * 64;
            size_t kv_fp8_sz = total_pos * cfg.num_key_value_heads * cfg.head_dim * sizeof(uint8_t);
            size_t sc_sz = total_pos * cfg.num_key_value_heads * sizeof(float);
            vc.k_cache_fp8 = (uint8_t*)alloc_gpu(kv_fp8_sz);
            vc.v_cache_fp8 = (uint8_t*)alloc_gpu(kv_fp8_sz);
            vc.k_scale_fp8 = (float*)alloc_gpu(sc_sz);
            vc.v_scale_fp8 = (float*)alloc_gpu(sc_sz);
            kv_bytes_total += 2 * kv_fp8_sz + 2 * sc_sz;
        }

        int kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        vc.k_conv_state = (__nv_bfloat16*)alloc_gpu((size_t)max_slots * 3 * kv_dim * sizeof(__nv_bfloat16));
        vc.v_conv_state = (__nv_bfloat16*)alloc_gpu((size_t)max_slots * 3 * kv_dim * sizeof(__nv_bfloat16));
        vc.attn_conv_state = (__nv_bfloat16*)alloc_gpu((size_t)max_slots * 3 * cfg.hidden_size * sizeof(__nv_bfloat16));
        vc.mlp_conv_state = (__nv_bfloat16*)alloc_gpu((size_t)max_slots * 3 * cfg.hidden_size * sizeof(__nv_bfloat16));

        int n_ds = std::min(2, max_slots);
        vc.k_conv_snap = (__nv_bfloat16*)alloc_gpu((size_t)n_ds * 8 * 3 * kv_dim * sizeof(__nv_bfloat16));
        vc.v_conv_snap = (__nv_bfloat16*)alloc_gpu((size_t)n_ds * 8 * 3 * kv_dim * sizeof(__nv_bfloat16));
        vc.attn_conv_snap = (__nv_bfloat16*)alloc_gpu((size_t)n_ds * 8 * 3 * cfg.hidden_size * sizeof(__nv_bfloat16));
        vc.mlp_conv_snap = (__nv_bfloat16*)alloc_gpu((size_t)n_ds * 8 * 3 * cfg.hidden_size * sizeof(__nv_bfloat16));

        // Cross-slot prefix-cache boundary snapshots (local SWA ring + conv states)
        if (vc.is_local) {
            size_t kv_sz = (size_t)num_snapshots * vc.cap * cfg.num_key_value_heads * cfg.head_dim * sizeof(__nv_bfloat16);
            vc.local_k_prefill_snap = (__nv_bfloat16*)alloc_gpu(kv_sz);
            vc.local_v_prefill_snap = (__nv_bfloat16*)alloc_gpu(kv_sz);
            kv_bytes_total += 2 * kv_sz;
        }
        vc.k_conv_prefill_snap = (__nv_bfloat16*)alloc_gpu((size_t)num_snapshots * 3 * kv_dim * sizeof(__nv_bfloat16));
        vc.v_conv_prefill_snap = (__nv_bfloat16*)alloc_gpu((size_t)num_snapshots * 3 * kv_dim * sizeof(__nv_bfloat16));
        vc.attn_conv_prefill_snap = (__nv_bfloat16*)alloc_gpu((size_t)num_snapshots * 3 * cfg.hidden_size * sizeof(__nv_bfloat16));
        vc.mlp_conv_prefill_snap = (__nv_bfloat16*)alloc_gpu((size_t)num_snapshots * 3 * cfg.hidden_size * sizeof(__nv_bfloat16));
    }

    // Allocate workspace buffers
    int T = max_batch_tokens;
    int D = cfg.hidden_size;
    int q_dim = cfg.num_attention_heads * cfg.head_dim;    // 3072
    int kv_dim = cfg.num_key_value_heads * cfg.head_dim;   // 1024
    int r_dim = cfg.num_attention_heads * cfg.d_rel;       // 384

    d_tokens = (int32_t*)alloc_gpu(T * sizeof(int32_t));
    d_pos = (int32_t*)alloc_gpu(sizeof(int32_t));
    d_batch_slots = (int32_t*)alloc_gpu(max_slots * sizeof(int32_t));
    d_batch_pos = (int32_t*)alloc_gpu(max_slots * sizeof(int32_t));
    d_block_tables = (int32_t*)alloc_gpu((size_t)max_slots * max_blocks_per_seq * sizeof(int32_t));
    {
        std::vector<int32_t> init_bt((size_t)max_slots * max_blocks_per_seq);
        for (int s = 0; s < max_slots; ++s) {
            for (int b = 0; b < max_blocks_per_seq; ++b) {
                init_bt[(size_t)s * max_blocks_per_seq + b] = s * max_blocks_per_seq + b;
            }
        }
        CHECK_CUDA(cudaMemcpyAsync(d_block_tables, init_bt.data(),
                                   init_bt.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    }
    d_stack = (__nv_bfloat16*)alloc_gpu((size_t)6 * T * D * sizeof(__nv_bfloat16));
    d_prelude_part = (__nv_bfloat16*)alloc_gpu((size_t)T * D * sizeof(__nv_bfloat16));
    d_bank = (__nv_bfloat16*)alloc_gpu((size_t)2 * T * D * sizeof(__nv_bfloat16));
    d_prelude_dest = (float*)alloc_gpu((size_t)T * D * sizeof(float));
    d_messages = (float*)alloc_gpu((size_t)4 * T * D * sizeof(float));
    d_fp32_mix = (float*)alloc_gpu((size_t)3 * T * D * sizeof(float));
    d_attn_in = (__nv_bfloat16*)alloc_gpu((size_t)3 * T * D * sizeof(__nv_bfloat16));
    d_mlp_in = (__nv_bfloat16*)alloc_gpu((size_t)T * D * sizeof(__nv_bfloat16));
    d_pair = (__nv_bfloat16*)alloc_gpu((size_t)T * 2 * D * sizeof(__nv_bfloat16));
    d_ctrl_h = (__nv_bfloat16*)alloc_gpu((size_t)T * cfg.loop_gate_rank * sizeof(__nv_bfloat16));
    d_ctrl_gain = (float*)alloc_gpu((size_t)T * D * sizeof(float));
    d_ctrl_forget = (float*)alloc_gpu((size_t)T * D * sizeof(float));

    d_qrgs = (__nv_bfloat16*)alloc_gpu((size_t)T * (2 * q_dim + r_dim + cfg.num_attention_heads) * sizeof(__nv_bfloat16));
    d_kv_out = (__nv_bfloat16*)alloc_gpu((size_t)T * 2 * kv_dim * sizeof(__nv_bfloat16));
    d_Q = (__nv_bfloat16*)alloc_gpu((size_t)T * q_dim * sizeof(__nv_bfloat16));
    d_K = (__nv_bfloat16*)alloc_gpu((size_t)T * kv_dim * sizeof(__nv_bfloat16));
    d_V = (__nv_bfloat16*)alloc_gpu((size_t)T * kv_dim * sizeof(__nv_bfloat16));
    d_K_conv = (__nv_bfloat16*)alloc_gpu((size_t)T * kv_dim * sizeof(__nv_bfloat16));
    d_V_conv = (__nv_bfloat16*)alloc_gpu((size_t)T * kv_dim * sizeof(__nv_bfloat16));
    d_dyn_w = (__nv_bfloat16*)alloc_gpu((size_t)T * cfg.num_key_value_heads * 4 * sizeof(__nv_bfloat16));
    d_R = (__nv_bfloat16*)alloc_gpu((size_t)T * r_dim * sizeof(__nv_bfloat16));
    d_rel_logits = (__nv_bfloat16*)alloc_gpu((size_t)T * cfg.num_attention_heads * cfg.rel_extent * sizeof(__nv_bfloat16));
    d_gate = (__nv_bfloat16*)alloc_gpu((size_t)T * q_dim * sizeof(__nv_bfloat16));
    d_sinks = (__nv_bfloat16*)alloc_gpu((size_t)T * cfg.num_attention_heads * sizeof(__nv_bfloat16));
    d_attn_raw = (__nv_bfloat16*)alloc_gpu((size_t)T * q_dim * sizeof(__nv_bfloat16));
    d_attn_proj = (__nv_bfloat16*)alloc_gpu((size_t)T * D * sizeof(__nv_bfloat16));
    d_attn_out = (__nv_bfloat16*)alloc_gpu((size_t)T * D * sizeof(__nv_bfloat16));
    d_mlp_raw = (__nv_bfloat16*)alloc_gpu((size_t)T * D * sizeof(__nv_bfloat16));
    d_mlp_out = (__nv_bfloat16*)alloc_gpu((size_t)T * D * sizeof(__nv_bfloat16));
    d_split_scratch = (float*)alloc_gpu((size_t)16 * cfg.num_attention_heads * 16 * (cfg.head_dim + 2) * sizeof(float));

    d_topk_ids = (int32_t*)alloc_gpu((size_t)T * cfg.num_experts_per_tok * sizeof(int32_t));
    d_topk_weights = (float*)alloc_gpu((size_t)T * cfg.num_experts_per_tok * sizeof(float));
    d_router_logits = (float*)alloc_gpu((size_t)T * cfg.n_routed_experts * sizeof(float));
    d_moe_act = (float*)alloc_gpu((size_t)T * cfg.num_experts_per_tok * cfg.intermediate_size * sizeof(float));
    d_moe_uniq = (int32_t*)alloc_gpu((size_t)(1 + 128 * 18) * sizeof(int32_t));

    d_exp_w13_bf16 = (__nv_bfloat16*)alloc_gpu((size_t)2 * cfg.intermediate_size * D * sizeof(__nv_bfloat16));
    d_exp_w2_bf16 = (__nv_bfloat16*)alloc_gpu((size_t)D * cfg.intermediate_size * sizeof(__nv_bfloat16));
    d_exp_x = (__nv_bfloat16*)alloc_gpu((size_t)T * D * sizeof(__nv_bfloat16));
    d_exp_gu = (__nv_bfloat16*)alloc_gpu((size_t)(T + 16) * 2 * cfg.intermediate_size * sizeof(__nv_bfloat16));
    d_exp_act = (__nv_bfloat16*)alloc_gpu((size_t)T * cfg.intermediate_size * sizeof(__nv_bfloat16));
    d_exp_down = (__nv_bfloat16*)alloc_gpu((size_t)(T + 16) * D * sizeof(__nv_bfloat16));
    int T_pad128 = ((T + 127) / 128) * 128;
    d_exp_x_fp4 = (uint8_t*)alloc_gpu((size_t)(T + 16) * (D / 2));
    d_exp_x_scale = (uint8_t*)alloc_gpu((size_t)T_pad128 * (D / 16));
    d_exp_act_fp4 = (uint8_t*)alloc_gpu((size_t)(T + 16) * (cfg.intermediate_size / 2));
    d_exp_act_scale = (uint8_t*)alloc_gpu((size_t)T_pad128 * (cfg.intermediate_size / 16));
    d_moe_out_fp32 = (float*)alloc_gpu((size_t)T * D * sizeof(float));
    d_exp_tok_idx = (int32_t*)alloc_gpu((size_t)T * cfg.num_experts_per_tok * sizeof(int32_t));
    d_exp_gammas = (float*)alloc_gpu((size_t)T * cfg.num_experts_per_tok * sizeof(float));
    d_logits = (float*)alloc_gpu((size_t)16 * cfg.vocab_size * sizeof(float));
    d_out_token = (int32_t*)alloc_gpu(16 * sizeof(int32_t));
    d_aux_states = (__nv_bfloat16*)alloc_gpu((size_t)4 * T * D * sizeof(__nv_bfloat16));
    int n_ds_slots = std::min(2, max_slots);
    d_last_aux = (__nv_bfloat16*)alloc_gpu((size_t)n_ds_slots * 4 * D * sizeof(__nv_bfloat16));

    for (int s = 0; s < max_slots; ++s) reset_cache(s);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    fprintf(stderr, "[warpgrid] Loaded Inkling-2-Mini in %.2f s (KV cache: %.2f GiB for %d slots x max_seq=%d)\n",
            sec, kv_bytes_total / (1024.0 * 1024.0 * 1024.0), max_slots, max_seq_len);

    if (!dflash2_dir.empty()) {
        if (draft_k <= 0) draft_k = 6;
        dspark.load_dflash2(dflash2_dir, embed_w, embed_norm_w, unembed_w, cublas, stream,
                            max_batch_tokens, draft_vocab_file, n_ds_slots);
        dspark.alloc_df2_snapshots(num_snapshots);
    } else if (draft_k > 0) {
        dspark.load(model_dir, embed_w, embed_norm_w, unembed_w, cublas, stream,
                    max_seq_len, max_batch_tokens, draft_vocab_file, n_ds_slots);
    }
}

void Inkling2Model::reset_cache(int slot_id) {
    int kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    for (auto& vc : caches) {
        CHECK_CUDA(cudaMemsetAsync(vc.k_conv_state + (int64_t)slot_id * 3 * kv_dim, 0, 3 * kv_dim * sizeof(__nv_bfloat16), stream));
        CHECK_CUDA(cudaMemsetAsync(vc.v_conv_state + (int64_t)slot_id * 3 * kv_dim, 0, 3 * kv_dim * sizeof(__nv_bfloat16), stream));
        CHECK_CUDA(cudaMemsetAsync(vc.attn_conv_state + (int64_t)slot_id * 3 * cfg.hidden_size, 0, 3 * cfg.hidden_size * sizeof(__nv_bfloat16), stream));
        CHECK_CUDA(cudaMemsetAsync(vc.mlp_conv_state + (int64_t)slot_id * 3 * cfg.hidden_size, 0, 3 * cfg.hidden_size * sizeof(__nv_bfloat16), stream));
    }
}

void Inkling2Model::save_boundary_snapshot(int slot_id, int snap_id) {
    if (snap_id < 0 || snap_id >= num_snapshots) return;
    int kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    int D = cfg.hidden_size;
    for (auto& vc : caches) {
        if (vc.is_local) {
            size_t one_kv = (size_t)vc.cap * kv_dim * sizeof(__nv_bfloat16);
            int64_t src_off = (int64_t)slot_id * vc.cap * kv_dim;
            int64_t dst_off = (int64_t)snap_id * vc.cap * kv_dim;
            CHECK_CUDA(cudaMemcpyAsync(vc.local_k_prefill_snap + dst_off, vc.k_cache + src_off, one_kv, cudaMemcpyDeviceToDevice, stream));
            CHECK_CUDA(cudaMemcpyAsync(vc.local_v_prefill_snap + dst_off, vc.v_cache + src_off, one_kv, cudaMemcpyDeviceToDevice, stream));
        }
        int64_t k_src = (int64_t)slot_id * 3 * kv_dim;
        int64_t k_dst = (int64_t)snap_id * 3 * kv_dim;
        int64_t d_src = (int64_t)slot_id * 3 * D;
        int64_t d_dst = (int64_t)snap_id * 3 * D;
        CHECK_CUDA(cudaMemcpyAsync(vc.k_conv_prefill_snap + k_dst, vc.k_conv_state + k_src, 3 * kv_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(vc.v_conv_prefill_snap + k_dst, vc.v_conv_state + k_src, 3 * kv_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(vc.attn_conv_prefill_snap + d_dst, vc.attn_conv_state + d_src, 3 * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(vc.mlp_conv_prefill_snap + d_dst, vc.mlp_conv_state + d_src, 3 * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
    }
    dspark.save_df2_snapshot(slot_id, snap_id, stream);
}

void Inkling2Model::restore_boundary_snapshot(int slot_id, int snap_id) {
    if (snap_id < 0 || snap_id >= num_snapshots) return;
    int kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    int D = cfg.hidden_size;
    for (auto& vc : caches) {
        if (vc.is_local) {
            size_t one_kv = (size_t)vc.cap * kv_dim * sizeof(__nv_bfloat16);
            int64_t src_off = (int64_t)snap_id * vc.cap * kv_dim;
            int64_t dst_off = (int64_t)slot_id * vc.cap * kv_dim;
            CHECK_CUDA(cudaMemcpyAsync(vc.k_cache + dst_off, vc.local_k_prefill_snap + src_off, one_kv, cudaMemcpyDeviceToDevice, stream));
            CHECK_CUDA(cudaMemcpyAsync(vc.v_cache + dst_off, vc.local_v_prefill_snap + src_off, one_kv, cudaMemcpyDeviceToDevice, stream));
        }
        int64_t k_src = (int64_t)snap_id * 3 * kv_dim;
        int64_t k_dst = (int64_t)slot_id * 3 * kv_dim;
        int64_t d_src = (int64_t)snap_id * 3 * D;
        int64_t d_dst = (int64_t)slot_id * 3 * D;
        CHECK_CUDA(cudaMemcpyAsync(vc.k_conv_state + k_dst, vc.k_conv_prefill_snap + k_src, 3 * kv_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(vc.v_conv_state + k_dst, vc.v_conv_prefill_snap + k_src, 3 * kv_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(vc.attn_conv_state + d_dst, vc.attn_conv_prefill_snap + d_src, 3 * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(vc.mlp_conv_state + d_dst, vc.mlp_conv_prefill_snap + d_src, 3 * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
    }
    dspark.restore_df2_snapshot(slot_id, snap_id, stream);
}

void Inkling2Model::save_prefill_snapshot(int slot_id) {
    save_boundary_snapshot(slot_id, slot_id);
}

void Inkling2Model::restore_prefill_snapshot(int slot_id) {
    restore_boundary_snapshot(slot_id, slot_id);
}

void Inkling2Model::set_slot_block_table(int slot_id, const int32_t* blocks, int num_blocks) {
    if (slot_id < 0 || slot_id >= max_slots || num_blocks <= 0) return;
    int n = std::min(num_blocks, max_blocks_per_seq);
    CHECK_CUDA(cudaMemcpyAsync(d_block_tables + (int64_t)slot_id * max_blocks_per_seq,
                               blocks, (size_t)n * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
}

void Inkling2Model::linear_bf16(
    const __nv_bfloat16* X, const __nv_bfloat16* W, __nv_bfloat16* Y, int M, int N, int K)
{
    if (use_bf16_gemv && (K & 7) == 0) {
        if (M == 1) {
            launch_bf16_gemv(X, W, Y, nullptr, N, K, stream);
            return;
        }
        if (M >= 2 && M <= 8) {
            if (M == 8 && N <= 128) {
                // cuBLAS is 2x faster than small_gemm for N=64, M=8 (0.0083ms vs 0.0164ms)
            } else {
                launch_bf16_small_gemm(X, W, Y, nullptr, M, N, K, stream);
                return;
            }
        }
    }
    // Row-major Y[M, N] = X[M, K] @ W[N, K]^T
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasGemmEx(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        W, CUDA_R_16BF, K,
        X, CUDA_R_16BF, K,
        &beta,
        Y, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

void Inkling2Model::linear_bf16_to_fp32(
    const __nv_bfloat16* X, const __nv_bfloat16* W, float* Y, int M, int N, int K)
{
    if (use_bf16_gemv && (K & 7) == 0) {
        if (M == 1) {
            launch_bf16_gemv(X, W, nullptr, Y, N, K, stream);
            return;
        }
        if (M >= 2 && M <= 8) {
            launch_bf16_small_gemm(X, W, nullptr, Y, M, N, K, stream);
            return;
        }
    }
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasGemmEx(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        W, CUDA_R_16BF, K,
        X, CUDA_R_16BF, K,
        &beta,
        Y, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

void Inkling2Model::run_controller(
    const LayerWeights& lw, int s, const float* dest, const float* msg,
    const __nv_bfloat16* norm_w, __nv_bfloat16* out, int T)
{
    int D = cfg.hidden_size;
    int r = cfg.loop_gate_rank;
    launch_pair_rms_norm(dest, msg, d_pair, T, D, 1e-6f, stream);
    linear_bf16(d_pair, lw.ctrl_down[s], d_ctrl_h, T, r, 2 * D);
    launch_fused_ctrl_blend_rmsnorm(d_ctrl_h, lw.ctrl_gain[s], lw.ctrl_forget[s],
                                    dest, msg, norm_w, out, T, D, r, cfg.rms_norm_eps, stream, d_ctrl_gain);
}

void Inkling2Model::linear_nvfp4_lt(
    const uint8_t* X_fp4, const uint8_t* X_scale,
    const uint8_t* W_fp4, const uint8_t* W_scale,
    __nv_bfloat16* Y_bf16, int M_pad, int N, int K, float alpha)
{
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(fp4_op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &W_scale, sizeof(W_scale)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(fp4_op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &X_scale, sizeof(X_scale)));

    uint64_t key = ((uint64_t)M_pad << 32) | ((uint64_t)N << 16) | (uint64_t)K;
    auto it = fp4_plan_cache.find(key);
    if (it == fp4_plan_cache.end()) {
        Fp4Plan plan;
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.layoutA, CUDA_R_4F_E2M1, K, N, K));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.layoutB, CUDA_R_4F_E2M1, K, M_pad, K));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.layoutD, CUDA_R_16BF, N, M_pad, N));
        cublasLtMatmulPreference_t pref;
        CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
        CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &lt_workspace_bytes, sizeof(lt_workspace_bytes)));
        cublasLtMatmulHeuristicResult_t heur;
        int returned = 0;
        CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(cublas_lt, fp4_op_desc, plan.layoutA, plan.layoutB, plan.layoutD, plan.layoutD, pref, 1, &heur, &returned));
        cublasLtMatmulPreferenceDestroy(pref);
        if (returned == 0) {
            cublasLtMatrixLayoutDestroy(plan.layoutA);
            cublasLtMatrixLayoutDestroy(plan.layoutB);
            cublasLtMatrixLayoutDestroy(plan.layoutD);
            throw std::runtime_error("cuBLASLt NVFP4: no heuristic found");
        }
        plan.algo = heur.algo;
        it = fp4_plan_cache.emplace(key, plan).first;
    }

    const Fp4Plan& plan = it->second;
    float beta = 0.0f;
    CHECK_CUBLAS(cublasLtMatmul(
        cublas_lt, fp4_op_desc, &alpha,
        W_fp4, plan.layoutA, X_fp4, plan.layoutB,
        &beta, Y_bf16, plan.layoutD, Y_bf16, plan.layoutD,
        &plan.algo, d_lt_workspace, lt_workspace_bytes, stream));
}

void Inkling2Model::run_moe_grouped_prefill(
    const LayerWeights& lw, const __nv_bfloat16* x, __nv_bfloat16* out, int T)
{
    int D = cfg.hidden_size;
    int inter = cfg.intermediate_size;
    int E = cfg.n_routed_experts;
    int K = cfg.num_experts_per_tok;

    std::vector<int32_t> h_ids((size_t)T * K);
    std::vector<float> h_wts((size_t)T * K);
    CHECK_CUDA(cudaMemcpyAsync(h_ids.data(), d_topk_ids, h_ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(h_wts.data(), d_topk_weights, h_wts.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemsetAsync(d_moe_out_fp32, 0, (size_t)T * D * sizeof(float), stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Bucket token indices and routing weights per expert
    std::vector<int> counts(E, 0);
    for (int i = 0; i < T * K; ++i) counts[h_ids[i]]++;
    std::vector<int> offsets(E + 1, 0);
    for (int e = 0; e < E; ++e) offsets[e + 1] = offsets[e] + counts[e];
    std::vector<int32_t> packed_tok((size_t)T * K);
    std::vector<float> packed_wt((size_t)T * K);
    std::vector<int> cur = offsets;
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < K; ++k) {
            int e = h_ids[t * K + k];
            int p = cur[e]++;
            packed_tok[p] = t;
            packed_wt[p] = h_wts[t * K + k];
        }
    }
    CHECK_CUDA(cudaMemcpyAsync(d_exp_tok_idx, packed_tok.data(), packed_tok.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_exp_gammas, packed_wt.data(), packed_wt.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

    for (int e = 0; e < E; ++e) {
        int M = counts[e];
        if (M == 0) continue;
        int off = offsets[e];
        const int32_t* idx_e = d_exp_tok_idx + off;
        const float* wt_e = d_exp_gammas + off;

        if (!lw.is_mxfp8 && M >= 16) {
            // Native Blackwell cuBLASLt NVFP4 Tensor Core GEMM (~196 TFLOPS, ZERO weight dequant!)
            int M_pad = ((M + 15) / 16) * 16;
            const uint8_t* w13_e = lw.w13_weight + (size_t)e * (2 * inter) * (D / 2);
            const uint8_t* s13_e = lw.w13_scale_swiz + (size_t)e * (2 * inter) * (D / 16);
            const uint8_t* w2_e  = lw.w2_weight  + (size_t)e * D * (inter / 2);
            const uint8_t* s2_e  = lw.w2_scale_swiz  + (size_t)e * D * (inter / 16);

            launch_gather_quant_nvfp4(x, idx_e, d_exp_x_fp4, d_exp_x_scale, M, M_pad, D, stream);
            linear_nvfp4_lt(d_exp_x_fp4, d_exp_x_scale, w13_e, s13_e, d_exp_gu, M_pad, 2 * inter, D, lw.h_w13_gscale[e]);
            launch_swiglu_softcap_quant_nvfp4(d_exp_gu, wt_e, d_exp_act_fp4, d_exp_act_scale, M, M_pad, inter,
                                              cfg.mlp_gate_softcap, cfg.mlp_up_softcap, stream);
            linear_nvfp4_lt(d_exp_act_fp4, d_exp_act_scale, w2_e, s2_e, d_exp_down, M_pad, D, inter, lw.h_w2_gscale[e]);
            launch_scatter_add_bf16_to_fp32(d_exp_down, idx_e, d_moe_out_fp32, M, D, stream);
            continue;
        }

        launch_gather_bf16(x, idx_e, d_exp_x, M, D, stream);
        if (lw.is_mxfp8) {
            launch_dequant_mxfp8_matrix(lw.w13_weight, lw.w13_scale, e, d_exp_w13_bf16, 2 * inter, D, stream);
        } else {
            launch_dequant_nvfp4_matrix(lw.w13_weight, lw.w13_scale, lw.w13_amax, lw.w13_denom, e,
                                        d_exp_w13_bf16, 2 * inter, D, stream);
        }
        linear_bf16(d_exp_x, d_exp_w13_bf16, d_exp_gu, M, 2 * inter, D);
        launch_swiglu_softcap_bf16(d_exp_gu, wt_e, d_exp_act, M, inter,
                                   cfg.mlp_gate_softcap, cfg.mlp_up_softcap, stream);
        if (lw.is_mxfp8) {
            launch_dequant_mxfp8_matrix(lw.w2_weight, lw.w2_scale, e, d_exp_w2_bf16, D, inter, stream);
        } else {
            launch_dequant_nvfp4_matrix(lw.w2_weight, lw.w2_scale, lw.w2_amax, lw.w2_denom, e,
                                        d_exp_w2_bf16, D, inter, stream);
        }
        linear_bf16(d_exp_act, d_exp_w2_bf16, d_exp_down, M, D, inter);
        launch_scatter_add_bf16_to_fp32(d_exp_down, idx_e, d_moe_out_fp32, M, D, stream);
    }
    launch_fp32_to_bf16(d_moe_out_fp32, out, T * D, stream);
}

void Inkling2Model::reset_profile() {
    prof_attn_res_ctrl_ms = prof_dense_proj_ms = prof_attn_sconv_ms = 0.0;
    prof_router_ms = prof_moe_nvfp4_ms = prof_moe_mxfp8_ms = prof_lm_head_ms = 0.0;
    prof_steps = 0;
}

void Inkling2Model::print_profile() const {
    if (prof_steps == 0) return;
    double n = (double)prof_steps;
    double tot = (prof_attn_res_ctrl_ms + prof_dense_proj_ms + prof_attn_sconv_ms +
                  prof_router_ms + prof_moe_nvfp4_ms + prof_moe_mxfp8_ms + prof_lm_head_ms) / n;
    fprintf(stderr, "\n=== STAGE PROFILE (avg over %d steps, total=%.2f ms/tok) ===\n", prof_steps, tot);
    auto row = [&](const char* name, double ms) {
        fprintf(stderr, "  %-22s: %6.2f ms (%5.1f%%)\n", name, ms / n, 100.0 * (ms / n) / tot);
    };
    row("AttnRes + LoopCtrl", prof_attn_res_ctrl_ms);
    row("Dense BF16 Projs",   prof_dense_proj_ms);
    row("Attention + SConv",  prof_attn_sconv_ms);
    row("MoE Router",         prof_router_ms);
    row("MoE NVFP4 (44x)",    prof_moe_nvfp4_ms);
    row("MoE MXFP8 (4x)",     prof_moe_mxfp8_ms);
    row("LM Head + Sample",   prof_lm_head_ms);
}

void Inkling2Model::forward_internal(const int32_t* h_tokens, int T, int start_pos, int slot_id) {
    if (T > max_batch_tokens) throw std::runtime_error("T exceeds max_batch_tokens");
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    if (profile_stages) {
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
    }
    auto mark_start = [&]() { if (profile_stages) cudaEventRecord(ev0, stream); };
    auto mark_end = [&](double& acc) {
        if (profile_stages) {
            cudaEventRecord(ev1, stream);
            cudaEventSynchronize(ev1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev0, ev1);
            acc += ms;
        }
    };

    if (h_tokens != nullptr) {
        CHECK_CUDA(cudaMemcpyAsync(d_tokens, h_tokens, T * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    }
    const int32_t* dev_pos = (h_tokens == nullptr) ? d_pos : nullptr;

    int D = cfg.hidden_size;
    int stack_stride = max_batch_tokens * D;
    int q_dim = cfg.num_attention_heads * cfg.head_dim;  // 3072
    int kv_dim = cfg.num_key_value_heads * cfg.head_dim; // 1024
    int r_dim = cfg.num_attention_heads * cfg.d_rel;     // 384

    // stack[0] = embed_norm(embed(tokens))
    launch_embed_norm(d_tokens, embed_w, embed_norm_w, d_stack, T, D, cfg.rms_norm_eps, stream);

    int num_blocks = 0;
    int prelude_blocks = 0;
    bool have_prelude_dest = false;

    for (size_t v = 0; v < visits.size(); ++v) {
        int i = visits[v].first;
        int pass_idx = visits[v].second;
        bool looped = (pass_idx > 0);
        const auto& lw = layers[i];
        auto& vc = caches[v];

        // Loop pass boundaries
        if (i == cfg.loop_core_start && !looped) {
            prelude_blocks = num_blocks;
            CHECK_CUDA(cudaMemcpyAsync(
                d_prelude_part, d_stack + (int64_t)num_blocks * stack_stride,
                (size_t)T * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
            have_prelude_dest = false;
        } else if (i == cfg.loop_core_start && looped) {
            // bank[0] = stack[prelude_blocks + 1], bank[1] = stack[prelude_blocks + 2]
            CHECK_CUDA(cudaMemcpyAsync(
                d_bank, d_stack + (int64_t)(prelude_blocks + 1) * stack_stride,
                (size_t)T * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
            CHECK_CUDA(cudaMemcpyAsync(
                d_bank + stack_stride, d_stack + (int64_t)(prelude_blocks + 2) * stack_stride,
                (size_t)T * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
            num_blocks = prelude_blocks;
            CHECK_CUDA(cudaMemcpyAsync(
                d_stack + (int64_t)num_blocks * stack_stride, d_prelude_part,
                (size_t)T * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
            if (!have_prelude_dest) {
                launch_attn_res(
                    d_stack, num_blocks, stack_stride,
                    d_prelude_part, nullptr,
                    attn_queries[i - 1], 1, nullptr,
                    nullptr, d_prelude_dest, T, D, cfg.rms_norm_eps, stream);
                have_prelude_dest = true;
            }
        }

        __nv_bfloat16* partial = d_stack + (int64_t)num_blocks * stack_stride;

        // Capture DSpark auxiliary target hidden states at visits {5, 23, 41}
        if (dspark.loaded) {
            int aux_slot = (v == 5) ? 0 : (v == 23 ? 1 : (v == 41 ? 2 : -1));
            if (aux_slot >= 0) {
                CHECK_CUDA(cudaMemcpyAsync(
                    d_aux_states + (int64_t)aux_slot * stack_stride, partial,
                    (size_t)T * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
            }
        }

        // 1. Attention inputs (q_in, k_in, v_in)
        mark_start();
        const __nv_bfloat16* q_in = nullptr;
        const __nv_bfloat16* k_in = nullptr;
        const __nv_bfloat16* v_in = nullptr;

        if (looped) {
            // Compute 4 cross-pass messages from bank[0] (stack) + bank[1] (last)
            const float* mq = message_queries + (int64_t)(i - cfg.loop_core_start) * 4 * D;
            launch_attn_res(
                d_bank, 1, stack_stride,
                d_bank + stack_stride, nullptr,
                mq, 4, nullptr,
                nullptr, d_messages, T, D, cfg.rms_norm_eps, stream);

            // Compute fp32 dest(s) for attention
            const float* dest_q = nullptr;
            const float* dest_k = nullptr;
            const float* dest_v = nullptr;
            if (i == cfg.loop_core_start) {
                dest_q = dest_k = dest_v = d_prelude_dest;
            } else {
                int nq = lw.is_local ? 1 : 3;
                launch_attn_res(
                    d_stack, num_blocks, stack_stride,
                    partial, nullptr,
                    attn_queries[i - 1], nq, nullptr,
                    nullptr, d_fp32_mix, T, D, cfg.rms_norm_eps, stream);
                dest_q = d_fp32_mix;
                dest_k = lw.is_local ? d_fp32_mix : (d_fp32_mix + (int64_t)T * D);
                dest_v = lw.is_local ? d_fp32_mix : (d_fp32_mix + (int64_t)2 * T * D);
            }
            const __nv_bfloat16* nw_q = lw.attn_norm;
            const __nv_bfloat16* nw_k = lw.is_local ? lw.attn_norm : (lw.attn_norm + D);
            const __nv_bfloat16* nw_v = lw.is_local ? lw.attn_norm : (lw.attn_norm + 2 * D);

            run_controller(lw, 0, dest_q, d_messages + (int64_t)0 * T * D, nw_q, d_attn_in + (int64_t)0 * T * D, T);
            run_controller(lw, 1, dest_k, d_messages + (int64_t)1 * T * D, nw_k, d_attn_in + (int64_t)1 * T * D, T);
            run_controller(lw, 2, dest_v, d_messages + (int64_t)2 * T * D, nw_v, d_attn_in + (int64_t)2 * T * D, T);
            q_in = d_attn_in + (int64_t)0 * T * D;
            k_in = d_attn_in + (int64_t)1 * T * D;
            v_in = d_attn_in + (int64_t)2 * T * D;
        } else if (i == 0) {
            launch_rmsnorm(partial, lw.attn_norm, d_attn_in, T, D, cfg.rms_norm_eps, stream);
            q_in = k_in = v_in = d_attn_in;
        } else {
            int nq = lw.is_local ? 1 : 3;
            launch_attn_res(
                d_stack, num_blocks, stack_stride,
                partial, nullptr,
                attn_queries[i - 1], nq, lw.attn_norm,
                d_attn_in, nullptr, T, D, cfg.rms_norm_eps, stream);
            q_in = d_attn_in;
            k_in = lw.is_local ? d_attn_in : (d_attn_in + (int64_t)T * D);
            v_in = lw.is_local ? d_attn_in : (d_attn_in + (int64_t)2 * T * D);
        }
        mark_end(prof_attn_res_ctrl_ms);

        // 2. Fused Attention Q/R/Gate/Sink + K/V projections (FP8 W8A16 for T<=8, BF16 Tensor Core for prefill)
        mark_start();
        int qrgs_dim = 2 * q_dim + r_dim + cfg.num_attention_heads;
        if (T <= 8) {
            launch_fp8_small_gemm(q_in, lw.w_qrgs_fp8, lw.w_qrgs_sc, d_qrgs, nullptr, T, qrgs_dim, D, stream);
            launch_split_qrgs(d_qrgs, d_Q, d_R, d_gate, d_sinks, T, q_dim, r_dim, cfg.num_attention_heads, stream);
            if (k_in == v_in) {
                launch_fp8_small_gemm(k_in, lw.w_kv_fp8, lw.w_kv_sc, d_kv_out, nullptr, T, 2 * kv_dim, D, stream);
                launch_split_kv(d_kv_out, d_K, d_V, T, kv_dim, stream);
            } else {
                launch_fp8_small_gemm(k_in, lw.w_kv_fp8, lw.w_kv_sc, d_K, nullptr, T, kv_dim, D, stream);
                launch_fp8_small_gemm(v_in, lw.w_kv_fp8 + (int64_t)kv_dim * D, lw.w_kv_sc + kv_dim, d_V, nullptr, T, kv_dim, D, stream);
            }
        } else {
            linear_bf16(q_in, lw.w_qrgs, d_qrgs, T, qrgs_dim, D);
            launch_split_qrgs(d_qrgs, d_Q, d_R, d_gate, d_sinks, T, q_dim, r_dim, cfg.num_attention_heads, stream);
            if (k_in == v_in) {
                linear_bf16(k_in, lw.w_kv, d_kv_out, T, 2 * kv_dim, D);
                launch_split_kv(d_kv_out, d_K, d_V, T, kv_dim, stream);
            } else {
                linear_bf16(k_in, lw.wk, d_K, T, kv_dim, D);
                linear_bf16(v_in, lw.wv, d_V, T, kv_dim, D);
            }
        }
        mark_end(prof_dense_proj_ms);

        // 3. K/V short convs
        mark_start();
        int64_t kv_slot_off = (int64_t)slot_id * vc.cap * kv_dim;
        __nv_bfloat16* slot_k_cache = vc.k_cache ? (vc.k_cache + kv_slot_off) : nullptr;
        __nv_bfloat16* slot_v_cache = vc.v_cache ? (vc.v_cache + kv_slot_off) : nullptr;
        uint8_t* slot_k_fp8 = vc.k_cache_fp8;
        uint8_t* slot_v_fp8 = vc.v_cache_fp8;
        float* slot_k_sc = vc.k_scale_fp8;
        float* slot_v_sc = vc.v_scale_fp8;
        const int32_t* slot_bt = (!lw.is_local) ? (d_block_tables + (int64_t)slot_id * max_blocks_per_seq) : nullptr;
        __nv_bfloat16* slot_k_cs = vc.k_conv_state + (int64_t)slot_id * 3 * kv_dim;
        __nv_bfloat16* slot_v_cs = vc.v_conv_state + (int64_t)slot_id * 3 * kv_dim;
        __nv_bfloat16* slot_attn_cs = vc.attn_conv_state + (int64_t)slot_id * 3 * D;
        __nv_bfloat16* slot_mlp_cs = vc.mlp_conv_state + (int64_t)slot_id * 3 * D;

        __nv_bfloat16* k_snap = save_conv_snaps ? vc.k_conv_snap : nullptr;
        __nv_bfloat16* v_snap = save_conv_snaps ? vc.v_conv_snap : nullptr;
        if (lw.is_local) {
            launch_sconv_static(d_K, lw.k_sconv_w, slot_k_cs, d_K_conv, T, kv_dim, stream, k_snap);
            launch_sconv_static(d_V, lw.v_sconv_w, slot_v_cs, d_V_conv, T, kv_dim, stream, v_snap);
        } else {
            int dyn_dim = cfg.num_key_value_heads * 4;
            linear_bf16(k_in, lw.k_sconv_proj, d_dyn_w, T, dyn_dim, D);
            launch_sconv_dynamic(d_K, lw.k_sconv_w, d_dyn_w, slot_k_cs, d_K_conv,
                                 T, cfg.num_key_value_heads, cfg.head_dim, stream, k_snap);
            linear_bf16(v_in, lw.v_sconv_proj, d_dyn_w, T, dyn_dim, D);
            launch_sconv_dynamic(d_V, lw.v_sconv_w, d_dyn_w, slot_v_cs, d_V_conv,
                                 T, cfg.num_key_value_heads, cfg.head_dim, stream, v_snap);
        }

        // 4. Q/K/V/R clip/norm/tau + KV cache update
        launch_attn_prep(
            d_Q, d_K_conv, d_V_conv, d_R,
            lw.q_clip_scale, lw.rel_clip_scale,
            slot_k_cache, slot_v_cache,
            start_pos, dev_pos, T, vc.cap, lw.is_local,
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.d_rel,
            cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold,
            cfg.log_scaling_n_floor, cfg.log_scaling_alpha, cfg.rms_norm_eps, stream,
            slot_k_fp8, slot_v_fp8, slot_k_sc, slot_v_sc, slot_bt);

        // 5. Relative position bias: rel_logits[T*24, 1024] = R[T*24, 16] @ rel_proj[16, 1024]
        {
            float alpha = 1.0f, beta = 0.0f;
            CHECK_CUBLAS(cublasGemmEx(
                cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                cfg.rel_extent, T * cfg.num_attention_heads, cfg.d_rel,
                &alpha,
                lw.rel_proj, CUDA_R_16BF, cfg.rel_extent,
                d_R, CUDA_R_16BF, cfg.d_rel,
                &beta,
                d_rel_logits, CUDA_R_16BF, cfg.rel_extent,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }

        // 6. Attention + sigmoid(gate) + o_proj + post-attn sconv
        launch_attention(
            d_Q, slot_k_cache, slot_v_cache, d_rel_logits, d_gate, d_sinks, d_attn_raw,
            d_split_scratch, start_pos, dev_pos, T, vc.cap, lw.is_local, cfg.sliding_window_size,
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.rel_extent, stream,
            slot_k_fp8, slot_v_fp8, slot_k_sc, slot_v_sc, slot_bt);
        mark_end(prof_attn_sconv_ms);

        mark_start();
        if (T <= 8) launch_fp8_small_gemm(d_attn_raw, lw.wo_fp8, lw.wo_sc, d_attn_proj, nullptr, T, D, q_dim, stream);
        else linear_bf16(d_attn_raw, lw.wo, d_attn_proj, T, D, q_dim);
        mark_end(prof_dense_proj_ms);

        mark_start();
        launch_sconv_static(d_attn_proj, lw.attn_sconv_w, slot_attn_cs, d_attn_out, T, D, stream,
                            save_conv_snaps ? vc.attn_conv_snap : nullptr);
        mark_end(prof_attn_sconv_ms);

        // 7. MLP input AttnRes (+ loop controller if looped)
        mark_start();
        bool first = (i % cfg.n_layers_per_attn_res_block == 0);
        int mlp_n_stack = num_blocks + (first ? 1 : 0);
        const __nv_bfloat16* mlp_last = first ? d_attn_out : partial;
        const __nv_bfloat16* mlp_add = first ? nullptr : d_attn_out;

        if (looped) {
            launch_attn_res(
                d_stack, mlp_n_stack, stack_stride,
                mlp_last, mlp_add,
                mlp_queries[i], 1, nullptr,
                nullptr, d_fp32_mix, T, D, cfg.rms_norm_eps, stream);
            run_controller(lw, 3, d_fp32_mix, d_messages + (int64_t)3 * T * D, lw.mlp_norm, d_mlp_in, T);
        } else {
            launch_attn_res(
                d_stack, mlp_n_stack, stack_stride,
                mlp_last, mlp_add,
                mlp_queries[i], 1, lw.mlp_norm,
                d_mlp_in, nullptr, T, D, cfg.rms_norm_eps, stream);
        }
        mark_end(prof_attn_res_ctrl_ms);

        // 8. MoE router + experts + post-mlp sconv
        mark_start();
        launch_moe_router(
            d_mlp_in, lw.gate_weight, lw.gate_bias,
            d_router_logits, d_topk_ids, d_topk_weights,
            T, D, cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.route_scale, stream);
        mark_end(prof_router_ms);

        mark_start();
        if (T >= moe_group_thresh) {
            run_moe_grouped_prefill(lw, d_mlp_in, d_mlp_raw, T);
            mark_end(lw.is_mxfp8 ? prof_moe_mxfp8_ms : prof_moe_nvfp4_ms);
        } else if (T >= 2 && T <= 8) {
            if (lw.is_mxfp8) {
                launch_moe_small_batch_mxfp8(
                    d_mlp_in, d_topk_ids, d_topk_weights,
                    lw.w13_weight, lw.w13_scale, lw.w2_weight, lw.w2_scale,
                    d_moe_uniq, d_moe_act, d_mlp_raw,
                    T, D, cfg.intermediate_size, cfg.num_experts_per_tok,
                    cfg.mlp_gate_softcap, cfg.mlp_up_softcap, stream);
                mark_end(prof_moe_mxfp8_ms);
            } else {
                launch_moe_small_batch_nvfp4(
                    d_mlp_in, d_topk_ids, d_topk_weights,
                    lw.w13_weight, lw.w13_scale, lw.w13_amax, lw.w13_denom,
                    lw.w2_weight, lw.w2_scale, lw.w2_amax, lw.w2_denom,
                    d_moe_uniq, d_moe_act, d_mlp_raw,
                    T, D, cfg.intermediate_size, cfg.num_experts_per_tok,
                    cfg.mlp_gate_softcap, cfg.mlp_up_softcap, stream);
                mark_end(prof_moe_nvfp4_ms);
            }
        } else if (lw.is_mxfp8) {
            launch_moe_mxfp8(
                d_mlp_in, d_topk_ids, d_topk_weights,
                lw.w13_weight, lw.w13_scale, lw.w2_weight, lw.w2_scale,
                d_moe_act, d_mlp_raw,
                T, D, cfg.intermediate_size, cfg.num_experts_per_tok,
                cfg.mlp_gate_softcap, cfg.mlp_up_softcap, stream);
            mark_end(prof_moe_mxfp8_ms);
        } else {
            launch_moe_nvfp4(
                d_mlp_in, d_topk_ids, d_topk_weights,
                lw.w13_weight, lw.w13_scale, lw.w13_amax, lw.w13_denom,
                lw.w2_weight, lw.w2_scale, lw.w2_amax, lw.w2_denom,
                d_moe_act, d_mlp_raw,
                T, D, cfg.intermediate_size, cfg.num_experts_per_tok,
                cfg.mlp_gate_softcap, cfg.mlp_up_softcap, stream);
            mark_end(prof_moe_nvfp4_ms);
        }
        mark_start();
        launch_sconv_static(d_mlp_raw, lw.mlp_sconv_w, slot_mlp_cs, d_mlp_out, T, D, stream,
                            save_conv_snaps ? vc.mlp_conv_snap : nullptr);
        mark_end(prof_attn_sconv_ms);

        // 9. Residual update
        if (first) {
            num_blocks += 1;
            __nv_bfloat16* new_part = d_stack + (int64_t)num_blocks * stack_stride;
            launch_add_bf16(d_attn_out, d_mlp_out, new_part, T * D, stream);
        } else {
            // partial = partial + attn_out + mlp_out
            launch_add_bf16(partial, d_attn_out, d_attn_proj, T * D, stream);
            launch_add_bf16(d_attn_proj, d_mlp_out, partial, T * D, stream);
        }
    }

    // Final AttnRes + norm -> d_mlp_in[T, D]
    mark_start();
    __nv_bfloat16* final_part = d_stack + (int64_t)num_blocks * stack_stride;
    launch_attn_res(
        d_stack, num_blocks, stack_stride,
        final_part, nullptr,
        output_query, 1, final_norm_w,
        d_mlp_in, nullptr, T, D, cfg.rms_norm_eps, stream);

    if (dspark.loaded) {
        // Visit 48 aux feature = final norm output (d_mlp_in)
        CHECK_CUDA(cudaMemcpyAsync(
            d_aux_states + (int64_t)3 * stack_stride, d_mlp_in,
            (size_t)T * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
    }

    // LM head (per-row FP8 W8A16: cuts 926 MB -> 463 MB, 2x faster!)
    if (!save_conv_snaps) {
        const __nv_bfloat16* last_h = d_mlp_in + (int64_t)(T - 1) * D;
        launch_fp8_small_gemm(last_h, unembed_fp8, unembed_sc, nullptr, d_logits, 1, cfg.unpadded_vocab_size, D, stream);
    } else if (T <= 8) {
        launch_fp8_small_gemm(d_mlp_in, unembed_fp8, unembed_sc, nullptr, d_logits, T, cfg.unpadded_vocab_size, D, stream);
    } else {
        if (T <= 8) launch_fp8_small_gemm(d_mlp_in, unembed_fp8, unembed_sc, nullptr, d_logits, T, cfg.unpadded_vocab_size, D, stream);
    else linear_bf16_to_fp32(d_mlp_in, unembed_w, d_logits, T, cfg.unpadded_vocab_size, D);
    }
    mark_end(prof_lm_head_ms);

    if (profile_stages) {
        prof_steps += 1;
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
    }
}

int32_t Inkling2Model::forward_and_greedy(const int32_t* h_tokens, int T, int start_pos, int slot_id) {
    if (T == 1 && slot_id == 0 && use_cuda_graph && !profile_stages) {
        CHECK_CUDA(cudaMemcpyAsync(d_tokens, h_tokens, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_pos, &start_pos, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        if (!graph_captured) {
            CHECK_CUDA(cudaStreamSynchronize(stream));
            CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            forward_internal(nullptr, 1, 0, 0);
            launch_softcap_argmax(d_logits, d_out_token, cfg.unpadded_vocab_size, cfg.final_logit_softcapping, stream);
            CHECK_CUDA(cudaStreamEndCapture(stream, &decode_graph));
            CHECK_CUDA(cudaGraphInstantiate(&decode_graph_exec, decode_graph, nullptr, nullptr, 0));
            graph_captured = true;
            fprintf(stderr, "[warpgrid] Captured T=1 decode CUDA graph!\n");
        }
        CHECK_CUDA(cudaGraphLaunch(decode_graph_exec, stream));
        int32_t next_tok = 0;
        CHECK_CUDA(cudaMemcpyAsync(&next_tok, d_out_token, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        return next_tok;
    }
    forward_internal(h_tokens, T, start_pos, slot_id);
    launch_softcap_argmax(d_logits, d_out_token, cfg.unpadded_vocab_size, cfg.final_logit_softcapping, stream);
    int32_t next_tok = 0;
    CHECK_CUDA(cudaMemcpyAsync(&next_tok, d_out_token, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return next_tok;
}

void Inkling2Model::forward_batch_decode_internal(int B) {
    int D = cfg.hidden_size;
    int stack_stride = max_batch_tokens * D;
    int q_dim = cfg.num_attention_heads * cfg.head_dim;
    int kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    int r_dim = cfg.num_attention_heads * cfg.d_rel;

    launch_embed_norm(d_tokens, embed_w, embed_norm_w, d_stack, B, D, cfg.rms_norm_eps, stream);

    int num_blocks = 0;
    int prelude_blocks = 0;
    bool have_prelude_dest = false;

    for (size_t v = 0; v < visits.size(); ++v) {
        int i = visits[v].first;
        int pass_idx = visits[v].second;
        bool looped = (pass_idx > 0);
        const auto& lw = layers[i];
        auto& vc = caches[v];

        if (i == cfg.loop_core_start && !looped) {
            prelude_blocks = num_blocks;
            CHECK_CUDA(cudaMemcpyAsync(d_prelude_part, d_stack + (int64_t)num_blocks * stack_stride,
                (size_t)B * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
            have_prelude_dest = false;
        } else if (i == cfg.loop_core_start && looped) {
            CHECK_CUDA(cudaMemcpyAsync(d_bank, d_stack + (int64_t)(prelude_blocks + 1) * stack_stride,
                (size_t)B * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
            CHECK_CUDA(cudaMemcpyAsync(d_bank + stack_stride, d_stack + (int64_t)(prelude_blocks + 2) * stack_stride,
                (size_t)B * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
            num_blocks = prelude_blocks;
            CHECK_CUDA(cudaMemcpyAsync(d_stack + (int64_t)num_blocks * stack_stride, d_prelude_part,
                (size_t)B * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
            if (!have_prelude_dest) {
                launch_attn_res(d_stack, num_blocks, stack_stride, d_prelude_part, nullptr,
                    attn_queries[i - 1], 1, nullptr, nullptr, d_prelude_dest, B, D, cfg.rms_norm_eps, stream);
                have_prelude_dest = true;
            }
        }

        __nv_bfloat16* partial = d_stack + (int64_t)num_blocks * stack_stride;
        const __nv_bfloat16 *q_in, *k_in, *v_in;
        if (looped) {
            const float* mq = message_queries + (int64_t)(i - cfg.loop_core_start) * 4 * D;
            launch_attn_res(d_bank, 1, stack_stride, d_bank + stack_stride, nullptr,
                mq, 4, nullptr, nullptr, d_messages, B, D, cfg.rms_norm_eps, stream);
            const float *dest_q, *dest_k, *dest_v;
            if (i == cfg.loop_core_start) {
                dest_q = dest_k = dest_v = d_prelude_dest;
            } else {
                int nq = lw.is_local ? 1 : 3;
                launch_attn_res(d_stack, num_blocks, stack_stride, partial, nullptr,
                    attn_queries[i - 1], nq, nullptr, nullptr, d_fp32_mix, B, D, cfg.rms_norm_eps, stream);
                dest_q = d_fp32_mix;
                dest_k = lw.is_local ? d_fp32_mix : (d_fp32_mix + (int64_t)B * D);
                dest_v = lw.is_local ? d_fp32_mix : (d_fp32_mix + (int64_t)2 * B * D);
            }
            const __nv_bfloat16* nw_q = lw.attn_norm;
            const __nv_bfloat16* nw_k = lw.is_local ? lw.attn_norm : (lw.attn_norm + D);
            const __nv_bfloat16* nw_v = lw.is_local ? lw.attn_norm : (lw.attn_norm + 2 * D);
            run_controller(lw, 0, dest_q, d_messages + (int64_t)0 * B * D, nw_q, d_attn_in + (int64_t)0 * B * D, B);
            run_controller(lw, 1, dest_k, d_messages + (int64_t)1 * B * D, nw_k, d_attn_in + (int64_t)1 * B * D, B);
            run_controller(lw, 2, dest_v, d_messages + (int64_t)2 * B * D, nw_v, d_attn_in + (int64_t)2 * B * D, B);
            q_in = d_attn_in; k_in = d_attn_in + (int64_t)B * D; v_in = d_attn_in + (int64_t)2 * B * D;
        } else if (i == 0) {
            launch_rmsnorm(partial, lw.attn_norm, d_attn_in, B, D, cfg.rms_norm_eps, stream);
            q_in = k_in = v_in = d_attn_in;
        } else {
            int nq = lw.is_local ? 1 : 3;
            launch_attn_res(d_stack, num_blocks, stack_stride, partial, nullptr,
                attn_queries[i - 1], nq, lw.attn_norm, d_attn_in, nullptr, B, D, cfg.rms_norm_eps, stream);
            q_in = d_attn_in;
            k_in = lw.is_local ? d_attn_in : (d_attn_in + (int64_t)B * D);
            v_in = lw.is_local ? d_attn_in : (d_attn_in + (int64_t)2 * B * D);
        }

        int qrgs_dim = 2 * q_dim + r_dim + cfg.num_attention_heads;
        launch_fp8_small_gemm(q_in, lw.w_qrgs_fp8, lw.w_qrgs_sc, d_qrgs, nullptr, B, qrgs_dim, D, stream);
        launch_split_qrgs(d_qrgs, d_Q, d_R, d_gate, d_sinks, B, q_dim, r_dim, cfg.num_attention_heads, stream);
        if (k_in == v_in) {
            launch_fp8_small_gemm(k_in, lw.w_kv_fp8, lw.w_kv_sc, d_kv_out, nullptr, B, 2 * kv_dim, D, stream);
            launch_split_kv(d_kv_out, d_K, d_V, B, kv_dim, stream);
        } else {
            launch_fp8_small_gemm(k_in, lw.w_kv_fp8, lw.w_kv_sc, d_K, nullptr, B, kv_dim, D, stream);
            launch_fp8_small_gemm(v_in, lw.w_kv_fp8 + (int64_t)kv_dim * D, lw.w_kv_sc + kv_dim, d_V, nullptr, B, kv_dim, D, stream);
        }

        if (lw.is_local) {
            launch_sconv_static_batched(d_K, lw.k_sconv_w, vc.k_conv_state, d_batch_slots, d_K_conv, B, kv_dim, stream);
            launch_sconv_static_batched(d_V, lw.v_sconv_w, vc.v_conv_state, d_batch_slots, d_V_conv, B, kv_dim, stream);
        } else {
            int dyn_dim = cfg.num_key_value_heads * 4;
            linear_bf16(k_in, lw.k_sconv_proj, d_dyn_w, B, dyn_dim, D);
            launch_sconv_dynamic_batched(d_K, lw.k_sconv_w, d_dyn_w, vc.k_conv_state, d_batch_slots, d_K_conv,
                                         B, cfg.num_key_value_heads, cfg.head_dim, stream);
            linear_bf16(v_in, lw.v_sconv_proj, d_dyn_w, B, dyn_dim, D);
            launch_sconv_dynamic_batched(d_V, lw.v_sconv_w, d_dyn_w, vc.v_conv_state, d_batch_slots, d_V_conv,
                                         B, cfg.num_key_value_heads, cfg.head_dim, stream);
        }

        launch_attn_prep_batched(
            d_Q, d_K_conv, d_V_conv, d_R, lw.q_clip_scale, lw.rel_clip_scale,
            vc.k_cache, vc.v_cache, d_batch_slots, d_batch_pos,
            B, vc.cap, lw.is_local, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.d_rel,
            cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold,
            cfg.log_scaling_n_floor, cfg.log_scaling_alpha, cfg.rms_norm_eps, stream,
            vc.k_cache_fp8, vc.v_cache_fp8, vc.k_scale_fp8, vc.v_scale_fp8,
            d_block_tables, max_blocks_per_seq);

        {
            float alpha = 1.0f, beta = 0.0f;
            CHECK_CUBLAS(cublasGemmEx(
                cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                cfg.rel_extent, B * cfg.num_attention_heads, cfg.d_rel,
                &alpha, lw.rel_proj, CUDA_R_16BF, cfg.rel_extent,
                d_R, CUDA_R_16BF, cfg.d_rel, &beta,
                d_rel_logits, CUDA_R_16BF, cfg.rel_extent, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }

        launch_attention_batched(
            d_Q, vc.k_cache, vc.v_cache, d_rel_logits, d_gate, d_sinks, d_attn_raw,
            d_batch_slots, d_batch_pos, B, vc.cap, lw.is_local, cfg.sliding_window_size,
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.rel_extent, stream,
            vc.k_cache_fp8, vc.v_cache_fp8, vc.k_scale_fp8, vc.v_scale_fp8,
            d_block_tables, max_blocks_per_seq);
        launch_fp8_small_gemm(d_attn_raw, lw.wo_fp8, lw.wo_sc, d_attn_proj, nullptr, B, D, q_dim, stream);
        launch_sconv_static_batched(d_attn_proj, lw.attn_sconv_w, vc.attn_conv_state, d_batch_slots, d_attn_out, B, D, stream);

        bool first = (i % cfg.n_layers_per_attn_res_block == 0);
        int mlp_n_stack = num_blocks + (first ? 1 : 0);
        const __nv_bfloat16* mlp_last = first ? d_attn_out : partial;
        const __nv_bfloat16* mlp_add = first ? nullptr : d_attn_out;
        if (looped) {
            launch_attn_res(d_stack, mlp_n_stack, stack_stride, mlp_last, mlp_add,
                mlp_queries[i], 1, nullptr, nullptr, d_fp32_mix, B, D, cfg.rms_norm_eps, stream);
            run_controller(lw, 3, d_fp32_mix, d_messages + (int64_t)3 * B * D, lw.mlp_norm, d_mlp_in, B);
        } else {
            launch_attn_res(d_stack, mlp_n_stack, stack_stride, mlp_last, mlp_add,
                mlp_queries[i], 1, lw.mlp_norm, d_mlp_in, nullptr, B, D, cfg.rms_norm_eps, stream);
        }

        launch_moe_router(d_mlp_in, lw.gate_weight, lw.gate_bias, d_router_logits, d_topk_ids, d_topk_weights,
                          B, D, cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.route_scale, stream);
        if (B == 1) {
            if (lw.is_mxfp8) {
                launch_moe_mxfp8(d_mlp_in, d_topk_ids, d_topk_weights,
                    lw.w13_weight, lw.w13_scale, lw.w2_weight, lw.w2_scale,
                    d_moe_act, d_mlp_raw, 1, D, cfg.intermediate_size, cfg.num_experts_per_tok,
                    cfg.mlp_gate_softcap, cfg.mlp_up_softcap, stream);
            } else {
                launch_moe_nvfp4(d_mlp_in, d_topk_ids, d_topk_weights,
                    lw.w13_weight, lw.w13_scale, lw.w13_amax, lw.w13_denom,
                    lw.w2_weight, lw.w2_scale, lw.w2_amax, lw.w2_denom,
                    d_moe_act, d_mlp_raw, 1, D, cfg.intermediate_size, cfg.num_experts_per_tok,
                    cfg.mlp_gate_softcap, cfg.mlp_up_softcap, stream);
            }
        } else if (lw.is_mxfp8) {
            launch_moe_small_batch_mxfp8(d_mlp_in, d_topk_ids, d_topk_weights,
                lw.w13_weight, lw.w13_scale, lw.w2_weight, lw.w2_scale,
                d_moe_uniq, d_moe_act, d_mlp_raw, B, D, cfg.intermediate_size, cfg.num_experts_per_tok,
                cfg.mlp_gate_softcap, cfg.mlp_up_softcap, stream);
        } else {
            launch_moe_small_batch_nvfp4(d_mlp_in, d_topk_ids, d_topk_weights,
                lw.w13_weight, lw.w13_scale, lw.w13_amax, lw.w13_denom,
                lw.w2_weight, lw.w2_scale, lw.w2_amax, lw.w2_denom,
                d_moe_uniq, d_moe_act, d_mlp_raw, B, D, cfg.intermediate_size, cfg.num_experts_per_tok,
                cfg.mlp_gate_softcap, cfg.mlp_up_softcap, stream);
        }
        launch_sconv_static_batched(d_mlp_raw, lw.mlp_sconv_w, vc.mlp_conv_state, d_batch_slots, d_mlp_out, B, D, stream);

        if (first) {
            num_blocks += 1;
            launch_add_bf16(d_attn_out, d_mlp_out, d_stack + (int64_t)num_blocks * stack_stride, B * D, stream);
        } else {
            launch_add_bf16(partial, d_attn_out, d_attn_proj, B * D, stream);
            launch_add_bf16(d_attn_proj, d_mlp_out, partial, B * D, stream);
        }
    }

    launch_attn_res(d_stack, num_blocks, stack_stride, d_stack + (int64_t)num_blocks * stack_stride, nullptr,
        output_query, 1, final_norm_w, d_mlp_in, nullptr, B, D, cfg.rms_norm_eps, stream);
    launch_fp8_small_gemm(d_mlp_in, unembed_fp8, unembed_sc, nullptr, d_logits, B, cfg.unpadded_vocab_size, D, stream);
    launch_batched_argmax(d_logits, d_out_token, B, cfg.unpadded_vocab_size, stream);
}

void Inkling2Model::forward_batch_decode_greedy(
    const int32_t* h_tokens, const int32_t* h_slots, const int32_t* h_positions,
    int B, int32_t* h_out_tokens)
{
    CHECK_CUDA(cudaMemcpyAsync(d_tokens, h_tokens, B * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_batch_slots, h_slots, B * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_batch_pos, h_positions, B * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    if (use_cuda_graph && B >= 1 && B <= 8) {
        if (!batch_graph_execs[B]) {
            CHECK_CUDA(cudaStreamSynchronize(stream));
            CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            forward_batch_decode_internal(B);
            CHECK_CUDA(cudaStreamEndCapture(stream, &batch_graphs[B]));
            CHECK_CUDA(cudaGraphInstantiate(&batch_graph_execs[B], batch_graphs[B], nullptr, nullptr, 0));
            fprintf(stderr, "[warpgrid] Captured B=%d parallel decode CUDA graph!\n", B);
        }
        CHECK_CUDA(cudaGraphLaunch(batch_graph_execs[B], stream));
    } else {
        forward_batch_decode_internal(B);
    }
    CHECK_CUDA(cudaMemcpyAsync(h_out_tokens, d_out_token, B * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

void Inkling2Model::forward_verify_greedy(
    const int32_t* h_tokens, int T, int start_pos, int32_t* h_out_tokens)
{
    if (use_cuda_graph && !profile_stages && T >= 2 && T <= 8) {
        CHECK_CUDA(cudaMemcpyAsync(d_tokens, h_tokens, T * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_pos, &start_pos, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        if (!verify_graph_execs[T]) {
            CHECK_CUDA(cudaStreamSynchronize(stream));
            CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            save_conv_snaps = true;
            forward_internal(nullptr, T, 0);
            save_conv_snaps = false;
            launch_batched_argmax(d_logits, d_out_token, T, cfg.unpadded_vocab_size, stream);
            CHECK_CUDA(cudaStreamEndCapture(stream, &verify_graphs[T]));
            CHECK_CUDA(cudaGraphInstantiate(&verify_graph_execs[T], verify_graphs[T], nullptr, nullptr, 0));
            fprintf(stderr, "[warpgrid] Captured T=%d verify CUDA graph!\n", T);
        }
        CHECK_CUDA(cudaGraphLaunch(verify_graph_execs[T], stream));
        CHECK_CUDA(cudaMemcpyAsync(h_out_tokens, d_out_token, T * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        return;
    }
    save_conv_snaps = true;
    forward_internal(h_tokens, T, start_pos);
    save_conv_snaps = false;
    launch_batched_argmax(d_logits, d_out_token, T, cfg.unpadded_vocab_size, stream);
    CHECK_CUDA(cudaMemcpyAsync(h_out_tokens, d_out_token, T * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

void Inkling2Model::forward_batch_verify_internal(int B, int L) {
    int T = B * L;
    int D = cfg.hidden_size;
    int stack_stride = max_batch_tokens * D;
    int q_dim = cfg.num_attention_heads * cfg.head_dim;
    int kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    int r_dim = cfg.num_attention_heads * cfg.d_rel;

    launch_embed_norm(d_tokens, embed_w, embed_norm_w, d_stack, T, D, cfg.rms_norm_eps, stream);

    int num_blocks = 0;
    int prelude_blocks = 0;
    bool have_prelude_dest = false;

    for (size_t v = 0; v < visits.size(); ++v) {
        int i = visits[v].first;
        int pass_idx = visits[v].second;
        bool looped = (pass_idx > 0);
        const auto& lw = layers[i];
        auto& vc = caches[v];

        if (i == cfg.loop_core_start && !looped) {
            prelude_blocks = num_blocks;
            CHECK_CUDA(cudaMemcpyAsync(d_prelude_part, d_stack + (int64_t)num_blocks * stack_stride,
                (size_t)T * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
            have_prelude_dest = false;
        } else if (i == cfg.loop_core_start && looped) {
            CHECK_CUDA(cudaMemcpyAsync(d_bank, d_stack + (int64_t)(prelude_blocks + 1) * stack_stride,
                (size_t)T * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
            CHECK_CUDA(cudaMemcpyAsync(d_bank + stack_stride, d_stack + (int64_t)(prelude_blocks + 2) * stack_stride,
                (size_t)T * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
            num_blocks = prelude_blocks;
            CHECK_CUDA(cudaMemcpyAsync(d_stack + (int64_t)num_blocks * stack_stride, d_prelude_part,
                (size_t)T * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
            if (!have_prelude_dest) {
                launch_attn_res(d_stack, num_blocks, stack_stride, d_prelude_part, nullptr,
                    attn_queries[i - 1], 1, nullptr, nullptr, d_prelude_dest, T, D, cfg.rms_norm_eps, stream);
                have_prelude_dest = true;
            }
        }

        __nv_bfloat16* partial = d_stack + (int64_t)num_blocks * stack_stride;

        // Capture DSpark target aux hidden states at visits {5, 23, 41}
        if (dspark.loaded) {
            int aux_tap = (v == 5) ? 0 : (v == 23 ? 1 : (v == 41 ? 2 : -1));
            if (aux_tap >= 0) {
                CHECK_CUDA(cudaMemcpyAsync(
                    d_aux_states + (int64_t)aux_tap * stack_stride, partial,
                    (size_t)T * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
            }
        }

        const __nv_bfloat16 *q_in, *k_in, *v_in;
        if (looped) {
            const float* mq = message_queries + (int64_t)(i - cfg.loop_core_start) * 4 * D;
            launch_attn_res(d_bank, 1, stack_stride, d_bank + stack_stride, nullptr,
                mq, 4, nullptr, nullptr, d_messages, T, D, cfg.rms_norm_eps, stream);
            const float *dest_q, *dest_k, *dest_v;
            if (i == cfg.loop_core_start) {
                dest_q = dest_k = dest_v = d_prelude_dest;
            } else {
                int nq = lw.is_local ? 1 : 3;
                launch_attn_res(d_stack, num_blocks, stack_stride, partial, nullptr,
                    attn_queries[i - 1], nq, nullptr, nullptr, d_fp32_mix, T, D, cfg.rms_norm_eps, stream);
                dest_q = d_fp32_mix;
                dest_k = lw.is_local ? d_fp32_mix : (d_fp32_mix + (int64_t)T * D);
                dest_v = lw.is_local ? d_fp32_mix : (d_fp32_mix + (int64_t)2 * T * D);
            }
            const __nv_bfloat16* nw_q = lw.attn_norm;
            const __nv_bfloat16* nw_k = lw.is_local ? lw.attn_norm : (lw.attn_norm + D);
            const __nv_bfloat16* nw_v = lw.is_local ? lw.attn_norm : (lw.attn_norm + 2 * D);
            run_controller(lw, 0, dest_q, d_messages + (int64_t)0 * T * D, nw_q, d_attn_in + (int64_t)0 * T * D, T);
            run_controller(lw, 1, dest_k, d_messages + (int64_t)1 * T * D, nw_k, d_attn_in + (int64_t)1 * T * D, T);
            run_controller(lw, 2, dest_v, d_messages + (int64_t)2 * T * D, nw_v, d_attn_in + (int64_t)2 * T * D, T);
            q_in = d_attn_in; k_in = d_attn_in + (int64_t)T * D; v_in = d_attn_in + (int64_t)2 * T * D;
        } else if (i == 0) {
            launch_rmsnorm(partial, lw.attn_norm, d_attn_in, T, D, cfg.rms_norm_eps, stream);
            q_in = k_in = v_in = d_attn_in;
        } else {
            int nq = lw.is_local ? 1 : 3;
            launch_attn_res(d_stack, num_blocks, stack_stride, partial, nullptr,
                attn_queries[i - 1], nq, lw.attn_norm, d_attn_in, nullptr, T, D, cfg.rms_norm_eps, stream);
            q_in = d_attn_in;
            k_in = lw.is_local ? d_attn_in : (d_attn_in + (int64_t)T * D);
            v_in = lw.is_local ? d_attn_in : (d_attn_in + (int64_t)2 * T * D);
        }

        int qrgs_dim = 2 * q_dim + r_dim + cfg.num_attention_heads;
        if (T <= 8) {
            launch_fp8_small_gemm(q_in, lw.w_qrgs_fp8, lw.w_qrgs_sc, d_qrgs, nullptr, T, qrgs_dim, D, stream);
            launch_split_qrgs(d_qrgs, d_Q, d_R, d_gate, d_sinks, T, q_dim, r_dim, cfg.num_attention_heads, stream);
            if (k_in == v_in) {
                launch_fp8_small_gemm(k_in, lw.w_kv_fp8, lw.w_kv_sc, d_kv_out, nullptr, T, 2 * kv_dim, D, stream);
                launch_split_kv(d_kv_out, d_K, d_V, T, kv_dim, stream);
            } else {
                launch_fp8_small_gemm(k_in, lw.w_kv_fp8, lw.w_kv_sc, d_K, nullptr, T, kv_dim, D, stream);
                launch_fp8_small_gemm(v_in, lw.w_kv_fp8 + (int64_t)kv_dim * D, lw.w_kv_sc + kv_dim, d_V, nullptr, T, kv_dim, D, stream);
            }
        } else {
            linear_bf16(q_in, lw.w_qrgs, d_qrgs, T, qrgs_dim, D);
            launch_split_qrgs(d_qrgs, d_Q, d_R, d_gate, d_sinks, T, q_dim, r_dim, cfg.num_attention_heads, stream);
            if (k_in == v_in) {
                linear_bf16(k_in, lw.w_kv, d_kv_out, T, 2 * kv_dim, D);
                launch_split_kv(d_kv_out, d_K, d_V, T, kv_dim, stream);
            } else {
                linear_bf16(k_in, lw.wk, d_K, T, kv_dim, D);
                linear_bf16(v_in, lw.wv, d_V, T, kv_dim, D);
            }
        }

        if (lw.is_local) {
            launch_sconv_static_batch_verify(d_K, lw.k_sconv_w, vc.k_conv_state, vc.k_conv_snap, d_batch_slots, d_K_conv, B, L, kv_dim, stream);
            launch_sconv_static_batch_verify(d_V, lw.v_sconv_w, vc.v_conv_state, vc.v_conv_snap, d_batch_slots, d_V_conv, B, L, kv_dim, stream);
        } else {
            int dyn_dim = cfg.num_key_value_heads * 4;
            linear_bf16(k_in, lw.k_sconv_proj, d_dyn_w, T, dyn_dim, D);
            launch_sconv_dynamic_batch_verify(d_K, lw.k_sconv_w, d_dyn_w, vc.k_conv_state, vc.k_conv_snap, d_batch_slots, d_K_conv,
                                              B, L, cfg.num_key_value_heads, cfg.head_dim, stream);
            linear_bf16(v_in, lw.v_sconv_proj, d_dyn_w, T, dyn_dim, D);
            launch_sconv_dynamic_batch_verify(d_V, lw.v_sconv_w, d_dyn_w, vc.v_conv_state, vc.v_conv_snap, d_batch_slots, d_V_conv,
                                              B, L, cfg.num_key_value_heads, cfg.head_dim, stream);
        }

        launch_attn_prep_batch_verify(
            d_Q, d_K_conv, d_V_conv, d_R, lw.q_clip_scale, lw.rel_clip_scale,
            vc.k_cache, vc.v_cache, d_batch_slots, d_batch_pos,
            B, L, vc.cap, lw.is_local, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.d_rel,
            cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold,
            cfg.log_scaling_n_floor, cfg.log_scaling_alpha, cfg.rms_norm_eps, stream,
            vc.k_cache_fp8, vc.v_cache_fp8, vc.k_scale_fp8, vc.v_scale_fp8,
            d_block_tables, max_blocks_per_seq);

        {
            float alpha = 1.0f, beta = 0.0f;
            CHECK_CUBLAS(cublasGemmEx(
                cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                cfg.rel_extent, T * cfg.num_attention_heads, cfg.d_rel,
                &alpha, lw.rel_proj, CUDA_R_16BF, cfg.rel_extent,
                d_R, CUDA_R_16BF, cfg.d_rel, &beta,
                d_rel_logits, CUDA_R_16BF, cfg.rel_extent, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }

        launch_attention_batch_verify(
            d_Q, vc.k_cache, vc.v_cache, d_rel_logits, d_gate, d_sinks, d_attn_raw,
            d_batch_slots, d_batch_pos, B, L, vc.cap, lw.is_local, cfg.sliding_window_size,
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.rel_extent, stream,
            vc.k_cache_fp8, vc.v_cache_fp8, vc.k_scale_fp8, vc.v_scale_fp8, d_split_scratch,
            d_block_tables, max_blocks_per_seq);
        if (T <= 8) launch_fp8_small_gemm(d_attn_raw, lw.wo_fp8, lw.wo_sc, d_attn_proj, nullptr, T, D, q_dim, stream);
        else linear_bf16(d_attn_raw, lw.wo, d_attn_proj, T, D, q_dim);
        launch_sconv_static_batch_verify(d_attn_proj, lw.attn_sconv_w, vc.attn_conv_state, vc.attn_conv_snap, d_batch_slots, d_attn_out, B, L, D, stream);

        bool first = (i % cfg.n_layers_per_attn_res_block == 0);
        int mlp_n_stack = num_blocks + (first ? 1 : 0);
        const __nv_bfloat16* mlp_last = first ? d_attn_out : partial;
        const __nv_bfloat16* mlp_add = first ? nullptr : d_attn_out;
        if (looped) {
            launch_attn_res(d_stack, mlp_n_stack, stack_stride, mlp_last, mlp_add,
                mlp_queries[i], 1, nullptr, nullptr, d_fp32_mix, T, D, cfg.rms_norm_eps, stream);
            run_controller(lw, 3, d_fp32_mix, d_messages + (int64_t)3 * T * D, lw.mlp_norm, d_mlp_in, T);
        } else {
            launch_attn_res(d_stack, mlp_n_stack, stack_stride, mlp_last, mlp_add,
                mlp_queries[i], 1, lw.mlp_norm, d_mlp_in, nullptr, T, D, cfg.rms_norm_eps, stream);
        }

        launch_moe_router(d_mlp_in, lw.gate_weight, lw.gate_bias, d_router_logits, d_topk_ids, d_topk_weights,
                          T, D, cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.route_scale, stream);
        if (lw.is_mxfp8) {
            launch_moe_small_batch_mxfp8(d_mlp_in, d_topk_ids, d_topk_weights,
                lw.w13_weight, lw.w13_scale, lw.w2_weight, lw.w2_scale,
                d_moe_uniq, d_moe_act, d_mlp_raw, T, D, cfg.intermediate_size, cfg.num_experts_per_tok,
                cfg.mlp_gate_softcap, cfg.mlp_up_softcap, stream);
        } else {
            launch_moe_small_batch_nvfp4(d_mlp_in, d_topk_ids, d_topk_weights,
                lw.w13_weight, lw.w13_scale, lw.w13_amax, lw.w13_denom,
                lw.w2_weight, lw.w2_scale, lw.w2_amax, lw.w2_denom,
                d_moe_uniq, d_moe_act, d_mlp_raw, T, D, cfg.intermediate_size, cfg.num_experts_per_tok,
                cfg.mlp_gate_softcap, cfg.mlp_up_softcap, stream);
        }
        launch_sconv_static_batch_verify(d_mlp_raw, lw.mlp_sconv_w, vc.mlp_conv_state, vc.mlp_conv_snap, d_batch_slots, d_mlp_out, B, L, D, stream);

        if (first) {
            num_blocks += 1;
            launch_add_bf16(d_attn_out, d_mlp_out, d_stack + (int64_t)num_blocks * stack_stride, T * D, stream);
        } else {
            launch_add_bf16(partial, d_attn_out, d_attn_proj, T * D, stream);
            launch_add_bf16(d_attn_proj, d_mlp_out, partial, T * D, stream);
        }
    }

    launch_attn_res(d_stack, num_blocks, stack_stride, d_stack + (int64_t)num_blocks * stack_stride, nullptr,
        output_query, 1, final_norm_w, d_mlp_in, nullptr, T, D, cfg.rms_norm_eps, stream);
    if (dspark.loaded) {
        CHECK_CUDA(cudaMemcpyAsync(
            d_aux_states + (int64_t)3 * stack_stride, d_mlp_in,
            (size_t)T * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
    }
    if (T <= 8) launch_fp8_small_gemm(d_mlp_in, unembed_fp8, unembed_sc, nullptr, d_logits, T, cfg.unpadded_vocab_size, D, stream);
    else linear_bf16_to_fp32(d_mlp_in, unembed_w, d_logits, T, cfg.unpadded_vocab_size, D);
    launch_batched_argmax(d_logits, d_out_token, T, cfg.unpadded_vocab_size, stream);
}

void Inkling2Model::forward_batch_verify_greedy(
    const int32_t* h_tokens, const int32_t* h_slots, const int32_t* h_positions,
    int B, int L, int32_t* h_out_tokens)
{
    int T = B * L;
    CHECK_CUDA(cudaMemcpyAsync(d_tokens, h_tokens, T * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_batch_slots, h_slots, B * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_batch_pos, h_positions, B * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    if (use_cuda_graph && B >= 1 && B <= 8 && L >= 2 && L <= 5) {
        if (!batch_verify_graph_execs[B][L]) {
            CHECK_CUDA(cudaStreamSynchronize(stream));
            CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            forward_batch_verify_internal(B, L);
            CHECK_CUDA(cudaStreamEndCapture(stream, &batch_verify_graphs[B][L]));
            CHECK_CUDA(cudaGraphInstantiate(&batch_verify_graph_execs[B][L], batch_verify_graphs[B][L], nullptr, nullptr, 0));
            fprintf(stderr, "[warpgrid] Captured B=%d L=%d (T=%d) batched verify CUDA graph!\n", B, L, T);
        }
        CHECK_CUDA(cudaGraphLaunch(batch_verify_graph_execs[B][L], stream));
    } else {
        forward_batch_verify_internal(B, L);
    }
    CHECK_CUDA(cudaMemcpyAsync(h_out_tokens, d_out_token, T * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

void Inkling2Model::rollback_conv_states(int accepted_len, int slot_id) {
    int idx = accepted_len - 1;
    int kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    int D = cfg.hidden_size;
    int64_t k_dst = (int64_t)slot_id * 3 * kv_dim;
    int64_t d_dst = (int64_t)slot_id * 3 * D;
    int64_t k_src = ((int64_t)slot_id * 8 + idx) * 3 * kv_dim;
    int64_t d_src = ((int64_t)slot_id * 8 + idx) * 3 * D;
    for (auto& vc : caches) {
        CHECK_CUDA(cudaMemcpyAsync(vc.k_conv_state + k_dst, vc.k_conv_snap + k_src,
                                   3 * kv_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(vc.v_conv_state + k_dst, vc.v_conv_snap + k_src,
                                   3 * kv_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(vc.attn_conv_state + d_dst, vc.attn_conv_snap + d_src,
                                   3 * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(vc.mlp_conv_state + d_dst, vc.mlp_conv_snap + d_src,
                                   3 * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
    }
}

void Inkling2Model::commit_dspark_context(int count, int start_pos, int slot_id, int aux_tok_offset) {
    if (!dspark.loaded || count <= 0 || slot_id >= dspark.max_dspark_slots) return;
    int D = cfg.hidden_size;
    int stack_stride = max_batch_tokens * D;
    dspark.store_context_kv(d_aux_states + (int64_t)aux_tok_offset * D, stack_stride, count, start_pos, slot_id);
    // Copy last committed token's aux features (index aux_tok_offset + count - 1) into d_last_aux[slot_id][4, D]
    __nv_bfloat16* dst_aux = d_last_aux + (int64_t)slot_id * 4 * D;
    for (int tap = 0; tap < 4; ++tap) {
        CHECK_CUDA(cudaMemcpyAsync(
            dst_aux + (int64_t)tap * D,
            d_aux_states + (int64_t)tap * stack_stride + (int64_t)(aux_tok_offset + count - 1) * D,
            D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
    }
}

void Inkling2Model::forward_logits(const int32_t* h_tokens, int T, int start_pos, std::vector<float>& h_logits, int slot_id) {
    forward_internal(h_tokens, T, start_pos, slot_id);
    launch_softcap_argmax(d_logits, d_out_token, cfg.unpadded_vocab_size, cfg.final_logit_softcapping, stream);
    h_logits.resize(cfg.unpadded_vocab_size);
    CHECK_CUDA(cudaMemcpyAsync(h_logits.data(), d_logits, (size_t)cfg.unpadded_vocab_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

} // namespace ink
