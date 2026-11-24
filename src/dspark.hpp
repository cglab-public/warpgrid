#pragma once
#include "config.hpp"
#include "safetensors.hpp"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace ink {

struct DraftLayerWeights {
    bool is_local = true;
    __nv_bfloat16* dflare_norm = nullptr;      // [2304]
    __nv_bfloat16* input_ln = nullptr;         // [2304]
    __nv_bfloat16* post_attn_ln = nullptr;     // [2304]
    __nv_bfloat16* q_proj = nullptr;           // [3072, 2304]
    __nv_bfloat16* k_proj = nullptr;           // [1024, 2304]
    __nv_bfloat16* v_proj = nullptr;           // [1024, 2304]
    __nv_bfloat16* k_proj_ctx = nullptr;       // [1024, 2304]
    __nv_bfloat16* v_proj_ctx = nullptr;       // [1024, 2304]
    __nv_bfloat16* o_proj = nullptr;           // [2304, 3072]
    __nv_bfloat16* q_norm = nullptr;           // [128]
    __nv_bfloat16* k_norm = nullptr;           // [128]
    __nv_bfloat16* k_norm_ctx = nullptr;       // [128]
    __nv_bfloat16* gate_proj = nullptr;        // [9216, 2304]
    __nv_bfloat16* up_proj = nullptr;          // [9216, 2304]
    __nv_bfloat16* down_proj = nullptr;        // [2304, 9216]
    // Per-row FP8 E4M3 quantized draft weights for 2x draft bandwidth!
    uint8_t *q_fp8 = nullptr, *k_fp8 = nullptr, *v_fp8 = nullptr, *o_fp8 = nullptr;
    uint8_t *gate_fp8 = nullptr, *up_fp8 = nullptr, *down_fp8 = nullptr;
    float   *q_sc = nullptr, *k_sc = nullptr, *v_sc = nullptr, *o_sc = nullptr;
    float   *gate_sc = nullptr, *up_sc = nullptr, *down_sc = nullptr;
};

struct DFlash2LayerWeights {
    __nv_bfloat16* input_ln = nullptr;
    __nv_bfloat16* post_ln = nullptr;
    __nv_bfloat16* q_norm = nullptr;
    __nv_bfloat16* k_norm = nullptr;
    __nv_bfloat16* attn_conv_base = nullptr;
    __nv_bfloat16* attn_conv_proj = nullptr;
    __nv_bfloat16* mlp_conv_base = nullptr;
    __nv_bfloat16* mlp_conv_proj = nullptr;
    __nv_bfloat16* k_proj = nullptr;
    __nv_bfloat16* v_proj = nullptr;
    uint8_t *q_fp8 = nullptr, *k_fp8 = nullptr, *v_fp8 = nullptr, *o_fp8 = nullptr;
    uint8_t *gate_fp8 = nullptr, *up_fp8 = nullptr, *down_fp8 = nullptr;
    float *q_sc = nullptr, *k_sc = nullptr, *v_sc = nullptr, *o_sc = nullptr;
    float *gate_sc = nullptr, *up_sc = nullptr, *down_sc = nullptr;
};

void launch_dflash2_concat_taps(
    const __nv_bfloat16* aux_states, int aux_stride,
    __nv_bfloat16* out, int T, int D, cudaStream_t stream);

void launch_dflash2_grouped_dyn_conv(
    const __nv_bfloat16* x, const __nv_bfloat16* base_k, const __nv_bfloat16* dyn_k,
    __nv_bfloat16* out, int T, int D, int group_size, int half_idx, cudaStream_t stream);

void launch_dflash2_block_attn(
    const __nv_bfloat16* q_blk, const __nv_bfloat16* k_blk, const __nv_bfloat16* v_blk,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, __nv_bfloat16* out,
    int anchor_pos, const int32_t* d_anchor_pos, int B, int cap, int window_size,
    int n_heads, int n_kv_heads, int head_dim, cudaStream_t stream);

void launch_dflash2_topk16(
    float* logits, float* top_vals, int32_t* top_ids,
    const int32_t* draft_to_target, int M, int V, cudaStream_t stream);

void launch_dflash2_select_path(
    const int32_t* d_anchor_tok, const float* top_vals, const int32_t* top_ids,
    const __nv_bfloat16* h_sel, const __nv_bfloat16* pred_cb, const __nv_bfloat16* succ_cb,
    int32_t* out_tok, int nd, int rank, cudaStream_t stream);

class DSparkDrafter {
public:
    SafetensorsFile st;
    cublasHandle_t cublas = nullptr;
    cudaStream_t stream = nullptr;

    // Shared pointers from target model (non-owning)
    const __nv_bfloat16* target_embed_w = nullptr;
    const __nv_bfloat16* target_embed_norm_w = nullptr;
    const __nv_bfloat16* target_unembed_w = nullptr;

    // DSpark config (from Inkling-2-Mini mtp_config)
    int num_layers = 3;
    int hidden_size = 2304;
    int num_heads = 24;
    int num_kv_heads = 8;
    int head_dim = 128;
    int inter_size = 9216;
    int block_size = 8;
    int markov_rank = 256;
    int vocab_size = 200058;
    int sliding_window = 1024;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    float softcap = 30.0f;
    int max_seq_len = 32768;

    // Weights
    float* d_fuse_alpha = nullptr;              // [4, 4] FP32 (softmax of fuse_weight)
    __nv_bfloat16* d_mask_embed = nullptr;      // [2304]
    __nv_bfloat16* d_final_norm = nullptr;      // [2304]
    std::vector<DraftLayerWeights> layers;

    // Markov head weights
    __nv_bfloat16* d_markov_dflare_norm = nullptr; // [2304]
    __nv_bfloat16* d_markov_w1 = nullptr;          // [201024, 256]
    __nv_bfloat16* d_rnn_proj_w = nullptr;         // [768, 5120]
    float* d_rnn_proj_b = nullptr;                 // [768] FP32
    // Folded LM head: fold_w = unembed @ hidden_out.weight ([draft_vocab, 256] BF16)
    //                 fold_b = unembed @ hidden_out.bias   ([draft_vocab] FP32)
    int draft_vocab_size = 200058;
    int draft_bpe_keep = 0; // 0 = full vocab; >0 = keep [0..draft_bpe_keep) + special [199998..200058)
    __nv_bfloat16* d_pruned_unembed = nullptr;     // [draft_vocab, 2304] (aliases target_unembed_w if unpruned)
    uint8_t* d_pruned_unembed_fp8 = nullptr;       // [draft_vocab, 2304] FP8 E4M3
    float*   d_pruned_unembed_sc = nullptr;        // [draft_vocab]
    int32_t* d_draft_to_target = nullptr;          // [draft_vocab] maps pruned idx -> target token id
    __nv_bfloat16* d_fold_w = nullptr;             // [draft_vocab, 256]
    uint8_t* d_fold_w_fp8 = nullptr;               // [draft_vocab, 256] FP8 E4M3
    float*   d_fold_w_sc = nullptr;                // [draft_vocab]
    float* d_fold_b = nullptr;                     // [draft_vocab]

    // Draft KV caches (3 layers: L0/L2 local, L1 global) x max_dspark_slots
    int max_dspark_slots = 4;
    __nv_bfloat16* d_k_cache[3] = {};
    __nv_bfloat16* d_v_cache[3] = {};
    int kv_cap[3] = {};

    // Workspace (for block_size = 8 and context ingestion)
    __nv_bfloat16* d_ctx_mix = nullptr;    // [max_ctx, 2304]
    __nv_bfloat16* d_ctx_k = nullptr;      // [max_ctx, 1024]
    __nv_bfloat16* d_ctx_v = nullptr;      // [max_ctx, 1024]
    __nv_bfloat16* d_x = nullptr;          // [8, 2304]
    __nv_bfloat16* d_res = nullptr;        // [8, 2304]
    __nv_bfloat16* d_q = nullptr;          // [8, 3072]
    __nv_bfloat16* d_k = nullptr;          // [8, 1024]
    __nv_bfloat16* d_v = nullptr;          // [8, 1024]
    __nv_bfloat16* d_attn_out = nullptr;   // [8, 3072]
    __nv_bfloat16* d_proj_out = nullptr;   // [8, 2304]
    __nv_bfloat16* d_mlp_g = nullptr;      // [8, 9216]
    __nv_bfloat16* d_mlp_u = nullptr;      // [8, 9216]
    __nv_bfloat16* d_f_rnn = nullptr;      // [2304]
    __nv_bfloat16* d_base_logits = nullptr;// [7, 200058]
    __nv_bfloat16* d_rnn_in = nullptr;     // [5120]
    float* d_rnn_raw = nullptr;            // [768]
    __nv_bfloat16* d_rnn_state = nullptr;  // [256]
    __nv_bfloat16* d_rnn_latent = nullptr; // [256]
    float* d_markov_bias = nullptr;        // [200058]
    int32_t* d_draft_tok = nullptr;        // [8]
    int32_t* d_anchor_tok = nullptr;       // [1] for CUDA graph capture
    int32_t* d_anchor_pos = nullptr;       // [1] for CUDA graph capture

    cudaGraph_t draft_graphs[8][8] = {};
    cudaGraphExec_t draft_graph_execs[8][8] = {};
    bool use_cuda_graph = true;

    std::vector<void*> allocs;
    bool loaded = false;
    bool df2_mode = false;

    int df2_num_layers = 5;
    int df2_heads = 32;
    int df2_kv_heads = 8;
    int df2_head_dim = 128;
    int df2_inter = 6912;
    int df2_block = 8;
    int df2_window = 2048;
    int df2_kv_cap = 0;
    int df2_group_size = 16;
    int df2_sel_rank = 256;
    int32_t df2_mask_id = 200056;
    float df2_rope_theta = 10000.0f;
    float df2_eps = 1e-6f;

    __nv_bfloat16* df2_fc = nullptr;
    uint8_t* df2_fc_fp8 = nullptr;
    float* df2_fc_sc = nullptr;
    __nv_bfloat16* df2_hidden_norm = nullptr;
    __nv_bfloat16* df2_final_norm = nullptr;
    __nv_bfloat16* df2_sel_hproj = nullptr;
    __nv_bfloat16* df2_sel_pred_cb = nullptr;
    __nv_bfloat16* df2_sel_succ_cb = nullptr;
    std::vector<DFlash2LayerWeights> df2_layers;
    __nv_bfloat16* df2_k_cache[5] = {};
    __nv_bfloat16* df2_v_cache[5] = {};

    __nv_bfloat16* df2_ctx_cat = nullptr;
    __nv_bfloat16* df2_ctx_fc = nullptr;
    __nv_bfloat16* df2_ctx_norm = nullptr;
    __nv_bfloat16* df2_ctx_k = nullptr;
    __nv_bfloat16* df2_ctx_v = nullptr;
    __nv_bfloat16* df2_res = nullptr;
    __nv_bfloat16* df2_x = nullptr;
    __nv_bfloat16* df2_xc = nullptr;
    __nv_bfloat16* df2_dyn = nullptr;
    __nv_bfloat16* df2_q = nullptr;
    __nv_bfloat16* df2_k = nullptr;
    __nv_bfloat16* df2_v = nullptr;
    __nv_bfloat16* df2_attn_out = nullptr;
    __nv_bfloat16* df2_proj_out = nullptr;
    __nv_bfloat16* df2_mlp_g = nullptr;
    __nv_bfloat16* df2_mlp_u = nullptr;
    __nv_bfloat16* df2_h_sel = nullptr;
    float* df2_logits = nullptr;
    float* df2_top_v = nullptr;
    int32_t* df2_top_i = nullptr;

    DSparkDrafter() = default;
    ~DSparkDrafter();

    void load(const std::string& model_dir,
              const __nv_bfloat16* embed_w,
              const __nv_bfloat16* embed_norm_w,
              const __nv_bfloat16* unembed_w,
              cublasHandle_t shared_cublas,
              cudaStream_t shared_stream,
              int max_seq, int max_ctx_chunk, const std::string& draft_vocab_file = "",
              int n_dspark_slots = 4);

    void load_dflash2(const std::string& df2_dir,
                      const __nv_bfloat16* embed_w,
                      const __nv_bfloat16* embed_norm_w,
                      const __nv_bfloat16* unembed_w,
                      cublasHandle_t shared_cublas,
                      cudaStream_t shared_stream,
                      int max_ctx_chunk, const std::string& draft_vocab_file = "",
                      int n_dspark_slots = 4);

    // Store context K/V from target's 4 aux hidden states [4, T, 2304] at [start_pos .. start_pos+T-1]
    void store_context_kv(const __nv_bfloat16* aux_states, int aux_stride, int T, int start_pos, int slot_id = 0);

    // Draft `num_draft` tokens (1..7) starting after `anchor_tok` at `anchor_pos`
    // using `last_aux` [4, 2304] (aux features of the anchor token's predecessor / last committed token)
    void draft_tokens(int32_t anchor_tok, int anchor_pos,
                      const __nv_bfloat16* last_aux, int aux_stride,
                      int num_draft, int32_t* h_out_drafts, int slot_id = 0);

private:
    void* alloc_gpu(size_t bytes);
    void linear_bf16(const __nv_bfloat16* X, const __nv_bfloat16* W, __nv_bfloat16* Y, int M, int N, int K);
    void linear_bf16_to_fp32(const __nv_bfloat16* X, const __nv_bfloat16* W, float* Y, int M, int N, int K);
    void store_dflash2_context_kv(const __nv_bfloat16* aux_states, int aux_stride, int T, int start_pos, int slot_id);
    void draft_dflash2_tokens(int32_t anchor_tok, int anchor_pos, int num_draft, int32_t* h_out_drafts, int slot_id);
};

} // namespace ink
