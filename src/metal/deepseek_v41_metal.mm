#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "deepseek_v41_metal.hpp"
#ifdef WARPGRID_METAL_PRECOMPILED
#include "kernels_gguf_metallib_embed.hpp"
#else
#include "kernels_gguf_source_embed.hpp"
#endif
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mach/mach.h>
#include <stdexcept>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace ink {

namespace {

struct GgufGemvParams {
    uint32_t M;
    uint32_t N;
    uint32_t K;
};

struct GgufGroupedGemvParams {
    uint32_t T;
    uint32_t G;
    uint32_t N;
    uint32_t K;
    uint32_t group_stride_bytes;
};

struct GgufMoeParams {
    uint32_t T;
    uint32_t N;
    uint32_t K;
    uint32_t top_k;
    uint32_t expert_stride_bytes;
};

struct GgufMoeSwigluParams {
    uint32_t T;
    uint32_t N;
    uint32_t K;
    uint32_t top_k;
    uint32_t expert_stride_bytes;
    float clamp_val;
};

struct DsRmsParams {
    uint32_t T;
    uint32_t D;
    float eps;
};

struct DsHcMixParams {
    uint32_t T;
    uint32_t hc_mult;
    uint32_t sinkhorn_iters;
    float eps;
};

struct DsHcApplyParams {
    uint32_t T;
    uint32_t D;
    uint32_t hc_mult;
};

struct DsEngramApplyParams {
    uint32_t T;
    uint32_t D;
    uint32_t hc_mult;
    float eps;
};

struct DsPartialRopeParams {
    uint32_t T;
    uint32_t start_pos;
    uint32_t pos_stride;
    uint32_t n_heads;
    uint32_t head_dim;
    uint32_t n_nope;
    uint32_t n_rot;
    uint32_t inverse;
    float freq_base;
};

struct DsMqaSinkAttnParams {
    uint32_t T;
    uint32_t start_pos;
    uint32_t n_heads;
    uint32_t head_dim;
    uint32_t sliding_window;
    uint32_t n_comp_sel;
    float scale;
};

struct DsCompGroupParams {
    uint32_t n_blocks;
    uint32_t ratio;
    uint32_t head_dim;
    float eps;
};

struct DsIndexerParams {
    uint32_t T;
    uint32_t start_pos;
    uint32_t n_comp;
    uint32_t max_comp;
    uint32_t n_idx_head;
    uint32_t idx_dim;
    uint32_t ratio;
    uint32_t sliding_window;
    uint32_t top_k;
    float iw_scale;
};

struct DsRouterParams {
    uint32_t T;
    uint32_t E;
    uint32_t top_k;
    uint32_t has_bias;
    float route_scale;
};

struct DsStreamCollapseParams {
    uint32_t T;
    uint32_t D;
    uint32_t hc_mult;
};

struct DsHcMeanTapParams {
    uint32_t T;
    uint32_t D;
    uint32_t hc;
    uint32_t n_taps;
    uint32_t tap_idx;
};

struct DsDsparkKvRingParams {
    uint32_t T;
    uint32_t start_pos;
    uint32_t n_swa;
    uint32_t head_dim;
};

struct DsDsparkAttnParams {
    uint32_t B;
    uint32_t n_heads;
    uint32_t head_dim;
    uint32_t anchor_pos;
    uint32_t n_swa;
    float scale;
};

struct DsMarkovBiasParams {
    uint32_t V;
    uint32_t R;
};

struct WeightBuf {
    id<MTLBuffer> buf = nil;
    size_t base_offset = 0;
    const uint8_t* raw_data = nullptr;
    int nocache_fd = -1;
    uint64_t file_offset = 0;
    GgmlType type = GgmlType::F32;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t groups = 1;
    uint32_t group_stride_bytes = 0;
    bool valid() const { return buf != nil; }
    bool is_streamed() const { return buf == nil && (raw_data != nullptr || nocache_fd >= 0); }
};

struct Ds41LayerWeights {
    id<MTLBuffer> attn_norm = nil;
    id<MTLBuffer> attn_sinks = nil;
    WeightBuf wq_a;
    id<MTLBuffer> attn_q_a_norm = nil;
    WeightBuf wq_b;
    WeightBuf wkv;
    id<MTLBuffer> attn_kv_norm = nil;
    WeightBuf wo_a;
    WeightBuf wo_b;

    WeightBuf hc_attn_fn;
    id<MTLBuffer> hc_attn_base = nil;
    id<MTLBuffer> hc_attn_scale = nil;
    WeightBuf hc_ffn_fn;
    id<MTLBuffer> hc_ffn_base = nil;
    id<MTLBuffer> hc_ffn_scale = nil;

    WeightBuf comp_wkv;
    WeightBuf comp_wgate;
    id<MTLBuffer> comp_norm = nil;
    WeightBuf idx_k;
    id<MTLBuffer> idx_k_norm = nil;
    WeightBuf idx_q_b;
    WeightBuf idx_proj;

    WeightBuf engram_kv;
    id<MTLBuffer> engram_gate = nil;

    id<MTLBuffer> ffn_norm = nil;
    WeightBuf router;
    id<MTLBuffer> router_bias = nil;
    WeightBuf ffn_gate_exps;
    WeightBuf ffn_up_exps;
    WeightBuf ffn_down_exps;
    WeightBuf ffn_gate_shexp;
    WeightBuf ffn_up_shexp;
    WeightBuf ffn_down_shexp;
};

struct Ds41LayerKv {
    id<MTLBuffer> kv_cache = nil;
    id<MTLBuffer> comp_raw_kv = nil;
    id<MTLBuffer> comp_raw_score = nil;
    id<MTLBuffer> comp_kv = nil;
    id<MTLBuffer> comp_ik = nil;
    id<MTLBuffer> topk_sel = nil;
};

static bool pread_all(int fd, void* dst, size_t len, off_t off) {
    uint8_t* p = static_cast<uint8_t*>(dst);
    size_t done = 0;
    while (done < len) {
        ssize_t n = ::pread(fd, p + done, len - done, off + (off_t)done);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        done += (size_t)n;
    }
    return true;
}

struct ExpLruCache {
    struct Entry {
        int32_t layer = -1;
        int32_t expert = -1;
        int32_t prev = -1;
        int32_t next = -1;
        uint64_t pin_gen = 0;
    };

    int num_layers = 0;
    int num_experts = 0;
    int num_slots = 0;
    size_t gu_stride = 0;
    size_t d_stride = 0;
    id<MTLBuffer> buf_g = nil;
    id<MTLBuffer> buf_u = nil;
    id<MTLBuffer> buf_d = nil;
    std::vector<int32_t> lut;
    std::vector<Entry> entries;
    int32_t head = -1;
    int32_t tail = -1;
    uint64_t cur_gen = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;

    bool enabled() const { return num_slots > 0 && buf_g != nil && buf_u != nil && buf_d != nil; }

    void reset() {
        num_layers = 0;
        num_experts = 0;
        num_slots = 0;
        gu_stride = 0;
        d_stride = 0;
        buf_g = nil;
        buf_u = nil;
        buf_d = nil;
        lut.clear();
        entries.clear();
        head = -1;
        tail = -1;
        cur_gen = 0;
        hits = 0;
        misses = 0;
    }

    bool init(id<MTLDevice> dev, int L, int E, size_t max_gu, size_t max_d, uint64_t budget_bytes) {
        if (L <= 0 || E <= 0 || max_gu == 0 || max_d == 0 || budget_bytes == 0) return false;
        size_t slot_bytes = max_gu * 2u + max_d;
        uint64_t max_slots_budget = budget_bytes / slot_bytes;
        uint64_t max_buf = (uint64_t)dev.maxBufferLength;
        if (max_buf > 0) {
            uint64_t max_slots_buf = max_buf / std::max(max_gu, max_d);
            if (max_slots_buf < max_slots_budget) max_slots_budget = max_slots_buf;
        }
        uint64_t total_exps = (uint64_t)L * (uint64_t)E;
        int slots = (int)std::min<uint64_t>(max_slots_budget, total_exps);
        if (slots < 64) return false;
        id<MTLBuffer> bg = [dev newBufferWithLength:(size_t)slots * max_gu options:MTLResourceStorageModeShared];
        id<MTLBuffer> bu = [dev newBufferWithLength:(size_t)slots * max_gu options:MTLResourceStorageModeShared];
        id<MTLBuffer> bd = [dev newBufferWithLength:(size_t)slots * max_d options:MTLResourceStorageModeShared];
        if (!bg || !bu || !bd) return false;
        num_layers = L;
        num_experts = E;
        num_slots = slots;
        gu_stride = max_gu;
        d_stride = max_d;
        buf_g = bg;
        buf_u = bu;
        buf_d = bd;
        lut.assign((size_t)L * (size_t)E, -1);
        entries.resize((size_t)slots);
        for (int i = 0; i < slots; ++i) {
            entries[(size_t)i].prev = i - 1;
            entries[(size_t)i].next = (i + 1 < slots) ? (i + 1) : -1;
        }
        head = 0;
        tail = slots - 1;
        return true;
    }

    void touch_mru(int32_t s) {
        if (s == head) return;
        auto& e = entries[(size_t)s];
        if (e.prev >= 0) entries[(size_t)e.prev].next = e.next;
        if (e.next >= 0) entries[(size_t)e.next].prev = e.prev;
        if (s == tail) tail = e.prev;
        e.prev = -1;
        e.next = head;
        if (head >= 0) entries[(size_t)head].prev = s;
        head = s;
        if (tail < 0) tail = s;
    }

    void pin_and_resolve(int layer, const int32_t* uniq_ids, size_t U,
                         std::vector<int32_t>& out_slots,
                         std::vector<std::pair<int32_t, int32_t>>& out_misses) {
        ++cur_gen;
        out_slots.assign(U, -1);
        out_misses.clear();
        size_t base = (size_t)layer * (size_t)num_experts;
        for (size_t u = 0; u < U; ++u) {
            int32_t eid = uniq_ids[u];
            int32_t s = lut[base + (size_t)eid];
            if (s >= 0) {
                entries[(size_t)s].pin_gen = cur_gen;
                touch_mru(s);
                out_slots[u] = s;
                hits++;
            }
        }
        int32_t scan = tail;
        for (size_t u = 0; u < U; ++u) {
            if (out_slots[u] >= 0) continue;
            while (scan >= 0 && entries[(size_t)scan].pin_gen == cur_gen) {
                scan = entries[(size_t)scan].prev;
            }
            if (scan < 0) throw std::runtime_error("ExpLruCache exhausted pinned slots");
            int32_t s = scan;
            scan = entries[(size_t)s].prev;
            auto& e = entries[(size_t)s];
            if (e.layer >= 0 && e.expert >= 0) {
                lut[(size_t)e.layer * (size_t)num_experts + (size_t)e.expert] = -1;
            }
            int32_t eid = uniq_ids[u];
            e.layer = layer;
            e.expert = eid;
            e.pin_gen = cur_gen;
            lut[base + (size_t)eid] = s;
            touch_mru(s);
            out_slots[u] = s;
            out_misses.push_back({s, eid});
            misses++;
        }
    }
};

}

struct DeepSeekV41Model::Impl {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> lib = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> psos;

    WeightBuf lm_head;
    id<MTLBuffer> output_norm = nil;
    bool engram_in_main_gguf = false;
    bool stream_moe_experts = false;
    std::vector<id<MTLBuffer>> shard_bufs;
    std::vector<Ds41LayerWeights> layers;
    std::vector<Ds41LayerKv> kv_caches;
    std::vector<std::vector<int32_t>> slot_tok_history;

    id<MTLBuffer> b_moe_stage_gate = nil;
    id<MTLBuffer> b_moe_stage_up = nil;
    id<MTLBuffer> b_moe_stage_down = nil;
    id<MTLBuffer> b_stage_topk_ids = nil;
    size_t moe_stage_cap_gate = 0;
    size_t moe_stage_cap_up = 0;
    size_t moe_stage_cap_down = 0;
    std::vector<int> nocache_fds;
    ExpLruCache exp_cache;

    struct DsparkLayer {
        id<MTLBuffer> attn_norm = nil;
        id<MTLBuffer> attn_sinks = nil;
        WeightBuf wq_a;
        id<MTLBuffer> attn_q_a_norm = nil;
        WeightBuf wq_b;
        WeightBuf wkv;
        id<MTLBuffer> attn_kv_norm = nil;
        WeightBuf wo_a;
        WeightBuf wo_b;
        WeightBuf hc_attn_fn;
        id<MTLBuffer> hc_attn_base = nil;
        id<MTLBuffer> hc_attn_scale = nil;
        WeightBuf hc_ffn_fn;
        id<MTLBuffer> hc_ffn_base = nil;
        id<MTLBuffer> hc_ffn_scale = nil;
        id<MTLBuffer> ffn_norm = nil;
        WeightBuf router;
        id<MTLBuffer> router_bias = nil;
        WeightBuf ffn_gate_exps;
        WeightBuf ffn_up_exps;
        WeightBuf ffn_down_exps;
        WeightBuf ffn_gate_shexp;
        WeightBuf ffn_up_shexp;
        WeightBuf ffn_down_shexp;
    };

    struct DsparkState {
        bool loaded = false;
        GgufFile gguf;
        int num_layers = 0;
        int hidden_size = 0;
        int n_heads = 0;
        int head_dim = 0;
        int n_rot = 0;
        int q_lora_rank = 0;
        int o_group_count = 0;
        int o_lora_rank = 0;
        int n_swa = 128;
        int n_experts = 0;
        int top_k = 0;
        int hc_mult = 4;
        int hc_sinkhorn_iters = 20;
        int block_size = 5;
        int mask_token_id = 128799;
        int markov_rank = 0;
        float rms_norm_eps = 1e-20f;
        float hc_eps = 1e-6f;
        float rope_freq_base = 10000.0f;
        float route_scale = 1.5f;
        float swiglu_clamp_exp = 10.0f;
        float swiglu_clamp_shexp = 10.0f;
        std::vector<int> target_layers;
        std::vector<int> layer_to_tap;
        WeightBuf fc;
        id<MTLBuffer> enc_output_norm = nil;
        id<MTLBuffer> output_norm = nil;
        WeightBuf markov_w1;
        WeightBuf markov_w2;
        std::vector<DsparkLayer> layers;
        std::vector<id<MTLBuffer>> kv_rings;
        id<MTLBuffer> b_taps = nil;
        id<MTLBuffer> b_inp_g = nil;
        id<MTLBuffer> b_logits_blk = nil;
        id<MTLBuffer> b_markov_logits = nil;
        id<MTLBuffer> b_markov_toks = nil;

        void reset() {
            loaded = false;
            layers.clear();
            kv_rings.clear();
            target_layers.clear();
            layer_to_tap.clear();
            fc = WeightBuf{};
            enc_output_norm = nil;
            output_norm = nil;
            markov_w1 = WeightBuf{};
            markov_w2 = WeightBuf{};
            b_taps = nil;
            b_inp_g = nil;
            b_logits_blk = nil;
            b_markov_logits = nil;
            b_markov_toks = nil;
            gguf.close();
        }
    } dspark;
    int last_verify_start_pos = 0;
    int32_t* verify_out_buf = nullptr;

    void close_nocache_fds() {
        for (int fd : nocache_fds) {
            if (fd >= 0) ::close(fd);
        }
        nocache_fds.clear();
    }

    ~Impl() {
        close_nocache_fds();
    }

    id<MTLBuffer> b_streams_a = nil;
    id<MTLBuffer> b_streams_b = nil;
    id<MTLBuffer> b_flat_norm = nil;
    id<MTLBuffer> b_hc_raw = nil;
    id<MTLBuffer> b_hc_pre_a = nil;
    id<MTLBuffer> b_hc_pre_b = nil;
    id<MTLBuffer> b_hc_post = nil;
    id<MTLBuffer> b_hc_comb = nil;

    id<MTLBuffer> b_hidden = nil;
    id<MTLBuffer> b_norm = nil;
    id<MTLBuffer> b_qr = nil;
    id<MTLBuffer> b_qr_norm = nil;
    id<MTLBuffer> b_q = nil;
    id<MTLBuffer> b_kv = nil;
    id<MTLBuffer> b_kv_norm = nil;
    id<MTLBuffer> b_attn_out = nil;
    id<MTLBuffer> b_oa = nil;
    id<MTLBuffer> b_sub_out = nil;

    id<MTLBuffer> b_comp_tmp_kv = nil;
    id<MTLBuffer> b_comp_tmp_score = nil;
    id<MTLBuffer> b_comp_block = nil;
    id<MTLBuffer> b_idx_k_tmp = nil;
    id<MTLBuffer> b_idx_q = nil;
    id<MTLBuffer> b_idx_w = nil;
    id<MTLBuffer> b_idx_scores = nil;

    std::vector<id<MTLBuffer>> b_engram_rows;
    id<MTLBuffer> b_engram_kv = nil;

    id<MTLBuffer> b_router_logits = nil;
    id<MTLBuffer> b_topk_ids = nil;
    id<MTLBuffer> b_topk_w = nil;
    id<MTLBuffer> b_ffn_gate = nil;
    id<MTLBuffer> b_ffn_up = nil;
    id<MTLBuffer> b_ffn_act = nil;
    id<MTLBuffer> b_ffn_tmp = nil;
    id<MTLBuffer> b_logits = nil;
    id<MTLBuffer> b_argmax = nil;
    id<MTLBuffer> b_dummy_bias = nil;

    void init_metal() {
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) throw std::runtime_error("No Apple Metal device available");
        queue = [dev newCommandQueue];
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
            throw std::runtime_error(std::string("Failed to load GGUF Metal library: ") +
                                     (err ? err.localizedDescription.UTF8String : "unknown"));
        }
    }

    id<MTLComputePipelineState> pso(const std::string& name) {
        auto it = psos.find(name);
        if (it != psos.end()) return it->second;
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
        if (!fn) throw std::runtime_error("Missing GGUF Metal kernel: " + name);
        NSError* err = nil;
        id<MTLComputePipelineState> st = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!st) {
            throw std::runtime_error("Failed to create PSO " + name + ": " +
                                     (err ? err.localizedDescription.UTF8String : "unknown"));
        }
        psos[name] = st;
        return st;
    }

    id<MTLBuffer> alloc_buf(size_t bytes) {
        size_t sz = std::max<size_t>(16, bytes);
        id<MTLBuffer> b = [dev newBufferWithLength:sz options:MTLResourceStorageModeShared];
        if (!b) throw std::runtime_error("Metal buffer allocation failed (" + std::to_string(sz) + " bytes)");
        memset(b.contents, 0, sz);
        return b;
    }

    id<MTLBuffer> load_f32_required(const GgufFile& gf, const std::string& name) {
        if (!gf.has_tensor(name)) {
            throw std::runtime_error("Missing required DeepSeek-V4.1 tensor: " + name);
        }
        const auto& ti = gf.tensor(name);
        size_t n = (size_t)ti.dims[0] * (size_t)ti.num_rows();
        id<MTLBuffer> b = alloc_buf(n * sizeof(float));
        dequant_row_fp32(ti.type, gf.tensor_data(name), static_cast<float*>(b.contents), n);
        return b;
    }

    void init_shard_buffers(const GgufFile& gf) {
        shard_bufs.clear();
        (void)gf;
    }

    WeightBuf load_weight_required(const GgufFile& gf, const std::string& name, uint32_t force_groups = 0) {
        if (!gf.has_tensor(name)) {
            throw std::runtime_error("Missing required DeepSeek-V4.1 tensor: " + name);
        }
        const auto& ti = gf.tensor(name);
        WeightBuf wb;
        wb.type = ti.type;
        wb.cols = ti.dims.size() >= 1 ? (uint32_t)ti.dims[0] : 0;
        wb.rows = ti.dims.size() >= 2 ? (uint32_t)ti.dims[1] : 1;
        wb.groups = ti.dims.size() >= 3 ? (uint32_t)ti.dims[2] : 1;
        if (force_groups > 1 && wb.groups == 1 && wb.rows % force_groups == 0) {
            wb.groups = force_groups;
            wb.rows /= force_groups;
        }
        if (wb.groups > 1 && wb.cols > 0) {
            wb.group_stride_bytes = (uint32_t)(ggml_row_nbytes(ti.type, wb.cols) * wb.rows);
        }
        if (ti.type == GgmlType::F16) {
            size_t total_elems = (size_t)wb.cols * wb.rows * wb.groups;
            wb.buf = alloc_buf(total_elems * sizeof(float));
            wb.base_offset = 0;
            dequant_row_fp32(ti.type, gf.tensor_data(name), static_cast<float*>(wb.buf.contents), total_elems);
            wb.type = GgmlType::F32;
            if (wb.groups > 1) wb.group_stride_bytes = wb.cols * wb.rows * (uint32_t)sizeof(float);
        } else {
            long ps = sysconf(_SC_PAGESIZE);
            size_t page_sz = ps > 0 ? (size_t)ps : 16384u;
            uintptr_t ptr_addr = reinterpret_cast<uintptr_t>(gf.tensor_data(name));
            uintptr_t page_base = ptr_addr & ~(uintptr_t)(page_sz - 1u);
            size_t intra = (size_t)(ptr_addr - page_base);
            size_t aligned_len = (intra + std::max<size_t>(16, ti.nbytes) + page_sz - 1u) & ~(page_sz - 1u);
            uint64_t max_buf = (uint64_t)dev.maxBufferLength;
            id<MTLBuffer> nocopy = nil;
            if (max_buf == 0 || aligned_len <= max_buf) {
                nocopy = [dev newBufferWithBytesNoCopy:reinterpret_cast<void*>(page_base)
                                                length:aligned_len
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
            }
            if (nocopy != nil) {
                wb.buf = nocopy;
                wb.base_offset = intra;
            } else {
                wb.buf = [dev newBufferWithBytes:gf.tensor_data(name)
                                          length:std::max<size_t>(16, ti.nbytes)
                                         options:MTLResourceStorageModeShared];
                wb.base_offset = 0;
                if (!wb.buf) throw std::runtime_error("Metal buffer allocation failed for tensor: " + name);
            }
        }
        return wb;
    }

    WeightBuf load_weight_streamable(const GgufFile& gf, const std::string& name, bool stream) {
        if (!stream) return load_weight_required(gf, name);
        if (!gf.has_tensor(name)) {
            throw std::runtime_error("Missing required DeepSeek-V4.1 tensor: " + name);
        }
        const auto& ti = gf.tensor(name);
        if (ti.type == GgmlType::F16) return load_weight_required(gf, name);
        WeightBuf wb;
        wb.type = ti.type;
        wb.cols = ti.dims.size() >= 1 ? (uint32_t)ti.dims[0] : 0;
        wb.rows = ti.dims.size() >= 2 ? (uint32_t)ti.dims[1] : 1;
        wb.groups = ti.dims.size() >= 3 ? (uint32_t)ti.dims[2] : 1;
        if (wb.groups > 1 && wb.cols > 0) {
            wb.group_stride_bytes = (uint32_t)(ggml_row_nbytes(ti.type, wb.cols) * wb.rows);
        }
        if (ti.shard_index < nocache_fds.size() && nocache_fds[ti.shard_index] >= 0) {
            wb.nocache_fd = nocache_fds[ti.shard_index];
            wb.file_offset = (uint64_t)gf.shard_data_offset(ti.shard_index) + ti.offset;
        } else {
            wb.raw_data = gf.tensor_data(name);
        }
        return wb;
    }

    void ensure_moe_stage(size_t g_bytes, size_t u_bytes, size_t d_bytes) {
        if (g_bytes > moe_stage_cap_gate) {
            b_moe_stage_gate = alloc_buf(g_bytes);
            moe_stage_cap_gate = g_bytes;
        }
        if (u_bytes > moe_stage_cap_up) {
            b_moe_stage_up = alloc_buf(u_bytes);
            moe_stage_cap_up = u_bytes;
        }
        if (d_bytes > moe_stage_cap_down) {
            b_moe_stage_down = alloc_buf(d_bytes);
            moe_stage_cap_down = d_bytes;
        }
    }

    static const char* gemv_kernel_for(GgmlType t) {
        switch (t) {
            case GgmlType::F32: return "gguf_gemv_f32";
            case GgmlType::BF16: return "gguf_gemv_bf16";
            case GgmlType::Q8_0: return "gguf_gemv_q8_0";
            case GgmlType::IQ4_NL: return "gguf_gemv_iq4_nl";
            case GgmlType::MXFP4: return "gguf_gemv_mxfp4";
            case GgmlType::Q4_K: return "gguf_gemv_q4_k";
            case GgmlType::Q5_K: return "gguf_gemv_q5_k";
            case GgmlType::Q6_K: return "gguf_gemv_q6_k";
            case GgmlType::Q2_K: return "gguf_gemv_q2_k";
            case GgmlType::IQ2_XXS: return "gguf_gemv_iq2_xxs";
            case GgmlType::IQ1_S: return "gguf_gemv_iq1_s";
            default:
                throw std::runtime_error("Unsupported Metal GEMV GGUF quant type: " + std::to_string((uint32_t)t));
        }
    }

    static const char* moe_indexed_kernel_for(GgmlType t) {
        switch (t) {
            case GgmlType::F32: return "gguf_moe_indexed_f32";
            case GgmlType::BF16: return "gguf_moe_indexed_bf16";
            case GgmlType::Q8_0: return "gguf_moe_indexed_q8_0";
            case GgmlType::IQ4_NL: return "gguf_moe_indexed_iq4_nl";
            case GgmlType::MXFP4: return "gguf_moe_indexed_mxfp4";
            case GgmlType::Q4_K: return "gguf_moe_indexed_q4_k";
            case GgmlType::Q5_K: return "gguf_moe_indexed_q5_k";
            case GgmlType::Q6_K: return "gguf_moe_indexed_q6_k";
            case GgmlType::Q2_K: return "gguf_moe_indexed_q2_k";
            case GgmlType::IQ2_XXS: return "gguf_moe_indexed_iq2_xxs";
            case GgmlType::IQ1_S: return "gguf_moe_indexed_iq1_s";
            default:
                throw std::runtime_error("Unsupported Metal MoE indexed GGUF quant type: " + std::to_string((uint32_t)t));
        }
    }

    static const char* moe_down_accum_kernel_for(GgmlType t) {
        switch (t) {
            case GgmlType::F32: return "gguf_moe_down_accum_f32";
            case GgmlType::BF16: return "gguf_moe_down_accum_bf16";
            case GgmlType::Q8_0: return "gguf_moe_down_accum_q8_0";
            case GgmlType::IQ4_NL: return "gguf_moe_down_accum_iq4_nl";
            case GgmlType::MXFP4: return "gguf_moe_down_accum_mxfp4";
            case GgmlType::Q4_K: return "gguf_moe_down_accum_q4_k";
            case GgmlType::Q5_K: return "gguf_moe_down_accum_q5_k";
            case GgmlType::Q6_K: return "gguf_moe_down_accum_q6_k";
            case GgmlType::Q2_K: return "gguf_moe_down_accum_q2_k";
            case GgmlType::IQ2_XXS: return "gguf_moe_down_accum_iq2_xxs";
            case GgmlType::IQ1_S: return "gguf_moe_down_accum_iq1_s";
            default:
                throw std::runtime_error("Unsupported Metal MoE down-accum GGUF quant type: " + std::to_string((uint32_t)t));
        }
    }

    void encode_gemv(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, const WeightBuf& w, id<MTLBuffer> y,
                     uint32_t M, size_t x_offset = 0, size_t w_offset = 0, size_t y_offset = 0, bool force_fp32_x = false) {
        if (!w.valid()) throw std::runtime_error("Attempted Metal GEMV on missing weight buffer");
        GgufGemvParams p{M, w.rows, w.cols};
        if (w.type == GgmlType::F32 && w.rows <= 64u && (w.cols & 3u) == 0u && ((w.base_offset + w_offset) & 15u) == 0u && (x_offset & 15u) == 0u) {
            [enc setComputePipelineState:pso("gguf_gemv_f32_wide")];
            [enc setBuffer:x offset:x_offset atIndex:0];
            [enc setBuffer:w.buf offset:w.base_offset + w_offset atIndex:1];
            [enc setBuffer:y offset:y_offset atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(w.rows, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            return;
        }
        if (!force_fp32_x && M >= 8u && (w.type == GgmlType::Q6_K || w.type == GgmlType::Q5_K || w.type == GgmlType::Q8_0)) {
            const char* kname = (w.type == GgmlType::Q6_K) ? "gguf_gemm_q6_k_tiled"
                              : (w.type == GgmlType::Q5_K) ? "gguf_gemm_q5_k_tiled"
                                                           : "gguf_gemm_q8_0_tiled";
            [enc setComputePipelineState:pso(kname)];
            [enc setBuffer:x offset:x_offset atIndex:0];
            [enc setBuffer:w.buf offset:w.base_offset + w_offset atIndex:1];
            [enc setBuffer:y offset:y_offset atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((w.rows + 31u) / 32u, (M + 31u) / 32u, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            return;
        }
        if (M >= 2u && M <= 6u && (x_offset & 15u) == 0u &&
            (w.type == GgmlType::Q6_K || w.type == GgmlType::Q5_K || w.type == GgmlType::Q8_0)) {
            const char* kname = (w.type == GgmlType::Q6_K) ? "gguf_gemv_q6_k_multi"
                              : (w.type == GgmlType::Q5_K) ? "gguf_gemv_q5_k_multi"
                                                           : "gguf_gemv_q8_0_multi";
            [enc setComputePipelineState:pso(kname)];
            [enc setBuffer:x offset:x_offset atIndex:0];
            [enc setBuffer:w.buf offset:w.base_offset + w_offset atIndex:1];
            [enc setBuffer:y offset:y_offset atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((w.rows + 1u) / 2u, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            return;
        }
        if (w.type == GgmlType::Q6_K && (x_offset & 15u) == 0u) {
            if (w.cols <= 2048u && w.rows >= 4096u) {
                [enc setComputePipelineState:pso("gguf_gemv_q6_k_smallk")];
                [enc setBuffer:x offset:x_offset atIndex:0];
                [enc setBuffer:w.buf offset:w.base_offset + w_offset atIndex:1];
                [enc setBuffer:y offset:y_offset atIndex:2];
                [enc setBytes:&p length:sizeof(p) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((w.rows + 7u) / 8u, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                return;
            }
            [enc setComputePipelineState:pso("gguf_gemv_q6_k_fast")];
            [enc setBuffer:x offset:x_offset atIndex:0];
            [enc setBuffer:w.buf offset:w.base_offset + w_offset atIndex:1];
            [enc setBuffer:y offset:y_offset atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((w.rows + 3u) / 4u, M, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            return;
        }
        if (w.type == GgmlType::Q8_0 && (x_offset & 15u) == 0u) {
            [enc setComputePipelineState:pso("gguf_gemv_q8_0_fast")];
            [enc setBuffer:x offset:x_offset atIndex:0];
            [enc setBuffer:w.buf offset:w.base_offset + w_offset atIndex:1];
            [enc setBuffer:y offset:y_offset atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((w.rows + 3u) / 4u, M, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            return;
        }
        if (w.type == GgmlType::Q5_K) {
            [enc setComputePipelineState:pso("gguf_gemv_q5_k_fast")];
            [enc setBuffer:x offset:x_offset atIndex:0];
            [enc setBuffer:w.buf offset:w.base_offset + w_offset atIndex:1];
            [enc setBuffer:y offset:y_offset atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((w.rows + 3u) / 4u, M, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            return;
        }
        if (w.type == GgmlType::BF16 && (w.cols & 3u) == 0u && ((w.base_offset + w_offset) & 7u) == 0u && (x_offset & 15u) == 0u) {
            [enc setComputePipelineState:pso("gguf_gemv_bf16_fast")];
            [enc setBuffer:x offset:x_offset atIndex:0];
            [enc setBuffer:w.buf offset:w.base_offset + w_offset atIndex:1];
            [enc setBuffer:y offset:y_offset atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((w.rows + 3u) / 4u, M, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            return;
        }
        [enc setComputePipelineState:pso(gemv_kernel_for(w.type))];
        [enc setBuffer:x offset:x_offset atIndex:0];
        [enc setBuffer:w.buf offset:w.base_offset + w_offset atIndex:1];
        [enc setBuffer:y offset:y_offset atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((w.rows + 3u) / 4u, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

    void encode_moe_indexed(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, const WeightBuf& w,
                            id<MTLBuffer> topk_ids, id<MTLBuffer> y, uint32_t T, uint32_t top_k) {
        if (!w.valid()) throw std::runtime_error("Attempted Metal MoE indexed GEMV on missing weight buffer");
        GgufMoeParams mp{T, w.rows, w.cols, top_k, w.group_stride_bytes};
        [enc setComputePipelineState:pso(moe_indexed_kernel_for(w.type))];
        [enc setBuffer:x offset:0 atIndex:0];
        [enc setBuffer:w.buf offset:w.base_offset atIndex:1];
        [enc setBuffer:topk_ids offset:0 atIndex:2];
        [enc setBuffer:y offset:0 atIndex:3];
        [enc setBytes:&mp length:sizeof(mp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((w.rows + 3u) / 4u, T * top_k, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

    void encode_moe_down_accum(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, const WeightBuf& w,
                               id<MTLBuffer> topk_ids, id<MTLBuffer> topk_w, id<MTLBuffer> y,
                               uint32_t T, uint32_t top_k) {
        if (!w.valid()) throw std::runtime_error("Attempted Metal MoE down-accum GEMV on missing weight buffer");
        GgufMoeParams mp{T, w.rows, w.cols, top_k, w.group_stride_bytes};
        if (w.type == GgmlType::IQ1_S || w.type == GgmlType::IQ2_XXS) {
            const char* kname = (w.type == GgmlType::IQ1_S) ? "gguf_moe_down_accum_iq1_s_4r" : "gguf_moe_down_accum_iq2_xxs_4r";
            [enc setComputePipelineState:pso(kname)];
            [enc setBuffer:x offset:0 atIndex:0];
            [enc setBuffer:w.buf offset:w.base_offset atIndex:1];
            [enc setBuffer:topk_ids offset:0 atIndex:2];
            [enc setBuffer:topk_w offset:0 atIndex:3];
            [enc setBuffer:y offset:0 atIndex:4];
            [enc setBytes:&mp length:sizeof(mp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((w.rows + 7u) / 8u, T, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            return;
        }
        [enc setComputePipelineState:pso(moe_down_accum_kernel_for(w.type))];
        [enc setBuffer:x offset:0 atIndex:0];
        [enc setBuffer:w.buf offset:w.base_offset atIndex:1];
        [enc setBuffer:topk_ids offset:0 atIndex:2];
        [enc setBuffer:topk_w offset:0 atIndex:3];
        [enc setBuffer:y offset:0 atIndex:4];
        [enc setBytes:&mp length:sizeof(mp) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((w.rows + 3u) / 4u, T, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

    void encode_moe_gate_up_swiglu(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x,
                                   const WeightBuf& wg, const WeightBuf& wu,
                                   id<MTLBuffer> topk_ids, id<MTLBuffer> y,
                                   uint32_t T, uint32_t top_k, float clamp_val) {
        GgufMoeSwigluParams sp{T, wg.rows, wg.cols, top_k, wg.group_stride_bytes, clamp_val};
        const char* kname = (wg.type == GgmlType::IQ1_S) ? "gguf_moe_gate_up_swiglu_iq1_s" : "gguf_moe_gate_up_swiglu_iq2_xxs";
        [enc setComputePipelineState:pso(kname)];
        [enc setBuffer:x offset:0 atIndex:0];
        [enc setBuffer:wg.buf offset:wg.base_offset atIndex:1];
        [enc setBuffer:wu.buf offset:wu.base_offset atIndex:2];
        [enc setBuffer:topk_ids offset:0 atIndex:3];
        [enc setBuffer:y offset:0 atIndex:4];
        [enc setBytes:&sp length:sizeof(sp) atIndex:5];
        uint32_t rows_per_tg = (wg.type == GgmlType::IQ1_S) ? 4u : 8u;
        [enc dispatchThreadgroups:MTLSizeMake((wg.rows + rows_per_tg - 1u) / rows_per_tg, T * top_k, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    void encode_rmsnorm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, id<MTLBuffer> w, id<MTLBuffer> y,
                        uint32_t T, uint32_t D, float eps, size_t x_offset = 0, size_t y_offset = 0) {
        DsRmsParams p{T, D, eps};
        uint32_t tptg = (D >= 1024u) ? 256u : 32u;
        [enc setComputePipelineState:pso("ds_rmsnorm_f32")];
        [enc setBuffer:x offset:x_offset atIndex:0];
        [enc setBuffer:w offset:0 atIndex:1];
        [enc setBuffer:y offset:y_offset atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(tptg, 1, 1)];
    }

    void encode_rmsnorm_unweighted(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, id<MTLBuffer> y,
                                   uint32_t T, uint32_t D, float eps) {
        DsRmsParams p{T, D, eps};
        uint32_t tptg = (D >= 1024u) ? 256u : 32u;
        [enc setComputePipelineState:pso("ds_rmsnorm_unweighted_f32")];
        [enc setBuffer:x offset:0 atIndex:0];
        [enc setBuffer:y offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(tptg, 1, 1)];
    }

    void encode_swiglu(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> gate, id<MTLBuffer> up, id<MTLBuffer> out, uint32_t n, float clamp_val = 0.0f) {
        struct { uint32_t n; float clamp_val; } sp{n, clamp_val};
        [enc setComputePipelineState:pso("ds_swiglu_f32")];
        [enc setBuffer:gate offset:0 atIndex:0];
        [enc setBuffer:up offset:0 atIndex:1];
        [enc setBuffer:out offset:0 atIndex:2];
        [enc setBytes:&sp length:sizeof(sp) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((n + 255u) / 256u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    void encode_hc_mixes(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> streams,
                         const WeightBuf& fn, id<MTLBuffer> scale, id<MTLBuffer> base,
                         id<MTLBuffer> out_pre, id<MTLBuffer> out_post, id<MTLBuffer> out_comb,
                         uint32_t T, uint32_t D, uint32_t hc, uint32_t iters, float rms_eps, float hc_eps) {
        encode_rmsnorm_unweighted(enc, streams, b_flat_norm, T, hc * D, rms_eps);
        encode_gemv(enc, b_flat_norm, fn, b_hc_raw, T);
        DsHcMixParams mp{T, hc, iters, hc_eps};
        [enc setComputePipelineState:pso("ds_hc_mixes_f32")];
        [enc setBuffer:b_hc_raw offset:0 atIndex:0];
        [enc setBuffer:scale offset:0 atIndex:1];
        [enc setBuffer:base offset:0 atIndex:2];
        [enc setBuffer:out_pre offset:0 atIndex:3];
        [enc setBuffer:out_post offset:0 atIndex:4];
        [enc setBuffer:out_comb offset:0 atIndex:5];
        [enc setBytes:&mp length:sizeof(mp) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    void encode_hc_pre(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> streams, id<MTLBuffer> pre_w,
                       id<MTLBuffer> out_hidden, uint32_t T, uint32_t D, uint32_t hc) {
        DsHcApplyParams ap{T, D, hc};
        [enc setComputePipelineState:pso("ds_hc_pre_f32")];
        [enc setBuffer:streams offset:0 atIndex:0];
        [enc setBuffer:pre_w offset:0 atIndex:1];
        [enc setBuffer:out_hidden offset:0 atIndex:2];
        [enc setBytes:&ap length:sizeof(ap) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(D, T, 1) threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(64u, D), 1, 1)];
    }

    void encode_hc_post(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> sub, id<MTLBuffer> residual,
                        id<MTLBuffer> post_w, id<MTLBuffer> comb_w, id<MTLBuffer> out_streams,
                        uint32_t T, uint32_t D, uint32_t hc) {
        DsHcApplyParams ap{T, D, hc};
        [enc setComputePipelineState:pso("ds_hc_post_f32")];
        [enc setBuffer:sub offset:0 atIndex:0];
        [enc setBuffer:residual offset:0 atIndex:1];
        [enc setBuffer:post_w offset:0 atIndex:2];
        [enc setBuffer:comb_w offset:0 atIndex:3];
        [enc setBuffer:out_streams offset:0 atIndex:4];
        [enc setBytes:&ap length:sizeof(ap) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(D, T, 1) threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(64u, D), 1, 1)];
    }
};

DeepSeekV41Model::DeepSeekV41Model() : impl_(std::make_unique<Impl>()) {}
DeepSeekV41Model::~DeepSeekV41Model() = default;

void DeepSeekV41Model::load_tokenizer(Tokenizer& tok) const {
    gguf.load_tokenizer(tok);
}

void DeepSeekV41Model::load(const std::string& model_path, int max_seq, int max_chunk, int n_slots) {
    impl_->init_metal();
    gguf.open(model_path, false);
    impl_->init_shard_buffers(gguf);
    cfg = DeepSeekConfig::from_gguf(gguf);

    if (!engram_path.empty()) {
        engram.open(engram_path, 4096);
        if (!engram.has_tensor("per_layer_token_embd.weight")) {
            throw std::runtime_error("Engram GGUF is missing per_layer_token_embd.weight: " + engram_path);
        }
    }
    impl_->engram_in_main_gguf = gguf.has_tensor("per_layer_token_embd.weight");
    if (impl_->engram_in_main_gguf) {
        size_t es = gguf.tensor("per_layer_token_embd.weight").shard_index;
        const void* ep = gguf.shard_mapped(es);
        size_t esz = gguf.shard_size(es);
        if (ep && esz > 0) {
            ::madvise(const_cast<void*>(ep), esz, MADV_RANDOM);
        }
    }
    if (!cfg.ple_layers.empty() && !engram.is_open() && !impl_->engram_in_main_gguf) {
        throw std::runtime_error("Model declares Engram layers (ple.layers) but per_layer_token_embd.weight is missing; pass --engram <path>");
    }

    uint64_t rec_ws = (uint64_t)impl_->dev.recommendedMaxWorkingSetSize;
    size_t kv_per_slot = cfg.kv_bytes_per_token() * (size_t)max_seq;
    int eff_slots = std::max(1, n_slots);
    if (rec_ws > 0 && kv_per_slot > 0) {
        size_t kv_budget = (size_t)(rec_ws / 4);
        while (eff_slots > 1 && (size_t)eff_slots * kv_per_slot > kv_budget) {
            --eff_slots;
        }
    }
    max_slots = eff_slots;
    max_seq_len = max_seq;
    max_batch_tokens = std::max(1, max_chunk);
    if (!dspark_path.empty()) {
        int min_spec_batch = std::max(8, draft_k + 1);
        if (max_batch_tokens < min_spec_batch) max_batch_tokens = min_spec_batch;
    } else {
        draft_k = 0;
    }
    max_blocks_per_seq = (max_seq_len + 63) / 64;
    num_kv_blocks = max_slots * max_blocks_per_seq;
    num_snapshots = 0;

    if (!gguf.has_tensor("token_embd.weight")) {
        throw std::runtime_error("Missing required DeepSeek-V4.1 tensor: token_embd.weight");
    }
    {
        auto et = gguf.tensor("token_embd.weight").type;
        bool ok_emb = (et == GgmlType::F32 || et == GgmlType::F16 || et == GgmlType::BF16 ||
                       et == GgmlType::Q8_0 || et == GgmlType::IQ4_NL || et == GgmlType::MXFP4 ||
                       et == GgmlType::Q4_K || et == GgmlType::Q5_K || et == GgmlType::Q6_K || et == GgmlType::Q2_K);
        if (!ok_emb) {
            throw std::runtime_error("Unsupported token_embd.weight GGUF type: " + std::to_string((uint32_t)et));
        }
    }
    impl_->output_norm = impl_->load_f32_required(gguf, "output_norm.weight");
    impl_->lm_head = impl_->load_weight_required(gguf, gguf.has_tensor("output.weight") ? "output.weight" : "token_embd.weight");

    impl_->layers.resize((size_t)cfg.num_hidden_layers);
    impl_->kv_caches.resize((size_t)cfg.num_hidden_layers);
    impl_->slot_tok_history.assign((size_t)max_slots, {});

    size_t kv_layer_bytes = (size_t)max_slots * (size_t)max_seq_len * (size_t)cfg.head_dim * sizeof(float);
    uint32_t topk_cap = (uint32_t)std::max(1, cfg.index_topk);

    uint64_t moe_budget_bytes = (rec_ws > 0) ? (rec_ws * 45u) / 100u : UINT64_MAX;
    {
        vm_statistics64_data_t vm_stat{};
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm_stat), &count) == KERN_SUCCESS) {
            long ps = sysconf(_SC_PAGESIZE);
            uint64_t page_sz = ps > 0 ? (uint64_t)ps : 16384ull;
            uint64_t avail_bytes = ((uint64_t)vm_stat.free_count + (uint64_t)vm_stat.inactive_count + (uint64_t)vm_stat.purgeable_count) * page_sz;
            uint64_t safe_avail = (avail_bytes * 60u) / 100u;
            if (safe_avail < moe_budget_bytes) moe_budget_bytes = safe_avail;
        }
    }
    if (!dspark_path.empty() && moe_budget_bytes > (8ull << 30)) {
        moe_budget_bytes -= (8ull << 30);
    }
    if (const char* env_gb = std::getenv("WARPGRID_DS_MOE_BUDGET_GB")) {
        double gb = std::atof(env_gb);
        if (gb >= 0.0) moe_budget_bytes = (uint64_t)(gb * 1024.0 * 1024.0 * 1024.0);
    }

    uint64_t total_moe_bytes = 0;
    size_t max_gu_stride = 0;
    size_t max_d_stride = 0;
    bool any_f16_exp = false;
    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
        std::string p = "blk." + std::to_string(l) + ".";
        for (const char* suf : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
            std::string nm = p + suf;
            if (!gguf.has_tensor(nm)) continue;
            const auto& t = gguf.tensor(nm);
            total_moe_bytes += t.nbytes;
            if (t.type == GgmlType::F16) any_f16_exp = true;
            uint32_t cols = t.dims.size() >= 1 ? (uint32_t)t.dims[0] : 0;
            uint32_t rows = t.dims.size() >= 2 ? (uint32_t)t.dims[1] : 1;
            uint32_t grps = t.dims.size() >= 3 ? (uint32_t)t.dims[2] : 1;
            if (grps > 1 && cols > 0) {
                size_t st = ggml_row_nbytes(t.type, cols) * rows;
                if (suf[4] == 'd') max_d_stride = std::max(max_d_stride, st);
                else max_gu_stride = std::max(max_gu_stride, st);
            }
        }
    }
    impl_->exp_cache.reset();
    impl_->close_nocache_fds();
    bool use_exp_cache = false;
    if (!any_f16_exp && total_moe_bytes > moe_budget_bytes && max_gu_stride > 0 && max_d_stride > 0) {
        use_exp_cache = impl_->exp_cache.init(
            impl_->dev, cfg.num_hidden_layers, cfg.n_routed_experts,
            max_gu_stride, max_d_stride, moe_budget_bytes);
        if (use_exp_cache) {
            impl_->nocache_fds.assign(gguf.num_shards(), -1);
            for (size_t s = 0; s < gguf.num_shards(); ++s) {
                const std::string& sp = gguf.shard_path(s);
                if (sp.empty()) continue;
                int fd = ::open(sp.c_str(), O_RDONLY);
                if (fd >= 0) {
                    ::fcntl(fd, F_NOCACHE, 1);
                    impl_->nocache_fds[s] = fd;
                }
            }
        }
    }

    uint64_t resident_moe_bytes = 0;
    int resident_moe_layers = 0;

    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
        std::string p = "blk." + std::to_string(l) + ".";
        auto& lw = impl_->layers[(size_t)l];

        lw.attn_norm = impl_->load_f32_required(gguf, p + "attn_norm.weight");
        lw.attn_sinks = impl_->load_f32_required(gguf, p + "attn_sinks.weight");
        lw.wq_a = impl_->load_weight_required(gguf, p + "attn_q_a.weight");
        lw.attn_q_a_norm = impl_->load_f32_required(gguf, p + "attn_q_a_norm.weight");
        lw.wq_b = impl_->load_weight_required(gguf, p + "attn_q_b.weight");
        lw.wkv = impl_->load_weight_required(gguf, p + "attn_kv.weight");
        lw.attn_kv_norm = impl_->load_f32_required(gguf, p + "attn_kv_a_norm.weight");
        lw.wo_a = impl_->load_weight_required(gguf, p + "attn_output_a.weight", (uint32_t)std::max(1, cfg.o_group_count));
        lw.wo_b = impl_->load_weight_required(gguf, p + "attn_output_b.weight");

        lw.hc_attn_fn = impl_->load_weight_required(gguf, p + "hc_attn_fn.weight");
        lw.hc_attn_base = impl_->load_f32_required(gguf, p + "hc_attn_base.weight");
        lw.hc_attn_scale = impl_->load_f32_required(gguf, p + "hc_attn_scale.weight");
        lw.hc_ffn_fn = impl_->load_weight_required(gguf, p + "hc_ffn_fn.weight");
        lw.hc_ffn_base = impl_->load_f32_required(gguf, p + "hc_ffn_base.weight");
        lw.hc_ffn_scale = impl_->load_f32_required(gguf, p + "hc_ffn_scale.weight");

        int ratio = cfg.compress_ratio_for(l);
        if (ratio > 0 && cfg.kv_src[(size_t)l] == l) {
            lw.comp_wkv = impl_->load_weight_required(gguf, p + "attn_compressor_kv.weight");
            lw.comp_norm = impl_->load_f32_required(gguf, p + "attn_compressor_norm.weight");
            if (ratio > 1) {
                lw.comp_wgate = impl_->load_weight_required(gguf, p + "attn_compressor_gate.weight");
            }
            lw.idx_k = impl_->load_weight_required(gguf, p + "indexer.attn_k.weight");
            lw.idx_k_norm = impl_->load_f32_required(gguf, p + "indexer.k_norm.weight");

            size_t max_comp = (size_t)(max_seq_len / ratio) + 1;
            impl_->kv_caches[(size_t)l].comp_raw_kv = impl_->alloc_buf(kv_layer_bytes);
            impl_->kv_caches[(size_t)l].comp_raw_score = impl_->alloc_buf(kv_layer_bytes);
            impl_->kv_caches[(size_t)l].comp_kv = impl_->alloc_buf((size_t)max_slots * max_comp * (size_t)cfg.head_dim * sizeof(float));
            impl_->kv_caches[(size_t)l].comp_ik = impl_->alloc_buf((size_t)max_slots * max_comp * (size_t)cfg.index_head_dim * sizeof(float));
        }
        if (ratio > 0 && cfg.topk_src[(size_t)l] == l) {
            lw.idx_q_b = impl_->load_weight_required(gguf, p + "indexer.attn_q_b.weight");
            lw.idx_proj = impl_->load_weight_required(gguf, p + "indexer.proj.weight");
            impl_->kv_caches[(size_t)l].topk_sel = impl_->alloc_buf((size_t)max_batch_tokens * topk_cap * sizeof(int32_t));
        }

        if (cfg.engram_index(l) >= 0) {
            lw.engram_kv = impl_->load_weight_required(gguf, p + "engram_kv.weight");
            lw.engram_gate = impl_->load_f32_required(gguf, p + "engram_gate.weight");
        }

        lw.ffn_norm = impl_->load_f32_required(gguf, p + "ffn_norm.weight");
        lw.router = impl_->load_weight_required(gguf, p + "ffn_gate_inp.weight");
        lw.router_bias = impl_->load_f32_required(gguf, p + "exp_probs_b.bias");
        uint64_t layer_moe_bytes = 0;
        bool has_f16_exp = false;
        if (gguf.has_tensor(p + "ffn_gate_exps.weight")) {
            const auto& t = gguf.tensor(p + "ffn_gate_exps.weight");
            layer_moe_bytes += t.nbytes;
            has_f16_exp = has_f16_exp || (t.type == GgmlType::F16);
        }
        if (gguf.has_tensor(p + "ffn_up_exps.weight")) {
            const auto& t = gguf.tensor(p + "ffn_up_exps.weight");
            layer_moe_bytes += t.nbytes;
            has_f16_exp = has_f16_exp || (t.type == GgmlType::F16);
        }
        if (gguf.has_tensor(p + "ffn_down_exps.weight")) {
            const auto& t = gguf.tensor(p + "ffn_down_exps.weight");
            layer_moe_bytes += t.nbytes;
            has_f16_exp = has_f16_exp || (t.type == GgmlType::F16);
        }
        bool stream_layer = use_exp_cache || (!has_f16_exp && (resident_moe_bytes + layer_moe_bytes > moe_budget_bytes));
        if (stream_layer) {
            impl_->stream_moe_experts = true;
        } else {
            resident_moe_bytes += layer_moe_bytes;
            resident_moe_layers++;
        }
        lw.ffn_gate_exps = impl_->load_weight_streamable(gguf, p + "ffn_gate_exps.weight", stream_layer);
        lw.ffn_up_exps = impl_->load_weight_streamable(gguf, p + "ffn_up_exps.weight", stream_layer);
        lw.ffn_down_exps = impl_->load_weight_streamable(gguf, p + "ffn_down_exps.weight", stream_layer);
        lw.ffn_gate_shexp = impl_->load_weight_required(gguf, p + "ffn_gate_shexp.weight");
        lw.ffn_up_shexp = impl_->load_weight_required(gguf, p + "ffn_up_shexp.weight");
        lw.ffn_down_shexp = impl_->load_weight_required(gguf, p + "ffn_down_shexp.weight");

        impl_->kv_caches[(size_t)l].kv_cache = impl_->alloc_buf(kv_layer_bytes);
    }

    size_t Tmax = (size_t)max_batch_tokens;
    size_t D = (size_t)cfg.hidden_size;
    size_t hc = (size_t)std::max(1, cfg.hc_mult);
    size_t Qdim = (size_t)cfg.num_attention_heads * (size_t)cfg.head_dim;
    size_t Inter = (size_t)std::max(cfg.ffn_intermediate_size, cfg.expert_intermediate_size);
    for (const auto& lw : impl_->layers) {
        Inter = std::max<size_t>(Inter, lw.ffn_gate_shexp.rows);
        Inter = std::max<size_t>(Inter, lw.ffn_up_shexp.rows);
        Inter = std::max<size_t>(Inter, lw.ffn_gate_exps.rows);
        Inter = std::max<size_t>(Inter, lw.ffn_up_exps.rows);
    }

    impl_->b_streams_a = impl_->alloc_buf(Tmax * hc * D * sizeof(float));
    impl_->b_streams_b = impl_->alloc_buf(Tmax * hc * D * sizeof(float));
    impl_->b_flat_norm = impl_->alloc_buf(Tmax * hc * D * sizeof(float));
    impl_->b_hc_raw = impl_->alloc_buf(Tmax * (2 + hc) * hc * sizeof(float));
    impl_->b_hc_pre_a = impl_->alloc_buf(Tmax * hc * sizeof(float));
    impl_->b_hc_pre_b = impl_->alloc_buf(Tmax * hc * sizeof(float));
    impl_->b_hc_post = impl_->alloc_buf(Tmax * hc * sizeof(float));
    impl_->b_hc_comb = impl_->alloc_buf(Tmax * hc * hc * sizeof(float));

    impl_->b_hidden = impl_->alloc_buf(Tmax * D * sizeof(float));
    impl_->b_norm = impl_->alloc_buf(Tmax * D * sizeof(float));
    impl_->b_qr = impl_->alloc_buf(Tmax * (size_t)std::max(1, cfg.q_lora_rank) * sizeof(float));
    impl_->b_qr_norm = impl_->alloc_buf(Tmax * (size_t)std::max(1, cfg.q_lora_rank) * sizeof(float));
    impl_->b_q = impl_->alloc_buf(Tmax * Qdim * sizeof(float));
    impl_->b_kv = impl_->alloc_buf(Tmax * (size_t)cfg.head_dim * sizeof(float));
    impl_->b_kv_norm = impl_->alloc_buf(Tmax * (size_t)cfg.head_dim * sizeof(float));
    impl_->b_attn_out = impl_->alloc_buf(Tmax * Qdim * sizeof(float));
    impl_->b_oa = impl_->alloc_buf(Tmax * (size_t)std::max(1, cfg.o_group_count * cfg.o_lora_rank) * sizeof(float));
    impl_->b_sub_out = impl_->alloc_buf(Tmax * D * sizeof(float));

    impl_->b_comp_tmp_kv = impl_->alloc_buf(Tmax * (size_t)cfg.head_dim * sizeof(float));
    impl_->b_comp_tmp_score = impl_->alloc_buf(Tmax * (size_t)cfg.head_dim * sizeof(float));
    impl_->b_comp_block = impl_->alloc_buf(Tmax * (size_t)cfg.head_dim * sizeof(float));
    impl_->b_idx_k_tmp = impl_->alloc_buf(Tmax * (size_t)std::max(1, cfg.index_head_dim) * sizeof(float));
    impl_->b_idx_q = impl_->alloc_buf(Tmax * (size_t)std::max(1, cfg.index_head_count * cfg.index_head_dim) * sizeof(float));
    impl_->b_idx_w = impl_->alloc_buf(Tmax * (size_t)std::max(1, cfg.index_head_count) * sizeof(float));
    impl_->b_idx_scores = impl_->alloc_buf(Tmax * (size_t)(max_seq_len + 1) * sizeof(float));

    size_t ple_dim = (size_t)std::max(1, cfg.ple_n_heads * cfg.ple_head_dim);
    size_t n_ple = std::max<size_t>(1, cfg.ple_layers.size());
    impl_->b_engram_rows.resize(n_ple);
    for (size_t i = 0; i < n_ple; ++i) {
        impl_->b_engram_rows[i] = impl_->alloc_buf(Tmax * ple_dim * sizeof(float));
    }
    impl_->b_engram_kv = impl_->alloc_buf(Tmax * (hc + 1) * D * sizeof(float));

    size_t top_k_sz = (size_t)std::max(1, cfg.num_experts_per_tok);
    impl_->b_router_logits = impl_->alloc_buf(Tmax * (size_t)std::max(1, cfg.n_routed_experts) * sizeof(float));
    impl_->b_topk_ids = impl_->alloc_buf(Tmax * top_k_sz * sizeof(int32_t));
    impl_->b_stage_topk_ids = impl_->alloc_buf(Tmax * top_k_sz * sizeof(int32_t));
    impl_->b_topk_w = impl_->alloc_buf(Tmax * top_k_sz * sizeof(float));
    impl_->b_ffn_gate = impl_->alloc_buf(Tmax * top_k_sz * Inter * sizeof(float));
    impl_->b_ffn_up = impl_->alloc_buf(Tmax * top_k_sz * Inter * sizeof(float));
    impl_->b_ffn_act = impl_->alloc_buf(Tmax * top_k_sz * Inter * sizeof(float));
    impl_->b_ffn_tmp = impl_->alloc_buf(Tmax * D * sizeof(float));
    size_t verify_cap = (size_t)std::max(1, draft_k + 1);
    impl_->b_logits = impl_->alloc_buf(verify_cap * (size_t)cfg.vocab_size * sizeof(float));
    impl_->b_argmax = impl_->alloc_buf(std::max<size_t>(8, verify_cap) * sizeof(int32_t));
    impl_->b_dummy_bias = impl_->alloc_buf((size_t)std::max(1, cfg.n_routed_experts) * sizeof(float));

    impl_->dspark.reset();
    if (!dspark_path.empty()) {
        auto& ds = impl_->dspark;
        ds.gguf.open(dspark_path);
        ds.num_layers = (int)ds.gguf.get_i64("dflash.block_count", 3);
        ds.hidden_size = (int)ds.gguf.get_i64("dflash.embedding_length", cfg.hidden_size);
        ds.n_heads = (int)ds.gguf.get_i64("dflash.attention.head_count", cfg.num_attention_heads);
        ds.head_dim = (int)ds.gguf.get_i64("dflash.attention.key_length", cfg.head_dim);
        ds.n_rot = (int)ds.gguf.get_i64("dflash.rope.dimension_count", cfg.n_rot);
        ds.q_lora_rank = (int)ds.gguf.get_i64("dflash.attention.q_lora_rank", cfg.q_lora_rank);
        ds.o_group_count = (int)ds.gguf.get_i64("dflash.attention.output_group_count", cfg.o_group_count);
        ds.o_lora_rank = (int)ds.gguf.get_i64("dflash.attention.output_lora_rank", cfg.o_lora_rank);
        ds.n_swa = (int)ds.gguf.get_i64("dflash.attention.sliding_window", 128);
        ds.n_experts = (int)ds.gguf.get_i64("dflash.expert_count", 128);
        ds.top_k = (int)ds.gguf.get_i64("dflash.expert_used_count", 3);
        ds.hc_mult = (int)ds.gguf.get_i64("dflash.hyper_connection.count", cfg.hc_mult);
        ds.hc_sinkhorn_iters = (int)ds.gguf.get_i64("dflash.hyper_connection.sinkhorn_iterations", cfg.hc_sinkhorn_iters);
        ds.block_size = (int)ds.gguf.get_i64("dflash.block_size", 5);
        ds.mask_token_id = (int)ds.gguf.get_i64("tokenizer.ggml.mask_token_id", 128799);
        ds.rms_norm_eps = (float)ds.gguf.get_f64("dflash.attention.layer_norm_rms_epsilon", cfg.rms_norm_eps);
        ds.hc_eps = (float)ds.gguf.get_f64("dflash.hyper_connection.epsilon", cfg.hc_eps);
        ds.rope_freq_base = (float)ds.gguf.get_f64("dflash.rope.freq_base", cfg.rope_freq_base);
        ds.route_scale = (float)ds.gguf.get_f64("dflash.expert_weights_scale", 1.5);
        if (ds.gguf.has_kv("dflash.swiglu_clamp_exp") && !ds.gguf.kv("dflash.swiglu_clamp_exp").f64_arr.empty()) {
            ds.swiglu_clamp_exp = (float)ds.gguf.kv("dflash.swiglu_clamp_exp").f64_arr[0];
        }
        if (ds.gguf.has_kv("dflash.swiglu_clamp_shexp") && !ds.gguf.kv("dflash.swiglu_clamp_shexp").f64_arr.empty()) {
            ds.swiglu_clamp_shexp = (float)ds.gguf.kv("dflash.swiglu_clamp_shexp").f64_arr[0];
        }
        if (ds.gguf.has_kv("dflash.target_layers")) {
            const auto& tv = ds.gguf.kv("dflash.target_layers");
            for (int64_t x : tv.i64_arr) ds.target_layers.push_back((int)x);
            for (uint64_t x : tv.u64_arr) ds.target_layers.push_back((int)x);
        }
        if (ds.target_layers.empty()) {
            throw std::runtime_error("DSpark GGUF missing dflash.target_layers");
        }
        ds.layer_to_tap.assign((size_t)cfg.num_hidden_layers, -1);
        for (size_t i = 0; i < ds.target_layers.size(); ++i) {
            int tl = ds.target_layers[i];
            if (tl < 0 || tl >= cfg.num_hidden_layers) {
                throw std::runtime_error("DSpark target layer out of range: " + std::to_string(tl));
            }
            ds.layer_to_tap[(size_t)tl] = (int)i;
        }
        ds.fc = impl_->load_weight_required(ds.gguf, "fc.weight");
        ds.enc_output_norm = impl_->load_f32_required(ds.gguf, "enc.output_norm.weight");
        ds.output_norm = impl_->load_f32_required(ds.gguf, "output_norm.weight");
        ds.markov_w1 = impl_->load_weight_required(ds.gguf, "markov_w1.weight");
        ds.markov_w2 = impl_->load_weight_required(ds.gguf, "markov_w2.weight");
        ds.markov_rank = (int)ds.markov_w1.cols;
        ds.layers.resize((size_t)ds.num_layers);
        ds.kv_rings.resize((size_t)ds.num_layers);
        size_t ring_bytes = (size_t)max_slots * (size_t)ds.n_swa * (size_t)ds.head_dim * sizeof(float);
        for (int il = 0; il < ds.num_layers; ++il) {
            std::string p = "blk." + std::to_string(il) + ".";
            auto& dl = ds.layers[(size_t)il];
            dl.attn_norm = impl_->load_f32_required(ds.gguf, p + "attn_norm.weight");
            dl.attn_sinks = impl_->load_f32_required(ds.gguf, p + "attn_sinks.weight");
            dl.wq_a = impl_->load_weight_required(ds.gguf, p + "attn_q_a.weight");
            dl.attn_q_a_norm = impl_->load_f32_required(ds.gguf, p + "attn_q_a_norm.weight");
            dl.wq_b = impl_->load_weight_required(ds.gguf, p + "attn_q_b.weight");
            dl.wkv = impl_->load_weight_required(ds.gguf, p + "attn_kv.weight");
            dl.attn_kv_norm = impl_->load_f32_required(ds.gguf, p + "attn_kv_a_norm.weight");
            dl.wo_a = impl_->load_weight_required(ds.gguf, p + "attn_output_a.weight", (uint32_t)std::max(1, ds.o_group_count));
            dl.wo_b = impl_->load_weight_required(ds.gguf, p + "attn_output_b.weight");
            dl.hc_attn_fn = impl_->load_weight_required(ds.gguf, p + "hc_attn_fn.weight");
            dl.hc_attn_base = impl_->load_f32_required(ds.gguf, p + "hc_attn_base.weight");
            dl.hc_attn_scale = impl_->load_f32_required(ds.gguf, p + "hc_attn_scale.weight");
            dl.hc_ffn_fn = impl_->load_weight_required(ds.gguf, p + "hc_ffn_fn.weight");
            dl.hc_ffn_base = impl_->load_f32_required(ds.gguf, p + "hc_ffn_base.weight");
            dl.hc_ffn_scale = impl_->load_f32_required(ds.gguf, p + "hc_ffn_scale.weight");
            dl.ffn_norm = impl_->load_f32_required(ds.gguf, p + "ffn_norm.weight");
            dl.router = impl_->load_weight_required(ds.gguf, p + "ffn_gate_inp.weight");
            dl.router_bias = impl_->load_f32_required(ds.gguf, p + "exp_probs_b.bias");
            dl.ffn_gate_exps = impl_->load_weight_required(ds.gguf, p + "ffn_gate_exps.weight");
            dl.ffn_up_exps = impl_->load_weight_required(ds.gguf, p + "ffn_up_exps.weight");
            dl.ffn_down_exps = impl_->load_weight_required(ds.gguf, p + "ffn_down_exps.weight");
            dl.ffn_gate_shexp = impl_->load_weight_required(ds.gguf, p + "ffn_gate_shexp.weight");
            dl.ffn_up_shexp = impl_->load_weight_required(ds.gguf, p + "ffn_up_shexp.weight");
            dl.ffn_down_shexp = impl_->load_weight_required(ds.gguf, p + "ffn_down_shexp.weight");
            ds.kv_rings[(size_t)il] = impl_->alloc_buf(ring_bytes);
        }
        size_t max_blk = (size_t)std::max(8, ds.block_size + 1);
        ds.b_taps = impl_->alloc_buf(Tmax * ds.target_layers.size() * D * sizeof(float));
        ds.b_inp_g = impl_->alloc_buf(Tmax * D * sizeof(float));
        ds.b_logits_blk = impl_->alloc_buf(max_blk * (size_t)cfg.vocab_size * sizeof(float));
        ds.b_markov_logits = impl_->alloc_buf((size_t)cfg.vocab_size * sizeof(float));
        ds.b_markov_toks = impl_->alloc_buf((max_blk + 1) * sizeof(int32_t));
        if (draft_k <= 0) draft_k = 3;
        if (draft_k > ds.block_size) draft_k = ds.block_size;
        ds.loaded = true;
        fprintf(stderr, "[warpgrid] Loaded DeepSeek DSpark: %d layers, E=%d (top-%d MXFP4), swa=%d, block=%d, draft_k=%d, taps=%zu, mask=%d\n",
                ds.num_layers, ds.n_experts, ds.top_k, ds.n_swa, ds.block_size, draft_k, ds.target_layers.size(), ds.mask_token_id);
    }

    if (max_slots == 1) {
        num_kv_blocks = std::max(1, (max_seq_len + 63) / 64);
        num_snapshots = 1;
    }

    if (impl_->exp_cache.enabled()) {
        double cache_gib = (double)impl_->exp_cache.num_slots *
                           (double)(impl_->exp_cache.gu_stride * 2u + impl_->exp_cache.d_stride) /
                           (1024.0 * 1024.0 * 1024.0);
        fprintf(stderr, "[warpgrid] Loaded DeepSeek-V4.1 (%s): %d layers, D=%d, H=%d, hc=%d, E=%d (top-%d), slots=%d, max_seq=%d, engram=%s, exp_cache=%d/%d (%.1fGiB F_NOCACHE), prefix_cache=%s\n",
                cfg.arch.c_str(), cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads,
                cfg.hc_mult, cfg.n_routed_experts, cfg.num_experts_per_tok, max_slots, max_seq_len,
                (engram.is_open() || impl_->engram_in_main_gguf) ? "ON" : "OFF",
                impl_->exp_cache.num_slots, cfg.num_hidden_layers * cfg.n_routed_experts, cache_gib,
                num_snapshots > 0 ? "ON" : "OFF");
    } else {
        fprintf(stderr, "[warpgrid] Loaded DeepSeek-V4.1 (%s): %d layers, D=%d, H=%d, hc=%d, E=%d (top-%d), slots=%d, max_seq=%d, engram=%s, moe_res=%d/%d, prefix_cache=%s\n",
                cfg.arch.c_str(), cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads,
                cfg.hc_mult, cfg.n_routed_experts, cfg.num_experts_per_tok, max_slots, max_seq_len,
                (engram.is_open() || impl_->engram_in_main_gguf) ? "ON" : "OFF",
                resident_moe_layers, cfg.num_hidden_layers,
                num_snapshots > 0 ? "ON" : "OFF");
    }
}

void DeepSeekV41Model::reset_cache(int slot_id) {
    if (slot_id < 0 || slot_id >= max_slots) return;
    if ((size_t)slot_id < impl_->slot_tok_history.size()) {
        impl_->slot_tok_history[(size_t)slot_id].clear();
    }
    size_t slot_bytes = (size_t)max_seq_len * (size_t)cfg.head_dim * sizeof(float);
    for (size_t l = 0; l < impl_->kv_caches.size(); ++l) {
        auto& kv = impl_->kv_caches[l];
        memset(static_cast<uint8_t*>(kv.kv_cache.contents) + (size_t)slot_id * slot_bytes, 0, slot_bytes);
        if (kv.comp_raw_kv != nil) {
            memset(static_cast<uint8_t*>(kv.comp_raw_kv.contents) + (size_t)slot_id * slot_bytes, 0, slot_bytes);
            memset(static_cast<uint8_t*>(kv.comp_raw_score.contents) + (size_t)slot_id * slot_bytes, 0, slot_bytes);
        }
    }
    if (impl_->dspark.loaded) {
        size_t ds_slot_bytes = (size_t)impl_->dspark.n_swa * (size_t)impl_->dspark.head_dim * sizeof(float);
        for (auto& ring : impl_->dspark.kv_rings) {
            if (ring != nil) {
                memset(static_cast<uint8_t*>(ring.contents) + (size_t)slot_id * ds_slot_bytes, 0, ds_slot_bytes);
            }
        }
    }
}

void DeepSeekV41Model::forward_logits(const int32_t* h_tokens, int T, int start_pos, std::vector<float>& h_logits, int slot_id) {
    forward_and_greedy(h_tokens, T, start_pos, slot_id);
    h_logits.resize((size_t)cfg.vocab_size);
    memcpy(h_logits.data(), impl_->b_logits.contents, (size_t)cfg.vocab_size * sizeof(float));
}

int32_t DeepSeekV41Model::forward_and_greedy(const int32_t* h_tokens, int T, int start_pos, int slot_id) {
    @autoreleasepool {
    if (T <= 0 || T > max_batch_tokens) throw std::runtime_error("Invalid token batch size T");
    if (slot_id < 0 || slot_id >= max_slots) throw std::runtime_error("Invalid slot_id");
    if (start_pos < 0 || start_pos + T > max_seq_len) throw std::runtime_error("Sequence length exceeds max_seq_len");

    auto& hist = impl_->slot_tok_history[(size_t)slot_id];
    if ((int)hist.size() < start_pos + T) hist.resize((size_t)(start_pos + T), 0);
    for (int t = 0; t < T; ++t) hist[(size_t)(start_pos + t)] = h_tokens[t];

    size_t D = (size_t)cfg.hidden_size;
    uint32_t hc = (uint32_t)std::max(1, cfg.hc_mult);
    float* hidden_ptr = static_cast<float*>(impl_->b_hidden.contents);
    const auto& emb_ti = gguf.tensor("token_embd.weight");
    size_t row_stride = ggml_row_nbytes(emb_ti.type, emb_ti.dims[0]);
    const uint8_t* base = gguf.tensor_data("token_embd.weight");
    uint64_t nrows = emb_ti.num_rows();
    for (int t = 0; t < T; ++t) {
        uint64_t tid = (uint64_t)std::clamp(h_tokens[t], 0, (int32_t)(nrows > 0 ? nrows - 1 : 0));
        dequant_row_fp32(emb_ti.type, base + tid * row_stride, hidden_ptr + (size_t)t * D, D);
    }

    if ((engram.is_open() || impl_->engram_in_main_gguf) && cfg.ple_n_heads > 0 && cfg.ple_head_dim > 0) {
        std::vector<int32_t> row_ids((size_t)cfg.ple_n_heads, 0);
        for (size_t e_idx = 0; e_idx < cfg.ple_layers.size(); ++e_idx) {
            float* erows_ptr = static_cast<float*>(impl_->b_engram_rows[e_idx].contents);
            for (int t = 0; t < T; ++t) {
                cfg.engram_hash_token((int)e_idx, hist, start_pos + t, row_ids.data());
                for (int h = 0; h < cfg.ple_n_heads; ++h) {
                    float* dst = erows_ptr + ((size_t)t * (size_t)cfg.ple_n_heads + (size_t)h) * (size_t)cfg.ple_head_dim;
                    if (engram.is_open()) {
                        engram.lookup_row("per_layer_token_embd.weight", (uint64_t)row_ids[(size_t)h], dst, (size_t)cfg.ple_head_dim);
                    } else {
                        const auto& pti = gguf.tensor("per_layer_token_embd.weight");
                        size_t pstride = ggml_row_nbytes(pti.type, pti.dims[0]);
                        const uint8_t* pbase = gguf.tensor_data("per_layer_token_embd.weight");
                        dequant_row_fp32(pti.type, pbase + (size_t)row_ids[(size_t)h] * pstride, dst, (size_t)cfg.ple_head_dim);
                    }
                }
            }
        }
    }

    id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    {
        DsStreamCollapseParams scp{(uint32_t)T, (uint32_t)D, hc};
        [enc setComputePipelineState:impl_->pso("ds_mhc_expand_f32")];
        [enc setBuffer:impl_->b_hidden offset:0 atIndex:0];
        [enc setBuffer:impl_->b_streams_a offset:0 atIndex:1];
        [enc setBytes:&scp length:sizeof(scp) atIndex:2];
        [enc dispatchThreads:MTLSizeMake((uint32_t)D, (uint32_t)T, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(64u, (uint32_t)D), 1, 1)];
    }

    id<MTLBuffer> pre_attn = impl_->b_hc_pre_a;
    id<MTLBuffer> pre_ffn = impl_->b_hc_pre_b;

    size_t slot_kv_bytes = (size_t)max_seq_len * (size_t)cfg.head_dim * sizeof(float);
    size_t slot_kv_off = (size_t)slot_id * slot_kv_bytes;
    uint32_t n_nope = (uint32_t)std::max(0, cfg.head_dim - cfg.n_rot);
    uint32_t n_rot = (uint32_t)std::max(0, cfg.n_rot);
    uint32_t topk_cap = (uint32_t)std::max(1, cfg.index_topk);

    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
        const auto& lw = impl_->layers[(size_t)l];
        auto& kv = impl_->kv_caches[(size_t)l];

        int e_idx = cfg.engram_index(l);
        if (e_idx >= 0 && (engram.is_open() || impl_->engram_in_main_gguf) && cfg.ple_n_heads > 0 && cfg.ple_head_dim > 0) {
            impl_->encode_gemv(enc, impl_->b_engram_rows[(size_t)e_idx], lw.engram_kv, impl_->b_engram_kv, (uint32_t)T);
            DsEngramApplyParams eap{(uint32_t)T, (uint32_t)D, hc, cfg.rms_norm_eps};
            [enc setComputePipelineState:impl_->pso("ds_engram_apply_f32")];
            [enc setBuffer:impl_->b_streams_a offset:0 atIndex:0];
            [enc setBuffer:impl_->b_engram_kv offset:0 atIndex:1];
            [enc setBuffer:lw.engram_gate offset:0 atIndex:2];
            [enc setBytes:&eap length:sizeof(eap) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(hc, (uint32_t)T, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }

        if (impl_->dspark.loaded && (size_t)l < impl_->dspark.layer_to_tap.size() && impl_->dspark.layer_to_tap[(size_t)l] >= 0) {
            DsHcMeanTapParams tp{(uint32_t)T, (uint32_t)D, hc, (uint32_t)impl_->dspark.target_layers.size(), (uint32_t)impl_->dspark.layer_to_tap[(size_t)l]};
            [enc setComputePipelineState:impl_->pso("ds_hc_mean_to_tap_f32")];
            [enc setBuffer:impl_->b_streams_a offset:0 atIndex:0];
            [enc setBuffer:impl_->dspark.b_taps offset:0 atIndex:1];
            [enc setBytes:&tp length:sizeof(tp) atIndex:2];
            [enc dispatchThreads:MTLSizeMake((uint32_t)D, (uint32_t)T, 1)
                threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(64u, (uint32_t)D), 1, 1)];
        }

        impl_->encode_hc_mixes(enc, impl_->b_streams_a, lw.hc_attn_fn, lw.hc_attn_scale, lw.hc_attn_base,
                               pre_attn, impl_->b_hc_post, impl_->b_hc_comb,
                               (uint32_t)T, (uint32_t)D, hc, (uint32_t)cfg.hc_sinkhorn_iters, cfg.rms_norm_eps, cfg.hc_eps);

        if (l > 0) {
            impl_->encode_hc_pre(enc, impl_->b_streams_a, pre_ffn, impl_->b_hidden, (uint32_t)T, (uint32_t)D, hc);
        } else {
            [enc endEncoding];
            id<MTLBlitCommandEncoder> bl0 = [cb blitCommandEncoder];
            for (int t = 0; t < T; ++t) {
                [bl0 copyFromBuffer:impl_->b_streams_a sourceOffset:(size_t)t * hc * D * sizeof(float)
                           toBuffer:impl_->b_hidden destinationOffset:(size_t)t * D * sizeof(float)
                               size:D * sizeof(float)];
            }
            [bl0 endEncoding];
            enc = [cb computeCommandEncoder];
        }

        impl_->encode_rmsnorm(enc, impl_->b_hidden, lw.attn_norm, impl_->b_norm, (uint32_t)T, (uint32_t)D, cfg.rms_norm_eps);

        impl_->encode_gemv(enc, impl_->b_norm, lw.wq_a, impl_->b_qr, (uint32_t)T);
        impl_->encode_rmsnorm(enc, impl_->b_qr, lw.attn_q_a_norm, impl_->b_qr_norm, (uint32_t)T, (uint32_t)cfg.q_lora_rank, cfg.rms_norm_eps);
        impl_->encode_gemv(enc, impl_->b_qr_norm, lw.wq_b, impl_->b_q, (uint32_t)T);

        impl_->encode_gemv(enc, impl_->b_norm, lw.wkv, impl_->b_kv, (uint32_t)T);
        impl_->encode_rmsnorm(enc, impl_->b_kv, lw.attn_kv_norm, impl_->b_kv_norm, (uint32_t)T, (uint32_t)cfg.head_dim, cfg.rms_norm_eps);

        int ratio = cfg.compress_ratio_for(l);
        float freq_base_l = (ratio != 0) ? cfg.compress_rope_freq_base : cfg.rope_freq_base;

        if (n_rot >= 2) {
            DsPartialRopeParams rq{(uint32_t)T, (uint32_t)start_pos, 1u, (uint32_t)cfg.num_attention_heads, (uint32_t)cfg.head_dim, n_nope, n_rot, 0u, freq_base_l};
            [enc setComputePipelineState:impl_->pso("ds_rope_partial_f32")];
            [enc setBuffer:impl_->b_q offset:0 atIndex:0];
            [enc setBytes:&rq length:sizeof(rq) atIndex:1];
            [enc dispatchThreads:MTLSizeMake(n_rot / 2u, (uint32_t)cfg.num_attention_heads, (uint32_t)T)
                threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(32u, n_rot / 2u), 1, 1)];

            DsPartialRopeParams rk{(uint32_t)T, (uint32_t)start_pos, 1u, 1u, (uint32_t)cfg.head_dim, n_nope, n_rot, 0u, freq_base_l};
            [enc setBuffer:impl_->b_kv_norm offset:0 atIndex:0];
            [enc setBytes:&rk length:sizeof(rk) atIndex:1];
            [enc dispatchThreads:MTLSizeMake(n_rot / 2u, 1u, (uint32_t)T)
                threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(32u, n_rot / 2u), 1, 1)];
        }

        if (ratio > 0 && cfg.kv_src[(size_t)l] == l) {
            impl_->encode_gemv(enc, impl_->b_norm, lw.comp_wkv, impl_->b_comp_tmp_kv, (uint32_t)T);
            if (ratio > 1) {
                impl_->encode_gemv(enc, impl_->b_norm, lw.comp_wgate, impl_->b_comp_tmp_score, (uint32_t)T);
            } else {
                uint32_t zn = (uint32_t)((size_t)T * (size_t)cfg.head_dim);
                [enc setComputePipelineState:impl_->pso("ds_zero_f32")];
                [enc setBuffer:impl_->b_comp_tmp_score offset:0 atIndex:0];
                [enc setBytes:&zn length:sizeof(zn) atIndex:1];
                [enc dispatchThreadgroups:MTLSizeMake((zn + 255u) / 256u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }

        [enc endEncoding];

        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        size_t kv_copy_bytes = (size_t)T * (size_t)cfg.head_dim * sizeof(float);
        size_t kv_dst_off = slot_kv_off + (size_t)start_pos * (size_t)cfg.head_dim * sizeof(float);
        [blit copyFromBuffer:impl_->b_kv_norm sourceOffset:0 toBuffer:kv.kv_cache destinationOffset:kv_dst_off size:kv_copy_bytes];
        if (ratio > 0 && cfg.kv_src[(size_t)l] == l) {
            [blit copyFromBuffer:impl_->b_comp_tmp_kv sourceOffset:0 toBuffer:kv.comp_raw_kv destinationOffset:kv_dst_off size:kv_copy_bytes];
            [blit copyFromBuffer:impl_->b_comp_tmp_score sourceOffset:0 toBuffer:kv.comp_raw_score destinationOffset:kv_dst_off size:kv_copy_bytes];
        }
        [blit endEncoding];

        enc = [cb computeCommandEncoder];

        uint32_t total_comp_blocks = (ratio > 0) ? (uint32_t)((start_pos + T) / ratio) : 0u;
        if (ratio > 0 && cfg.kv_src[(size_t)l] == l && total_comp_blocks > 0) {
            uint32_t first_blk = (uint32_t)(start_pos / ratio);
            uint32_t n_new = total_comp_blocks - first_blk;
            if (n_new > 0) {
                size_t max_comp = (size_t)(max_seq_len / ratio) + 1;
                size_t raw_off = slot_kv_off + (size_t)first_blk * (size_t)ratio * (size_t)cfg.head_dim * sizeof(float);
                size_t comp_off = (size_t)slot_id * max_comp * (size_t)cfg.head_dim * sizeof(float) +
                                  (size_t)first_blk * (size_t)cfg.head_dim * sizeof(float);
                size_t ik_off = (size_t)slot_id * max_comp * (size_t)cfg.index_head_dim * sizeof(float) +
                                (size_t)first_blk * (size_t)cfg.index_head_dim * sizeof(float);

                DsCompGroupParams cgp{n_new, (uint32_t)ratio, (uint32_t)cfg.head_dim, cfg.rms_norm_eps};
                [enc setComputePipelineState:impl_->pso("ds_comp_group_reduce_f32")];
                [enc setBuffer:kv.comp_raw_kv offset:raw_off atIndex:0];
                [enc setBuffer:kv.comp_raw_score offset:raw_off atIndex:1];
                [enc setBuffer:lw.comp_norm offset:0 atIndex:2];
                [enc setBuffer:kv.comp_kv offset:comp_off atIndex:3];
                [enc setBytes:&cgp length:sizeof(cgp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(n_new, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                impl_->encode_gemv(enc, kv.comp_kv, lw.idx_k, impl_->b_idx_k_tmp, n_new, comp_off, 0, 0);
                impl_->encode_rmsnorm(enc, impl_->b_idx_k_tmp, lw.idx_k_norm, kv.comp_ik, n_new, (uint32_t)cfg.index_head_dim, cfg.rms_norm_eps, 0, ik_off);

                if (n_rot >= 2) {
                    uint32_t ik_nope = (uint32_t)std::max(0, cfg.index_head_dim - cfg.n_rot);
                    DsPartialRopeParams rik{n_new, first_blk * (uint32_t)ratio, (uint32_t)ratio, 1u, (uint32_t)cfg.index_head_dim, ik_nope, n_rot, 0u, freq_base_l};
                    [enc setComputePipelineState:impl_->pso("ds_rope_partial_f32")];
                    [enc setBuffer:kv.comp_ik offset:ik_off atIndex:0];
                    [enc setBytes:&rik length:sizeof(rik) atIndex:1];
                    [enc dispatchThreads:MTLSizeMake(n_rot / 2u, 1u, n_new)
                        threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(32u, n_rot / 2u), 1, 1)];

                    DsPartialRopeParams rc{n_new, first_blk * (uint32_t)ratio, (uint32_t)ratio, 1u, (uint32_t)cfg.head_dim, n_nope, n_rot, 0u, freq_base_l};
                    [enc setBuffer:kv.comp_kv offset:comp_off atIndex:0];
                    [enc setBytes:&rc length:sizeof(rc) atIndex:1];
                    [enc dispatchThreads:MTLSizeMake(n_rot / 2u, 1u, n_new)
                        threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(32u, n_rot / 2u), 1, 1)];
                }
            }
        }

        if (ratio > 0 && cfg.topk_src[(size_t)l] == l && total_comp_blocks > 0) {
            int src_kv_l = cfg.kv_src[(size_t)l];
            size_t max_comp = (size_t)(max_seq_len / ratio) + 1;
            size_t slot_ik_off = (size_t)slot_id * max_comp * (size_t)cfg.index_head_dim * sizeof(float);

            impl_->encode_gemv(enc, impl_->b_qr_norm, lw.idx_q_b, impl_->b_idx_q, (uint32_t)T);
            if (n_rot >= 2) {
                uint32_t iq_nope = (uint32_t)std::max(0, cfg.index_head_dim - cfg.n_rot);
                DsPartialRopeParams riq{(uint32_t)T, (uint32_t)start_pos, 1u, (uint32_t)cfg.index_head_count, (uint32_t)cfg.index_head_dim, iq_nope, n_rot, 0u, freq_base_l};
                [enc setComputePipelineState:impl_->pso("ds_rope_partial_f32")];
                [enc setBuffer:impl_->b_idx_q offset:0 atIndex:0];
                [enc setBytes:&riq length:sizeof(riq) atIndex:1];
                [enc dispatchThreads:MTLSizeMake(n_rot / 2u, (uint32_t)cfg.index_head_count, (uint32_t)T)
                    threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(32u, n_rot / 2u), 1, 1)];
            }
            impl_->encode_gemv(enc, impl_->b_norm, lw.idx_proj, impl_->b_idx_w, (uint32_t)T);

            DsIndexerParams ip{
                (uint32_t)T, (uint32_t)start_pos, total_comp_blocks, (uint32_t)(max_seq_len + 1),
                (uint32_t)cfg.index_head_count, (uint32_t)cfg.index_head_dim,
                (uint32_t)ratio, (uint32_t)cfg.sliding_window, topk_cap,
                1.0f / std::sqrt((float)(cfg.index_head_dim * cfg.index_head_count))
            };
            [enc setComputePipelineState:impl_->pso("ds_indexer_score_topk_f32")];
            [enc setBuffer:impl_->b_idx_q offset:0 atIndex:0];
            [enc setBuffer:impl_->kv_caches[(size_t)src_kv_l].comp_ik offset:slot_ik_off atIndex:1];
            [enc setBuffer:impl_->b_idx_w offset:0 atIndex:2];
            [enc setBuffer:impl_->b_idx_scores offset:0 atIndex:3];
            [enc setBuffer:kv.topk_sel offset:0 atIndex:4];
            [enc setBytes:&ip length:sizeof(ip) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((uint32_t)T, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        if (ratio > 0 && total_comp_blocks > 0 && cfg.kv_src[(size_t)l] >= 0 && cfg.topk_src[(size_t)l] >= 0) {
            int src_kv_l = cfg.kv_src[(size_t)l];
            int src_tk_l = cfg.topk_src[(size_t)l];
            size_t max_comp = (size_t)(max_seq_len / ratio) + 1;
            size_t slot_comp_off = (size_t)slot_id * max_comp * (size_t)cfg.head_dim * sizeof(float);
            DsMqaSinkAttnParams ap{
                (uint32_t)T, (uint32_t)start_pos, (uint32_t)cfg.num_attention_heads, (uint32_t)cfg.head_dim,
                (uint32_t)cfg.sliding_window, topk_cap, 1.0f / std::sqrt((float)cfg.head_dim)
            };
            [enc setComputePipelineState:impl_->pso("ds_mqa_sink_comp_attn_f32")];
            [enc setBuffer:impl_->b_q offset:0 atIndex:0];
            [enc setBuffer:kv.kv_cache offset:slot_kv_off atIndex:1];
            [enc setBuffer:impl_->kv_caches[(size_t)src_kv_l].comp_kv offset:slot_comp_off atIndex:2];
            [enc setBuffer:impl_->kv_caches[(size_t)src_tk_l].topk_sel offset:0 atIndex:3];
            [enc setBuffer:lw.attn_sinks offset:0 atIndex:4];
            [enc setBuffer:impl_->b_attn_out offset:0 atIndex:5];
            [enc setBytes:&ap length:sizeof(ap) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((uint32_t)cfg.num_attention_heads, (uint32_t)T, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        } else {
            DsMqaSinkAttnParams ap{
                (uint32_t)T, (uint32_t)start_pos, (uint32_t)cfg.num_attention_heads, (uint32_t)cfg.head_dim,
                (uint32_t)cfg.sliding_window, 0u, 1.0f / std::sqrt((float)cfg.head_dim)
            };
            [enc setComputePipelineState:impl_->pso("ds_mqa_sink_attn_f32")];
            [enc setBuffer:impl_->b_q offset:0 atIndex:0];
            [enc setBuffer:kv.kv_cache offset:slot_kv_off atIndex:1];
            [enc setBuffer:lw.attn_sinks offset:0 atIndex:2];
            [enc setBuffer:impl_->b_attn_out offset:0 atIndex:3];
            [enc setBytes:&ap length:sizeof(ap) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((uint32_t)cfg.num_attention_heads, (uint32_t)T, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }

        if (n_rot >= 2) {
            DsPartialRopeParams rinv{(uint32_t)T, (uint32_t)start_pos, 1u, (uint32_t)cfg.num_attention_heads, (uint32_t)cfg.head_dim, n_nope, n_rot, 1u, freq_base_l};
            [enc setComputePipelineState:impl_->pso("ds_rope_partial_f32")];
            [enc setBuffer:impl_->b_attn_out offset:0 atIndex:0];
            [enc setBytes:&rinv length:sizeof(rinv) atIndex:1];
            [enc dispatchThreads:MTLSizeMake(n_rot / 2u, (uint32_t)cfg.num_attention_heads, (uint32_t)T)
                threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(32u, n_rot / 2u), 1, 1)];
        }

        uint32_t o_groups = lw.wo_a.groups;
        if (lw.wo_a.type == GgmlType::Q6_K && (lw.wo_a.cols & 255u) == 0u) {
            GgufGroupedGemvParams gp{(uint32_t)T, o_groups, lw.wo_a.rows, lw.wo_a.cols, lw.wo_a.group_stride_bytes};
            [enc setComputePipelineState:impl_->pso("gguf_grouped_gemv_q6_k_fast")];
            [enc setBuffer:impl_->b_attn_out offset:0 atIndex:0];
            [enc setBuffer:lw.wo_a.buf offset:lw.wo_a.base_offset atIndex:1];
            [enc setBuffer:impl_->b_oa offset:0 atIndex:2];
            [enc setBytes:&gp length:sizeof(gp) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((lw.wo_a.rows + 3u) / 4u, (uint32_t)T * o_groups, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        } else {
            for (int t = 0; t < T; ++t) {
                for (uint32_t g = 0; g < o_groups; ++g) {
                    size_t x_off = ((size_t)t * o_groups + g) * lw.wo_a.cols * sizeof(float);
                    size_t w_off = (size_t)g * lw.wo_a.group_stride_bytes;
                    size_t y_off = ((size_t)t * o_groups + g) * lw.wo_a.rows * sizeof(float);
                    impl_->encode_gemv(enc, impl_->b_attn_out, lw.wo_a, impl_->b_oa, 1u, x_off, w_off, y_off);
                }
            }
        }
        impl_->encode_gemv(enc, impl_->b_oa, lw.wo_b, impl_->b_sub_out, (uint32_t)T);

        impl_->encode_hc_post(enc, impl_->b_sub_out, impl_->b_streams_a, impl_->b_hc_post, impl_->b_hc_comb,
                              impl_->b_streams_b, (uint32_t)T, (uint32_t)D, hc);
        std::swap(impl_->b_streams_a, impl_->b_streams_b);

        impl_->encode_hc_mixes(enc, impl_->b_streams_a, lw.hc_ffn_fn, lw.hc_ffn_scale, lw.hc_ffn_base,
                               pre_ffn, impl_->b_hc_post, impl_->b_hc_comb,
                               (uint32_t)T, (uint32_t)D, hc, (uint32_t)cfg.hc_sinkhorn_iters, cfg.rms_norm_eps, cfg.hc_eps);
        impl_->encode_hc_pre(enc, impl_->b_streams_a, pre_attn, impl_->b_hidden, (uint32_t)T, (uint32_t)D, hc);

        impl_->encode_rmsnorm(enc, impl_->b_hidden, lw.ffn_norm, impl_->b_norm, (uint32_t)T, (uint32_t)D, cfg.rms_norm_eps);

        impl_->encode_gemv(enc, impl_->b_norm, lw.ffn_gate_shexp, impl_->b_ffn_gate, (uint32_t)T);
        impl_->encode_gemv(enc, impl_->b_norm, lw.ffn_up_shexp, impl_->b_ffn_up, (uint32_t)T);
        impl_->encode_swiglu(enc, impl_->b_ffn_gate, impl_->b_ffn_up, impl_->b_ffn_act, (uint32_t)((size_t)T * lw.ffn_gate_shexp.rows), cfg.swiglu_clamp_shexp);
        impl_->encode_gemv(enc, impl_->b_ffn_act, lw.ffn_down_shexp, impl_->b_sub_out, (uint32_t)T);

        impl_->encode_gemv(enc, impl_->b_norm, lw.router, impl_->b_router_logits, (uint32_t)T);
        DsRouterParams rp{
            (uint32_t)T, (uint32_t)cfg.n_routed_experts, (uint32_t)cfg.num_experts_per_tok,
            lw.router_bias != nil ? 1u : 0u, cfg.route_scale
        };
        [enc setComputePipelineState:impl_->pso("ds_router_sqrtsoftplus_topk_f32")];
        [enc setBuffer:impl_->b_router_logits offset:0 atIndex:0];
        [enc setBuffer:(lw.router_bias != nil ? lw.router_bias : impl_->b_dummy_bias) offset:0 atIndex:1];
        [enc setBuffer:impl_->b_topk_ids offset:0 atIndex:2];
        [enc setBuffer:impl_->b_topk_w offset:0 atIndex:3];
        [enc setBytes:&rp length:sizeof(rp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((uint32_t)T, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        uint32_t top_k_u = (uint32_t)cfg.num_experts_per_tok;
        if (top_k_u > 0 && lw.ffn_gate_exps.groups > 0) {
            WeightBuf gate_w = lw.ffn_gate_exps;
            WeightBuf up_w = lw.ffn_up_exps;
            WeightBuf down_w = lw.ffn_down_exps;
            id<MTLBuffer> ids_buf = impl_->b_topk_ids;
            if (lw.ffn_gate_exps.is_streamed()) {
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];

                const int32_t* raw_ids = static_cast<const int32_t*>(impl_->b_topk_ids.contents);
                int32_t* st_ids = static_cast<int32_t*>(impl_->b_stage_topk_ids.contents);
                uint32_t n_sel = (uint32_t)T * top_k_u;
                std::vector<int32_t> remap((size_t)std::max(1, cfg.n_routed_experts), -1);
                std::vector<int32_t> uniq_ids;
                std::vector<int32_t> full_loc(n_sel, 0);
                uniq_ids.reserve(std::min<size_t>(n_sel, (size_t)cfg.n_routed_experts));
                for (uint32_t i = 0; i < n_sel; ++i) {
                    int32_t eid = raw_ids[i];
                    if (eid < 0 || eid >= cfg.n_routed_experts) eid = 0;
                    int32_t loc = remap[(size_t)eid];
                    if (loc < 0) {
                        loc = (int32_t)uniq_ids.size();
                        remap[(size_t)eid] = loc;
                        uniq_ids.push_back(eid);
                    }
                    full_loc[i] = loc;
                }
                size_t U = uniq_ids.size();
                size_t g_stride = lw.ffn_gate_exps.group_stride_bytes;
                size_t u_stride = lw.ffn_up_exps.group_stride_bytes;
                size_t d_stride = lw.ffn_down_exps.group_stride_bytes;
                const uint8_t* src_g = lw.ffn_gate_exps.raw_data;
                const uint8_t* src_u = lw.ffn_up_exps.raw_data;
                const uint8_t* src_d = lw.ffn_down_exps.raw_data;
                int fd_g = lw.ffn_gate_exps.nocache_fd;
                int fd_u = lw.ffn_up_exps.nocache_fd;
                int fd_d = lw.ffn_down_exps.nocache_fd;
                uint64_t off_g = lw.ffn_gate_exps.file_offset;
                uint64_t off_u = lw.ffn_up_exps.file_offset;
                uint64_t off_d = lw.ffn_down_exps.file_offset;

                if (impl_->exp_cache.enabled() && U <= (size_t)impl_->exp_cache.num_slots) {
                    std::vector<int32_t> slot_for_u;
                    std::vector<std::pair<int32_t, int32_t>> miss_list;
                    impl_->exp_cache.pin_and_resolve(l, uniq_ids.data(), U, slot_for_u, miss_list);
                    for (uint32_t i = 0; i < n_sel; ++i) {
                        st_ids[i] = slot_for_u[(size_t)full_loc[i]];
                    }
                    if (!miss_list.empty()) {
                        uint8_t* dst_g = static_cast<uint8_t*>(impl_->exp_cache.buf_g.contents);
                        uint8_t* dst_u = static_cast<uint8_t*>(impl_->exp_cache.buf_u.contents);
                        uint8_t* dst_d = static_cast<uint8_t*>(impl_->exp_cache.buf_d.contents);
                        size_t c_gu = impl_->exp_cache.gu_stride;
                        size_t c_d = impl_->exp_cache.d_stride;
                        const auto* misses = miss_list.data();
                        std::atomic<bool> io_err{false};
                        std::atomic<bool>* io_err_p = &io_err;
                        dispatch_apply(miss_list.size(), dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t m) {
                            size_t slot = (size_t)misses[m].first;
                            size_t eid = (size_t)misses[m].second;
                            bool ok = true;
                            if (fd_g >= 0) ok = ok && pread_all(fd_g, dst_g + slot * c_gu, g_stride, (off_t)(off_g + eid * g_stride));
                            else memcpy(dst_g + slot * c_gu, src_g + eid * g_stride, g_stride);
                            if (fd_u >= 0) ok = ok && pread_all(fd_u, dst_u + slot * c_gu, u_stride, (off_t)(off_u + eid * u_stride));
                            else memcpy(dst_u + slot * c_gu, src_u + eid * u_stride, u_stride);
                            if (fd_d >= 0) ok = ok && pread_all(fd_d, dst_d + slot * c_d, d_stride, (off_t)(off_d + eid * d_stride));
                            else memcpy(dst_d + slot * c_d, src_d + eid * d_stride, d_stride);
                            if (!ok) io_err_p->store(true, std::memory_order_relaxed);
                        });
                        if (io_err.load(std::memory_order_relaxed)) {
                            throw std::runtime_error("pread failed on GGUF shard");
                        }
                    }
                    gate_w.buf = impl_->exp_cache.buf_g;
                    gate_w.base_offset = 0;
                    gate_w.groups = (uint32_t)impl_->exp_cache.num_slots;
                    gate_w.group_stride_bytes = (uint32_t)impl_->exp_cache.gu_stride;
                    up_w.buf = impl_->exp_cache.buf_u;
                    up_w.base_offset = 0;
                    up_w.groups = (uint32_t)impl_->exp_cache.num_slots;
                    up_w.group_stride_bytes = (uint32_t)impl_->exp_cache.gu_stride;
                    down_w.buf = impl_->exp_cache.buf_d;
                    down_w.base_offset = 0;
                    down_w.groups = (uint32_t)impl_->exp_cache.num_slots;
                    down_w.group_stride_bytes = (uint32_t)impl_->exp_cache.d_stride;
                    ids_buf = impl_->b_stage_topk_ids;

                    cb = [impl_->queue commandBuffer];
                    enc = [cb computeCommandEncoder];
                    if (gate_w.type == up_w.type &&
                        (gate_w.type == GgmlType::IQ1_S || gate_w.type == GgmlType::IQ2_XXS) &&
                        gate_w.group_stride_bytes == up_w.group_stride_bytes) {
                        impl_->encode_moe_gate_up_swiglu(enc, impl_->b_norm, gate_w, up_w,
                                                         ids_buf, impl_->b_ffn_act,
                                                         (uint32_t)T, top_k_u, cfg.swiglu_clamp_exp);
                    } else {
                        impl_->encode_moe_indexed(enc, impl_->b_norm, gate_w, ids_buf, impl_->b_ffn_gate, (uint32_t)T, top_k_u);
                        impl_->encode_moe_indexed(enc, impl_->b_norm, up_w, ids_buf, impl_->b_ffn_up, (uint32_t)T, top_k_u);
                        impl_->encode_swiglu(enc, impl_->b_ffn_gate, impl_->b_ffn_up, impl_->b_ffn_act,
                                             (uint32_t)((size_t)T * top_k_u * gate_w.rows), cfg.swiglu_clamp_exp);
                    }
                    impl_->encode_moe_down_accum(enc, impl_->b_ffn_act, down_w, ids_buf, impl_->b_topk_w,
                                                 impl_->b_sub_out, (uint32_t)T, top_k_u);
                } else {
                constexpr size_t kMaxStageChunk = 16u;
                size_t cap_u = std::min<size_t>(U, kMaxStageChunk);
                impl_->ensure_moe_stage(cap_u * g_stride, cap_u * u_stride, cap_u * d_stride);

                uint8_t* dst_g = static_cast<uint8_t*>(impl_->b_moe_stage_gate.contents);
                uint8_t* dst_u = static_cast<uint8_t*>(impl_->b_moe_stage_up.contents);
                uint8_t* dst_d = static_cast<uint8_t*>(impl_->b_moe_stage_down.contents);
                const int32_t* uids = uniq_ids.data();

                for (size_t base_u = 0; base_u < U; base_u += kMaxStageChunk) {
                    size_t chunk_u = std::min<size_t>(kMaxStageChunk, U - base_u);
                    for (uint32_t i = 0; i < n_sel; ++i) {
                        int32_t loc = full_loc[i];
                        st_ids[i] = (loc >= (int32_t)base_u && loc < (int32_t)(base_u + chunk_u))
                                        ? (loc - (int32_t)base_u)
                                        : -1;
                    }
                    std::atomic<bool> io_err{false};
                    std::atomic<bool>* io_err_p = &io_err;
                    dispatch_apply(chunk_u, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t u) {
                        size_t eid = (size_t)uids[base_u + u];
                        bool ok = true;
                        if (fd_g >= 0) ok = ok && pread_all(fd_g, dst_g + u * g_stride, g_stride, (off_t)(off_g + eid * g_stride));
                        else memcpy(dst_g + u * g_stride, src_g + eid * g_stride, g_stride);
                        if (fd_u >= 0) ok = ok && pread_all(fd_u, dst_u + u * u_stride, u_stride, (off_t)(off_u + eid * u_stride));
                        else memcpy(dst_u + u * u_stride, src_u + eid * u_stride, u_stride);
                        if (fd_d >= 0) ok = ok && pread_all(fd_d, dst_d + u * d_stride, d_stride, (off_t)(off_d + eid * d_stride));
                        else memcpy(dst_d + u * d_stride, src_d + eid * d_stride, d_stride);
                        if (!ok) io_err_p->store(true, std::memory_order_relaxed);
                    });
                    if (io_err.load(std::memory_order_relaxed)) {
                        throw std::runtime_error("pread failed on GGUF shard");
                    }

                    gate_w.buf = impl_->b_moe_stage_gate;
                    gate_w.base_offset = 0;
                    gate_w.groups = (uint32_t)chunk_u;
                    up_w.buf = impl_->b_moe_stage_up;
                    up_w.base_offset = 0;
                    up_w.groups = (uint32_t)chunk_u;
                    down_w.buf = impl_->b_moe_stage_down;
                    down_w.base_offset = 0;
                    down_w.groups = (uint32_t)chunk_u;
                    ids_buf = impl_->b_stage_topk_ids;

                    cb = [impl_->queue commandBuffer];
                    enc = [cb computeCommandEncoder];
                    if (gate_w.type == up_w.type &&
                        (gate_w.type == GgmlType::IQ1_S || gate_w.type == GgmlType::IQ2_XXS) &&
                        gate_w.group_stride_bytes == up_w.group_stride_bytes) {
                        impl_->encode_moe_gate_up_swiglu(enc, impl_->b_norm, gate_w, up_w,
                                                         ids_buf, impl_->b_ffn_act,
                                                         (uint32_t)T, top_k_u, cfg.swiglu_clamp_exp);
                    } else {
                        impl_->encode_moe_indexed(enc, impl_->b_norm, gate_w, ids_buf, impl_->b_ffn_gate, (uint32_t)T, top_k_u);
                        impl_->encode_moe_indexed(enc, impl_->b_norm, up_w, ids_buf, impl_->b_ffn_up, (uint32_t)T, top_k_u);
                        impl_->encode_swiglu(enc, impl_->b_ffn_gate, impl_->b_ffn_up, impl_->b_ffn_act,
                                             (uint32_t)((size_t)T * top_k_u * gate_w.rows), cfg.swiglu_clamp_exp);
                    }
                    impl_->encode_moe_down_accum(enc, impl_->b_ffn_act, down_w, ids_buf, impl_->b_topk_w,
                                                 impl_->b_sub_out, (uint32_t)T, top_k_u);
                    if (base_u + chunk_u < U) {
                        [enc endEncoding];
                        [cb commit];
                        [cb waitUntilCompleted];
                    }
                }
                }
            } else {
                if (gate_w.type == up_w.type &&
                    (gate_w.type == GgmlType::IQ1_S || gate_w.type == GgmlType::IQ2_XXS) &&
                    gate_w.group_stride_bytes == up_w.group_stride_bytes) {
                    impl_->encode_moe_gate_up_swiglu(enc, impl_->b_norm, gate_w, up_w,
                                                     ids_buf, impl_->b_ffn_act,
                                                     (uint32_t)T, top_k_u, cfg.swiglu_clamp_exp);
                } else {
                    impl_->encode_moe_indexed(enc, impl_->b_norm, gate_w, ids_buf, impl_->b_ffn_gate, (uint32_t)T, top_k_u);
                    impl_->encode_moe_indexed(enc, impl_->b_norm, up_w, ids_buf, impl_->b_ffn_up, (uint32_t)T, top_k_u);
                    impl_->encode_swiglu(enc, impl_->b_ffn_gate, impl_->b_ffn_up, impl_->b_ffn_act,
                                         (uint32_t)((size_t)T * top_k_u * gate_w.rows), cfg.swiglu_clamp_exp);
                }
                impl_->encode_moe_down_accum(enc, impl_->b_ffn_act, down_w, ids_buf, impl_->b_topk_w,
                                             impl_->b_sub_out, (uint32_t)T, top_k_u);
            }
        }

        impl_->encode_hc_post(enc, impl_->b_sub_out, impl_->b_streams_a, impl_->b_hc_post, impl_->b_hc_comb,
                              impl_->b_streams_b, (uint32_t)T, (uint32_t)D, hc);
        std::swap(impl_->b_streams_a, impl_->b_streams_b);
        if (lw.ffn_gate_exps.is_streamed() || ((l + 1) % 8 == 0)) {
            [enc endEncoding];
            [cb commit];
            cb = [impl_->queue commandBuffer];
            enc = [cb computeCommandEncoder];
        }
    }

    if (cfg.num_hidden_layers > 0) {
        impl_->encode_hc_pre(enc, impl_->b_streams_a, pre_ffn, impl_->b_hidden, (uint32_t)T, (uint32_t)D, hc);
    }
    impl_->encode_rmsnorm(enc, impl_->b_hidden, impl_->output_norm, impl_->b_norm, (uint32_t)T, (uint32_t)D, cfg.rms_norm_eps);
    uint32_t v_sz = (uint32_t)cfg.vocab_size;
    if (impl_->verify_out_buf != nullptr) {
        impl_->encode_gemv(enc, impl_->b_norm, impl_->lm_head, impl_->b_logits, (uint32_t)T);
        [enc setComputePipelineState:impl_->pso("ds_argmax_f32")];
        [enc setBytes:&v_sz length:sizeof(v_sz) atIndex:2];
        for (int t = 0; t < T; ++t) {
            [enc setBuffer:impl_->b_logits offset:(size_t)t * (size_t)v_sz * sizeof(float) atIndex:0];
            [enc setBuffer:impl_->b_argmax offset:(size_t)t * sizeof(int32_t) atIndex:1];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const int32_t* argp = static_cast<const int32_t*>(impl_->b_argmax.contents);
        memcpy(impl_->verify_out_buf, argp, (size_t)T * sizeof(int32_t));
        return argp[T - 1];
    }
    size_t last_off = (size_t)(T - 1) * D * sizeof(float);
    impl_->encode_gemv(enc, impl_->b_norm, impl_->lm_head, impl_->b_logits, 1u, last_off, 0, 0);
    [enc setComputePipelineState:impl_->pso("ds_argmax_f32")];
    [enc setBuffer:impl_->b_logits offset:0 atIndex:0];
    [enc setBuffer:impl_->b_argmax offset:0 atIndex:1];
    [enc setBytes:&v_sz length:sizeof(v_sz) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return *static_cast<const int32_t*>(impl_->b_argmax.contents);
    }
}

void DeepSeekV41Model::forward_verify_greedy(const int32_t* h_tokens, int T, int start_pos, int32_t* h_out_tokens) {
    if (T <= 0) return;
    if ((size_t)T * (size_t)cfg.vocab_size * sizeof(float) > impl_->b_logits.length) {
        impl_->b_logits = impl_->alloc_buf((size_t)T * (size_t)cfg.vocab_size * sizeof(float));
    }
    if ((size_t)T * sizeof(int32_t) > impl_->b_argmax.length) {
        impl_->b_argmax = impl_->alloc_buf((size_t)T * sizeof(int32_t));
    }
    impl_->last_verify_start_pos = start_pos;
    struct ResetGuard {
        int32_t*& ptr;
        ~ResetGuard() { ptr = nullptr; }
    } guard{impl_->verify_out_buf};
    impl_->verify_out_buf = h_out_tokens;
    forward_and_greedy(h_tokens, T, start_pos, 0);
}

void DeepSeekV41Model::rollback_conv_states(int accepted_len, int slot_id) {
    if (slot_id < 0 || slot_id >= max_slots) return;
    auto& hist = impl_->slot_tok_history[(size_t)slot_id];
    size_t keep = (size_t)std::max(0, impl_->last_verify_start_pos + accepted_len);
    if (hist.size() > keep) hist.resize(keep);
}

int DeepSeekV41Model::max_dspark_slots() const {
    return impl_->dspark.loaded ? 1 : 0;
}

void DeepSeekV41Model::commit_dspark_context(int count, int start_pos, int slot_id, int aux_tok_offset) {
    if (!impl_->dspark.loaded || count <= 0 || slot_id < 0 || slot_id >= max_slots) return;
    @autoreleasepool {
    auto& ds = impl_->dspark;
    if (ds.n_swa > 0 && count > ds.n_swa) {
        int skip = count - ds.n_swa;
        aux_tok_offset += skip;
        start_pos += skip;
        count = ds.n_swa;
    }
    size_t D = (size_t)ds.hidden_size;
    size_t n_taps = ds.target_layers.size();
    size_t tap_off = (size_t)aux_tok_offset * n_taps * D * sizeof(float);
    size_t slot_ring_off = (size_t)slot_id * (size_t)ds.n_swa * (size_t)ds.head_dim * sizeof(float);
    uint32_t n_nope = (uint32_t)std::max(0, ds.head_dim - ds.n_rot);
    uint32_t n_rot = (uint32_t)std::max(0, ds.n_rot);

    id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    impl_->encode_gemv(enc, ds.b_taps, ds.fc, ds.b_inp_g, (uint32_t)count, tap_off, 0, 0, true);
    impl_->encode_rmsnorm(enc, ds.b_inp_g, ds.enc_output_norm, ds.b_inp_g, (uint32_t)count, (uint32_t)D, ds.rms_norm_eps);

    for (int il = 0; il < ds.num_layers; ++il) {
        const auto& dl = ds.layers[(size_t)il];
        impl_->encode_gemv(enc, ds.b_inp_g, dl.wkv, impl_->b_kv, (uint32_t)count);
        impl_->encode_rmsnorm(enc, impl_->b_kv, dl.attn_kv_norm, impl_->b_kv_norm, (uint32_t)count, (uint32_t)ds.head_dim, ds.rms_norm_eps);
        if (n_rot >= 2) {
            DsPartialRopeParams rk{(uint32_t)count, (uint32_t)start_pos, 1u, 1u, (uint32_t)ds.head_dim, n_nope, n_rot, 0u, ds.rope_freq_base};
            [enc setComputePipelineState:impl_->pso("ds_rope_partial_f32")];
            [enc setBuffer:impl_->b_kv_norm offset:0 atIndex:0];
            [enc setBytes:&rk length:sizeof(rk) atIndex:1];
            [enc dispatchThreads:MTLSizeMake(n_rot / 2u, 1u, (uint32_t)count)
                threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(32u, n_rot / 2u), 1, 1)];
        }
        DsDsparkKvRingParams krp{(uint32_t)count, (uint32_t)start_pos, (uint32_t)ds.n_swa, (uint32_t)ds.head_dim};
        [enc setComputePipelineState:impl_->pso("ds_dspark_kv_ring_write_f32")];
        [enc setBuffer:impl_->b_kv_norm offset:0 atIndex:0];
        [enc setBuffer:ds.kv_rings[(size_t)il] offset:slot_ring_off atIndex:1];
        [enc setBytes:&krp length:sizeof(krp) atIndex:2];
        [enc dispatchThreads:MTLSizeMake((uint32_t)ds.head_dim, (uint32_t)count, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(64u, (uint32_t)ds.head_dim), 1, 1)];
    }

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    }
}

void DeepSeekV41Model::draft_tokens(int32_t anchor_tok, int anchor_pos, int num_draft, int32_t* h_out_drafts, int slot_id) {
    if (!impl_->dspark.loaded || num_draft <= 0 || slot_id < 0 || slot_id >= max_slots) return;
    @autoreleasepool {
    auto& ds = impl_->dspark;
    int B = std::clamp(num_draft, 1, ds.block_size);
    size_t D = (size_t)ds.hidden_size;
    uint32_t hc = (uint32_t)std::max(1, ds.hc_mult);
    uint32_t n_nope = (uint32_t)std::max(0, ds.head_dim - ds.n_rot);
    uint32_t n_rot = (uint32_t)std::max(0, ds.n_rot);
    size_t slot_ring_off = (size_t)slot_id * (size_t)ds.n_swa * (size_t)ds.head_dim * sizeof(float);

    float* hidden_ptr = static_cast<float*>(impl_->b_hidden.contents);
    const auto& emb_ti = gguf.tensor("token_embd.weight");
    size_t row_stride = ggml_row_nbytes(emb_ti.type, emb_ti.dims[0]);
    const uint8_t* emb_base = gguf.tensor_data("token_embd.weight");
    uint64_t nrows = emb_ti.num_rows();
    for (int b = 0; b < B; ++b) {
        int32_t tok = (b == 0) ? anchor_tok : ds.mask_token_id;
        uint64_t tid = (uint64_t)std::clamp(tok, 0, (int32_t)(nrows > 0 ? nrows - 1 : 0));
        dequant_row_fp32(emb_ti.type, emb_base + tid * row_stride, hidden_ptr + (size_t)b * D, D);
    }

    int32_t* markov_toks = static_cast<int32_t*>(ds.b_markov_toks.contents);
    markov_toks[0] = anchor_tok;

    id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    {
        DsStreamCollapseParams scp{(uint32_t)B, (uint32_t)D, hc};
        [enc setComputePipelineState:impl_->pso("ds_mhc_expand_f32")];
        [enc setBuffer:impl_->b_hidden offset:0 atIndex:0];
        [enc setBuffer:impl_->b_streams_a offset:0 atIndex:1];
        [enc setBytes:&scp length:sizeof(scp) atIndex:2];
        [enc dispatchThreads:MTLSizeMake((uint32_t)D, (uint32_t)B, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(64u, (uint32_t)D), 1, 1)];
    }

    id<MTLBuffer> pre_attn = impl_->b_hc_pre_a;
    id<MTLBuffer> pre_ffn = impl_->b_hc_pre_b;
    float attn_scale = 1.0f / std::sqrt((float)ds.head_dim);

    for (int il = 0; il < ds.num_layers; ++il) {
        const auto& dl = ds.layers[(size_t)il];

        impl_->encode_hc_mixes(enc, impl_->b_streams_a, dl.hc_attn_fn, dl.hc_attn_scale, dl.hc_attn_base,
                               pre_attn, impl_->b_hc_post, impl_->b_hc_comb,
                               (uint32_t)B, (uint32_t)D, hc, (uint32_t)ds.hc_sinkhorn_iters, ds.rms_norm_eps, ds.hc_eps);

        if (il > 0) {
            impl_->encode_hc_pre(enc, impl_->b_streams_a, pre_ffn, impl_->b_hidden, (uint32_t)B, (uint32_t)D, hc);
        } else {
            [enc endEncoding];
            id<MTLBlitCommandEncoder> bl0 = [cb blitCommandEncoder];
            for (int b = 0; b < B; ++b) {
                [bl0 copyFromBuffer:impl_->b_streams_a sourceOffset:(size_t)b * hc * D * sizeof(float)
                           toBuffer:impl_->b_hidden destinationOffset:(size_t)b * D * sizeof(float)
                               size:D * sizeof(float)];
            }
            [bl0 endEncoding];
            enc = [cb computeCommandEncoder];
        }

        impl_->encode_rmsnorm(enc, impl_->b_hidden, dl.attn_norm, impl_->b_norm, (uint32_t)B, (uint32_t)D, ds.rms_norm_eps);
        impl_->encode_gemv(enc, impl_->b_norm, dl.wq_a, impl_->b_qr, (uint32_t)B);
        impl_->encode_rmsnorm(enc, impl_->b_qr, dl.attn_q_a_norm, impl_->b_qr_norm, (uint32_t)B, (uint32_t)ds.q_lora_rank, ds.rms_norm_eps);
        impl_->encode_gemv(enc, impl_->b_qr_norm, dl.wq_b, impl_->b_q, (uint32_t)B);

        impl_->encode_gemv(enc, impl_->b_norm, dl.wkv, impl_->b_kv, (uint32_t)B);
        impl_->encode_rmsnorm(enc, impl_->b_kv, dl.attn_kv_norm, impl_->b_kv_norm, (uint32_t)B, (uint32_t)ds.head_dim, ds.rms_norm_eps);

        if (n_rot >= 2) {
            DsPartialRopeParams rq{(uint32_t)B, (uint32_t)anchor_pos, 1u, (uint32_t)ds.n_heads, (uint32_t)ds.head_dim, n_nope, n_rot, 0u, ds.rope_freq_base};
            [enc setComputePipelineState:impl_->pso("ds_rope_partial_f32")];
            [enc setBuffer:impl_->b_q offset:0 atIndex:0];
            [enc setBytes:&rq length:sizeof(rq) atIndex:1];
            [enc dispatchThreads:MTLSizeMake(n_rot / 2u, (uint32_t)ds.n_heads, (uint32_t)B)
                threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(32u, n_rot / 2u), 1, 1)];

            DsPartialRopeParams rk{(uint32_t)B, (uint32_t)anchor_pos, 1u, 1u, (uint32_t)ds.head_dim, n_nope, n_rot, 0u, ds.rope_freq_base};
            [enc setBuffer:impl_->b_kv_norm offset:0 atIndex:0];
            [enc setBytes:&rk length:sizeof(rk) atIndex:1];
            [enc dispatchThreads:MTLSizeMake(n_rot / 2u, 1u, (uint32_t)B)
                threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(32u, n_rot / 2u), 1, 1)];
        }

        {
            DsDsparkAttnParams ap{(uint32_t)B, (uint32_t)ds.n_heads, (uint32_t)ds.head_dim, (uint32_t)anchor_pos, (uint32_t)ds.n_swa, attn_scale};
            [enc setComputePipelineState:impl_->pso("ds_dspark_mqa_sink_attn_f32")];
            [enc setBuffer:impl_->b_q offset:0 atIndex:0];
            [enc setBuffer:ds.kv_rings[(size_t)il] offset:slot_ring_off atIndex:1];
            [enc setBuffer:impl_->b_kv_norm offset:0 atIndex:2];
            [enc setBuffer:dl.attn_sinks offset:0 atIndex:3];
            [enc setBuffer:impl_->b_attn_out offset:0 atIndex:4];
            [enc setBytes:&ap length:sizeof(ap) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((uint32_t)ds.n_heads, (uint32_t)B, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }

        if (n_rot >= 2) {
            DsPartialRopeParams rd{(uint32_t)B, (uint32_t)anchor_pos, 1u, (uint32_t)ds.n_heads, (uint32_t)ds.head_dim, n_nope, n_rot, 1u, ds.rope_freq_base};
            [enc setComputePipelineState:impl_->pso("ds_rope_partial_f32")];
            [enc setBuffer:impl_->b_attn_out offset:0 atIndex:0];
            [enc setBytes:&rd length:sizeof(rd) atIndex:1];
            [enc dispatchThreads:MTLSizeMake(n_rot / 2u, (uint32_t)ds.n_heads, (uint32_t)B)
                threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(32u, n_rot / 2u), 1, 1)];
        }

        uint32_t o_groups = dl.wo_a.groups;
        if (dl.wo_a.type == GgmlType::Q8_0 && (dl.wo_a.cols & 31u) == 0u) {
            GgufGroupedGemvParams gp{(uint32_t)B, o_groups, dl.wo_a.rows, dl.wo_a.cols, dl.wo_a.group_stride_bytes};
            [enc setComputePipelineState:impl_->pso("gguf_grouped_gemv_q8_0_fast")];
            [enc setBuffer:impl_->b_attn_out offset:0 atIndex:0];
            [enc setBuffer:dl.wo_a.buf offset:dl.wo_a.base_offset atIndex:1];
            [enc setBuffer:impl_->b_oa offset:0 atIndex:2];
            [enc setBytes:&gp length:sizeof(gp) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((dl.wo_a.rows + 3u) / 4u, (uint32_t)B * o_groups, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        } else {
            for (int b = 0; b < B; ++b) {
                for (uint32_t g = 0; g < o_groups; ++g) {
                    size_t x_off = ((size_t)b * o_groups + g) * dl.wo_a.cols * sizeof(float);
                    size_t w_off = (size_t)g * dl.wo_a.group_stride_bytes;
                    size_t y_off = ((size_t)b * o_groups + g) * dl.wo_a.rows * sizeof(float);
                    impl_->encode_gemv(enc, impl_->b_attn_out, dl.wo_a, impl_->b_oa, 1u, x_off, w_off, y_off);
                }
            }
        }
        impl_->encode_gemv(enc, impl_->b_oa, dl.wo_b, impl_->b_sub_out, (uint32_t)B);
        impl_->encode_hc_post(enc, impl_->b_sub_out, impl_->b_streams_a, impl_->b_hc_post, impl_->b_hc_comb,
                              impl_->b_streams_b, (uint32_t)B, (uint32_t)D, hc);
        std::swap(impl_->b_streams_a, impl_->b_streams_b);

        impl_->encode_hc_mixes(enc, impl_->b_streams_a, dl.hc_ffn_fn, dl.hc_ffn_scale, dl.hc_ffn_base,
                               pre_ffn, impl_->b_hc_post, impl_->b_hc_comb,
                               (uint32_t)B, (uint32_t)D, hc, (uint32_t)ds.hc_sinkhorn_iters, ds.rms_norm_eps, ds.hc_eps);
        impl_->encode_hc_pre(enc, impl_->b_streams_a, pre_attn, impl_->b_hidden, (uint32_t)B, (uint32_t)D, hc);
        impl_->encode_rmsnorm(enc, impl_->b_hidden, dl.ffn_norm, impl_->b_norm, (uint32_t)B, (uint32_t)D, ds.rms_norm_eps);

        impl_->encode_gemv(enc, impl_->b_norm, dl.ffn_gate_shexp, impl_->b_ffn_gate, (uint32_t)B);
        impl_->encode_gemv(enc, impl_->b_norm, dl.ffn_up_shexp, impl_->b_ffn_up, (uint32_t)B);
        impl_->encode_swiglu(enc, impl_->b_ffn_gate, impl_->b_ffn_up, impl_->b_ffn_act,
                             (uint32_t)((size_t)B * dl.ffn_gate_shexp.rows), ds.swiglu_clamp_shexp);
        impl_->encode_gemv(enc, impl_->b_ffn_act, dl.ffn_down_shexp, impl_->b_sub_out, (uint32_t)B);

        impl_->encode_gemv(enc, impl_->b_norm, dl.router, impl_->b_router_logits, (uint32_t)B);
        {
            DsRouterParams rp{(uint32_t)B, (uint32_t)ds.n_experts, (uint32_t)ds.top_k, 1u, ds.route_scale};
            [enc setComputePipelineState:impl_->pso("ds_router_topk_f32")];
            [enc setBuffer:impl_->b_router_logits offset:0 atIndex:0];
            [enc setBuffer:dl.router_bias offset:0 atIndex:1];
            [enc setBuffer:impl_->b_topk_ids offset:0 atIndex:2];
            [enc setBuffer:impl_->b_topk_w offset:0 atIndex:3];
            [enc setBytes:&rp length:sizeof(rp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((uint32_t)B, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        if (dl.ffn_gate_exps.type == GgmlType::MXFP4 && dl.ffn_up_exps.type == GgmlType::MXFP4 &&
            dl.ffn_gate_exps.group_stride_bytes == dl.ffn_up_exps.group_stride_bytes) {
            GgufMoeSwigluParams sp{(uint32_t)B, dl.ffn_gate_exps.rows, dl.ffn_gate_exps.cols,
                                   (uint32_t)ds.top_k, dl.ffn_gate_exps.group_stride_bytes, ds.swiglu_clamp_exp};
            [enc setComputePipelineState:impl_->pso("gguf_moe_gate_up_swiglu_mxfp4")];
            [enc setBuffer:impl_->b_norm offset:0 atIndex:0];
            [enc setBuffer:dl.ffn_gate_exps.buf offset:dl.ffn_gate_exps.base_offset atIndex:1];
            [enc setBuffer:dl.ffn_up_exps.buf offset:dl.ffn_up_exps.base_offset atIndex:2];
            [enc setBuffer:impl_->b_topk_ids offset:0 atIndex:3];
            [enc setBuffer:impl_->b_ffn_act offset:0 atIndex:4];
            [enc setBytes:&sp length:sizeof(sp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((dl.ffn_gate_exps.rows + 3u) / 4u, (uint32_t)B * (uint32_t)ds.top_k, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        } else {
            impl_->encode_moe_indexed(enc, impl_->b_norm, dl.ffn_gate_exps, impl_->b_topk_ids, impl_->b_ffn_gate, (uint32_t)B, (uint32_t)ds.top_k);
            impl_->encode_moe_indexed(enc, impl_->b_norm, dl.ffn_up_exps, impl_->b_topk_ids, impl_->b_ffn_up, (uint32_t)B, (uint32_t)ds.top_k);
            impl_->encode_swiglu(enc, impl_->b_ffn_gate, impl_->b_ffn_up, impl_->b_ffn_act,
                                 (uint32_t)((size_t)B * (size_t)ds.top_k * dl.ffn_gate_exps.rows), ds.swiglu_clamp_exp);
        }
        if (dl.ffn_down_exps.type == GgmlType::MXFP4) {
            GgufMoeParams mp{(uint32_t)B, dl.ffn_down_exps.rows, dl.ffn_down_exps.cols,
                             (uint32_t)ds.top_k, dl.ffn_down_exps.group_stride_bytes};
            [enc setComputePipelineState:impl_->pso("gguf_moe_down_accum_mxfp4_4r")];
            [enc setBuffer:impl_->b_ffn_act offset:0 atIndex:0];
            [enc setBuffer:dl.ffn_down_exps.buf offset:dl.ffn_down_exps.base_offset atIndex:1];
            [enc setBuffer:impl_->b_topk_ids offset:0 atIndex:2];
            [enc setBuffer:impl_->b_topk_w offset:0 atIndex:3];
            [enc setBuffer:impl_->b_sub_out offset:0 atIndex:4];
            [enc setBytes:&mp length:sizeof(mp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((dl.ffn_down_exps.rows + 7u) / 8u, (uint32_t)B, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        } else {
            impl_->encode_moe_down_accum(enc, impl_->b_ffn_act, dl.ffn_down_exps, impl_->b_topk_ids, impl_->b_topk_w,
                                         impl_->b_sub_out, (uint32_t)B, (uint32_t)ds.top_k);
        }

        impl_->encode_hc_post(enc, impl_->b_sub_out, impl_->b_streams_a, impl_->b_hc_post, impl_->b_hc_comb,
                              impl_->b_streams_b, (uint32_t)B, (uint32_t)D, hc);
        std::swap(impl_->b_streams_a, impl_->b_streams_b);
    }

    impl_->encode_hc_pre(enc, impl_->b_streams_a, pre_ffn, impl_->b_hidden, (uint32_t)B, (uint32_t)D, hc);
    impl_->encode_rmsnorm(enc, impl_->b_hidden, ds.output_norm, impl_->b_norm, (uint32_t)B, (uint32_t)D, ds.rms_norm_eps);
    impl_->encode_gemv(enc, impl_->b_norm, impl_->lm_head, ds.b_logits_blk, (uint32_t)B);

    uint32_t v_sz = (uint32_t)cfg.vocab_size;
    DsMarkovBiasParams mp{v_sz, (uint32_t)ds.markov_rank};
    uint32_t markov_tg = (v_sz + 3u) / 4u;
    for (int i = 0; i < B; ++i) {
        [enc setComputePipelineState:impl_->pso("ds_markov_bias_add_bf16")];
        [enc setBuffer:ds.b_logits_blk offset:(size_t)i * (size_t)v_sz * sizeof(float) atIndex:0];
        [enc setBuffer:ds.markov_w1.buf offset:ds.markov_w1.base_offset atIndex:1];
        [enc setBuffer:ds.markov_w2.buf offset:ds.markov_w2.base_offset atIndex:2];
        [enc setBuffer:ds.b_markov_toks offset:(size_t)i * sizeof(int32_t) atIndex:3];
        [enc setBuffer:ds.b_markov_logits offset:0 atIndex:4];
        [enc setBytes:&mp length:sizeof(mp) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(markov_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        [enc setComputePipelineState:impl_->pso("ds_argmax_f32")];
        [enc setBuffer:ds.b_markov_logits offset:0 atIndex:0];
        [enc setBuffer:ds.b_markov_toks offset:(size_t)(i + 1) * sizeof(int32_t) atIndex:1];
        [enc setBytes:&v_sz length:sizeof(v_sz) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    for (int i = 0; i < B; ++i) h_out_drafts[i] = markov_toks[i + 1];
    }
}

void DeepSeekV41Model::forward_batch_decode_greedy(
    const int32_t* h_tokens, const int32_t* h_slots, const int32_t* h_positions,
    int B, int32_t* h_out_tokens) {
    for (int b = 0; b < B; ++b) {
        h_out_tokens[b] = forward_and_greedy(&h_tokens[b], 1, h_positions[b], h_slots[b]);
    }
}

void DeepSeekV41Model::enable_imatrix_capture(bool enable) {
    if (enable) {
        throw std::runtime_error("--dump-imatrix is not supported for DeepSeek GGUF models");
    }
}

void DeepSeekV41Model::save_imatrix(const std::string& path) const {
    (void)path;
    throw std::runtime_error("--dump-imatrix is not supported for DeepSeek GGUF models");
}

}
