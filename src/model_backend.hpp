#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace ink {

class ModelBackend {
public:
    int max_slots = 8;
    int max_seq_len = 32768;
    int max_batch_tokens = 2048;
    int max_blocks_per_seq = 0;
    int num_kv_blocks = 0;
    int num_snapshots = 0;

    int draft_k = 0;
    std::string draft_vocab_file;
    std::string dflash2_dir;
    std::string dspark_path;
    std::string engram_path;
    int moe_group_thresh = 64;
    bool use_bf16_gemv = true;
    bool use_cuda_graph = true;
    bool profile_stages = false;

    virtual ~ModelBackend() = default;

    virtual std::string arch_name() const = 0;

    virtual void load(const std::string& model_path, int max_seq = 32768, int max_chunk = 512, int n_slots = 8) = 0;
    virtual void reset_cache(int slot_id = 0) = 0;
    virtual void save_prefill_snapshot(int slot_id) { (void)slot_id; }
    virtual void restore_prefill_snapshot(int slot_id) { (void)slot_id; }
    virtual void save_boundary_snapshot(int slot_id, int snap_id) {
        (void)slot_id;
        (void)snap_id;
    }
    virtual void restore_boundary_snapshot(int slot_id, int snap_id) {
        (void)slot_id;
        (void)snap_id;
    }
    virtual void set_slot_block_table(int slot_id, const int32_t* blocks, int num_blocks) {
        (void)slot_id;
        (void)blocks;
        (void)num_blocks;
    }

    virtual int32_t forward_and_greedy(const int32_t* h_tokens, int T, int start_pos, int slot_id = 0) = 0;

    virtual void forward_batch_decode_greedy(
        const int32_t* h_tokens, const int32_t* h_slots, const int32_t* h_positions,
        int B, int32_t* h_out_tokens) = 0;

    virtual void forward_verify_greedy(const int32_t* h_tokens, int T, int start_pos, int32_t* h_out_tokens) {
        (void)h_tokens;
        (void)T;
        (void)start_pos;
        (void)h_out_tokens;
    }

    virtual void forward_batch_verify_greedy(
        const int32_t* h_tokens, const int32_t* h_slots, const int32_t* h_positions,
        int B, int L, int32_t* h_out_tokens) {
        (void)h_tokens;
        (void)h_slots;
        (void)h_positions;
        (void)B;
        (void)L;
        (void)h_out_tokens;
    }

    virtual void rollback_conv_states(int accepted_len, int slot_id = 0) {
        (void)accepted_len;
        (void)slot_id;
    }

    virtual void commit_dspark_context(int count, int start_pos, int slot_id = 0, int aux_tok_offset = 0) {
        (void)count;
        (void)start_pos;
        (void)slot_id;
        (void)aux_tok_offset;
    }

    virtual int max_dspark_slots() const { return 0; }
    virtual bool in_place_prefix_cache() const { return false; }

    virtual void draft_tokens(int32_t anchor_tok, int anchor_pos, int num_draft, int32_t* h_out_drafts, int slot_id = 0) {
        (void)anchor_tok;
        (void)anchor_pos;
        (void)num_draft;
        (void)h_out_drafts;
        (void)slot_id;
    }

    virtual void forward_logits(const int32_t* h_tokens, int T, int start_pos, std::vector<float>& h_logits, int slot_id = 0) = 0;

    virtual void reset_profile() {}
    virtual void print_profile() const {}
    virtual void enable_imatrix_capture(bool enable) { (void)enable; }
    virtual void save_imatrix(const std::string& path) const { (void)path; }
};

}
