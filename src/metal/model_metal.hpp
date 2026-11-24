#pragma once
#include "../config.hpp"
#include "../safetensors.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ink {

class Inkling2Model {
public:
    struct Impl;

    InklingConfig cfg;
    SafetensorsFile st;

    int max_slots = 8;
    int max_seq_len = 32768;
    int max_batch_tokens = 2048;

    int draft_k = 0;
    std::string draft_vocab_file;
    std::string dflash2_dir;
    int moe_group_thresh = 64;
    bool use_bf16_gemv = true;
    bool use_cuda_graph = true;
    bool profile_stages = false;

    Inkling2Model();
    ~Inkling2Model();
    Inkling2Model(const Inkling2Model&) = delete;
    Inkling2Model& operator=(const Inkling2Model&) = delete;

    void load(const std::string& model_dir, int max_seq = 32768, int max_chunk = 512, int n_slots = 8);
    void reset_cache(int slot_id = 0);
    void save_prefill_snapshot(int slot_id);
    void restore_prefill_snapshot(int slot_id);

    int32_t forward_and_greedy(const int32_t* h_tokens, int T, int start_pos, int slot_id = 0);

    void forward_batch_decode_greedy(
        const int32_t* h_tokens, const int32_t* h_slots, const int32_t* h_positions,
        int B, int32_t* h_out_tokens);

    void forward_verify_greedy(const int32_t* h_tokens, int T, int start_pos, int32_t* h_out_tokens);

    void forward_batch_verify_greedy(
        const int32_t* h_tokens, const int32_t* h_slots, const int32_t* h_positions,
        int B, int L, int32_t* h_out_tokens);

    void rollback_conv_states(int accepted_len, int slot_id = 0);

    void commit_dspark_context(int count, int start_pos, int slot_id = 0, int aux_tok_offset = 0);

    int max_dspark_slots() const;

    void draft_tokens(int32_t anchor_tok, int anchor_pos, int num_draft, int32_t* h_out_drafts, int slot_id = 0);

    void forward_logits(const int32_t* h_tokens, int T, int start_pos, std::vector<float>& h_logits, int slot_id = 0);

    void reset_profile();
    void print_profile() const;

private:
    std::unique_ptr<Impl> impl_;
};

}
