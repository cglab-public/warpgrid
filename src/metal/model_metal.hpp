#pragma once
#include "../config.hpp"
#include "../model_backend.hpp"
#include "../safetensors.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ink {

class Inkling2Model : public ModelBackend {
public:
    struct Impl;

    InklingConfig cfg;
    SafetensorsFile st;

    Inkling2Model();
    ~Inkling2Model() override;
    Inkling2Model(const Inkling2Model&) = delete;
    Inkling2Model& operator=(const Inkling2Model&) = delete;

    std::string arch_name() const override { return "inkling2"; }

    void load(const std::string& model_dir, int max_seq = 32768, int max_chunk = 512, int n_slots = 8) override;
    void reset_cache(int slot_id = 0) override;
    void save_prefill_snapshot(int slot_id) override;
    void restore_prefill_snapshot(int slot_id) override;
    void save_boundary_snapshot(int slot_id, int snap_id) override;
    void restore_boundary_snapshot(int slot_id, int snap_id) override;
    void set_slot_block_table(int slot_id, const int32_t* blocks, int num_blocks) override;

    int32_t forward_and_greedy(const int32_t* h_tokens, int T, int start_pos, int slot_id = 0) override;

    void forward_batch_decode_greedy(
        const int32_t* h_tokens, const int32_t* h_slots, const int32_t* h_positions,
        int B, int32_t* h_out_tokens) override;

    void forward_verify_greedy(const int32_t* h_tokens, int T, int start_pos, int32_t* h_out_tokens) override;

    void forward_batch_verify_greedy(
        const int32_t* h_tokens, const int32_t* h_slots, const int32_t* h_positions,
        int B, int L, int32_t* h_out_tokens) override;

    void rollback_conv_states(int accepted_len, int slot_id = 0) override;

    void commit_dspark_context(int count, int start_pos, int slot_id = 0, int aux_tok_offset = 0) override;

    int max_dspark_slots() const override;

    void draft_tokens(int32_t anchor_tok, int anchor_pos, int num_draft, int32_t* h_out_drafts, int slot_id = 0) override;

    void forward_logits(const int32_t* h_tokens, int T, int start_pos, std::vector<float>& h_logits, int slot_id = 0) override;

    void reset_profile() override;
    void print_profile() const override;
    void enable_imatrix_capture(bool enable) override;
    void save_imatrix(const std::string& path) const override;

private:
    std::unique_ptr<Impl> impl_;
};

}
