#pragma once
#include "../deepseek_config.hpp"
#include "../engram.hpp"
#include "../gguf.hpp"
#include "../model_backend.hpp"
#include "../tokenizer.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ink {

class DeepSeekV41Model : public ModelBackend {
public:
    struct Impl;

    DeepSeekConfig cfg;
    GgufFile gguf;
    EngramTable engram;

    DeepSeekV41Model();
    ~DeepSeekV41Model() override;
    DeepSeekV41Model(const DeepSeekV41Model&) = delete;
    DeepSeekV41Model& operator=(const DeepSeekV41Model&) = delete;

    std::string arch_name() const override { return cfg.arch; }
    bool in_place_prefix_cache() const override { return max_slots == 1 && num_snapshots > 0; }

    void load(const std::string& model_path, int max_seq = 32768, int max_chunk = 512, int n_slots = 1) override;
    void load_tokenizer(Tokenizer& tok) const;
    void reset_cache(int slot_id = 0) override;

    int32_t forward_and_greedy(const int32_t* h_tokens, int T, int start_pos, int slot_id = 0) override;

    void forward_batch_decode_greedy(
        const int32_t* h_tokens, const int32_t* h_slots, const int32_t* h_positions,
        int B, int32_t* h_out_tokens) override;

    void forward_logits(const int32_t* h_tokens, int T, int start_pos, std::vector<float>& h_logits, int slot_id = 0) override;
    void enable_imatrix_capture(bool enable) override;
    void save_imatrix(const std::string& path) const override;

private:
    std::unique_ptr<Impl> impl_;
};

}
