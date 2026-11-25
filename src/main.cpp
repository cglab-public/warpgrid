#include "cli.hpp"
#include "model.hpp"
#include "model_backend.hpp"
#ifdef WARPGRID_METAL
#include "metal/deepseek_v41_metal.hpp"
#else
#include "cuda/deepseek_v41_cuda.hpp"
#endif
#include "server.hpp"
#include "tokenizer.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    ink::CliOptions opts;
    std::string parse_err;
    if (!ink::parse_cli_args(argc, argv, opts, parse_err)) {
        fprintf(stderr, "%s\n", parse_err.c_str());
        return 1;
    }

    if (opts.model_dir.empty()) {
        fprintf(stderr, "Usage: %s --model <snapshot_dir_or_gguf> [--engram <engram.gguf>] [--port 8888] [--api-key <key>] [--max-seq-len 32768] [--bench]\n", argv[0]);
        return 1;
    }

    try {
        ink::ModelFormat fmt = ink::detect_model_format(opts.model_dir);
        if (fmt == ink::ModelFormat::Unknown) {
            throw std::runtime_error("Unsupported or missing model path: " + opts.model_dir);
        }

        ink::Tokenizer tok;
        std::unique_ptr<ink::ModelBackend> model;

        if (fmt == ink::ModelFormat::GgufDeepSeekV41) {
            if (!opts.served_name_explicit) opts.served_name = "DeepSeek-V4.1-Flash";
            auto ds = std::make_unique<ink::DeepSeekV41Model>();
            ds->engram_path = opts.engram_path;
            ds->load(opts.model_dir, opts.max_seq, opts.max_chunk, opts.max_slots);
            ds->load_tokenizer(tok);
            model = std::move(ds);
        } else {
            tok.load(opts.model_dir + "/tokenizer.json");
            model = std::make_unique<ink::Inkling2Model>();
            model->use_bf16_gemv = !opts.no_gemv;
            model->use_cuda_graph = !opts.no_graph;
            model->draft_k = (opts.draft_k == 0 && !opts.dflash2_dir.empty()) ? 7 : opts.draft_k;
            model->draft_vocab_file = opts.draft_vocab_file;
            model->dflash2_dir = opts.dflash2_dir;
            model->engram_path = opts.engram_path;
            model->moe_group_thresh = opts.moe_group_thresh;
            model->load(opts.model_dir, opts.max_seq, opts.max_chunk, opts.max_slots);
        }

        if (!opts.dump_imatrix_path.empty()) {
#ifdef WARPGRID_METAL
            std::string ctext = opts.bench_prompt;
            if (!opts.calib_file.empty()) {
                std::ifstream cf(opts.calib_file);
                if (!cf) throw std::runtime_error("Cannot open calib file: " + opts.calib_file);
                ctext.assign((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
            }
            auto cids = tok.encode(ctext);
            int max_t = std::min((int)cids.size(), model->max_seq_len);
            fprintf(stderr, "[imatrix] Calibrating on %d tokens...\n", max_t);
            model->reset_cache();
            model->enable_imatrix_capture(true);
            int pos = 0;
            while (pos < max_t) {
                int n = std::min(model->max_batch_tokens, max_t - pos);
                model->forward_and_greedy(cids.data() + pos, n, pos);
                pos += n;
            }
            model->enable_imatrix_capture(false);
            model->save_imatrix(opts.dump_imatrix_path);
            return 0;
#else
            throw std::runtime_error("--dump-imatrix is only supported on the Metal backend");
#endif
        }

        if (opts.bench_prefill > 0) {
            std::vector<int32_t> ptoks(opts.bench_prefill);
            for (int i = 0; i < opts.bench_prefill; ++i) ptoks[i] = 1000 + (i % 5000);
            model->reset_cache();
            if (opts.profile_mode) { model->reset_profile(); model->profile_stages = true; }
            auto t0 = std::chrono::steady_clock::now();
            int pos = 0;
            while (pos < opts.bench_prefill) {
                int n = std::min(model->max_batch_tokens, opts.bench_prefill - pos);
                model->forward_and_greedy(ptoks.data() + pos, n, pos);
                pos += n;
            }
            auto t1 = std::chrono::steady_clock::now();
            if (opts.profile_mode) { model->print_profile(); model->profile_stages = false; }
            double s = std::chrono::duration<double>(t1 - t0).count();
            printf("Prefill %d tokens (chunk=%d): %.3f s -> %.1f tok/s\n",
                   opts.bench_prefill, model->max_batch_tokens, s, opts.bench_prefill / s);
            return 0;
        }

        if (opts.bench_mode) {
            json::Value req;
            req["reasoning_effort"] = json::Value{json::Value::String, false, 0, "low"};
            req["messages"].type = json::Value::Array;
            json::Value m;
            m["role"] = json::Value{json::Value::String, false, 0, "user"};
            m["content"] = json::Value{json::Value::String, false, 0, opts.bench_prompt};
            req["messages"].arr.push_back(std::move(m));

            std::string rendered = tok.render_chat_for_format(req);
            auto ids = tok.encode(rendered);
            fprintf(stderr, "[bench] Prompt tokens: %zu\n", ids.size());

            model->reset_cache();
            auto t0 = std::chrono::steady_clock::now();
            int32_t next_tok = model->forward_and_greedy(ids.data(), (int)ids.size(), 0);
            if (model->draft_k > 0) model->commit_dspark_context((int)ids.size(), 0);
            auto t1 = std::chrono::steady_clock::now();
            double prefill_s = std::chrono::duration<double>(t1 - t0).count();

            std::vector<int32_t> gen;
            gen.push_back(next_tok);
            int pos = (int)ids.size();

            int spec_steps = 0, spec_drafted = 0, spec_accepted = 0;
            double total_draft_ms = 0.0, total_verify_ms = 0.0;
            auto td0 = std::chrono::steady_clock::now();
            if (opts.profile_mode) { model->reset_profile(); model->profile_stages = true; }

            if (model->draft_k > 0) {
                int k = model->draft_k;
                int32_t drafts[8];
                int32_t verify_in[8];
                int32_t verify_out[8];
                while ((int)gen.size() < opts.bench_tokens) {
                    if (next_tok == tok.eos_id) break;
                    int cur_k = std::min(k, opts.bench_tokens - (int)gen.size());
                    if (cur_k <= 0) break;

                    auto s0 = std::chrono::steady_clock::now();
                    model->draft_tokens(next_tok, pos, cur_k, drafts, 0);
                    auto s1 = std::chrono::steady_clock::now();
                    verify_in[0] = next_tok;
                    for (int j = 0; j < cur_k; ++j) verify_in[j + 1] = drafts[j];

                    model->forward_verify_greedy(verify_in, cur_k + 1, pos, verify_out);
                    auto s2 = std::chrono::steady_clock::now();
                    total_draft_ms += std::chrono::duration<double, std::milli>(s1 - s0).count();
                    total_verify_ms += std::chrono::duration<double, std::milli>(s2 - s1).count();

                    int acc = 0;
                    for (int j = 0; j < cur_k; ++j) {
                        if (verify_out[j] == drafts[j]) {
                            ++acc;
                        } else {
                            break;
                        }
                    }
                    int emitted = acc + 1;
                    if (emitted < cur_k + 1) {
                        model->rollback_conv_states(emitted);
                    }
                    model->commit_dspark_context(emitted, pos);

                    spec_steps += 1;
                    spec_drafted += cur_k;
                    spec_accepted += acc;

                    bool hit_eos = false;
                    for (int j = 0; j < acc; ++j) {
                        gen.push_back(drafts[j]);
                        if (drafts[j] == tok.eos_id) { hit_eos = true; break; }
                    }
                    if (!hit_eos && (int)gen.size() < opts.bench_tokens) {
                        next_tok = verify_out[acc];
                        gen.push_back(next_tok);
                        if (next_tok == tok.eos_id) hit_eos = true;
                    }
                    pos += emitted;
                    if (hit_eos) break;
                }
            } else {
                for (int step = 1; step < opts.bench_tokens; ++step) {
                    if (next_tok == tok.eos_id) break;
                    next_tok = model->forward_and_greedy(&next_tok, 1, pos++);
                    gen.push_back(next_tok);
                }
            }
            auto td1 = std::chrono::steady_clock::now();
            if (opts.profile_mode) { model->print_profile(); model->profile_stages = false; }
            double dec_s = std::chrono::duration<double>(td1 - td0).count();
            int dec_n = (int)gen.size() - 1;

            printf("=== OUTPUT (%zu tokens) ===\n%s\n", gen.size(), tok.decode(gen, false).c_str());
            printf("=== METRICS ===\n");
            printf("Prefill: %zu tokens in %.3f s (%.1f tok/s)\n", ids.size(), prefill_s, ids.size() / prefill_s);
            if (dec_n > 0) {
                printf("Decode : %d tokens in %.3f s (%.2f tok/s, %.2f ms/tok)\n",
                       dec_n, dec_s, dec_n / dec_s, (dec_s * 1000.0) / dec_n);
            }
            if (spec_steps > 0) {
                printf("DSpark : k=%d, steps=%d, accept=%d/%d (%.1f%%), mean_a=%.2f tok/step, step_ms=%.2f (draft=%.2f + verify=%.2f)\n",
                       model->draft_k, spec_steps, spec_accepted, spec_drafted,
                       100.0 * spec_accepted / std::max(1, spec_drafted),
                       (double)(spec_accepted + spec_steps) / spec_steps,
                       (dec_s * 1000.0) / spec_steps,
                       total_draft_ms / spec_steps, total_verify_ms / spec_steps);
            }
            return 0;
        }

        ink::OpenAIServer server(*model, tok, opts.served_name, opts.port, opts.otlp_endpoint, opts.api_key);
        server.run();
    } catch (const std::exception& e) {
        fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return 0;
}
