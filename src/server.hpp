#pragma once
#include "auth.hpp"
#include "json.hpp"
#include "model.hpp"
#include "telemetry.hpp"
#include "tokenizer.hpp"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace ink {

class OpenAIServer {
    Inkling2Model& model_;
    Tokenizer& tok_;
    std::string served_name_;
    int port_;
    std::string api_key_;
    ServerMetrics metrics_;
    OtlpExporter otlp_;

    struct RequestState {
        std::vector<int32_t> prompt_ids;
        int max_tokens = 512;
        float temperature = 0.0f;
        float top_p = 1.0f;
        uint32_t seed = 42;
        std::chrono::steady_clock::time_point t_start{};
        uint64_t start_unix_ns = 0;

        // Runtime state owned by GPU worker
        int slot_id = -1;
        int pos = 0;
        int cached_tokens = 0;
        bool use_prefill_snap = false;
        int generated = 0;
        int32_t last_tok = 0;
        bool dspark_ready = false;
        int dyn_k = 5;
        std::atomic<int> spec_steps{0};
        std::atomic<int> spec_drafted{0};
        std::atomic<int> spec_accepted{0};
        std::atomic<bool> used_dflash2{false};

        // Output token stream consumed by HTTP thread
        std::mutex mu;
        std::condition_variable cv;
        std::deque<int32_t> tokens;
        bool finished = false;
        std::atomic<bool> cancelled{false};
        std::string error;

        void push_token(int32_t tid) {
            std::lock_guard<std::mutex> lk(mu);
            tokens.push_back(tid);
            cv.notify_one();
        }
        void mark_done(const std::string& err = "") {
            std::lock_guard<std::mutex> lk(mu);
            if (!err.empty()) error = err;
            finished = true;
            cv.notify_one();
        }
    };

    std::mutex sched_mu_;
    std::condition_variable sched_cv_;
    std::deque<std::shared_ptr<RequestState>> pending_;
    std::atomic<bool> stop_worker_{false};
    std::thread worker_;

    static bool send_all(int fd, const std::string& data) {
        size_t sent = 0;
        while (sent < data.size()) {
#ifdef MSG_NOSIGNAL
            int flags = MSG_NOSIGNAL;
#else
            int flags = 0;
#endif
            ssize_t n = ::send(fd, data.data() + sent, data.size() - sent, flags);
            if (n <= 0) return false;
            sent += (size_t)n;
        }
        return true;
    }

    static const char* http_status_text(int status) {
        switch (status) {
            case 200: return "OK";
            case 400: return "Bad Request";
            case 401: return "Unauthorized";
            case 404: return "Not Found";
            default:  return "Error";
        }
    }

    static void send_text(int fd, int status, const char* content_type, const std::string& body,
                          const std::string& extra_headers = "") {
        std::string hdr = "HTTP/1.1 " + std::to_string(status) + " " + http_status_text(status) +
                          "\r\nContent-Type: " + content_type + "\r\nContent-Length: " +
                          std::to_string(body.size()) + "\r\n" + extra_headers +
                          "Connection: close\r\n\r\n";
        send_all(fd, hdr + body);
    }

    static void send_json(int fd, int status, const std::string& body) {
        send_text(fd, status, "application/json", body);
    }

    static void send_unauthorized(int fd) {
        send_text(fd, 401, "application/json",
                  "{\"error\":{\"message\":\"Unauthorized: invalid or missing API key\","
                  "\"type\":\"authentication_error\",\"code\":\"invalid_api_key\"}}",
                  "WWW-Authenticate: Bearer\r\n");
    }

    static int32_t sample_token(const std::vector<float>& logits, float temp, float top_p, std::mt19937& rng) {
        if (temp <= 0.0f) {
            int32_t best = 0;
            float bv = logits[0];
            for (size_t i = 1; i < logits.size(); ++i) {
                if (logits[i] > bv) { bv = logits[i]; best = (int32_t)i; }
            }
            return best;
        }
        float inv_t = 1.0f / temp;
        float max_l = logits[0];
        for (float v : logits) if (v > max_l) max_l = v;
        std::vector<std::pair<float, int32_t>> probs;
        probs.reserve(logits.size());
        double sum = 0.0;
        for (size_t i = 0; i < logits.size(); ++i) {
            float p = std::exp((logits[i] - max_l) * inv_t);
            sum += p;
            if (p > 1e-7f) probs.push_back({p, (int32_t)i});
        }
        if (top_p < 1.0f) {
            std::sort(probs.begin(), probs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            double cum = 0.0;
            size_t cut = probs.size();
            for (size_t i = 0; i < probs.size(); ++i) {
                cum += probs[i].first / sum;
                if (cum >= top_p) { cut = i + 1; break; }
            }
            probs.resize(cut);
            sum = 0.0;
            for (const auto& p : probs) sum += p.first;
        }
        std::uniform_real_distribution<double> dist(0.0, sum);
        double r = dist(rng);
        double acc = 0.0;
        for (const auto& p : probs) {
            acc += p.first;
            if (acc >= r) return p.second;
        }
        return probs.empty() ? 0 : probs.back().second;
    }

    void worker_loop() {
        int max_slots = model_.max_slots;
        std::vector<bool> slot_used(max_slots, false);
        std::vector<bool> slot_dspark_valid(max_slots, false);
        std::vector<int> slot_dspark_written(max_slots, 0);
        std::vector<std::vector<int32_t>> slot_full_tokens(max_slots);
        std::vector<std::vector<int32_t>> slot_prefill_tokens(max_slots);
        std::vector<std::shared_ptr<RequestState>> active;
        std::vector<float> logits;

        auto publish_slot = [&](const std::shared_ptr<RequestState>& r) {
            SlotTelemetry st;
            st.slot_id = r->slot_id;
            st.active = true;
            st.pos = r->pos;
            st.cached_tokens = r->cached_tokens;
            st.generated = r->generated;
            st.dflash2_active = r->dspark_ready && !model_.dflash2_dir.empty();
            metrics_.update_slot(st);
        };
        auto release_slot = [&](int sid) {
            slot_used[sid] = false;
            metrics_.clear_slot(sid);
        };

        auto is_strict_prefix = [](const std::vector<int32_t>& cached, const std::vector<int32_t>& prompt) -> int {
            if (cached.empty() || cached.size() >= prompt.size()) return 0;
            for (size_t i = 0; i < cached.size(); ++i) {
                if (cached[i] != prompt[i]) return 0;
            }
            return (int)cached.size();
        };

        while (!stop_worker_) {
            // 1. Admit pending requests into free slots (choosing best prefix-cache match!)
            {
                std::unique_lock<std::mutex> lk(sched_mu_);
                if (active.empty() && pending_.empty()) {
                    sched_cv_.wait(lk, [&]() { return stop_worker_ || !pending_.empty(); });
                    if (stop_worker_) break;
                }
                while (!pending_.empty() && (int)active.size() < max_slots) {
                    auto req = pending_.front();
                    int best_slot = -1;
                    int best_hit = 0;
                    bool best_snap = false;
                    int first_free = -1;
                    for (int s = 0; s < max_slots; ++s) {
                        if (slot_used[s]) continue;
                        if (first_free < 0) first_free = s;
                        int h_full = is_strict_prefix(slot_full_tokens[s], req->prompt_ids);
                        if (h_full > best_hit) { best_hit = h_full; best_slot = s; best_snap = false; }
                        int h_pre = is_strict_prefix(slot_prefill_tokens[s], req->prompt_ids);
                        if (h_pre > best_hit) { best_hit = h_pre; best_slot = s; best_snap = true; }
                    }
                    if (first_free < 0) break;
                    pending_.pop_front();
                    int chosen = (best_slot >= 0) ? best_slot : first_free;
                    slot_used[chosen] = true;
                    req->slot_id = chosen;
                    req->cached_tokens = best_hit;
                    req->use_prefill_snap = best_snap;
                    active.push_back(req);
                    metrics_.on_dequeue();
                    publish_slot(req);
                }
            }

            // 2. Prefill any newly admitted requests (pos == 0)
            for (auto it = active.begin(); it != active.end(); ) {
                auto& r = *it;
                if (r->pos > 0) { ++it; continue; }
                int plen = (int)r->prompt_ids.size();
                if (plen + r->max_tokens > model_.max_seq_len) {
                    r->mark_done("prompt + max_tokens exceeds max_seq_len");
                    release_slot(r->slot_id);
                    it = active.erase(it);
                    continue;
                }
                int p = r->cached_tokens;
                if (p > 0) {
                    if (r->use_prefill_snap) {
                        model_.restore_prefill_snapshot(r->slot_id);
                    }
                } else {
                    model_.reset_cache(r->slot_id);
                }
                bool can_commit_ds = false;
                if (r->slot_id < model_.max_dspark_slots() && model_.draft_k > 0) {
                    if (r->cached_tokens == 0) {
                        slot_dspark_valid[r->slot_id] = true;
                        slot_dspark_written[r->slot_id] = 0;
                        can_commit_ds = true;
                    } else if (!r->use_prefill_snap && slot_dspark_valid[r->slot_id] &&
                               slot_dspark_written[r->slot_id] == r->cached_tokens) {
                        can_commit_ds = true;
                    } else if (r->use_prefill_snap && slot_dspark_valid[r->slot_id] &&
                               slot_dspark_written[r->slot_id] < (model_.dflash2_dir.empty() ? 1024 : 2048)) {
                        can_commit_ds = true;
                    } else {
                        slot_dspark_valid[r->slot_id] = false;
                    }
                }
                int chunk = model_.max_batch_tokens;
                int32_t first_tok = 0;
                std::mt19937 rng(r->seed);
                while (p < plen) {
                    int n = std::min(chunk, plen - p);
                    bool last = (p + n == plen);
                    if (last && r->temperature > 0.0f) {
                        model_.forward_logits(r->prompt_ids.data() + p, n, p, logits, r->slot_id);
                        first_tok = sample_token(logits, r->temperature, r->top_p, rng);
                    } else {
                        first_tok = model_.forward_and_greedy(r->prompt_ids.data() + p, n, p, r->slot_id);
                    }
                    if (can_commit_ds) {
                        model_.commit_dspark_context(n, p, r->slot_id);
                        slot_dspark_written[r->slot_id] = p + n;
                        r->dspark_ready = true;
                    }
                    p += n;
                    r->pos = p;
                    publish_slot(r);
                }
                model_.save_prefill_snapshot(r->slot_id);
                slot_prefill_tokens[r->slot_id] = r->prompt_ids;
                slot_full_tokens[r->slot_id] = r->prompt_ids;

                r->pos = plen;
                r->last_tok = first_tok;
                r->generated = 1;
                r->push_token(first_tok);
                publish_slot(r);
                if (first_tok == tok_.eos_id || r->generated >= r->max_tokens) {
                    r->mark_done();
                    release_slot(r->slot_id);
                    it = active.erase(it);
                } else {
                    ++it;
                }
            }

            if (active.empty()) continue;

            // If more requests arrived while we were prefilling and we have free slots, admit & prefill them first!
            bool has_pending = false;
            {
                std::lock_guard<std::mutex> lk(sched_mu_);
                has_pending = !pending_.empty();
            }
            if (has_pending && (int)active.size() < max_slots) continue;

            // 3. Single active request on slot 0 with DSpark enabled -> speculative decode!
            if (active.size() == 1 && !has_pending && active[0]->slot_id == 0 &&
                active[0]->dspark_ready && model_.draft_k > 0 && active[0]->temperature <= 0.0f)
            {
                auto& r = active[0];
                if (r->cancelled) {
                    r->mark_done();
                    release_slot(r->slot_id);
                    active.clear();
                    continue;
                }
                bool is_df2 = !model_.dflash2_dir.empty();
                int cur_k = std::min(is_df2 ? model_.draft_k : std::min(r->dyn_k, model_.draft_k), r->max_tokens - r->generated);
                if (cur_k <= 0) {
                    r->mark_done();
                    release_slot(r->slot_id);
                    active.clear();
                    continue;
                }
                int32_t drafts[8], verify_in[8], verify_out[8];
                model_.draft_tokens(r->last_tok, r->pos, cur_k, drafts, 0);
                verify_in[0] = r->last_tok;
                for (int j = 0; j < cur_k; ++j) verify_in[j + 1] = drafts[j];
                model_.forward_verify_greedy(verify_in, cur_k + 1, r->pos, verify_out);

                int acc = 0;
                for (int j = 0; j < cur_k; ++j) {
                    if (verify_out[j] == drafts[j]) ++acc;
                    else break;
                }
                r->spec_steps.fetch_add(1, std::memory_order_relaxed);
                r->spec_drafted.fetch_add(cur_k, std::memory_order_relaxed);
                r->spec_accepted.fetch_add(acc, std::memory_order_relaxed);
                if (is_df2) r->used_dflash2.store(true, std::memory_order_relaxed);
                if (!is_df2) {
                    if (acc == cur_k && r->dyn_k < model_.draft_k) r->dyn_k++;
                    else if (acc <= cur_k - 2 && r->dyn_k > 3) r->dyn_k--;
                }
                int emitted = acc + 1;
                if (emitted < cur_k + 1) model_.rollback_conv_states(emitted);
                model_.commit_dspark_context(emitted, r->pos);
                slot_dspark_written[r->slot_id] = r->pos + emitted;
                // Record the `emitted` tokens that were committed to slot 0's KV/conv cache
                for (int j = 0; j < emitted; ++j) {
                    slot_full_tokens[r->slot_id].push_back(verify_in[j]);
                }

                bool done = false;
                for (int j = 0; j < acc && !done && r->generated < r->max_tokens; ++j) {
                    r->push_token(drafts[j]);
                    r->generated++;
                    if (drafts[j] == tok_.eos_id) done = true;
                }
                if (!done && r->generated < r->max_tokens) {
                    r->last_tok = verify_out[acc];
                    r->push_token(r->last_tok);
                    r->generated++;
                    if (r->last_tok == tok_.eos_id) done = true;
                }
                r->pos += emitted;
                publish_slot(r);
                if (done || r->generated >= r->max_tokens) {
                    r->mark_done();
                    release_slot(r->slot_id);
                    active.clear();
                }
                continue;
            }

            // 3b. Batched DSpark speculative decode for B = 1..2 active sequences on DSpark slots (0..1)!
            // (Physics crossover verified in r24: at B>=3, k=1 adds ~6ms draft overhead + 1.6-2x MoE expert fanout,
            //  making standard B=3..8 parallel decode +17% faster: 110.40 vs 92.17 agg at c=8.)
            if (active.size() >= 1 && active.size() <= 2 && !has_pending && model_.draft_k > 0) {
                int B = (int)active.size();
                bool all_ds = true;
                int min_rem = 1000000;
                for (int b = 0; b < B; ++b) {
                    auto& r = active[b];
                    if (r->cancelled || r->slot_id >= model_.max_dspark_slots() ||
                        !r->dspark_ready || r->temperature > 0.0f) {
                        all_ds = false;
                        break;
                    }
                    min_rem = std::min(min_rem, r->max_tokens - r->generated);
                }
                int batch_k = std::min(3, std::min(model_.draft_k, min_rem));
                if (all_ds && batch_k >= 1) {
                    int L = batch_k + 1;
                    int32_t drafts[8][8];
                    int32_t verify_in[16], verify_out[16];
                    int32_t h_slots[8], h_pos[8];
                    for (int b = 0; b < B; ++b) {
                        auto& r = active[b];
                        h_slots[b] = r->slot_id;
                        h_pos[b] = r->pos;
                        model_.draft_tokens(r->last_tok, r->pos, batch_k, drafts[b], r->slot_id);
                        verify_in[b * L + 0] = r->last_tok;
                        for (int j = 0; j < batch_k; ++j) verify_in[b * L + j + 1] = drafts[b][j];
                    }
                    model_.forward_batch_verify_greedy(verify_in, h_slots, h_pos, B, L, verify_out);

                    for (int b = B - 1; b >= 0; --b) {
                        auto& r = active[b];
                        int acc = 0;
                        for (int j = 0; j < batch_k; ++j) {
                            if (verify_out[b * L + j] == drafts[b][j]) ++acc;
                            else break;
                        }
                        int emitted = acc + 1;
                        r->spec_steps.fetch_add(1, std::memory_order_relaxed);
                        r->spec_drafted.fetch_add(batch_k, std::memory_order_relaxed);
                        r->spec_accepted.fetch_add(acc, std::memory_order_relaxed);
                        if (!model_.dflash2_dir.empty()) r->used_dflash2.store(true, std::memory_order_relaxed);
                        if (emitted < L) model_.rollback_conv_states(emitted, r->slot_id);
                        model_.commit_dspark_context(emitted, r->pos, r->slot_id, b * L);
                        slot_dspark_written[r->slot_id] = r->pos + emitted;
                        for (int j = 0; j < emitted; ++j) {
                            slot_full_tokens[r->slot_id].push_back(verify_in[b * L + j]);
                        }
                        bool done = false;
                        for (int j = 0; j < acc && !done && r->generated < r->max_tokens; ++j) {
                            r->push_token(drafts[b][j]);
                            r->generated++;
                            if (drafts[b][j] == tok_.eos_id) done = true;
                        }
                        if (!done && r->generated < r->max_tokens) {
                            r->last_tok = verify_out[b * L + acc];
                            r->push_token(r->last_tok);
                            r->generated++;
                            if (r->last_tok == tok_.eos_id) done = true;
                        }
                        r->pos += emitted;
                        publish_slot(r);
                        if (done || r->generated >= r->max_tokens) {
                            r->mark_done();
                            release_slot(r->slot_id);
                            active.erase(active.begin() + b);
                        }
                    }
                    continue;
                }
            }

            // 4. Parallel continuous-batching decode step across ALL B active sequences!
            // (Invalidate dspark_ready on slot 0 since batched decode doesn't update DSpark KV)
            int B = (int)active.size();
            int32_t h_toks[8], h_slots[8], h_pos[8], h_out[8];
            for (int b = 0; b < B; ++b) {
                active[b]->dspark_ready = false;
                slot_dspark_valid[active[b]->slot_id] = false;
                h_toks[b] = active[b]->last_tok;
                h_slots[b] = active[b]->slot_id;
                h_pos[b] = active[b]->pos;
                slot_full_tokens[active[b]->slot_id].push_back(active[b]->last_tok);
            }
            model_.forward_batch_decode_greedy(h_toks, h_slots, h_pos, B, h_out);

            for (int b = B - 1; b >= 0; --b) {
                auto& r = active[b];
                if (r->cancelled) {
                    r->mark_done();
                    release_slot(r->slot_id);
                    active.erase(active.begin() + b);
                    continue;
                }
                r->last_tok = h_out[b];
                r->pos += 1;
                r->generated += 1;
                r->push_token(r->last_tok);
                publish_slot(r);
                if (r->last_tok == tok_.eos_id || r->generated >= r->max_tokens) {
                    r->mark_done();
                    release_slot(r->slot_id);
                    active.erase(active.begin() + b);
                }
            }
        }
    }

    enum class ParseMode { None, Header, Thinking, Content, ToolJson };

    void handle_chat(int fd, const json::Value& req) {
        bool stream = req.contains("stream") && req["stream"].as_bool(false);
        auto st = std::make_shared<RequestState>();
        st->max_tokens = req.contains("max_tokens") ? req["max_tokens"].as_int(512)
                       : (req.contains("max_completion_tokens") ? req["max_completion_tokens"].as_int(512) : 512);
        st->temperature = req.contains("temperature") ? (float)req["temperature"].as_double(0.0) : 0.0f;
        st->top_p = req.contains("top_p") ? (float)req["top_p"].as_double(1.0) : 1.0f;
        st->seed = req.contains("seed") ? (uint32_t)req["seed"].as_i64(42) : 42;
        st->t_start = std::chrono::steady_clock::now();
        st->start_unix_ns = now_unix_ns();

        std::string prompt = Tokenizer::render_chat(req);
        st->prompt_ids = tok_.encode(prompt);
        int prompt_len = (int)st->prompt_ids.size();

        if (prompt_len + st->max_tokens > model_.max_seq_len) {
            metrics_.on_request_error();
            fprintf(stderr, "[warpgrid] req_error 400 prompt=%d max_tokens=%d exceeds max_seq_len=%d\n",
                    prompt_len, st->max_tokens, model_.max_seq_len);
            send_json(fd, 400, "{\"error\":\"prompt + max_tokens exceeds max_seq_len\"}");
            return;
        }

        if (stream) {
            std::string hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                              "Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
            if (!send_all(fd, hdr)) return;
        }

        metrics_.on_enqueue();
        {
            std::lock_guard<std::mutex> lk(sched_mu_);
            pending_.push_back(st);
        }
        sched_cv_.notify_one();

        ParseMode mode = ParseMode::None;
        std::string full_reasoning, full_content, tool_buf;
        std::vector<json::Value> tool_calls;
        int completion_tokens = 0;
        std::string finish_reason = "length";

        auto emit_sse_delta = [&](const std::string& field, const std::string& text) -> bool {
            if (!stream || text.empty()) return true;
            std::string chunk_json = "{\"id\":\"chatcmpl-warpgrid\",\"object\":\"chat.completion.chunk\",\"model\":"
                + json::escape_str(served_name_) + ",\"choices\":[{\"index\":0,\"delta\":{"
                + json::escape_str(field) + ":" + json::escape_str(text) + "},\"finish_reason\":null}]}";
            return send_all(fd, "data: " + chunk_json + "\n\n");
        };

        auto finish_tool_buf = [&]() {
            if (tool_buf.empty()) return;
            try {
                auto obj = json::parse(tool_buf);
                std::string fname = obj["name"].as_str();
                std::string fargs = json::dump(obj["args"]);
                json::Value tc;
                tc["id"] = json::Value{json::Value::String, false, 0, "call_" + std::to_string(tool_calls.size())};
                tc["type"] = json::Value{json::Value::String, false, 0, "function"};
                tc["function"]["name"] = json::Value{json::Value::String, false, 0, fname};
                tc["function"]["arguments"] = json::Value{json::Value::String, false, 0, fargs};
                if (stream) {
                    json::Value delta_tc = tc;
                    delta_tc["index"] = json::Value{json::Value::Number, false, (double)tool_calls.size()};
                    std::string chunk_json = "{\"id\":\"chatcmpl-warpgrid\",\"object\":\"chat.completion.chunk\",\"model\":"
                        + json::escape_str(served_name_) + ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":["
                        + json::dump(delta_tc) + "]},\"finish_reason\":null}]}";
                    send_all(fd, "data: " + chunk_json + "\n\n");
                }
                tool_calls.push_back(std::move(tc));
            } catch (...) {}
            tool_buf.clear();
        };

        bool client_ok = true;
        double ttft_ms = 0.0;
        bool got_first = false;
        while (client_ok) {
            int32_t tid = -1;
            {
                std::unique_lock<std::mutex> lk(st->mu);
                st->cv.wait(lk, [&]() { return !st->tokens.empty() || st->finished; });
                if (st->tokens.empty()) break;
                tid = st->tokens.front();
                st->tokens.pop_front();
            }
            if (!got_first) {
                got_first = true;
                ttft_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - st->t_start).count();
            }
            ++completion_tokens;
            if (tid == tok_.eos_id) {
                finish_tool_buf();
                finish_reason = tool_calls.empty() ? "stop" : "tool_calls";
                break;
            }
            if (tid == tok_.message_model_id) {
                finish_tool_buf();
                mode = ParseMode::Header;
            } else if (tid == tok_.thinking_id) {
                mode = ParseMode::Thinking;
            } else if (tid == tok_.content_text_id) {
                mode = ParseMode::Content;
            } else if (tid == tok_.invoke_tool_id) {
                mode = ParseMode::ToolJson;
                tool_buf.clear();
            } else if (tid == tok_.end_message_id) {
                if (mode == ParseMode::ToolJson) finish_tool_buf();
                mode = ParseMode::None;
            } else {
                std::string piece = tok_.decode_token(tid, true);
                if (mode == ParseMode::Thinking) {
                    full_reasoning += piece;
                    client_ok = emit_sse_delta("reasoning", piece);
                } else if (mode == ParseMode::Content || mode == ParseMode::None) {
                    full_content += piece;
                    client_ok = emit_sse_delta("content", piece);
                } else if (mode == ParseMode::ToolJson) {
                    tool_buf += piece;
                }
            }
        }
        auto record_telemetry = [&](bool cancelled, const std::string& fin_reason) {
            double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - st->t_start).count();
            double gen_ms = std::max(0.0, total_ms - ttft_ms);
            double gen_tps = (completion_tokens > 1 && gen_ms > 0.0)
                ? ((completion_tokens - 1) * 1000.0 / gen_ms)
                : (total_ms > 0.0 ? (completion_tokens * 1000.0 / total_ms) : 0.0);
            int s_steps = st->spec_steps.load(std::memory_order_relaxed);
            int s_drafted = st->spec_drafted.load(std::memory_order_relaxed);
            int s_accepted = st->spec_accepted.load(std::memory_order_relaxed);
            bool used_df2 = st->used_dflash2.load(std::memory_order_relaxed);
            double mean_a = s_steps > 0 ? (double)(s_accepted + s_steps) / (double)s_steps : 0.0;

            RequestSpan sp;
            sp.model = served_name_;
            sp.slot_id = st->slot_id;
            sp.prompt_tokens = prompt_len;
            sp.cached_tokens = st->cached_tokens;
            sp.completion_tokens = completion_tokens;
            sp.ttft_ms = ttft_ms;
            sp.total_ms = total_ms;
            sp.gen_tok_per_sec = gen_tps;
            sp.spec_mean_a = mean_a;
            sp.dflash2_active = used_df2;
            sp.finish_reason = cancelled ? "cancelled" : fin_reason;
            sp.cancelled = cancelled;
            sp.error = st->error;
            sp.start_unix_ns = st->start_unix_ns;
            sp.end_unix_ns = now_unix_ns();

            metrics_.on_request_complete(sp, s_steps, s_drafted, s_accepted);
            otlp_.emit(sp);
            const char* spec_lbl = s_steps > 0 ? (used_df2 ? "df2" : "dspark") : "off";
            fprintf(stderr, "[warpgrid] req slot=%d prompt=%d cached=%d gen=%d ttft=%.1fms tps=%.2f spec=%s(mean_a=%.2f) temp=%.2f finish=%s\n",
                    st->slot_id, prompt_len, st->cached_tokens, completion_tokens, ttft_ms, gen_tps,
                    spec_lbl, mean_a, st->temperature, sp.finish_reason.c_str());
        };

        if (!client_ok) {
            st->cancelled.store(true, std::memory_order_relaxed);
            record_telemetry(true, "cancelled");
            return;
        }
        finish_tool_buf();
        record_telemetry(false, finish_reason);

        if (stream) {
            std::string end_chunk = "{\"id\":\"chatcmpl-warpgrid\",\"object\":\"chat.completion.chunk\",\"model\":"
                + json::escape_str(served_name_) + ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":"
                + json::escape_str(finish_reason) + "}],\"usage\":{\"prompt_tokens\":"
                + std::to_string(prompt_len) + ",\"completion_tokens\":" + std::to_string(completion_tokens)
                + ",\"total_tokens\":" + std::to_string(prompt_len + completion_tokens)
                + ",\"prompt_tokens_details\":{\"cached_tokens\":" + std::to_string(st->cached_tokens) + "}}}";
            send_all(fd, "data: " + end_chunk + "\n\ndata: [DONE]\n\n");
        } else {
            json::Value msg;
            msg["role"] = json::Value{json::Value::String, false, 0, "assistant"};
            msg["content"] = json::Value{json::Value::String, false, 0, full_content};
            if (!full_reasoning.empty()) {
                msg["reasoning"] = json::Value{json::Value::String, false, 0, full_reasoning};
                msg["reasoning_content"] = json::Value{json::Value::String, false, 0, full_reasoning};
            }
            if (!tool_calls.empty()) {
                msg["tool_calls"].type = json::Value::Array;
                msg["tool_calls"].arr = std::move(tool_calls);
            }
            json::Value choice;
            choice["index"] = json::Value{json::Value::Number, false, 0};
            choice["message"] = std::move(msg);
            choice["finish_reason"] = json::Value{json::Value::String, false, 0, finish_reason};

            json::Value resp;
            resp["id"] = json::Value{json::Value::String, false, 0, "chatcmpl-warpgrid"};
            resp["object"] = json::Value{json::Value::String, false, 0, "chat.completion"};
            resp["model"] = json::Value{json::Value::String, false, 0, served_name_};
            resp["choices"].type = json::Value::Array;
            resp["choices"].arr.push_back(std::move(choice));
            resp["usage"]["prompt_tokens"] = json::Value{json::Value::Number, false, (double)prompt_len};
            resp["usage"]["completion_tokens"] = json::Value{json::Value::Number, false, (double)completion_tokens};
            resp["usage"]["total_tokens"] = json::Value{json::Value::Number, false, (double)(prompt_len + completion_tokens)};
            resp["usage"]["prompt_tokens_details"]["cached_tokens"] = json::Value{json::Value::Number, false, (double)st->cached_tokens};
            send_json(fd, 200, json::dump(resp));
        }
    }

    void handle_client(int fd) {
        std::string buf;
        char tmp[4096];
        size_t header_end = std::string::npos;
        size_t content_len = 0;
        std::string method, path;

        while (true) {
            ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
            if (n <= 0) break;
            buf.append(tmp, (size_t)n);
            if (header_end == std::string::npos) {
                header_end = buf.find("\r\n\r\n");
                if (header_end != std::string::npos) {
                    std::string_view hdrs(buf.data(), header_end);
                    std::string_view req_line = hdrs.substr(0, hdrs.find("\r\n"));
                    bool valid_req_line = parse_request_line(req_line, method, path);
                    if (!api_key_.empty()) {
                        bool is_pub = valid_req_line && is_public_endpoint(method, path);
                        if (!is_pub && !is_request_authorized(hdrs, api_key_)) {
                            send_unauthorized(fd);
                            ::close(fd);
                            return;
                        }
                    }
                    auto cl_pos = hdrs.find("Content-Length:");
                    if (cl_pos == std::string_view::npos) cl_pos = hdrs.find("content-length:");
                    if (cl_pos != std::string_view::npos) {
                        content_len = (size_t)std::strtoull(buf.c_str() + cl_pos + 15, nullptr, 10);
                    }
                }
            }
            if (header_end != std::string::npos && buf.size() >= header_end + 4 + content_len) break;
        }
        if (header_end == std::string::npos) { ::close(fd); return; }

        std::string body = buf.substr(header_end + 4, content_len);

        if (method == "GET" && path == "/health") {
            send_json(fd, 200, "{\"status\":\"ok\"}");
        } else if (method == "GET" && path == "/status") {
            send_json(fd, 200, metrics_.format_status_json());
        } else if (method == "GET" && path == "/metrics") {
            send_text(fd, 200, "text/plain; version=0.0.4; charset=utf-8", metrics_.format_prometheus());
        } else if (method == "GET" && path == "/v1/models") {
            send_json(fd, 200, "{\"object\":\"list\",\"data\":[{\"id\":" + json::escape_str(served_name_) + ",\"object\":\"model\"}]}");
        } else if (method == "POST" && path == "/v1/chat/completions") {
            try {
                auto req = json::parse(body);
                handle_chat(fd, req);
            } catch (const std::exception& e) {
                metrics_.on_request_error();
                fprintf(stderr, "[warpgrid] req_error 400 %s\n", e.what());
                send_json(fd, 400, std::string("{\"error\":") + json::escape_str(e.what()) + "}");
            }
        } else {
            send_json(fd, 404, "{\"error\":\"not found\"}");
        }
        ::close(fd);
    }

public:
    OpenAIServer(Inkling2Model& m, Tokenizer& t, std::string name, int port,
                 std::string otlp_endpoint = "", std::string api_key = "")
        : model_(m),
          tok_(t),
          served_name_(std::move(name)),
          port_(port),
          api_key_(std::move(api_key)),
          metrics_(served_name_,
#ifdef WARPGRID_METAL
                   "metal",
#else
                   "cuda",
#endif
                   m.max_slots, m.max_seq_len, !m.dflash2_dir.empty(),
                   parse_http_url(otlp_endpoint).valid),
          otlp_(otlp_endpoint) {}

    ~OpenAIServer() {
        stop_worker_ = true;
        sched_cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    void run() {
        std::signal(SIGPIPE, SIG_IGN);
        worker_ = std::thread(&OpenAIServer::worker_loop, this);
        int sfd = ::socket(AF_INET, SOCK_STREAM, 0);
        int opt = 1;
        ::setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons((uint16_t)port_);
        if (::bind(sfd, (sockaddr*)&addr, sizeof(addr)) < 0) {
            throw std::runtime_error("bind failed on port " + std::to_string(port_));
        }
        ::listen(sfd, 128);
        fprintf(stderr, "[warpgrid] Listening on 0.0.0.0:%d (model: %s, max_slots=%d, continuous batching ON, auth=%s)\n",
                port_, served_name_.c_str(), model_.max_slots, api_key_.empty() ? "OFF" : "ON");
        while (true) {
            int cfd = ::accept(sfd, nullptr, nullptr);
            if (cfd < 0) continue;
#ifdef SO_NOSIGPIPE
            int nosig = 1;
            ::setsockopt(cfd, SOL_SOCKET, SO_NOSIGPIPE, &nosig, sizeof(nosig));
#endif
            std::thread(&OpenAIServer::handle_client, this, cfd).detach();
        }
    }
};

} // namespace ink
