#pragma once
#include "json.hpp"
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iomanip>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace ink {

struct SlotTelemetry {
    int slot_id = 0;
    bool active = false;
    int pos = 0;
    int cached_tokens = 0;
    int generated = 0;
    bool dflash2_active = false;
};

struct RequestSpan {
    std::string model;
    int slot_id = -1;
    int prompt_tokens = 0;
    int cached_tokens = 0;
    int completion_tokens = 0;
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    double gen_tok_per_sec = 0.0;
    double spec_mean_a = 0.0;
    bool dflash2_active = false;
    std::string finish_reason;
    bool cancelled = false;
    std::string error;
    uint64_t start_unix_ns = 0;
    uint64_t end_unix_ns = 0;
};

struct ParsedHttpUrl {
    bool valid = false;
    std::string host;
    int port = 80;
    std::string path = "/v1/traces";
};

inline uint64_t now_unix_ns() {
    auto now = std::chrono::system_clock::now().time_since_epoch();
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

inline ParsedHttpUrl parse_http_url(const std::string& url) {
    ParsedHttpUrl out;
    const std::string prefix = "http://";
    if (url.rfind(prefix, 0) != 0) return out;
    std::string rest = url.substr(prefix.size());
    if (rest.empty()) return out;

    size_t slash = rest.find('/');
    std::string hostport = (slash == std::string::npos) ? rest : rest.substr(0, slash);
    std::string path = (slash == std::string::npos) ? "" : rest.substr(slash);

    if (hostport.empty()) return out;
    if (hostport[0] == '[') {
        size_t close = hostport.find(']');
        if (close == std::string::npos || close <= 1) return out;
        out.host = hostport.substr(1, close - 1);
        std::string tail = hostport.substr(close + 1);
        if (tail.empty()) {
            out.port = 80;
        } else if (tail[0] == ':' && tail.size() > 1) {
            try {
                int p = std::stoi(tail.substr(1));
                if (p <= 0 || p > 65535) return out;
                out.port = p;
            } catch (...) {
                return out;
            }
        } else {
            return out;
        }
    } else {
        size_t colon = hostport.rfind(':');
        if (colon != std::string::npos) {
            out.host = hostport.substr(0, colon);
            std::string pstr = hostport.substr(colon + 1);
            if (pstr.empty()) return out;
            try {
                int p = std::stoi(pstr);
                if (p <= 0 || p > 65535) return out;
                out.port = p;
            } catch (...) {
                return out;
            }
        } else {
            out.host = hostport;
            out.port = 80;
        }
    }

    if (out.host.empty()) return out;
    out.path = (path.empty() || path == "/") ? "/v1/traces" : path;
    out.valid = true;
    return out;
}

class ServerMetrics {
    std::string model_name_;
    std::string backend_;
    int max_slots_;
    int max_seq_len_;
    bool dflash2_enabled_;
    bool otlp_enabled_;
    std::chrono::steady_clock::time_point start_time_;

    mutable std::mutex mu_;
    uint64_t requests_total_ = 0;
    int64_t requests_active_ = 0;
    int64_t requests_pending_ = 0;
    uint64_t requests_cancelled_ = 0;
    uint64_t requests_error_ = 0;
    uint64_t prompt_tokens_total_ = 0;
    uint64_t cached_tokens_total_ = 0;
    uint64_t generated_tokens_total_ = 0;
    uint64_t spec_steps_total_ = 0;
    uint64_t spec_drafted_total_ = 0;
    uint64_t spec_accepted_total_ = 0;
    double ttft_ms_sum_ = 0.0;
    uint64_t ttft_count_ = 0;
    uint64_t decode_tokens_total_ = 0;
    double decode_ms_sum_ = 0.0;
    std::vector<SlotTelemetry> slots_;

public:
    ServerMetrics(std::string model_name, std::string backend, int max_slots,
                  int max_seq_len, bool dflash2_enabled, bool otlp_enabled)
        : model_name_(std::move(model_name)),
          backend_(std::move(backend)),
          max_slots_(std::max(1, max_slots)),
          max_seq_len_(max_seq_len),
          dflash2_enabled_(dflash2_enabled),
          otlp_enabled_(otlp_enabled),
          start_time_(std::chrono::steady_clock::now()),
          slots_(max_slots_)
    {
        for (int i = 0; i < max_slots_; ++i) slots_[i].slot_id = i;
    }

    void on_enqueue() {
        std::lock_guard<std::mutex> lk(mu_);
        ++requests_pending_;
    }

    void on_dequeue() {
        std::lock_guard<std::mutex> lk(mu_);
        if (requests_pending_ > 0) --requests_pending_;
        ++requests_active_;
    }

    void on_request_complete(const RequestSpan& sp, int spec_steps, int spec_drafted, int spec_accepted) {
        std::lock_guard<std::mutex> lk(mu_);
        ++requests_total_;
        if (requests_active_ > 0) --requests_active_;
        if (sp.cancelled) ++requests_cancelled_;
        if (!sp.error.empty()) ++requests_error_;
        prompt_tokens_total_ += (uint64_t)std::max(0, sp.prompt_tokens);
        cached_tokens_total_ += (uint64_t)std::max(0, sp.cached_tokens);
        generated_tokens_total_ += (uint64_t)std::max(0, sp.completion_tokens);
        spec_steps_total_ += (uint64_t)std::max(0, spec_steps);
        spec_drafted_total_ += (uint64_t)std::max(0, spec_drafted);
        spec_accepted_total_ += (uint64_t)std::max(0, spec_accepted);
        if (sp.ttft_ms > 0.0) {
            ttft_ms_sum_ += sp.ttft_ms;
            ++ttft_count_;
        }
        double dec_ms = std::max(0.0, sp.total_ms - sp.ttft_ms);
        if (sp.completion_tokens > 1 && dec_ms > 0.0) {
            decode_tokens_total_ += (uint64_t)(sp.completion_tokens - 1);
            decode_ms_sum_ += dec_ms;
        }
    }

    void on_request_error() {
        std::lock_guard<std::mutex> lk(mu_);
        ++requests_total_;
        ++requests_error_;
    }

    void update_slot(const SlotTelemetry& st) {
        std::lock_guard<std::mutex> lk(mu_);
        if (st.slot_id >= 0 && st.slot_id < max_slots_) {
            slots_[st.slot_id] = st;
        }
    }

    void clear_slot(int slot_id) {
        std::lock_guard<std::mutex> lk(mu_);
        if (slot_id >= 0 && slot_id < max_slots_) {
            slots_[slot_id] = SlotTelemetry{};
            slots_[slot_id].slot_id = slot_id;
        }
    }

    std::string format_status_json() const {
        std::lock_guard<std::mutex> lk(mu_);
        double uptime_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time_).count();
        double cache_ratio = prompt_tokens_total_ > 0
            ? (double)cached_tokens_total_ / (double)prompt_tokens_total_ : 0.0;
        double accept_ratio = spec_drafted_total_ > 0
            ? (double)spec_accepted_total_ / (double)spec_drafted_total_ : 0.0;
        double mean_a = spec_steps_total_ > 0
            ? (double)(spec_accepted_total_ + spec_steps_total_) / (double)spec_steps_total_ : 0.0;
        double ttft_avg = ttft_count_ > 0 ? ttft_ms_sum_ / (double)ttft_count_ : 0.0;
        double dec_tps = decode_ms_sum_ > 0.0 ? (double)decode_tokens_total_ / (decode_ms_sum_ / 1000.0) : 0.0;

        json::Value root;
        root["model"] = json::Value{json::Value::String, false, 0, model_name_};
        root["backend"] = json::Value{json::Value::String, false, 0, backend_};
        root["uptime_s"] = json::Value{json::Value::Number, false, uptime_s};
        root["max_slots"] = json::Value{json::Value::Number, false, (double)max_slots_};
        root["max_seq_len"] = json::Value{json::Value::Number, false, (double)max_seq_len_};
        root["dflash2_enabled"] = json::Value{json::Value::Bool, dflash2_enabled_, 0};
        root["otlp_enabled"] = json::Value{json::Value::Bool, otlp_enabled_, 0};

        root["requests"]["total"] = json::Value{json::Value::Number, false, (double)requests_total_};
        root["requests"]["active"] = json::Value{json::Value::Number, false, (double)requests_active_};
        root["requests"]["pending"] = json::Value{json::Value::Number, false, (double)requests_pending_};
        root["requests"]["cancelled"] = json::Value{json::Value::Number, false, (double)requests_cancelled_};
        root["requests"]["error"] = json::Value{json::Value::Number, false, (double)requests_error_};

        root["tokens"]["prompt_total"] = json::Value{json::Value::Number, false, (double)prompt_tokens_total_};
        root["tokens"]["cached_total"] = json::Value{json::Value::Number, false, (double)cached_tokens_total_};
        root["tokens"]["generated_total"] = json::Value{json::Value::Number, false, (double)generated_tokens_total_};
        root["tokens"]["cache_hit_ratio"] = json::Value{json::Value::Number, false, cache_ratio};

        root["latency"]["ttft_ms_avg"] = json::Value{json::Value::Number, false, ttft_avg};
        root["latency"]["decode_tok_per_sec"] = json::Value{json::Value::Number, false, dec_tps};

        root["speculative"]["steps"] = json::Value{json::Value::Number, false, (double)spec_steps_total_};
        root["speculative"]["drafted"] = json::Value{json::Value::Number, false, (double)spec_drafted_total_};
        root["speculative"]["accepted"] = json::Value{json::Value::Number, false, (double)spec_accepted_total_};
        root["speculative"]["accept_ratio"] = json::Value{json::Value::Number, false, accept_ratio};
        root["speculative"]["mean_tokens_per_step"] = json::Value{json::Value::Number, false, mean_a};

        root["slots"].type = json::Value::Array;
        for (const auto& sl : slots_) {
            json::Value sv;
            sv["slot_id"] = json::Value{json::Value::Number, false, (double)sl.slot_id};
            sv["active"] = json::Value{json::Value::Bool, sl.active, 0};
            sv["pos"] = json::Value{json::Value::Number, false, (double)sl.pos};
            sv["cached_tokens"] = json::Value{json::Value::Number, false, (double)sl.cached_tokens};
            sv["generated"] = json::Value{json::Value::Number, false, (double)sl.generated};
            sv["dflash2_active"] = json::Value{json::Value::Bool, sl.dflash2_active, 0};
            root["slots"].arr.push_back(std::move(sv));
        }
        return json::dump(root);
    }

    std::string format_prometheus() const {
        std::lock_guard<std::mutex> lk(mu_);
        double uptime_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time_).count();
        int active_slots = 0;
        int64_t kv_tokens_in_use = 0;
        for (const auto& sl : slots_) {
            if (sl.active) {
                ++active_slots;
                kv_tokens_in_use += sl.pos;
            }
        }
        double cache_ratio = prompt_tokens_total_ > 0
            ? (double)cached_tokens_total_ / (double)prompt_tokens_total_ : 0.0;
        double accept_ratio = spec_drafted_total_ > 0
            ? (double)spec_accepted_total_ / (double)spec_drafted_total_ : 0.0;
        double mean_a = spec_steps_total_ > 0
            ? (double)(spec_accepted_total_ + spec_steps_total_) / (double)spec_steps_total_ : 0.0;
        double ttft_avg = ttft_count_ > 0 ? ttft_ms_sum_ / (double)ttft_count_ : 0.0;
        double dec_tps = decode_ms_sum_ > 0.0 ? (double)decode_tokens_total_ / (decode_ms_sum_ / 1000.0) : 0.0;

        std::ostringstream o;
        o << std::fixed << std::setprecision(4);
        o << "# HELP warpgrid_uptime_seconds Server uptime in seconds.\n"
          << "# TYPE warpgrid_uptime_seconds gauge\n"
          << "warpgrid_uptime_seconds " << uptime_s << "\n"
          << "# HELP warpgrid_requests_total Total completed requests.\n"
          << "# TYPE warpgrid_requests_total counter\n"
          << "warpgrid_requests_total " << requests_total_ << "\n"
          << "# HELP warpgrid_requests_active Currently executing requests.\n"
          << "# TYPE warpgrid_requests_active gauge\n"
          << "warpgrid_requests_active " << requests_active_ << "\n"
          << "# HELP warpgrid_pending_requests Queued requests awaiting a free slot.\n"
          << "# TYPE warpgrid_pending_requests gauge\n"
          << "warpgrid_pending_requests " << requests_pending_ << "\n"
          << "# HELP warpgrid_requests_cancelled_total Total client-cancelled requests.\n"
          << "# TYPE warpgrid_requests_cancelled_total counter\n"
          << "warpgrid_requests_cancelled_total " << requests_cancelled_ << "\n"
          << "# HELP warpgrid_requests_errors_total Total failed requests.\n"
          << "# TYPE warpgrid_requests_errors_total counter\n"
          << "warpgrid_requests_errors_total " << requests_error_ << "\n"
          << "# HELP warpgrid_active_slots Currently occupied KV slots.\n"
          << "# TYPE warpgrid_active_slots gauge\n"
          << "warpgrid_active_slots " << active_slots << "\n"
          << "# HELP warpgrid_max_slots Configured maximum concurrent KV slots.\n"
          << "# TYPE warpgrid_max_slots gauge\n"
          << "warpgrid_max_slots " << max_slots_ << "\n"
          << "# HELP warpgrid_kv_tokens_in_use Total KV cache tokens held by active slots.\n"
          << "# TYPE warpgrid_kv_tokens_in_use gauge\n"
          << "warpgrid_kv_tokens_in_use " << kv_tokens_in_use << "\n"
          << "# HELP warpgrid_prompt_tokens_total Total prompt tokens processed.\n"
          << "# TYPE warpgrid_prompt_tokens_total counter\n"
          << "warpgrid_prompt_tokens_total " << prompt_tokens_total_ << "\n"
          << "# HELP warpgrid_prefix_cache_hit_tokens_total Total prompt tokens served from prefix cache.\n"
          << "# TYPE warpgrid_prefix_cache_hit_tokens_total counter\n"
          << "warpgrid_prefix_cache_hit_tokens_total " << cached_tokens_total_ << "\n"
          << "# HELP warpgrid_prefix_cache_hit_ratio Ratio of cached prompt tokens to total prompt tokens.\n"
          << "# TYPE warpgrid_prefix_cache_hit_ratio gauge\n"
          << "warpgrid_prefix_cache_hit_ratio " << cache_ratio << "\n"
          << "# HELP warpgrid_generated_tokens_total Total completion tokens generated.\n"
          << "# TYPE warpgrid_generated_tokens_total counter\n"
          << "warpgrid_generated_tokens_total " << generated_tokens_total_ << "\n"
          << "# HELP warpgrid_spec_steps_total Total speculative verification steps.\n"
          << "# TYPE warpgrid_spec_steps_total counter\n"
          << "warpgrid_spec_steps_total " << spec_steps_total_ << "\n"
          << "# HELP warpgrid_spec_drafted_tokens_total Total speculative tokens drafted.\n"
          << "# TYPE warpgrid_spec_drafted_tokens_total counter\n"
          << "warpgrid_spec_drafted_tokens_total " << spec_drafted_total_ << "\n"
          << "# HELP warpgrid_spec_accepted_tokens_total Total speculative tokens accepted.\n"
          << "# TYPE warpgrid_spec_accepted_tokens_total counter\n"
          << "warpgrid_spec_accepted_tokens_total " << spec_accepted_total_ << "\n"
          << "# HELP warpgrid_spec_accept_ratio Speculative draft token acceptance ratio.\n"
          << "# TYPE warpgrid_spec_accept_ratio gauge\n"
          << "warpgrid_spec_accept_ratio " << accept_ratio << "\n"
          << "# HELP warpgrid_spec_mean_tokens_per_step Mean tokens emitted per speculative step.\n"
          << "# TYPE warpgrid_spec_mean_tokens_per_step gauge\n"
          << "warpgrid_spec_mean_tokens_per_step " << mean_a << "\n"
          << "# HELP warpgrid_ttft_ms_avg Mean time-to-first-token in milliseconds.\n"
          << "# TYPE warpgrid_ttft_ms_avg gauge\n"
          << "warpgrid_ttft_ms_avg " << ttft_avg << "\n"
          << "# HELP warpgrid_decode_tok_per_sec Cumulative decode throughput in tokens per second.\n"
          << "# TYPE warpgrid_decode_tok_per_sec gauge\n"
          << "warpgrid_decode_tok_per_sec " << dec_tps << "\n";
        return o.str();
    }
};

class OtlpExporter {
    ParsedHttpUrl url_;
    bool enabled_ = false;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<RequestSpan> queue_;
    bool stop_ = false;
    std::thread worker_;
    static constexpr size_t kMaxQueue = 256;

    static std::string random_hex(size_t n_bytes) {
        static thread_local std::mt19937_64 rng(std::random_device{}());
        static const char* hex = "0123456789abcdef";
        std::string out;
        out.resize(n_bytes * 2);
        for (size_t i = 0; i < n_bytes; ++i) {
            uint8_t b = (uint8_t)(rng() & 0xFF);
            out[2 * i] = hex[b >> 4];
            out[2 * i + 1] = hex[b & 0xF];
        }
        return out;
    }

    static json::Value kv_str(const std::string& k, const std::string& v) {
        json::Value o;
        o["key"] = json::Value{json::Value::String, false, 0, k};
        o["value"]["stringValue"] = json::Value{json::Value::String, false, 0, v};
        return o;
    }

    static json::Value kv_int(const std::string& k, int64_t v) {
        json::Value o;
        o["key"] = json::Value{json::Value::String, false, 0, k};
        o["value"]["intValue"] = json::Value{json::Value::String, false, 0, std::to_string(v)};
        return o;
    }

    static json::Value kv_double(const std::string& k, double v) {
        json::Value o;
        o["key"] = json::Value{json::Value::String, false, 0, k};
        o["value"]["doubleValue"] = json::Value{json::Value::Number, false, v};
        return o;
    }

    static json::Value kv_bool(const std::string& k, bool v) {
        json::Value o;
        o["key"] = json::Value{json::Value::String, false, 0, k};
        o["value"]["boolValue"] = json::Value{json::Value::Bool, v, 0};
        return o;
    }

    void post_json(const std::string& body) const {
        addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        addrinfo* res = nullptr;
        std::string port_str = std::to_string(url_.port);
        if (::getaddrinfo(url_.host.c_str(), port_str.c_str(), &hints, &res) != 0 || !res) {
            return;
        }
        int fd = -1;
        for (addrinfo* p = res; p != nullptr; p = p->ai_next) {
            fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
            if (fd < 0) continue;
            timeval tv{};
            tv.tv_sec = 2;
            ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
            ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#ifdef SO_NOSIGPIPE
            int nosig = 1;
            ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &nosig, sizeof(nosig));
#endif
            if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
            ::close(fd);
            fd = -1;
        }
        ::freeaddrinfo(res);
        if (fd < 0) return;

        std::string host_hdr = (url_.host.find(':') != std::string::npos)
            ? ("[" + url_.host + "]") : url_.host;
        std::string req = "POST " + url_.path + " HTTP/1.1\r\n"
            + "Host: " + host_hdr + ":" + port_str + "\r\n"
            + "Content-Type: application/json\r\n"
            + "Content-Length: " + std::to_string(body.size()) + "\r\n"
            + "Connection: close\r\n\r\n" + body;
        size_t sent = 0;
        while (sent < req.size()) {
#ifdef MSG_NOSIGNAL
            int flags = MSG_NOSIGNAL;
#else
            int flags = 0;
#endif
            ssize_t n = ::send(fd, req.data() + sent, req.size() - sent, flags);
            if (n <= 0) break;
            sent += (size_t)n;
        }
        char drain[256];
        (void)::recv(fd, drain, sizeof(drain), 0);
        ::close(fd);
    }

    void run_loop() {
        while (true) {
            RequestSpan sp;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&]() { return stop_ || !queue_.empty(); });
                if (queue_.empty()) {
                    if (stop_) break;
                    continue;
                }
                sp = std::move(queue_.front());
                queue_.pop_front();
            }
            post_json(build_otlp_span_json(sp));
        }
    }

public:
    explicit OtlpExporter(const std::string& endpoint) {
        if (endpoint.empty()) return;
        url_ = parse_http_url(endpoint);
        if (!url_.valid) {
            fprintf(stderr, "[warpgrid] Invalid --otlp-endpoint '%s' (expected http://host:port[/path]); OTLP disabled\n",
                    endpoint.c_str());
            return;
        }
        enabled_ = true;
        worker_ = std::thread(&OtlpExporter::run_loop, this);
    }

    ~OtlpExporter() {
        if (!enabled_) return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    bool enabled() const { return enabled_; }

    static std::string build_otlp_span_json(const RequestSpan& sp) {
        json::Value attrs;
        attrs.type = json::Value::Array;
        attrs.arr.push_back(kv_str("gen_ai.system", "warpgrid"));
        attrs.arr.push_back(kv_str("gen_ai.request.model", sp.model));
        attrs.arr.push_back(kv_int("gen_ai.usage.input_tokens", sp.prompt_tokens));
        attrs.arr.push_back(kv_int("gen_ai.usage.output_tokens", sp.completion_tokens));
        attrs.arr.push_back(kv_int("warpgrid.cached_tokens", sp.cached_tokens));
        attrs.arr.push_back(kv_double("warpgrid.ttft_ms", sp.ttft_ms));
        attrs.arr.push_back(kv_double("warpgrid.gen_tok_per_sec", sp.gen_tok_per_sec));
        attrs.arr.push_back(kv_double("warpgrid.spec_mean_a", sp.spec_mean_a));
        attrs.arr.push_back(kv_bool("warpgrid.dflash2_active", sp.dflash2_active));
        attrs.arr.push_back(kv_int("warpgrid.slot_id", sp.slot_id));
        attrs.arr.push_back(kv_str("warpgrid.finish_reason", sp.finish_reason));
        attrs.arr.push_back(kv_bool("warpgrid.cancelled", sp.cancelled));

        json::Value span;
        span["traceId"] = json::Value{json::Value::String, false, 0, random_hex(16)};
        span["spanId"] = json::Value{json::Value::String, false, 0, random_hex(8)};
        span["name"] = json::Value{json::Value::String, false, 0, "chat.completions"};
        span["kind"] = json::Value{json::Value::Number, false, 2};
        span["startTimeUnixNano"] = json::Value{json::Value::String, false, 0, std::to_string(sp.start_unix_ns)};
        span["endTimeUnixNano"] = json::Value{json::Value::String, false, 0, std::to_string(sp.end_unix_ns)};
        span["attributes"] = std::move(attrs);
        span["status"]["code"] = json::Value{json::Value::Number, false, sp.error.empty() ? 1.0 : 2.0};
        if (!sp.error.empty()) {
            span["status"]["message"] = json::Value{json::Value::String, false, 0, sp.error};
        }

        json::Value scope_span;
        scope_span["scope"]["name"] = json::Value{json::Value::String, false, 0, "warpgrid"};
        scope_span["spans"].type = json::Value::Array;
        scope_span["spans"].arr.push_back(std::move(span));

        json::Value res_attrs;
        res_attrs.type = json::Value::Array;
        res_attrs.arr.push_back(kv_str("service.name", "warpgrid"));

        json::Value res_span;
        res_span["resource"]["attributes"] = std::move(res_attrs);
        res_span["scopeSpans"].type = json::Value::Array;
        res_span["scopeSpans"].arr.push_back(std::move(scope_span));

        json::Value root;
        root["resourceSpans"].type = json::Value::Array;
        root["resourceSpans"].arr.push_back(std::move(res_span));
        return json::dump(root);
    }

    void emit(const RequestSpan& sp) {
        if (!enabled_) return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (queue_.size() >= kMaxQueue) queue_.pop_front();
            queue_.push_back(sp);
        }
        cv_.notify_one();
    }
};

}
