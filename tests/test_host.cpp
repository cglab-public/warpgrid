#include "../src/config.hpp"
#include "../src/json.hpp"
#include "../src/safetensors.hpp"
#include "../src/telemetry.hpp"
#include "../src/tokenizer.hpp"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

static int g_passed = 0;
static int g_failed = 0;

#define EXPECT_TRUE(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failed; \
        return; \
    } \
} while (0)

#define EXPECT_EQ(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (!(_a == _b)) { \
        fprintf(stderr, "FAIL %s:%d: %s != %s\n", __FILE__, __LINE__, #a, #b); \
        ++g_failed; \
        return; \
    } \
} while (0)

static void run_test(const char* name, void (*fn)()) {
    int before = g_failed;
    fn();
    if (g_failed == before) {
        ++g_passed;
        printf("[PASS] %s\n", name);
    } else {
        printf("[FAIL] %s\n", name);
    }
}

static void test_json_roundtrip_and_escaping() {
    std::string raw = R"({"str":"hello\n\"world\"","num":42.5,"flag":true,"nil":null,"arr":[1,2,3],"obj":{"k":"v"}})";
    auto v = json::parse(raw);
    EXPECT_TRUE(v.is_object());
    EXPECT_EQ(v["str"].as_str(), std::string("hello\n\"world\""));
    EXPECT_TRUE(std::abs(v["num"].as_double() - 42.5) < 1e-9);
    EXPECT_EQ(v["flag"].as_bool(), true);
    EXPECT_TRUE(v["nil"].is_null());
    EXPECT_EQ(v["arr"].arr.size(), (size_t)3);
    EXPECT_EQ(v["arr"].arr[2].as_int(), 3);
    EXPECT_EQ(v["obj"]["k"].as_str(), std::string("v"));

    std::string dumped = json::dump(v);
    auto v2 = json::parse(dumped);
    EXPECT_EQ(v2["str"].as_str(), std::string("hello\n\"world\""));
    EXPECT_EQ(v2["arr"].arr[1].as_int(), 2);
}

static void test_config_inkling2_parse() {
    std::string tmp = "/tmp/warpgrid_test_config_" + std::to_string((int)getpid()) + ".json";
    {
        std::ofstream f(tmp);
        f << R"({
            "eos_token_id": 200006,
            "text_config": {
                "vocab_size": 201024,
                "unpadded_vocab_size": 200058,
                "hidden_size": 2304,
                "num_hidden_layers": 24,
                "num_attention_heads": 24,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "d_rel": 16,
                "rel_extent": 1024,
                "sliding_window_size": 1024,
                "sconv_kernel_size": 4,
                "rms_norm_eps": 1e-6,
                "n_routed_experts": 128,
                "num_experts_per_tok": 8,
                "intermediate_size": 1152,
                "local_layer_ids": [0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22],
                "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64}
            }
        })";
    }
    auto cfg = InklingConfig::load(tmp);
    ::unlink(tmp.c_str());

    EXPECT_EQ(cfg.hidden_size, 2304);
    EXPECT_EQ(cfg.num_hidden_layers, 24);
    EXPECT_EQ(cfg.num_visits(), 48);
    EXPECT_TRUE(cfg.is_local(0));
    EXPECT_TRUE(!cfg.is_local(5));
    EXPECT_TRUE(!cfg.is_local(11));
    EXPECT_TRUE(!cfg.is_local(17));
    EXPECT_TRUE(!cfg.is_local(23));
    EXPECT_TRUE(cfg.is_loop_core(6));
    EXPECT_TRUE(cfg.is_loop_core(17));
    EXPECT_TRUE(!cfg.is_loop_core(5));
    EXPECT_TRUE(!cfg.is_loop_core(18));
}

static void test_safetensors_portable_read_into_and_errors() {
    std::string tmp = "/tmp/warpgrid_test_" + std::to_string((int)getpid()) + ".safetensors";
    std::vector<float> payload = {1.25f, -2.5f, 3.75f, 4.0f};
    std::string hdr = R"({"t0":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"__metadata__":{"fmt":"pt"}})";
    {
        std::ofstream f(tmp, std::ios::binary);
        uint64_t hlen = (uint64_t)hdr.size();
        f.write(reinterpret_cast<const char*>(&hlen), 8);
        f.write(hdr.data(), (std::streamsize)hdr.size());
        f.write(reinterpret_cast<const char*>(payload.data()), (std::streamsize)(payload.size() * sizeof(float)));
    }

    SafetensorsFile st;
    st.open(tmp);
    EXPECT_TRUE(st.has("t0"));
    EXPECT_TRUE(!st.has("missing_tensor"));
    const auto& info = st.info("t0");
    EXPECT_EQ(info.dtype, std::string("F32"));
    EXPECT_EQ(info.nbytes, (size_t)16);
    EXPECT_EQ(info.shape.size(), (size_t)2);

    std::vector<uint8_t> host_bytes;
    st.load_to_host("t0", host_bytes);
    EXPECT_EQ(host_bytes.size(), (size_t)16);

    std::vector<float> direct(4, 0.0f);
    st.read_into("t0", direct.data(), direct.size() * sizeof(float));
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::abs(direct[i] - payload[i]) < 1e-7f);
    }

    bool threw_missing = false;
    try {
        (void)st.info("does_not_exist");
    } catch (const std::runtime_error&) {
        threw_missing = true;
    }
    EXPECT_TRUE(threw_missing);

    bool threw_small = false;
    try {
        uint8_t tiny[4];
        st.read_into("t0", tiny, sizeof(tiny));
    } catch (const std::runtime_error&) {
        threw_small = true;
    }
    EXPECT_TRUE(threw_small);

    ::unlink(tmp.c_str());
}

static void test_tokenizer_render_chat_and_tools() {
    auto req = json::parse(R"({
        "reasoning_effort": "low",
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        }],
        "messages": [
            {"role": "user", "content": "Weather in Tokyo?"}
        ]
    })");
    std::string rendered = ink::Tokenizer::render_chat(req);
    EXPECT_TRUE(rendered.find("tool_declare") != std::string::npos);
    EXPECT_TRUE(rendered.find("get_weather") != std::string::npos);
    EXPECT_TRUE(rendered.find("Thinking effort level: 0.2") != std::string::npos);
    EXPECT_TRUE(rendered.find("<|message_user|><|content_text|>Weather in Tokyo?<|end_message|>") != std::string::npos);
}

static void test_tokenizer_bpe_roundtrip_synthetic() {
    std::string tmp = "/tmp/warpgrid_test_tok_" + std::to_string((int)getpid()) + ".json";
    {
        std::ofstream f(tmp);
        f << R"({
            "added_tokens": [
                {"id": 10, "content": "<|message_user|>", "special": true},
                {"id": 11, "content": "<|end_message|>", "special": true}
            ],
            "model": {
                "vocab": {
                    "h": 0, "i": 1, "hi": 2, "!": 3, "Ġ": 4
                },
                "merges": ["h i"]
            }
        })";
    }
    ink::Tokenizer tok;
    tok.load(tmp);
    ::unlink(tmp.c_str());

    auto ids = tok.encode("<|message_user|>hi!<|end_message|>");
    EXPECT_EQ(ids.size(), (size_t)4);
    EXPECT_EQ(ids[0], 10);
    EXPECT_EQ(ids[1], 2);
    EXPECT_EQ(ids[2], 3);
    EXPECT_EQ(ids[3], 11);
    EXPECT_EQ(tok.decode(ids, false), std::string("<|message_user|>hi!<|end_message|>"));
    EXPECT_EQ(tok.decode(ids, true), std::string("hi!"));
}

static void test_telemetry_metrics_and_formats() {
    ink::ServerMetrics m("Inkling-2-Mini", "metal", 8, 139264, true, false);
    m.on_enqueue();
    m.on_dequeue();
    ink::SlotTelemetry st;
    st.slot_id = 0;
    st.active = true;
    st.pos = 128;
    st.cached_tokens = 64;
    st.generated = 32;
    st.dflash2_active = true;
    m.update_slot(st);

    ink::RequestSpan sp;
    sp.model = "Inkling-2-Mini";
    sp.slot_id = 0;
    sp.prompt_tokens = 100;
    sp.cached_tokens = 60;
    sp.completion_tokens = 50;
    sp.ttft_ms = 250.0;
    sp.total_ms = 1250.0;
    sp.gen_tok_per_sec = 50.0;
    sp.spec_mean_a = 4.5;
    sp.dflash2_active = true;
    sp.finish_reason = "stop";
    m.on_request_complete(sp, 10, 70, 35);
    m.clear_slot(0);

    auto status = json::parse(m.format_status_json());
    EXPECT_TRUE(status.is_object());
    EXPECT_EQ(status["model"].as_str(), std::string("Inkling-2-Mini"));
    EXPECT_EQ(status["max_slots"].as_int(), 8);
    EXPECT_EQ(status["requests"]["total"].as_int(), 1);
    EXPECT_EQ(status["tokens"]["prompt_total"].as_int(), 100);
    EXPECT_EQ(status["tokens"]["cached_total"].as_int(), 60);
    EXPECT_EQ(status["tokens"]["generated_total"].as_int(), 50);
    EXPECT_TRUE(status["speculative"]["accept_ratio"].as_double() > 0.49);

    std::string prom = m.format_prometheus();
    EXPECT_TRUE(prom.find("warpgrid_requests_total 1") != std::string::npos);
    EXPECT_TRUE(prom.find("warpgrid_prompt_tokens_total 100") != std::string::npos);
    EXPECT_TRUE(prom.find("warpgrid_prefix_cache_hit_tokens_total 60") != std::string::npos);
    EXPECT_TRUE(prom.find("warpgrid_generated_tokens_total 50") != std::string::npos);
    EXPECT_TRUE(prom.find("warpgrid_spec_accept_ratio") != std::string::npos);
    EXPECT_TRUE(prom.find("# TYPE warpgrid_requests_total counter") != std::string::npos);
}

static void test_telemetry_otlp_payload_and_loopback() {
    ink::OtlpExporter disabled("");
    EXPECT_TRUE(!disabled.enabled());

    auto u1 = ink::parse_http_url("http://127.0.0.1:4318");
    EXPECT_TRUE(u1.valid);
    EXPECT_EQ(u1.host, std::string("127.0.0.1"));
    EXPECT_EQ(u1.port, 4318);
    EXPECT_EQ(u1.path, std::string("/v1/traces"));

    auto u2 = ink::parse_http_url("http://127.0.0.1:4318/custom/traces");
    EXPECT_TRUE(u2.valid);
    EXPECT_EQ(u2.path, std::string("/custom/traces"));

    auto u3 = ink::parse_http_url("http://[::1]:4318");
    EXPECT_TRUE(u3.valid);
    EXPECT_EQ(u3.host, std::string("::1"));
    EXPECT_EQ(u3.port, 4318);
    EXPECT_EQ(u3.path, std::string("/v1/traces"));

    ink::RequestSpan sp;
    sp.model = "Inkling-2-Mini";
    sp.slot_id = 0;
    sp.prompt_tokens = 120;
    sp.cached_tokens = 80;
    sp.completion_tokens = 40;
    sp.ttft_ms = 180.0;
    sp.total_ms = 980.0;
    sp.gen_tok_per_sec = 50.0;
    sp.spec_mean_a = 4.2;
    sp.dflash2_active = true;
    sp.finish_reason = "stop";
    sp.start_unix_ns = 1700000000000000000ULL;
    sp.end_unix_ns = 1700000000980000000ULL;

    std::string payload = ink::OtlpExporter::build_otlp_span_json(sp);
    auto root = json::parse(payload);
    EXPECT_TRUE(root.is_object());
    EXPECT_EQ(root["resourceSpans"].arr.size(), (size_t)1);
    auto& span = root["resourceSpans"].arr[0]["scopeSpans"].arr[0]["spans"].arr[0];
    EXPECT_EQ(span["name"].as_str(), std::string("chat.completions"));
    EXPECT_EQ(span["traceId"].as_str().size(), (size_t)32);
    EXPECT_EQ(span["spanId"].as_str().size(), (size_t)16);

    int lfd = ::socket(AF_INET, SOCK_STREAM, 0);
    EXPECT_TRUE(lfd >= 0);
    int opt = 1;
    ::setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    EXPECT_EQ(::bind(lfd, (sockaddr*)&addr, sizeof(addr)), 0);
    socklen_t alen = sizeof(addr);
    EXPECT_EQ(::getsockname(lfd, (sockaddr*)&addr, &alen), 0);
    int port = ntohs(addr.sin_port);
    EXPECT_EQ(::listen(lfd, 4), 0);

    std::string received;
    std::thread srv([&]() {
        int cfd = ::accept(lfd, nullptr, nullptr);
        if (cfd >= 0) {
            char buf[4096];
            ssize_t n = 0;
            while ((n = ::recv(cfd, buf, sizeof(buf), 0)) > 0) {
                received.append(buf, (size_t)n);
                auto he = received.find("\r\n\r\n");
                if (he != std::string::npos) {
                    auto cl = received.find("Content-Length:");
                    if (cl != std::string::npos) {
                        size_t len = (size_t)std::strtoull(received.c_str() + cl + 15, nullptr, 10);
                        if (received.size() >= he + 4 + len) break;
                    }
                }
            }
            const char* ok = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";
            (void)::send(cfd, ok, std::strlen(ok), 0);
            ::close(cfd);
        }
        ::close(lfd);
    });

    {
        ink::OtlpExporter exp("http://127.0.0.1:" + std::to_string(port));
        EXPECT_TRUE(exp.enabled());
        exp.emit(sp);
    }
    srv.join();

    EXPECT_TRUE(received.find("POST /v1/traces HTTP/1.1") != std::string::npos);
    EXPECT_TRUE(received.find("Content-Type: application/json") != std::string::npos);
    EXPECT_TRUE(received.find("chat.completions") != std::string::npos);
}

int main() {
    run_test("test_json_roundtrip_and_escaping", test_json_roundtrip_and_escaping);
    run_test("test_config_inkling2_parse", test_config_inkling2_parse);
    run_test("test_safetensors_portable_read_into_and_errors", test_safetensors_portable_read_into_and_errors);
    run_test("test_tokenizer_render_chat_and_tools", test_tokenizer_render_chat_and_tools);
    run_test("test_tokenizer_bpe_roundtrip_synthetic", test_tokenizer_bpe_roundtrip_synthetic);
    run_test("test_telemetry_metrics_and_formats", test_telemetry_metrics_and_formats);
    run_test("test_telemetry_otlp_payload_and_loopback", test_telemetry_otlp_payload_and_loopback);
    printf("Host tests: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
