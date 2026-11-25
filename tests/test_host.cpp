#include "../src/auth.hpp"
#include "../src/config.hpp"
#include "../src/json.hpp"
#include "../src/prefix_cache.hpp"
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

static void test_auth_request_line_and_api_key() {
    // Constant-time equality
    EXPECT_TRUE(ink::constant_time_eq("secret-key-123", "secret-key-123"));
    EXPECT_TRUE(!ink::constant_time_eq("secret-key-123", "secret-key-124"));
    EXPECT_TRUE(!ink::constant_time_eq("secret-key-123", "secret-key-12"));
    EXPECT_TRUE(!ink::constant_time_eq("secret", ""));
    EXPECT_TRUE(ink::constant_time_eq("", ""));

    // Request line parsing & public endpoint rules
    std::string method, path;
    EXPECT_TRUE(ink::parse_request_line("GET /health HTTP/1.1", method, path));
    EXPECT_EQ(method, std::string("GET"));
    EXPECT_EQ(path, std::string("/health"));
    EXPECT_TRUE(ink::is_public_endpoint(method, path));

    EXPECT_TRUE(ink::parse_request_line("GET /health?probe=1 HTTP/1.1", method, path));
    EXPECT_EQ(path, std::string("/health"));
    EXPECT_TRUE(ink::is_public_endpoint(method, path));

    EXPECT_TRUE(ink::parse_request_line("GET /healthcheck HTTP/1.1", method, path));
    EXPECT_TRUE(!ink::is_public_endpoint(method, path));

    EXPECT_TRUE(ink::parse_request_line("POST /health HTTP/1.1", method, path));
    EXPECT_TRUE(!ink::is_public_endpoint(method, path));

    EXPECT_TRUE(ink::parse_request_line("POST /v1/chat/completions HTTP/1.1", method, path));
    EXPECT_EQ(method, std::string("POST"));
    EXPECT_EQ(path, std::string("/v1/chat/completions"));
    EXPECT_TRUE(!ink::is_public_endpoint(method, path));

    EXPECT_TRUE(!ink::parse_request_line("MALFORMED", method, path));

    // Auth disabled when expected key is empty
    const std::string no_auth_req =
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost:8888\r\n"
        "Content-Type: application/json";
    EXPECT_TRUE(ink::is_request_authorized(no_auth_req, ""));
    EXPECT_TRUE(!ink::is_request_authorized(no_auth_req, "sk-warpgrid-secret"));

    // Standard OpenAI Bearer auth (including case-insensitive header/scheme and OWS)
    const std::string bearer_ok =
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost:8888\r\n"
        "Authorization: Bearer sk-warpgrid-secret";
    EXPECT_TRUE(ink::is_request_authorized(bearer_ok, "sk-warpgrid-secret"));
    EXPECT_TRUE(!ink::is_request_authorized(bearer_ok, "sk-wrong"));

    const std::string bearer_ci =
        "GET /v1/models HTTP/1.1\r\n"
        "authorization:   bearer\tsk-warpgrid-secret  ";
    EXPECT_TRUE(ink::is_request_authorized(bearer_ci, "sk-warpgrid-secret"));

    // Reject missing space after Bearer or wrong scheme
    const std::string bearer_bad_scheme =
        "GET /v1/models HTTP/1.1\r\n"
        "Authorization: Basic sk-warpgrid-secret";
    EXPECT_TRUE(!ink::is_request_authorized(bearer_bad_scheme, "sk-warpgrid-secret"));

    const std::string bearer_no_space =
        "GET /v1/models HTTP/1.1\r\n"
        "Authorization: Bearersk-warpgrid-secret";
    EXPECT_TRUE(!ink::is_request_authorized(bearer_no_space, "sk-warpgrid-secret"));

    // X-API-Key header support
    const std::string x_api_key_ok =
        "GET /status HTTP/1.1\r\n"
        "x-api-key: sk-warpgrid-secret";
    EXPECT_TRUE(ink::is_request_authorized(x_api_key_ok, "sk-warpgrid-secret"));
    EXPECT_TRUE(!ink::is_request_authorized(x_api_key_ok, "sk-other"));

    // Must not match key appearing in an unrelated header or request line
    const std::string spoofed =
        "GET /v1/models?Authorization=Bearer%20sk-warpgrid-secret HTTP/1.1\r\n"
        "X-Custom: Authorization: Bearer sk-warpgrid-secret";
    EXPECT_TRUE(!ink::is_request_authorized(spoofed, "sk-warpgrid-secret"));

    // Case-insensitive Content-Length parsing (e.g. Python urllib sends "Content-length:")
    EXPECT_EQ(ink::parse_content_length("POST /v1/chat/completions HTTP/1.1\r\nContent-length: 123456"), (size_t)123456);
    EXPECT_EQ(ink::parse_content_length("POST /v1/chat/completions HTTP/1.1\r\nCONTENT-LENGTH: 42"), (size_t)42);
    EXPECT_EQ(ink::parse_content_length("GET /health HTTP/1.1\r\nHost: localhost"), (size_t)0);
}

static void test_radix_prefix_cache_cross_slot_and_eviction() {
    ink::RadixPrefixCache cache;
    cache.init(8, 2); // 8 physical blocks (512 tokens), 2 boundary snapshots

    // Build two prompts sharing a 128-token system prompt + different suffixes
    std::vector<int32_t> sys(128);
    for (int i = 0; i < 128; ++i) sys[i] = 1000 + i;
    std::vector<int32_t> p1 = sys;
    for (int i = 0; i < 70; ++i) p1.push_back(2000 + i); // len = 198
    std::vector<int32_t> p2 = sys;
    for (int i = 0; i < 40; ++i) p2.push_back(3000 + i); // len = 168

    EXPECT_EQ(ink::compute_lcp_aligned(p1, p2), 128);
    EXPECT_EQ(cache.peek_prefix_hit(p1), 0);

    // Slot 0 prefills p1: allocates 4 blocks (256 tokens), publishes shared 128-tok boundary
    std::vector<int32_t> s0;
    for (int i = 0; i < 4; ++i) s0.push_back(cache.alloc_block());
    EXPECT_EQ(cache.free_block_count(), 4);

    int32_t snap0 = cache.alloc_snapshot();
    EXPECT_TRUE(snap0 >= 0);
    int32_t attached = cache.publish_prefix(p1, 2, s0, snap0);
    EXPECT_EQ(attached, snap0);
    EXPECT_EQ(cache.cached_block_count(), 2);

    // Slot 1 arrives concurrently with p2 -> hits exact 128-token shared prefix & pins same physical blocks!
    auto m2 = cache.acquire_prefix(p2);
    EXPECT_EQ(m2.hit_tokens, 128);
    EXPECT_EQ(m2.snap_id, snap0);
    EXPECT_EQ(m2.phys_blocks.size(), (size_t)2);
    EXPECT_EQ(m2.phys_blocks[0], s0[0]);
    EXPECT_EQ(m2.phys_blocks[1], s0[1]);

    // While m2 holds a snapshot pin on snap0, exhausting the snapshot pool must NOT evict snap0!
    int32_t snap1 = cache.alloc_snapshot();
    EXPECT_TRUE(snap1 >= 0 && snap1 != snap0);
    // Attach snap1 to a second node? No free snaps left, and snap0 is pinned, so alloc_snapshot returns -1
    EXPECT_EQ(cache.alloc_snapshot(), -1);
    // Once m2 finishes restoring and releases its snapshot pin, snap0 becomes evictable again
    cache.release_snapshot_pin(m2.snap_id);
    int32_t recycled = cache.alloc_snapshot();
    EXPECT_EQ(recycled, snap0);
    // Re-attach snap0 for the rest of the test
    cache.publish_prefix(p1, 2, s0, recycled);

    // Slot 0 and Slot 1 both hold blocks 0..1; Slot 1 allocates 1 private suffix block
    std::vector<int32_t> s1 = m2.phys_blocks;
    s1.push_back(cache.alloc_block());
    EXPECT_EQ(cache.free_block_count(), 3);

    // Release Slot 0 -> shared blocks stay pinned by Slot 1; Slot 0's 2 private blocks return to free pool
    cache.release_slot_blocks(s0);
    EXPECT_EQ(cache.free_block_count(), 5);

    // Release Slot 1 -> shared blocks stay in cache (unreferenced, evictable), private block returns
    cache.release_slot_blocks(s1);
    EXPECT_EQ(cache.free_block_count(), 6);
    EXPECT_EQ(cache.cached_block_count(), 2);
    EXPECT_EQ(cache.peek_prefix_hit(p2), 128);

    // Allocate all 8 blocks -> forces LRU eviction of the 2 unreferenced cached blocks
    std::vector<int32_t> s2;
    for (int i = 0; i < 8; ++i) {
        int32_t b = cache.alloc_block();
        EXPECT_TRUE(b >= 0);
        s2.push_back(b);
    }
    EXPECT_EQ(cache.alloc_block(), -1); // Pool exhausted while all 8 blocks are active
    EXPECT_EQ(cache.cached_block_count(), 0);
    EXPECT_EQ(cache.peek_prefix_hit(p2), 0);

    cache.release_slot_blocks(s2);
    EXPECT_EQ(cache.free_block_count(), 8);
}

static void test_clamp_max_tokens() {
    EXPECT_EQ(ink::clamp_max_tokens(16, 32768, 32768), 32752);
    EXPECT_EQ(ink::clamp_max_tokens(13600, 32768, 32768), 19168);
    EXPECT_EQ(ink::clamp_max_tokens(32768, 16, 32768), 0);
    EXPECT_EQ(ink::clamp_max_tokens(100, 512, 32768), 512);
    EXPECT_EQ(ink::clamp_max_tokens(100, 0, 32768), 1);
    EXPECT_EQ(ink::clamp_max_tokens(100, -10, 32768), 1);
    EXPECT_EQ(ink::clamp_max_tokens(32767, 16, 32768), 1);
    EXPECT_EQ(ink::clamp_max_tokens(32769, 16, 32768), 0);
}

int main() {
    run_test("test_json_roundtrip_and_escaping", test_json_roundtrip_and_escaping);
    run_test("test_config_inkling2_parse", test_config_inkling2_parse);
    run_test("test_safetensors_portable_read_into_and_errors", test_safetensors_portable_read_into_and_errors);
    run_test("test_tokenizer_render_chat_and_tools", test_tokenizer_render_chat_and_tools);
    run_test("test_tokenizer_bpe_roundtrip_synthetic", test_tokenizer_bpe_roundtrip_synthetic);
    run_test("test_telemetry_metrics_and_formats", test_telemetry_metrics_and_formats);
    run_test("test_telemetry_otlp_payload_and_loopback", test_telemetry_otlp_payload_and_loopback);
    run_test("test_auth_request_line_and_api_key", test_auth_request_line_and_api_key);
    run_test("test_radix_prefix_cache_cross_slot_and_eviction", test_radix_prefix_cache_cross_slot_and_eviction);
    run_test("test_clamp_max_tokens", test_clamp_max_tokens);
    printf("Host tests: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
