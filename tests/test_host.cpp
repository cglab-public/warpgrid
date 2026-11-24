#include "../src/config.hpp"
#include "../src/json.hpp"
#include "../src/safetensors.hpp"
#include "../src/tokenizer.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
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

int main() {
    run_test("test_json_roundtrip_and_escaping", test_json_roundtrip_and_escaping);
    run_test("test_config_inkling2_parse", test_config_inkling2_parse);
    run_test("test_safetensors_portable_read_into_and_errors", test_safetensors_portable_read_into_and_errors);
    run_test("test_tokenizer_render_chat_and_tools", test_tokenizer_render_chat_and_tools);
    run_test("test_tokenizer_bpe_roundtrip_synthetic", test_tokenizer_bpe_roundtrip_synthetic);
    printf("Host tests: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
