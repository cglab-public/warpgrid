#pragma once
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace ink {

enum class ModelFormat {
    InklingSafetensors,
    GgufDeepSeekV41,
    Unknown
};

struct CliOptions {
    std::string model_dir;
    std::string engram_path;
    std::string served_name = "Inkling-2-Mini";
    bool served_name_explicit = false;
    int port = 8888;
    int max_seq = 32768;
    int max_chunk = 512;
    int max_slots = 8;
    bool bench_mode = false;
    bool profile_mode = false;
    bool no_gemv = false;
    bool no_graph = false;
    int bench_tokens = 128;
    int bench_prefill = 0;
    int draft_k = 0;
    std::string draft_vocab_file;
    std::string dflash2_dir;
    std::string dump_imatrix_path;
    std::string calib_file;
    std::string otlp_endpoint;
    std::string api_key;
    int moe_group_thresh = 64;
    std::string bench_prompt = "What is 17 * 19? Reply with just the number.";
};

inline bool parse_cli_args(int argc, char** argv, CliOptions& opts, std::string& err) {
    err.clear();
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) opts.model_dir = argv[++i];
        else if ((a == "--engram" || a == "--ple") && i + 1 < argc) opts.engram_path = argv[++i];
        else if (a == "--served-model-name" && i + 1 < argc) {
            opts.served_name = argv[++i];
            opts.served_name_explicit = true;
        }
        else if (a == "--port" && i + 1 < argc) opts.port = std::stoi(argv[++i]);
        else if (a == "--max-seq-len" && i + 1 < argc) opts.max_seq = std::stoi(argv[++i]);
        else if (a == "--max-chunk" && i + 1 < argc) opts.max_chunk = std::stoi(argv[++i]);
        else if (a == "--max-slots" && i + 1 < argc) opts.max_slots = std::stoi(argv[++i]);
        else if (a == "--bench") opts.bench_mode = true;
        else if (a == "--profile") opts.profile_mode = true;
        else if (a == "--bench-tokens" && i + 1 < argc) opts.bench_tokens = std::stoi(argv[++i]);
        else if (a == "--bench-prefill" && i + 1 < argc) opts.bench_prefill = std::stoi(argv[++i]);
        else if (a == "--draft-k" && i + 1 < argc) opts.draft_k = std::stoi(argv[++i]);
        else if (a == "--draft-vocab-file" && i + 1 < argc) opts.draft_vocab_file = argv[++i];
        else if (a == "--dflash2" && i + 1 < argc) opts.dflash2_dir = argv[++i];
        else if (a == "--dump-imatrix" && i + 1 < argc) opts.dump_imatrix_path = argv[++i];
        else if (a == "--calib-file" && i + 1 < argc) opts.calib_file = argv[++i];
        else if (a == "--otlp-endpoint" && i + 1 < argc) opts.otlp_endpoint = argv[++i];
        else if (a == "--api-key") {
            if (i + 1 >= argc || std::strlen(argv[i + 1]) == 0) {
                err = "Error: --api-key requires a non-empty value";
                return false;
            }
            opts.api_key = argv[++i];
        }
        else if (a == "--moe-group-thresh" && i + 1 < argc) opts.moe_group_thresh = std::stoi(argv[++i]);
        else if (a == "--prompt" && i + 1 < argc) opts.bench_prompt = argv[++i];
        else if (a == "--no-gemv") opts.no_gemv = true;
        else if (a == "--no-graph") opts.no_graph = true;
    }

    if (opts.api_key.empty()) {
        if (const char* env_key = std::getenv("WARPGRID_API_KEY"); env_key && env_key[0] != '\0') {
            opts.api_key = env_key;
        }
    }
    return true;
}

inline bool has_gguf_extension(const std::string& p) {
    if (p.size() < 5) return false;
    std::string ext = p.substr(p.size() - 5);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return ext == ".gguf";
}

inline bool has_gguf_magic(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    char magic[4] = {};
    f.read(magic, 4);
    return f.gcount() == 4 && std::memcmp(magic, "GGUF", 4) == 0;
}

inline ModelFormat detect_model_format(const std::string& model_path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (model_path.empty() || !fs::exists(model_path, ec)) return ModelFormat::Unknown;

    if (fs::is_regular_file(model_path, ec)) {
        if (has_gguf_magic(model_path) || has_gguf_extension(model_path)) {
            return ModelFormat::GgufDeepSeekV41;
        }
        return ModelFormat::Unknown;
    }

    if (fs::is_directory(model_path, ec)) {
        if (fs::exists(fs::path(model_path) / "config.json", ec)) {
            return ModelFormat::InklingSafetensors;
        }
        for (const auto& entry : fs::directory_iterator(model_path, ec)) {
            if (entry.is_regular_file(ec) && has_gguf_extension(entry.path().string())) {
                return ModelFormat::GgufDeepSeekV41;
            }
        }
    }
    return ModelFormat::Unknown;
}

}
