#pragma once
#include "tokenizer.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ink {

enum class GgmlType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39
};

inline size_t ggml_block_size(GgmlType t) {
    switch (t) {
        case GgmlType::F32:
        case GgmlType::F16:
        case GgmlType::BF16:
        case GgmlType::I8:
        case GgmlType::I16:
        case GgmlType::I32:
        case GgmlType::I64:
        case GgmlType::F64:
            return 1;
        case GgmlType::Q4_0:
        case GgmlType::Q4_1:
        case GgmlType::Q5_0:
        case GgmlType::Q5_1:
        case GgmlType::Q8_0:
        case GgmlType::Q8_1:
        case GgmlType::IQ4_NL:
        case GgmlType::MXFP4:
            return 32;
        case GgmlType::Q2_K:
        case GgmlType::Q3_K:
        case GgmlType::Q4_K:
        case GgmlType::Q5_K:
        case GgmlType::Q6_K:
        case GgmlType::Q8_K:
        case GgmlType::IQ2_XXS:
        case GgmlType::IQ2_XS:
        case GgmlType::IQ3_XXS:
        case GgmlType::IQ1_S:
        case GgmlType::IQ3_S:
        case GgmlType::IQ2_S:
        case GgmlType::IQ4_XS:
        case GgmlType::IQ1_M:
        case GgmlType::TQ1_0:
        case GgmlType::TQ2_0:
            return 256;
    }
    return 1;
}

inline size_t ggml_type_size(GgmlType t) {
    switch (t) {
        case GgmlType::F32: return 4;
        case GgmlType::F16: return 2;
        case GgmlType::BF16: return 2;
        case GgmlType::I8: return 1;
        case GgmlType::I16: return 2;
        case GgmlType::I32: return 4;
        case GgmlType::I64: return 8;
        case GgmlType::F64: return 8;
        case GgmlType::Q4_0: return 18;
        case GgmlType::Q4_1: return 20;
        case GgmlType::Q5_0: return 22;
        case GgmlType::Q5_1: return 24;
        case GgmlType::Q8_0: return 34;
        case GgmlType::Q8_1: return 36;
        case GgmlType::IQ4_NL: return 18;
        case GgmlType::MXFP4: return 17;
        case GgmlType::Q2_K: return 84;
        case GgmlType::Q3_K: return 110;
        case GgmlType::Q4_K: return 144;
        case GgmlType::Q5_K: return 176;
        case GgmlType::Q6_K: return 210;
        case GgmlType::Q8_K: return 292;
        case GgmlType::IQ2_XXS: return 66;
        case GgmlType::IQ2_XS: return 74;
        case GgmlType::IQ3_XXS: return 98;
        case GgmlType::IQ1_S: return 50;
        case GgmlType::IQ3_S: return 110;
        case GgmlType::IQ2_S: return 82;
        case GgmlType::IQ4_XS: return 136;
        case GgmlType::IQ1_M: return 56;
        case GgmlType::TQ1_0: return 54;
        case GgmlType::TQ2_0: return 66;
    }
    throw std::runtime_error("Unsupported GGML tensor type: " + std::to_string((uint32_t)t));
}

inline size_t ggml_row_nbytes(GgmlType t, uint64_t ne0) {
    size_t bs = ggml_block_size(t);
    size_t ts = ggml_type_size(t);
    if (ne0 % bs != 0) {
        throw std::runtime_error("Tensor row length not divisible by GGML block size");
    }
    uint64_t nblocks = ne0 / bs;
    if (nblocks > (uint64_t)(SIZE_MAX / ts)) {
        throw std::runtime_error("Tensor row byte size overflow");
    }
    return (size_t)nblocks * ts;
}

enum class GgufValType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12
};

struct GgufValue {
    GgufValType type = GgufValType::UINT32;
    GgufValType elem_type = GgufValType::UINT32;
    int64_t i64 = 0;
    uint64_t u64 = 0;
    double f64 = 0.0;
    bool b = false;
    std::string str;
    std::vector<std::string> str_arr;
    std::vector<int64_t> i64_arr;
    std::vector<uint64_t> u64_arr;
    std::vector<double> f64_arr;

    int64_t as_i64(int64_t def = 0) const {
        switch (type) {
            case GgufValType::INT8:
            case GgufValType::INT16:
            case GgufValType::INT32:
            case GgufValType::INT64:
                return i64;
            case GgufValType::UINT8:
            case GgufValType::UINT16:
            case GgufValType::UINT32:
            case GgufValType::UINT64:
                return (int64_t)u64;
            case GgufValType::BOOL:
                return b ? 1 : 0;
            default:
                return def;
        }
    }

    uint64_t as_u64(uint64_t def = 0) const {
        switch (type) {
            case GgufValType::UINT8:
            case GgufValType::UINT16:
            case GgufValType::UINT32:
            case GgufValType::UINT64:
                return u64;
            case GgufValType::INT8:
            case GgufValType::INT16:
            case GgufValType::INT32:
            case GgufValType::INT64:
                return i64 >= 0 ? (uint64_t)i64 : def;
            default:
                return def;
        }
    }

    double as_f64(double def = 0.0) const {
        if (type == GgufValType::FLOAT32 || type == GgufValType::FLOAT64) return f64;
        return (double)as_i64((int64_t)def);
    }

    bool as_bool(bool def = false) const {
        if (type == GgufValType::BOOL) return b;
        return as_i64(def ? 1 : 0) != 0;
    }

    const std::string& as_str() const { return str; }
};

struct GgufTensorInfo {
    std::string name;
    std::vector<uint64_t> dims;
    GgmlType type = GgmlType::F32;
    size_t shard_index = 0;
    uint64_t offset = 0;
    size_t nbytes = 0;

    uint64_t num_rows() const {
        uint64_t rows = 1;
        for (size_t i = 1; i < dims.size(); ++i) {
            if (dims[i] != 0 && rows > UINT64_MAX / dims[i]) {
                throw std::runtime_error("GGUF tensor row count overflow: " + name);
            }
            rows *= dims[i];
        }
        return rows;
    }
};

class GgufFile {
    struct Shard {
        std::string path;
        int fd = -1;
        void* mapped = nullptr;
        size_t file_size = 0;
        size_t data_offset = 0;

        Shard() = default;
        ~Shard() {
            if (mapped) ::munmap(mapped, file_size);
            if (fd >= 0) ::close(fd);
        }
        Shard(const Shard&) = delete;
        Shard& operator=(const Shard&) = delete;
        Shard(Shard&& o) noexcept
            : path(std::move(o.path)), fd(o.fd), mapped(o.mapped),
              file_size(o.file_size), data_offset(o.data_offset) {
            o.fd = -1;
            o.mapped = nullptr;
            o.file_size = 0;
            o.data_offset = 0;
        }
        Shard& operator=(Shard&& o) noexcept {
            if (this != &o) {
                if (mapped) ::munmap(mapped, file_size);
                if (fd >= 0) ::close(fd);
                path = std::move(o.path);
                fd = o.fd;
                mapped = o.mapped;
                file_size = o.file_size;
                data_offset = o.data_offset;
                o.fd = -1;
                o.mapped = nullptr;
                o.file_size = 0;
                o.data_offset = 0;
            }
            return *this;
        }
    };

    std::vector<Shard> shards_;
    std::unordered_map<std::string, GgufValue> kv_;
    std::unordered_map<std::string, GgufTensorInfo> tensors_;

    struct Cursor {
        const uint8_t* ptr = nullptr;
        size_t pos = 0;
        size_t len = 0;

        void ensure(size_t n) const {
            if (pos > len || n > len - pos) {
                throw std::runtime_error("Unexpected EOF while parsing GGUF header");
            }
        }

        template <typename T>
        T read_pod() {
            ensure(sizeof(T));
            T v{};
            std::memcpy(&v, ptr + pos, sizeof(T));
            pos += sizeof(T);
            return v;
        }

        std::string read_string() {
            uint64_t n = read_pod<uint64_t>();
            ensure((size_t)n);
            std::string s(reinterpret_cast<const char*>(ptr + pos), (size_t)n);
            pos += (size_t)n;
            return s;
        }
    };

    static GgufValue read_scalar(Cursor& c, GgufValType vt) {
        GgufValue v;
        v.type = vt;
        switch (vt) {
            case GgufValType::UINT8: v.u64 = c.read_pod<uint8_t>(); break;
            case GgufValType::INT8: v.i64 = c.read_pod<int8_t>(); break;
            case GgufValType::UINT16: v.u64 = c.read_pod<uint16_t>(); break;
            case GgufValType::INT16: v.i64 = c.read_pod<int16_t>(); break;
            case GgufValType::UINT32: v.u64 = c.read_pod<uint32_t>(); break;
            case GgufValType::INT32: v.i64 = c.read_pod<int32_t>(); break;
            case GgufValType::FLOAT32: v.f64 = (double)c.read_pod<float>(); break;
            case GgufValType::BOOL: v.b = c.read_pod<uint8_t>() != 0; break;
            case GgufValType::STRING: v.str = c.read_string(); break;
            case GgufValType::UINT64: v.u64 = c.read_pod<uint64_t>(); break;
            case GgufValType::INT64: v.i64 = c.read_pod<int64_t>(); break;
            case GgufValType::FLOAT64: v.f64 = c.read_pod<double>(); break;
            case GgufValType::ARRAY: {
                auto et = (GgufValType)c.read_pod<uint32_t>();
                uint64_t count = c.read_pod<uint64_t>();
                v.elem_type = et;
                if (et == GgufValType::STRING) {
                    v.str_arr.reserve((size_t)count);
                    for (uint64_t i = 0; i < count; ++i) v.str_arr.push_back(c.read_string());
                } else if (et == GgufValType::FLOAT32 || et == GgufValType::FLOAT64) {
                    v.f64_arr.reserve((size_t)count);
                    for (uint64_t i = 0; i < count; ++i) {
                        v.f64_arr.push_back(read_scalar(c, et).f64);
                    }
                } else if (et == GgufValType::INT8 || et == GgufValType::INT16 ||
                           et == GgufValType::INT32 || et == GgufValType::INT64) {
                    v.i64_arr.reserve((size_t)count);
                    for (uint64_t i = 0; i < count; ++i) {
                        v.i64_arr.push_back(read_scalar(c, et).i64);
                    }
                } else {
                    v.u64_arr.reserve((size_t)count);
                    for (uint64_t i = 0; i < count; ++i) {
                        v.u64_arr.push_back(read_scalar(c, et).as_u64());
                    }
                }
                break;
            }
            default:
                throw std::runtime_error("Unknown GGUF value type: " + std::to_string((uint32_t)vt));
        }
        return v;
    }

    static std::vector<std::string> discover_shards(const std::string& path) {
        namespace fs = std::filesystem;
        std::error_code ec;
        if (fs::is_directory(path, ec)) {
            std::vector<std::string> files;
            for (const auto& entry : fs::directory_iterator(path, ec)) {
                if (!entry.is_regular_file(ec)) continue;
                std::string p = entry.path().string();
                if (p.size() >= 5 && p.substr(p.size() - 5) == ".gguf") {
                    files.push_back(p);
                }
            }
            std::sort(files.begin(), files.end());
            if (files.empty()) throw std::runtime_error("No .gguf files found in directory: " + path);
            return files;
        }

        static const std::regex shard_re(R"(^(.*)-(\d{5})-of-(\d{5})\.gguf$)");
        std::smatch m;
        if (std::regex_match(path, m, shard_re)) {
            std::string prefix = m[1].str();
            int total = std::stoi(m[3].str());
            if (total > 1) {
                std::vector<std::string> files;
                files.reserve((size_t)total);
                for (int i = 1; i <= total; ++i) {
                    char buf[32];
                    std::snprintf(buf, sizeof(buf), "-%05d-of-%05d.gguf", i, total);
                    std::string candidate = prefix + buf;
                    if (!fs::exists(candidate, ec)) {
                        throw std::runtime_error("Missing GGUF shard: " + candidate);
                    }
                    files.push_back(candidate);
                }
                return files;
            }
        }
        return {path};
    }

    void parse_shard(size_t shard_idx, const std::string& path, bool random_access) {
        Shard sh;
        sh.path = path;
        sh.fd = ::open(path.c_str(), O_RDONLY);
        if (sh.fd < 0) throw std::runtime_error("Failed to open GGUF file: " + path);

        struct stat st{};
        if (::fstat(sh.fd, &st) != 0 || st.st_size <= 0) {
            throw std::runtime_error("Failed to stat GGUF file: " + path);
        }
        sh.file_size = (size_t)st.st_size;
        sh.mapped = ::mmap(nullptr, sh.file_size, PROT_READ, MAP_PRIVATE, sh.fd, 0);
        if (sh.mapped == MAP_FAILED) {
            sh.mapped = nullptr;
            throw std::runtime_error("mmap failed for GGUF file: " + path);
        }
        if (random_access) {
            ::madvise(sh.mapped, sh.file_size, MADV_RANDOM);
        }

        Cursor c{static_cast<const uint8_t*>(sh.mapped), 0, sh.file_size};
        c.ensure(4);
        if (std::memcmp(c.ptr, "GGUF", 4) != 0) {
            throw std::runtime_error("Invalid GGUF magic in: " + path);
        }
        c.pos += 4;
        uint32_t version = c.read_pod<uint32_t>();
        if (version < 2 || version > 3) {
            throw std::runtime_error("Unsupported GGUF version: " + std::to_string(version));
        }

        uint64_t tensor_count = c.read_pod<uint64_t>();
        uint64_t kv_count = c.read_pod<uint64_t>();
        size_t alignment = 32;

        for (uint64_t i = 0; i < kv_count; ++i) {
            std::string key = c.read_string();
            auto vt = (GgufValType)c.read_pod<uint32_t>();
            GgufValue val = read_scalar(c, vt);
            if (key == "general.alignment") {
                size_t a = (size_t)val.as_u64(32);
                if (a > 0 && (a & (a - 1)) == 0) alignment = a;
            }
            if (shard_idx == 0 || kv_.find(key) == kv_.end()) {
                kv_[key] = std::move(val);
            }
        }

        std::vector<GgufTensorInfo> shard_tensors;
        shard_tensors.reserve((size_t)tensor_count);
        for (uint64_t i = 0; i < tensor_count; ++i) {
            GgufTensorInfo ti;
            ti.name = c.read_string();
            uint32_t n_dims = c.read_pod<uint32_t>();
            ti.dims.resize(n_dims);
            for (uint32_t d = 0; d < n_dims; ++d) {
                ti.dims[d] = c.read_pod<uint64_t>();
            }
            ti.type = (GgmlType)c.read_pod<uint32_t>();
            ti.offset = c.read_pod<uint64_t>();
            ti.shard_index = shard_idx;
            uint64_t ne0 = ti.dims.empty() ? 0 : ti.dims[0];
            if (ne0 == 0) {
                ti.nbytes = 0;
            } else {
                size_t row_bytes = ggml_row_nbytes(ti.type, ne0);
                uint64_t nrows = ti.num_rows();
                if (nrows > (uint64_t)SIZE_MAX || (row_bytes != 0 && (size_t)nrows > SIZE_MAX / row_bytes)) {
                    throw std::runtime_error("GGUF tensor byte size overflow: " + ti.name);
                }
                ti.nbytes = row_bytes * (size_t)nrows;
            }
            shard_tensors.push_back(std::move(ti));
        }

        sh.data_offset = (c.pos + alignment - 1) & ~(alignment - 1);
        if (sh.data_offset > sh.file_size) {
            throw std::runtime_error("GGUF data offset out of bounds in: " + path);
        }
        size_t data_avail = sh.file_size - sh.data_offset;
        for (auto& ti : shard_tensors) {
            if (ti.offset > data_avail || ti.nbytes > data_avail - (size_t)ti.offset) {
                throw std::runtime_error("Tensor data out of bounds in GGUF: " + ti.name);
            }
            tensors_[ti.name] = std::move(ti);
        }
        shards_.push_back(std::move(sh));
    }

public:
    GgufFile() = default;
    ~GgufFile() { close(); }
    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    void close() {
        shards_.clear();
        kv_.clear();
        tensors_.clear();
    }

    void open(const std::string& path, bool random_access = false) {
        close();
        auto files = discover_shards(path);
        for (size_t i = 0; i < files.size(); ++i) {
            parse_shard(i, files[i], random_access);
        }
    }

    size_t num_shards() const { return shards_.size(); }

    bool has_kv(const std::string& key) const { return kv_.count(key) > 0; }

    const GgufValue& kv(const std::string& key) const {
        auto it = kv_.find(key);
        if (it == kv_.end()) throw std::runtime_error("Missing GGUF metadata key: " + key);
        return it->second;
    }

    std::string get_str(const std::string& key, const std::string& def = "") const {
        auto it = kv_.find(key);
        return (it != kv_.end() && it->second.type == GgufValType::STRING) ? it->second.str : def;
    }

    int64_t get_i64(const std::string& key, int64_t def = 0) const {
        auto it = kv_.find(key);
        return it != kv_.end() ? it->second.as_i64(def) : def;
    }

    double get_f64(const std::string& key, double def = 0.0) const {
        auto it = kv_.find(key);
        return it != kv_.end() ? it->second.as_f64(def) : def;
    }

    bool has_tensor(const std::string& name) const { return tensors_.count(name) > 0; }

    const GgufTensorInfo& tensor(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) throw std::runtime_error("Missing GGUF tensor: " + name);
        return it->second;
    }

    const uint8_t* tensor_data(const std::string& name) const {
        const auto& ti = tensor(name);
        const auto& sh = shards_[ti.shard_index];
        return static_cast<const uint8_t*>(sh.mapped) + sh.data_offset + ti.offset;
    }

    void read_tensor_into(const std::string& name, void* dst, size_t dst_bytes) const {
        const auto& ti = tensor(name);
        if (dst_bytes < ti.nbytes) {
            throw std::runtime_error("Destination buffer too small for GGUF tensor: " + name);
        }
        std::memcpy(dst, tensor_data(name), ti.nbytes);
    }

    const std::unordered_map<std::string, GgufTensorInfo>& all_tensors() const { return tensors_; }

    void load_tokenizer(Tokenizer& tok) const {
        if (!has_kv("tokenizer.ggml.tokens")) {
            throw std::runtime_error("GGUF file is missing tokenizer.ggml.tokens");
        }
        const auto& toks = kv("tokenizer.ggml.tokens").str_arr;
        std::vector<std::string> merges;
        if (has_kv("tokenizer.ggml.merges")) {
            merges = kv("tokenizer.ggml.merges").str_arr;
        }
        std::vector<int32_t> ttypes;
        if (has_kv("tokenizer.ggml.token_type")) {
            const auto& tv = kv("tokenizer.ggml.token_type");
            if (!tv.i64_arr.empty()) {
                ttypes.reserve(tv.i64_arr.size());
                for (int64_t x : tv.i64_arr) ttypes.push_back((int32_t)x);
            } else if (!tv.u64_arr.empty()) {
                ttypes.reserve(tv.u64_arr.size());
                for (uint64_t x : tv.u64_arr) ttypes.push_back((int32_t)x);
            }
        }
        int32_t bos = (int32_t)get_i64("tokenizer.ggml.bos_token_id", -1);
        int32_t eos = (int32_t)get_i64("tokenizer.ggml.eos_token_id", -1);
        std::string arch = get_str("general.architecture", "");
        ChatFormat fmt = (arch.find("inkling") != std::string::npos) ? ChatFormat::Inkling : ChatFormat::DeepSeek;
        tok.load_from_vectors(toks, merges, ttypes, bos, eos, fmt);
    }
};

}
