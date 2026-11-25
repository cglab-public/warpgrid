#pragma once
#include "gguf.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ink {

inline float ggml_fp16_to_fp32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t u = 0;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FFu;
            u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &u, sizeof(out));
    return out;
}

inline float ggml_bf16_to_fp32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float out = 0.0f;
    std::memcpy(&out, &u, sizeof(out));
    return out;
}

inline void dequant_row_fp32(GgmlType type, const uint8_t* src, float* dst, size_t n) {
    static constexpr int8_t kIq4NlValues[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113
    };

    if (type == GgmlType::F32) {
        std::memcpy(dst, src, n * sizeof(float));
        return;
    }
    if (type == GgmlType::F16) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
        for (size_t i = 0; i < n; ++i) dst[i] = ggml_fp16_to_fp32(p[i]);
        return;
    }
    if (type == GgmlType::BF16) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
        for (size_t i = 0; i < n; ++i) dst[i] = ggml_bf16_to_fp32(p[i]);
        return;
    }
    if (type == GgmlType::Q8_0) {
        size_t nb = n / 32;
        for (size_t b = 0; b < nb; ++b) {
            const uint8_t* blk = src + b * 34;
            uint16_t d_bits = 0;
            std::memcpy(&d_bits, blk, 2);
            float d = ggml_fp16_to_fp32(d_bits);
            const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
            for (int i = 0; i < 32; ++i) {
                dst[b * 32 + i] = d * (float)qs[i];
            }
        }
        return;
    }
    if (type == GgmlType::IQ4_NL) {
        size_t nb = n / 32;
        for (size_t b = 0; b < nb; ++b) {
            const uint8_t* blk = src + b * 18;
            uint16_t d_bits = 0;
            std::memcpy(&d_bits, blk, 2);
            float d = ggml_fp16_to_fp32(d_bits);
            const uint8_t* qs = blk + 2;
            for (int i = 0; i < 16; ++i) {
                dst[b * 32 + i] = d * (float)kIq4NlValues[qs[i] & 0x0Fu];
                dst[b * 32 + 16 + i] = d * (float)kIq4NlValues[qs[i] >> 4];
            }
        }
        return;
    }
    if (type == GgmlType::MXFP4) {
        static constexpr float kE2M1[16] = {
            0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
           -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
        };
        size_t nb = n / 32;
        for (size_t b = 0; b < nb; ++b) {
            const uint8_t* blk = src + b * 17;
            uint8_t e = blk[0];
            float sc = 5.877471754111438e-39f;
            if (e != 0) {
                uint32_t bits = (uint32_t)e << 23;
                std::memcpy(&sc, &bits, 4);
            }
            const uint8_t* qs = blk + 1;
            for (int i = 0; i < 16; ++i) {
                dst[b * 32 + i] = sc * kE2M1[qs[i] & 0x0Fu];
                dst[b * 32 + 16 + i] = sc * kE2M1[qs[i] >> 4];
            }
        }
        return;
    }
    if (type == GgmlType::Q4_K) {
        size_t nb = n / 256;
        for (size_t b = 0; b < nb; ++b) {
            const uint8_t* blk = src + b * 144;
            uint16_t d_bits = 0, dmin_bits = 0;
            std::memcpy(&d_bits, blk, 2);
            std::memcpy(&dmin_bits, blk + 2, 2);
            float d = ggml_fp16_to_fp32(d_bits);
            float dmin = ggml_fp16_to_fp32(dmin_bits);
            const uint8_t* scales = blk + 4;
            const uint8_t* qs = blk + 16;
            float* y = dst + b * 256;
            auto get_sm = [&](int j, uint8_t& sc_out, uint8_t& m_out) {
                if (j < 4) {
                    sc_out = scales[j] & 63u;
                    m_out = scales[j + 4] & 63u;
                } else {
                    sc_out = (scales[j + 4] & 0x0Fu) | ((scales[j - 4] >> 6) << 4);
                    m_out = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
                }
            };
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc0 = 0, m0 = 0, sc1 = 0, m1 = 0;
                get_sm(is + 0, sc0, m0);
                get_sm(is + 1, sc1, m1);
                float d1 = d * (float)sc0, min1 = dmin * (float)m0;
                float d2 = d * (float)sc1, min2 = dmin * (float)m1;
                for (int l = 0; l < 32; ++l) {
                    y[j + l] = d1 * (float)(qs[l] & 0x0Fu) - min1;
                    y[j + 32 + l] = d2 * (float)(qs[l] >> 4) - min2;
                }
                qs += 32;
                is += 2;
            }
        }
        return;
    }
    if (type == GgmlType::Q2_K) {
        size_t nb = n / 256;
        for (size_t b = 0; b < nb; ++b) {
            const uint8_t* blk = src + b * 84;
            const uint8_t* scales = blk;
            const uint8_t* q = blk + 16;
            uint16_t d_bits = 0, dmin_bits = 0;
            std::memcpy(&d_bits, blk + 80, 2);
            std::memcpy(&dmin_bits, blk + 82, 2);
            float d = ggml_fp16_to_fp32(d_bits);
            float dmin = ggml_fp16_to_fp32(dmin_bits);
            float* y = dst + b * 256;
            int is = 0;
            for (int n128 = 0; n128 < 256; n128 += 128) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    uint8_t sc = scales[is++];
                    float dl = d * (float)(sc & 0x0Fu);
                    float ml = dmin * (float)(sc >> 4);
                    for (int l = 0; l < 16; ++l) *y++ = dl * (float)((q[l] >> shift) & 3u) - ml;
                    sc = scales[is++];
                    dl = d * (float)(sc & 0x0Fu);
                    ml = dmin * (float)(sc >> 4);
                    for (int l = 0; l < 16; ++l) *y++ = dl * (float)((q[l + 16] >> shift) & 3u) - ml;
                    shift += 2;
                }
                q += 32;
            }
        }
        return;
    }
    if (type == GgmlType::Q6_K) {
        size_t nb = n / 256;
        for (size_t b = 0; b < nb; ++b) {
            const uint8_t* blk = src + b * 210;
            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);
            uint16_t d_bits = 0;
            std::memcpy(&d_bits, blk + 208, 2);
            float d = ggml_fp16_to_fp32(d_bits);
            float* y = dst + b * 256;
            for (int n128 = 0; n128 < 256; n128 += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = l / 16;
                    int8_t q1 = (int8_t)((ql[l + 0] & 0xFu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
                    int8_t q2 = (int8_t)((ql[l + 32] & 0xFu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
                    int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32;
                    int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32;
                    y[l + 0] = d * (float)sc[is + 0] * (float)q1;
                    y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                    y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                    y[l + 96] = d * (float)sc[is + 6] * (float)q4;
                }
                y += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        return;
    }
    if (type == GgmlType::Q5_K) {
        size_t nb = n / 256;
        for (size_t b = 0; b < nb; ++b) {
            const uint8_t* blk = src + b * 176;
            uint16_t d_bits = 0, dmin_bits = 0;
            std::memcpy(&d_bits, blk, 2);
            std::memcpy(&dmin_bits, blk + 2, 2);
            float d = ggml_fp16_to_fp32(d_bits);
            float dmin = ggml_fp16_to_fp32(dmin_bits);
            const uint8_t* scales = blk + 4;
            const uint8_t* qh = blk + 16;
            const uint8_t* ql = blk + 48;
            float* y = dst + b * 256;

            auto get_scale_min = [&](int j, uint8_t& sc_out, uint8_t& m_out) {
                if (j < 4) {
                    sc_out = scales[j] & 63u;
                    m_out = scales[j + 4] & 63u;
                } else {
                    sc_out = (scales[j + 4] & 0xFu) | ((scales[j - 4] >> 6) << 4);
                    m_out = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
                }
            };

            uint8_t u1 = 1, u2 = 2;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc0 = 0, m0 = 0, sc1 = 0, m1 = 0;
                get_scale_min(is + 0, sc0, m0);
                get_scale_min(is + 1, sc1, m1);
                float d1 = d * (float)sc0;
                float min1 = dmin * (float)m0;
                float d2 = d * (float)sc1;
                float min2 = dmin * (float)m1;
                for (int l = 0; l < 32; ++l) {
                    uint8_t qlo = (ql[l] & 0x0Fu) + ((qh[l] & u1) ? 16u : 0u);
                    y[j + l] = d1 * (float)qlo - min1;
                }
                for (int l = 0; l < 32; ++l) {
                    uint8_t qhi = (ql[l] >> 4) + ((qh[l] & u2) ? 16u : 0u);
                    y[j + 32 + l] = d2 * (float)qhi - min2;
                }
                ql += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        return;
    }
    throw std::runtime_error("Unsupported Engram row quantization type: " + std::to_string((uint32_t)type));
}

inline uint64_t engram_hash_row(const int32_t* tok_ids, int n, uint64_t seed, uint64_t table_rows) {
    if (table_rows == 0) return 0;
    uint64_t h = 0x9E3779B97F4A7C15ULL ^ seed;
    for (int i = 0; i < n; ++i) {
        uint64_t k = (uint64_t)(uint32_t)tok_ids[i] + 0x9E3779B97F4A7C15ULL;
        k ^= (k >> 30);
        k *= 0xBF58476D1CE4E5B9ULL;
        k ^= (k >> 27);
        k *= 0x94D049BB133111EBULL;
        k ^= (k >> 31);
        h ^= k;
        h = (h << 27) | (h >> 37);
        h = h * 5ULL + 0x52DCE729ULL;
    }
    h ^= (h >> 33);
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= (h >> 33);
    h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= (h >> 33);
    return h % table_rows;
}

class EngramTable {
    struct CacheKey {
        std::string tensor_name;
        uint64_t row_idx = 0;
        bool operator==(const CacheKey& o) const {
            return row_idx == o.row_idx && tensor_name == o.tensor_name;
        }
    };

    struct CacheKeyHash {
        size_t operator()(const CacheKey& k) const {
            size_t h1 = std::hash<std::string>{}(k.tensor_name);
            size_t h2 = std::hash<uint64_t>{}(k.row_idx);
            return h1 ^ (h2 + 0x9E3779B97F4A7C15ULL + (h1 << 6) + (h1 >> 2));
        }
    };

    struct CacheEntry {
        CacheKey key;
        std::vector<float> values;
    };

    GgufFile file_;
    size_t max_cached_rows_ = 4096;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    std::list<CacheEntry> lru_;
    std::unordered_map<CacheKey, std::list<CacheEntry>::iterator, CacheKeyHash> index_;

public:
    void open(const std::string& path, size_t max_cached_rows = 4096) {
        clear_cache();
        max_cached_rows_ = max_cached_rows;
        file_.open(path, true);
    }

    bool is_open() const { return file_.num_shards() > 0; }
    bool has_tensor(const std::string& name) const { return file_.has_tensor(name); }
    const GgufTensorInfo& tensor_info(const std::string& name) const { return file_.tensor(name); }
    uint64_t cache_hits() const { return hits_; }
    uint64_t cache_misses() const { return misses_; }

    void clear_cache() {
        lru_.clear();
        index_.clear();
        hits_ = 0;
        misses_ = 0;
    }

    void lookup_row(const std::string& tensor_name, uint64_t row_idx, float* out_fp32, size_t row_elems) {
        const auto& ti = file_.tensor(tensor_name);
        if (ti.dims.empty() || ti.dims[0] != row_elems) {
            throw std::runtime_error("Engram row width mismatch for " + tensor_name);
        }
        uint64_t nrows = ti.num_rows();
        if (row_idx >= nrows) {
            throw std::runtime_error("Engram row index out of bounds for " + tensor_name);
        }

        CacheKey key{tensor_name, row_idx};
        if (max_cached_rows_ > 0) {
            auto it = index_.find(key);
            if (it != index_.end()) {
                ++hits_;
                lru_.splice(lru_.begin(), lru_, it->second);
                std::memcpy(out_fp32, it->second->values.data(), row_elems * sizeof(float));
                return;
            }
        }

        ++misses_;
        size_t row_stride = ggml_row_nbytes(ti.type, ti.dims[0]);
        const uint8_t* base = file_.tensor_data(tensor_name);
        const uint8_t* row_ptr = base + (size_t)row_idx * row_stride;
        dequant_row_fp32(ti.type, row_ptr, out_fp32, row_elems);

        if (max_cached_rows_ > 0) {
            if (lru_.size() >= max_cached_rows_) {
                index_.erase(lru_.back().key);
                lru_.pop_back();
            }
            CacheEntry entry;
            entry.key = key;
            entry.values.assign(out_fp32, out_fp32 + row_elems);
            lru_.push_front(std::move(entry));
            index_[key] = lru_.begin();
        }
    }
};

}
