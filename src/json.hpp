#pragma once
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace json {

struct Value {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<Value> arr;
    std::map<std::string, Value> obj;

    bool is_null() const { return type == Null; }
    bool is_bool() const { return type == Bool; }
    bool is_number() const { return type == Number; }
    bool is_string() const { return type == String; }
    bool is_array() const { return type == Array; }
    bool is_object() const { return type == Object; }

    int64_t as_i64(int64_t def = 0) const { return type == Number ? (int64_t)num : def; }
    int as_int(int def = 0) const { return type == Number ? (int)num : def; }
    double as_double(double def = 0.0) const { return type == Number ? num : def; }
    bool as_bool(bool def = false) const { return type == Bool ? b : def; }
    const std::string& as_str() const { static const std::string e; return type == String ? str : e; }

    bool contains(const std::string& k) const { return type == Object && obj.count(k) > 0; }
    const Value& operator[](const std::string& k) const {
        static const Value null_v;
        if (type != Object) return null_v;
        auto it = obj.find(k);
        return it == obj.end() ? null_v : it->second;
    }
    Value& operator[](const std::string& k) {
        if (type != Object) { type = Object; obj.clear(); }
        return obj[k];
    }
};

class Parser {
    const char* p_;
    const char* end_;
    void skip_ws() {
        while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\r' || *p_ == '\n')) ++p_;
    }
    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) out.push_back((char)cp);
        else if (cp < 0x800) {
            out.push_back((char)(0xC0 | (cp >> 6)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back((char)(0xE0 | (cp >> 12)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            out.push_back((char)(0xF0 | (cp >> 18)));
            out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        }
    }
    uint32_t parse_hex4() {
        if (p_ + 4 > end_) throw std::runtime_error("JSON: truncated \\u escape");
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = *p_++;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
            else throw std::runtime_error("JSON: bad hex digit");
        }
        return v;
    }
    std::string parse_string() {
        if (p_ >= end_ || *p_ != '"') throw std::runtime_error("JSON: expected '\"'");
        ++p_;
        std::string s;
        while (p_ < end_) {
            char c = *p_++;
            if (c == '"') return s;
            if (c == '\\') {
                if (p_ >= end_) throw std::runtime_error("JSON: trailing backslash");
                char e = *p_++;
                switch (e) {
                    case '"': s.push_back('"'); break;
                    case '\\': s.push_back('\\'); break;
                    case '/': s.push_back('/'); break;
                    case 'b': s.push_back('\b'); break;
                    case 'f': s.push_back('\f'); break;
                    case 'n': s.push_back('\n'); break;
                    case 'r': s.push_back('\r'); break;
                    case 't': s.push_back('\t'); break;
                    case 'u': {
                        uint32_t cp = parse_hex4();
                        if (cp >= 0xD800 && cp <= 0xDBFF && p_ + 6 <= end_ && p_[0] == '\\' && p_[1] == 'u') {
                            p_ += 2;
                            uint32_t lo = parse_hex4();
                            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + (((cp - 0xD800) << 10) | (lo - 0xDC00));
                            }
                        }
                        append_utf8(s, cp);
                        break;
                    }
                    default: s.push_back(e); break;
                }
            } else {
                s.push_back(c);
            }
        }
        throw std::runtime_error("JSON: unterminated string");
    }
    Value parse_value() {
        skip_ws();
        if (p_ >= end_) throw std::runtime_error("JSON: unexpected EOF");
        char c = *p_;
        if (c == '"') {
            Value v; v.type = Value::String; v.str = parse_string(); return v;
        }
        if (c == '{') {
            ++p_;
            Value v; v.type = Value::Object;
            skip_ws();
            if (p_ < end_ && *p_ == '}') { ++p_; return v; }
            while (true) {
                skip_ws();
                std::string k = parse_string();
                skip_ws();
                if (p_ >= end_ || *p_ != ':') throw std::runtime_error("JSON: expected ':'");
                ++p_;
                v.obj.emplace(std::move(k), parse_value());
                skip_ws();
                if (p_ < end_ && *p_ == ',') { ++p_; continue; }
                if (p_ < end_ && *p_ == '}') { ++p_; return v; }
                throw std::runtime_error("JSON: expected ',' or '}'");
            }
        }
        if (c == '[') {
            ++p_;
            Value v; v.type = Value::Array;
            skip_ws();
            if (p_ < end_ && *p_ == ']') { ++p_; return v; }
            while (true) {
                v.arr.push_back(parse_value());
                skip_ws();
                if (p_ < end_ && *p_ == ',') { ++p_; continue; }
                if (p_ < end_ && *p_ == ']') { ++p_; return v; }
                throw std::runtime_error("JSON: expected ',' or ']'");
            }
        }
        if (c == 't' && p_ + 4 <= end_ && std::string(p_, 4) == "true") {
            p_ += 4; Value v; v.type = Value::Bool; v.b = true; return v;
        }
        if (c == 'f' && p_ + 5 <= end_ && std::string(p_, 5) == "false") {
            p_ += 5; Value v; v.type = Value::Bool; v.b = false; return v;
        }
        if (c == 'n' && p_ + 4 <= end_ && std::string(p_, 4) == "null") {
            p_ += 4; Value v; v.type = Value::Null; return v;
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            char* ep = nullptr;
            double d = std::strtod(p_, &ep);
            if (ep == p_) throw std::runtime_error("JSON: bad number");
            p_ = ep;
            Value v; v.type = Value::Number; v.num = d; return v;
        }
        throw std::runtime_error(std::string("JSON: unexpected char '") + c + "'");
    }
public:
    Parser(const char* data, size_t len) : p_(data), end_(data + len) {}
    Value parse() { return parse_value(); }
};

inline Value parse(const std::string& s) {
    Parser p(s.data(), s.size());
    return p.parse();
}

inline std::string escape_str(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back((char)c);
                }
        }
    }
    out.push_back('"');
    return out;
}

inline std::string dump(const Value& v) {
    switch (v.type) {
        case Value::Null: return "null";
        case Value::Bool: return v.b ? "true" : "false";
        case Value::Number: {
            if (std::floor(v.num) == v.num && std::fabs(v.num) < 1e15) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%lld", (long long)v.num);
                return buf;
            }
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.17g", v.num);
            return buf;
        }
        case Value::String: return escape_str(v.str);
        case Value::Array: {
            std::string out = "[";
            for (size_t i = 0; i < v.arr.size(); ++i) {
                if (i) out += ",";
                out += dump(v.arr[i]);
            }
            out += "]";
            return out;
        }
        case Value::Object: {
            std::string out = "{";
            bool first = true;
            for (const auto& kv : v.obj) {
                if (!first) out += ",";
                first = false;
                out += escape_str(kv.first) + ":" + dump(kv.second);
            }
            out += "}";
            return out;
        }
    }
    return "null";
}

} // namespace json
