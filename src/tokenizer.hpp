#pragma once
#include "json.hpp"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace ink {

enum class ChatFormat {
    Inkling,
    DeepSeek
};

// Native C++ Byte-Level BPE Tokenizer + Inkling Chat Template + Reasoning/Tool Parser
class Tokenizer {
    std::unordered_map<std::string, int32_t> vocab_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> merges_;
    std::vector<std::pair<std::string, int32_t>> special_tokens_;
    std::unordered_map<int32_t, std::string> special_id_to_str_;

    // GPT-2 ByteLevel mapping: byte [0..255] <-> UTF-8 string
    std::string byte_to_u8_[256];
    std::unordered_map<std::string, uint8_t> u8_to_byte_;

    void init_byte_table() {
        std::vector<int> bs;
        for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
        for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
        for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }
        for (size_t i = 0; i < bs.size(); ++i) {
            int cp = cs[i];
            std::string u8;
            if (cp < 0x80) u8.push_back((char)cp);
            else {
                u8.push_back((char)(0xC0 | (cp >> 6)));
                u8.push_back((char)(0x80 | (cp & 0x3F)));
            }
            byte_to_u8_[bs[i]] = u8;
            u8_to_byte_[u8] = (uint8_t)bs[i];
        }
    }

    std::string bytes_to_bpe_str(const std::string& raw) const {
        std::string out;
        out.reserve(raw.size() * 2);
        for (unsigned char c : raw) out += byte_to_u8_[c];
        return out;
    }

    std::string bpe_str_to_bytes(const std::string& s) const {
        std::string out;
        out.reserve(s.size());
        size_t i = 0;
        while (i < s.size()) {
            unsigned char c = (unsigned char)s[i];
            size_t len = 1;
            if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            if (i + len > s.size()) len = 1;
            std::string ch = s.substr(i, len);
            auto it = u8_to_byte_.find(ch);
            if (it != u8_to_byte_.end()) out.push_back((char)it->second);
            else out += ch;
            i += len;
        }
        return out;
    }

    void bpe_encode_piece(const std::string& piece, std::vector<int32_t>& out) const {
        if (piece.empty()) return;
        std::string bpe = bytes_to_bpe_str(piece);
        // Split into UTF-8 symbols
        std::vector<std::string> syms;
        for (size_t i = 0; i < bpe.size(); ) {
            unsigned char c = (unsigned char)bpe[i];
            size_t len = 1;
            if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            if (i + len > bpe.size()) len = 1;
            syms.push_back(bpe.substr(i, len));
            i += len;
        }
        if (syms.size() == 1) {
            auto it = vocab_.find(syms[0]);
            if (it != vocab_.end()) out.push_back(it->second);
            return;
        }
        while (syms.size() > 1) {
            int32_t best_rank = INT32_MAX;
            size_t best_idx = SIZE_MAX;
            for (size_t i = 0; i + 1 < syms.size(); ++i) {
                std::string pair = syms[i] + " " + syms[i + 1];
                auto it = merges_.find(pair);
                if (it != merges_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = i;
                }
            }
            if (best_idx == SIZE_MAX) break;
            std::string merged = syms[best_idx] + syms[best_idx + 1];
            syms[best_idx] = std::move(merged);
            syms.erase(syms.begin() + (ptrdiff_t)(best_idx + 1));
        }
        for (const auto& s : syms) {
            auto it = vocab_.find(s);
            if (it != vocab_.end()) out.push_back(it->second);
        }
    }

    // Simple GPT-4o / cl100k-style pre-tokenizer splitting ASCII words, numbers, punctuation, whitespace
    static void pretokenize_plain(const std::string& text, std::vector<std::string>& pieces) {
        size_t i = 0, n = text.size();
        auto is_letter = [](unsigned char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c >= 0x80; };
        auto is_digit = [](unsigned char c) { return c >= '0' && c <= '9'; };
        auto is_ws = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };

        while (i < n) {
            unsigned char c = (unsigned char)text[i];
            if (c == '\r' || c == '\n') {
                size_t j = i;
                while (j < n && (text[j] == '\r' || text[j] == '\n')) ++j;
                pieces.push_back(text.substr(i, j - i));
                i = j;
            } else if (c == ' ' && i + 1 < n && is_letter((unsigned char)text[i + 1])) {
                size_t j = i + 1;
                while (j < n && is_letter((unsigned char)text[j])) ++j;
                pieces.push_back(text.substr(i, j - i));
                i = j;
            } else if (is_letter(c)) {
                size_t j = i + 1;
                while (j < n && is_letter((unsigned char)text[j])) ++j;
                pieces.push_back(text.substr(i, j - i));
                i = j;
            } else if (is_digit(c)) {
                size_t j = i + 1;
                while (j < n && is_digit((unsigned char)text[j]) && (j - i) < 3) ++j;
                pieces.push_back(text.substr(i, j - i));
                i = j;
            } else if (is_ws(c)) {
                size_t j = i + 1;
                while (j < n && text[j] == c && !(j + 1 < n && !is_ws((unsigned char)text[j + 1]))) ++j;
                pieces.push_back(text.substr(i, j - i));
                i = j;
            } else {
                size_t j = i + 1;
                while (j < n && !is_letter((unsigned char)text[j]) && !is_digit((unsigned char)text[j]) && !is_ws((unsigned char)text[j])) ++j;
                pieces.push_back(text.substr(i, j - i));
                i = j;
            }
        }
    }

public:
    ChatFormat chat_format = ChatFormat::Inkling;
    int32_t bos_id = -1;
    int32_t eos_id = 200006;         // <|content_model_end_sampling|>
    int32_t end_message_id = 200008; // <|end_message|>
    int32_t thinking_id = 200007;    // <|content_thinking|>
    int32_t content_text_id = 200004;// <|content_text|>
    int32_t invoke_tool_id = 200012; // <|content_invoke_tool_json|>
    int32_t message_model_id = 200001;// <|message_model|>
    int32_t think_start_id = -1;
    int32_t think_end_id = -1;

    void load_from_vectors(
        const std::vector<std::string>& tokens,
        const std::vector<std::string>& merges,
        const std::vector<int32_t>& token_types,
        int32_t bos,
        int32_t eos,
        ChatFormat fmt = ChatFormat::DeepSeek) {
        init_byte_table();
        vocab_.clear();
        merges_.clear();
        special_tokens_.clear();
        special_id_to_str_.clear();
        id_to_token_.assign(tokens.size(), "");

        chat_format = fmt;
        bos_id = bos;
        eos_id = eos;
        end_message_id = -1;
        thinking_id = -1;
        content_text_id = -1;
        invoke_tool_id = -1;
        message_model_id = -1;
        think_start_id = -1;
        think_end_id = -1;

        vocab_.reserve(tokens.size() * 2);
        for (size_t i = 0; i < tokens.size(); ++i) {
            int32_t id = (int32_t)i;
            const std::string& t = tokens[i];
            id_to_token_[i] = t;
            int32_t ttype = (i < token_types.size()) ? token_types[i] : 1;
            bool is_special = (ttype == 3 || ttype == 4);
            if (is_special) {
                special_tokens_.push_back({t, id});
                special_id_to_str_[id] = t;
            } else {
                vocab_[t] = id;
            }
            if (t == "<think>") think_start_id = id;
            else if (t == "</think>") think_end_id = id;
            else if (t == "<｜begin▁of▁sentence｜>" && bos_id < 0) bos_id = id;
            else if (t == "<｜end▁of▁sentence｜>" && eos_id < 0) eos_id = id;
        }
        std::sort(special_tokens_.begin(), special_tokens_.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

        merges_.reserve(merges.size() * 2);
        for (size_t i = 0; i < merges.size(); ++i) {
            merges_[merges[i]] = (int32_t)i;
        }
        fprintf(stderr, "[warpgrid] Loaded GGUF tokenizer: %zu vocab, %zu merges, %zu special tokens\n",
                vocab_.size(), merges_.size(), special_tokens_.size());
    }

    void load(const std::string& path) {
        init_byte_table();
        std::ifstream f(path);
        if (!f) throw std::runtime_error("Cannot open tokenizer.json: " + path);
        std::stringstream ss;
        ss << f.rdbuf();
        auto root = json::parse(ss.str());

        const auto& vocab = root["model"]["vocab"];
        int32_t max_id = 0;
        for (const auto& kv : vocab.obj) {
            int32_t id = kv.second.as_int();
            vocab_[kv.first] = id;
            if (id > max_id) max_id = id;
        }
        const auto& added = root["added_tokens"];
        for (const auto& tok : added.arr) {
            int32_t id = tok["id"].as_int();
            std::string content = tok["content"].as_str();
            special_tokens_.push_back({content, id});
            special_id_to_str_[id] = content;
            if (id > max_id) max_id = id;
            if (content == "<|content_model_end_sampling|>") eos_id = id;
            else if (content == "<|end_message|>") end_message_id = id;
            else if (content == "<|content_thinking|>") thinking_id = id;
            else if (content == "<|content_text|>") content_text_id = id;
            else if (content == "<|content_invoke_tool_json|>") invoke_tool_id = id;
            else if (content == "<|message_model|>") message_model_id = id;
        }
        // Sort special tokens longest-first for greedy matching
        std::sort(special_tokens_.begin(), special_tokens_.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

        id_to_token_.resize((size_t)max_id + 1);
        for (const auto& kv : vocab_) id_to_token_[kv.second] = kv.first;
        for (const auto& kv : special_id_to_str_) id_to_token_[kv.first] = kv.second;

        const auto& merges = root["model"]["merges"];
        merges_.reserve(merges.arr.size() * 2);
        for (size_t i = 0; i < merges.arr.size(); ++i) {
            if (merges.arr[i].is_string()) {
                merges_[merges.arr[i].as_str()] = (int32_t)i;
            } else if (merges.arr[i].is_array() && merges.arr[i].arr.size() == 2) {
                merges_[merges.arr[i].arr[0].as_str() + " " + merges.arr[i].arr[1].as_str()] = (int32_t)i;
            }
        }
        fprintf(stderr, "[warpgrid] Loaded tokenizer: %zu vocab, %zu merges, %zu special tokens\n",
                vocab_.size(), merges_.size(), special_tokens_.size());
    }

    std::vector<int32_t> encode(const std::string& text) const {
        std::vector<int32_t> ids;
        size_t pos = 0;
        while (pos < text.size()) {
            // Find earliest special token match
            size_t next_pos = std::string::npos;
            int32_t next_id = -1;
            size_t next_len = 0;
            for (const auto& sp : special_tokens_) {
                size_t p = text.find(sp.first, pos);
                if (p != std::string::npos && (p < next_pos || (p == next_pos && sp.first.size() > next_len))) {
                    next_pos = p;
                    next_id = sp.second;
                    next_len = sp.first.size();
                }
            }
            std::string chunk = (next_pos == std::string::npos) ? text.substr(pos) : text.substr(pos, next_pos - pos);
            if (!chunk.empty()) {
                std::vector<std::string> pieces;
                pretokenize_plain(chunk, pieces);
                for (const auto& pc : pieces) bpe_encode_piece(pc, ids);
            }
            if (next_pos == std::string::npos) break;
            ids.push_back(next_id);
            pos = next_pos + next_len;
        }
        return ids;
    }

    std::string decode_token(int32_t id, bool skip_special = false) const {
        if (id < 0 || (size_t)id >= id_to_token_.size()) return "";
        auto sit = special_id_to_str_.find(id);
        if (sit != special_id_to_str_.end()) return skip_special ? "" : sit->second;
        return bpe_str_to_bytes(id_to_token_[id]);
    }

    std::string decode(const std::vector<int32_t>& ids, bool skip_special = false) const {
        std::string out;
        for (int32_t id : ids) out += decode_token(id, skip_special);
        return out;
    }

    // Exact Inkling-2-Mini chat template renderer
    static std::string render_chat(const json::Value& req) {
        std::string out;
        // 1. Tools declaration
        if (req.contains("tools") && req["tools"].is_array() && !req["tools"].arr.empty()) {
            json::Value specs;
            specs.type = json::Value::Array;
            for (const auto& t : req["tools"].arr) {
                const auto& fn = t.contains("function") ? t["function"] : t;
                json::Value spec;
                spec["description"] = fn.contains("description") && fn["description"].is_string()
                    ? fn["description"] : json::Value{json::Value::String, false, 0, ""};
                spec["name"] = fn["name"];
                spec["parameters"] = fn.contains("parameters") ? fn["parameters"] : json::Value{json::Value::Object};
                spec["type"] = t.contains("type") && t["type"].is_string()
                    ? t["type"] : json::Value{json::Value::String, false, 0, "function"};
                specs.arr.push_back(std::move(spec));
            }
            out += "<|message_system|>tool_declare<|content_xml|>" + json::dump(specs) + "<|end_message|>";
        }

        // Thinking effort mapping
        std::string effort_str = "0.7";
        if (req.contains("reasoning_effort")) {
            const auto& re = req["reasoning_effort"];
            if (re.is_string()) {
                const std::string& k = re.as_str();
                if (k == "none") effort_str = "0";
                else if (k == "minimal") effort_str = "0.1";
                else if (k == "low") effort_str = "0.2";
                else if (k == "medium") effort_str = "0.7";
                else if (k == "high") effort_str = "0.9";
                else if (k == "xhigh" || k == "max") effort_str = "0.99";
            } else if (re.is_number()) {
                double v = re.as_double(0.7);
                if (v == 0.0) effort_str = "0";
                else {
                    char buf[32];
                    std::snprintf(buf, sizeof(buf), "%.2g", v);
                    effort_str = buf;
                }
            }
        }
        if (req.contains("chat_template_kwargs") && req["chat_template_kwargs"].is_object()) {
            const auto& ctk = req["chat_template_kwargs"];
            if (ctk.contains("enable_thinking") && !ctk["enable_thinking"].as_bool(true)) {
                effort_str = "0";
            }
        }

        auto emit_effort = [&]() {
            out += "<|message_system|><|content_text|>Thinking effort level: " + effort_str + "<|end_message|>";
        };

        bool effort_emitted = false;
        const auto& msgs = req["messages"];
        for (const auto& m : msgs.arr) {
            std::string role = m["role"].as_str();
            if (!effort_emitted && role != "system") {
                emit_effort();
                effort_emitted = true;
            }
            std::string rtok;
            if (role == "user") rtok = "<|message_user|>";
            else if (role == "assistant") rtok = "<|message_model|>";
            else if (role == "system") rtok = "<|message_system|>";
            else if (role == "tool") rtok = "<|message_tool|>";
            if (role == "tool" && m.contains("name") && m["name"].is_string()) {
                rtok += m["name"].as_str();
            }
            if (role == "assistant" && m.contains("reasoning_content") && m["reasoning_content"].is_string() && !m["reasoning_content"].as_str().empty()) {
                out += rtok + "<|content_thinking|>" + m["reasoning_content"].as_str() + "<|end_message|>";
            }
            if (m.contains("content") && m["content"].is_string()) {
                out += rtok + "<|content_text|>" + m["content"].as_str() + "<|end_message|>";
            } else if (m.contains("content") && m["content"].is_array()) {
                for (const auto& part : m["content"].arr) {
                    if (part.is_string()) out += rtok + "<|content_text|>" + part.as_str() + "<|end_message|>";
                    else if (part.contains("text") && part["text"].is_string()) {
                        out += rtok + "<|content_text|>" + part["text"].as_str() + "<|end_message|>";
                    }
                }
            }
            if (role == "assistant") out += "<|content_model_end_sampling|>";
        }
        if (!effort_emitted) emit_effort();
        return out;
    }

    static std::string render_deepseek_chat(const json::Value& req) {
        std::string out = "<｜begin▁of▁sentence｜>";
        bool enable_thinking = true;
        if (req.contains("chat_template_kwargs") && req["chat_template_kwargs"].is_object()) {
            const auto& ctk = req["chat_template_kwargs"];
            if (ctk.contains("enable_thinking")) {
                enable_thinking = ctk["enable_thinking"].as_bool(true);
            }
        }
        if (req.contains("reasoning_effort") && req["reasoning_effort"].is_string() &&
            req["reasoning_effort"].as_str() == "none") {
            enable_thinking = false;
        }

        const auto& msgs = req["messages"];
        for (const auto& m : msgs.arr) {
            std::string role = m["role"].as_str();
            std::string content;
            if (m.contains("content") && m["content"].is_string()) {
                content = m["content"].as_str();
            } else if (m.contains("content") && m["content"].is_array()) {
                for (const auto& part : m["content"].arr) {
                    if (part.is_string()) content += part.as_str();
                    else if (part.contains("text") && part["text"].is_string()) {
                        content += part["text"].as_str();
                    }
                }
            }
            if (role == "system") {
                out += content;
            } else if (role == "user") {
                out += "<｜User｜>" + content;
            } else if (role == "assistant") {
                out += "<｜Assistant｜>" + content + "<｜end▁of▁sentence｜>";
            } else if (role == "tool") {
                out += "<｜User｜>" + content;
            }
        }
        out += "<｜Assistant｜>";
        if (enable_thinking) {
            out += "<think>\n";
        } else {
            out += "<think>\n\n</think>\n";
        }
        return out;
    }

    std::string render_chat_for_format(const json::Value& req) const {
        if (chat_format == ChatFormat::DeepSeek) {
            return render_deepseek_chat(req);
        }
        return render_chat(req);
    }
};

} // namespace ink
