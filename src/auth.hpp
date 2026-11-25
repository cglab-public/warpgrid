#pragma once
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace ink {

// Constant-time string comparison to avoid timing side channels on API keys.
inline bool constant_time_eq(std::string_view a, std::string_view b) {
    const size_t max_len = std::max(a.size(), b.size());
    volatile uint8_t diff = (a.size() != b.size()) ? 1u : 0u;
    for (size_t i = 0; i < max_len; ++i) {
        uint8_t ca = (i < a.size()) ? static_cast<uint8_t>(a[i]) : 0u;
        uint8_t cb = (i < b.size()) ? static_cast<uint8_t>(b[i]) : 0u;
        diff = static_cast<uint8_t>(diff | static_cast<uint8_t>(ca ^ cb));
    }
    return diff == 0u;
}

inline std::string_view trim_http_ows(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.remove_suffix(1);
    return s;
}

inline bool ascii_iequals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i])) !=
            std::tolower(static_cast<unsigned char>(b[i]))) {
            return false;
        }
    }
    return true;
}

// Parse "METHOD /path[?query] HTTP/1.1" into method and path (without query string).
inline bool parse_request_line(std::string_view req_line, std::string& method, std::string& path) {
    method.clear();
    path.clear();
    if (!req_line.empty() && req_line.back() == '\r') req_line.remove_suffix(1);
    size_t sp1 = req_line.find(' ');
    if (sp1 == std::string_view::npos || sp1 == 0) return false;
    method.assign(req_line.substr(0, sp1));

    size_t target_start = sp1 + 1;
    if (target_start >= req_line.size()) return false;
    size_t sp2 = req_line.find(' ', target_start);
    std::string_view target = (sp2 == std::string_view::npos)
        ? req_line.substr(target_start)
        : req_line.substr(target_start, sp2 - target_start);
    if (target.empty()) return false;

    size_t q = target.find('?');
    if (q != std::string_view::npos) target = target.substr(0, q);
    if (target.empty()) return false;

    path.assign(target);
    return true;
}

// Only liveness probe GET /health is unauthenticated when an API key is configured.
inline bool is_public_endpoint(std::string_view method, std::string_view path) {
    return method == "GET" && path == "/health";
}

// Validate an HTTP header block (request line + headers, without trailing \r\n\r\n)
// against an optional expected API key.
// Accepts:
//   Authorization: Bearer <key>   (case-insensitive header name and scheme)
//   X-API-Key: <key>              (case-insensitive header name)
// When expected_key is empty, authentication is disabled and all requests are allowed.
inline bool is_request_authorized(std::string_view header_block, std::string_view expected_key) {
    if (expected_key.empty()) return true;

    size_t pos = header_block.find('\n');
    if (pos == std::string_view::npos) return false;
    ++pos; // Skip request line

    while (pos < header_block.size()) {
        size_t eol = header_block.find('\n', pos);
        std::string_view line = (eol == std::string_view::npos)
            ? header_block.substr(pos)
            : header_block.substr(pos, eol - pos);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);

        size_t colon = line.find(':');
        if (colon != std::string_view::npos && colon > 0) {
            std::string_view name = line.substr(0, colon);
            std::string_view val = trim_http_ows(line.substr(colon + 1));

            if (ascii_iequals(name, "authorization")) {
                if (val.size() > 6 && ascii_iequals(val.substr(0, 6), "bearer") &&
                    (val[6] == ' ' || val[6] == '\t')) {
                    std::string_view token = trim_http_ows(val.substr(7));
                    if (constant_time_eq(token, expected_key)) return true;
                }
            } else if (ascii_iequals(name, "x-api-key")) {
                if (constant_time_eq(val, expected_key)) return true;
            }
        }

        if (eol == std::string_view::npos) break;
        pos = eol + 1;
    }
    return false;
}

} // namespace ink
