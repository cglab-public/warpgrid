#pragma once
#include "json.hpp"
#ifndef WARPGRID_METAL
#include <cuda_runtime.h>
#endif
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct TensorMeta {
    std::string dtype;
    std::vector<int64_t> shape;
    size_t offset = 0; // relative to data_base
    size_t nbytes = 0;
};

class SafetensorsFile {
    int fd_ = -1;
    size_t data_base_ = 0;
    std::unordered_map<std::string, TensorMeta> meta_;
#ifndef WARPGRID_METAL
    std::vector<void*> gpu_allocs_;
    std::unordered_map<std::string, void*> gpu_ptrs_;
#endif

public:
    SafetensorsFile() = default;
    ~SafetensorsFile() {
#ifndef WARPGRID_METAL
        for (void* p : gpu_allocs_) cudaFree(p);
#endif
        if (fd_ >= 0) close(fd_);
    }
    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;

    void open(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("Failed to open safetensors: " + path);
        uint64_t hlen = 0;
        if (::pread(fd_, &hlen, 8, 0) != 8) throw std::runtime_error("Failed to read safetensors header length");
        std::string hbuf(hlen, '\0');
        size_t got = 0;
        while (got < hlen) {
            ssize_t r = ::pread(fd_, &hbuf[got], hlen - got, 8 + got);
            if (r <= 0) throw std::runtime_error("Failed to read safetensors header");
            got += (size_t)r;
        }
        data_base_ = 8 + hlen;
        auto hdr = json::parse(hbuf);
        for (const auto& kv : hdr.obj) {
            if (kv.first == "__metadata__") continue;
            TensorMeta m;
            m.dtype = kv.second["dtype"].as_str();
            for (const auto& s : kv.second["shape"].arr) m.shape.push_back(s.as_i64());
            size_t s0 = (size_t)kv.second["data_offsets"].arr[0].as_i64();
            size_t s1 = (size_t)kv.second["data_offsets"].arr[1].as_i64();
            m.offset = s0;
            m.nbytes = s1 - s0;
            meta_.emplace(kv.first, std::move(m));
        }
    }

    bool has(const std::string& name) const { return meta_.count(name) > 0; }
    const TensorMeta& info(const std::string& name) const {
        auto it = meta_.find(name);
        if (it == meta_.end()) throw std::runtime_error("Missing tensor: " + name);
        return it->second;
    }

#ifndef WARPGRID_METAL
    void* load_to_gpu(const std::string& name, std::vector<uint8_t>& host_scratch) {
        auto it = gpu_ptrs_.find(name);
        if (it != gpu_ptrs_.end()) return it->second;
        const auto& m = info(name);
        void* d_ptr = nullptr;
        cudaError_t err = cudaMalloc(&d_ptr, m.nbytes);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed for " + name + ": " + cudaGetErrorString(err));
        }
        gpu_allocs_.push_back(d_ptr);
        gpu_ptrs_[name] = d_ptr;

        if (host_scratch.size() < m.nbytes) host_scratch.resize(m.nbytes);
        size_t got = 0;
        while (got < m.nbytes) {
            ssize_t r = ::pread(fd_, host_scratch.data() + got, m.nbytes - got, (off_t)(data_base_ + m.offset + got));
            if (r <= 0) throw std::runtime_error("pread failed for " + name);
            got += (size_t)r;
        }
        err = cudaMemcpy(d_ptr, host_scratch.data(), m.nbytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy failed for " + name + ": " + cudaGetErrorString(err));
        }
        return d_ptr;
    }
#endif

    void read_into(const std::string& name, void* dst, size_t dst_bytes) const {
        const auto& m = info(name);
        if (dst_bytes < m.nbytes) {
            throw std::runtime_error("Destination buffer too small for " + name);
        }
        uint8_t* out = static_cast<uint8_t*>(dst);
        size_t got = 0;
        while (got < m.nbytes) {
            ssize_t r = ::pread(fd_, out + got, m.nbytes - got, (off_t)(data_base_ + m.offset + got));
            if (r <= 0) throw std::runtime_error("pread failed for " + name);
            got += (size_t)r;
        }
    }

    void load_to_host(const std::string& name, std::vector<uint8_t>& out) const {
        const auto& m = info(name);
        out.resize(m.nbytes);
        read_into(name, out.data(), out.size());
    }

    const std::unordered_map<std::string, TensorMeta>& all_meta() const { return meta_; }
};
