<p align="center">
  <img src="assets/warpgrid-logo.svg" alt="warpgrid" width="680" />
</p>

<h3 align="center">
  Bare-metal C++20/GPU LLM inference engine for unified-memory workstations &amp; local accelerators.
</h3>

<p align="center">
  <strong>Zero Python · Zero PyTorch · 3.8s Cold Boot · 1.11M Live KV Tokens · Beats vLLM</strong>
</p>

---

## Why `warpgrid`?

Modern LLM serving frameworks like vLLM and SGLang are extraordinary pieces of software — engineered for datacenter fleets of H100/B200 GPUs with 3–8 TB/s of HBM bandwidth and hundreds of supported model architectures.

**Unified-memory workstations and local GPUs play by completely different physics:**

1. **Memory bandwidth is the entire game.** Machines like NVIDIA DGX Spark (GB10, `222 GB/s` LPDDR5X, `121 GiB` unified RAM) and Apple Silicon workstations pair huge unified memory pools with modest bandwidth compared to datacenter HBM. Every redundant DRAM read, extra kernel launch, or unquantized BF16 weight matrix directly caps `tok/s`.
2. **Novel architectures outrun generic frameworks.** Next-generation models like Thinking Machines' `Inkling-2-Mini` — with 4× looped transformer layers, Attention Residuals (`AttnRes`), dynamic short-convolutions (`sconv`), 128-expert NVFP4 MoE, and `DSpark` Markov speculative drafting — pay a steep tax in Python dispatch and generic fallback kernels.
3. **Single-binary simplicity.** Sometimes you don't want a 15 GB container, a 60-second Python boot, or a dependency tree of 200 packages just to serve a team of 8 developers from a single workstation.

**`warpgrid`** is a 100% from-scratch standalone **C++20/GPU** inference engine with native **NVIDIA Blackwell CUDA** and **Apple Silicon Metal** backends sharing a single portable C++20 host/HTTP/tokenizer/scheduler stack — engineered from first principles around GPU memory hierarchies, where every **warp** / **SIMDgroup** and every GPU **grid** launch is hand-tuned to push close to the hardware bandwidth ceiling.

---

## Reference Benchmarks (NVIDIA DGX Spark — GB10, `sm_121a`)

Measured end-to-end over the live OpenAI-compatible HTTP/SSE server (`./warpgrid` serving `thinkingmachines/Inkling-2-Mini` with **8 concurrent slots × 139,264 tokens = 1.11M live KV tokens**, **30.03 GiB KV**, **57 GiB RAM free**):

### Single-User Decode (`c=1`) vs. vLLM

| Workload | vLLM | **`warpgrid`** | Delta |
|---|---:|---:|---|
| **`c=1` Ladder (per-user)** | 59.8 tok/s (code) | **65.8 – 69.4 tok/s** | **+10% to +16%** vs vLLM |
| **`c=1` Code Panel (med / peak)** | 59.8 tok/s | **58.8 / 69.0 tok/s** | Peak **+15.3%** vs vLLM |
| **`c=1` Prose Panel (med / peak)** | 33.2 tok/s | **44.2 / 51.8 tok/s** | **+33% med / +56% peak** vs vLLM |
| **Cold Server Load** | ~45–60 s | **3.81 s** | **~15×** faster startup |

### Parallel Continuous-Batching Ladder (`c = 1..8`)

| Concurrency | Per-User Median | Aggregate Throughput | Median TTFT |
|---:|---:|---:|---:|
| **`c = 1`** *(Speculative adaptive `k=3..7`)* | **65.78 tok/s** | 59.57 tok/s | 267 ms |
| **`c = 2`** *(Batched speculative `k=3`)* | **41.51 tok/s** | **68.86 tok/s** | 482 ms |
| **`c = 4`** *(Continuous batching)* | **23.86 tok/s** | **84.85 tok/s** | 880 ms |
| **`c = 6`** *(Continuous batching)* | **18.69 tok/s** | **97.94 tok/s** | 1,294 ms |
| **`c = 8`** *(Continuous batching)* | **16.07 tok/s** | **110.40 tok/s** | 1,695 ms |

### Long Context & Prefix Caching (`14.5K` & `132K` — `3/3` Exact Needles)

| Benchmark | Prefill | Decode | Warm Prefix Hit TTFT | Accuracy |
|---|---:|---:|---:|---|
| **4.1K Multi-Turn Chat** | 2,809 ms cold | — | **297 ms** (**9.4× faster**) | 3/3 turns exact |
| **14.5K Needle-in-a-Haystack** | **1,311 tok/s** | **65.20 tok/s** | — | **3/3 exact PASS** |
| **132K Needle-in-a-Haystack** | **513.3 tok/s** | **24.1 tok/s** | **1,045.8 ms** (**246.1× faster**) | **3/3 exact PASS** |

---

## Reference Benchmarks (Apple Silicon Metal — M5 Max, 128 GiB Unified RAM)

Measured end-to-end over the live OpenAI-compatible HTTP/SSE server (`./warpgrid` built with `make -j` on macOS — zero Python/PyTorch/MLX dependencies — serving official `thinkingmachines/Inkling-2-Mini` NVFP4/MXFP8 + `DSpark` `mtp.safetensors` with **8 concurrent slots × 139,264 tokens**, **18.5 GiB peak RSS**):

| Workload | `llama.cpp` (`IQ4_NL`) | `oMLX 0.8.3` (`oQ4e`) | `MTPLX 2.13.2` (MTP) | **`warpgrid` (Metal + DFlash2)** | Notes |
|---|---:|---:|---:|---:|---|
| **Cold Server Load** | — | — | — | **1.18 s** (+0.09 s DFlash2) | Direct unified-memory safetensors load |
| **`c=1` Code SSE (med / peak)** | 41.0 tok/s | 59.5 tok/s | 83.3 tok/s | **80.16 / 107.91 tok/s** | **+34.7% med / +81.4% peak vs oMLX** *(on battery)* |
| **`c=1` Prose SSE (med)** | 41.0 tok/s | 57.2 tok/s | 57.0 tok/s | **55.60 tok/s** | Native DFlash2 (`--dflash2`, `k=7`, *on battery*) |
| **4.1K Multi-Turn Prefix Cache** | — | — | — | **358.2 ms** warm (**27.5×**) | Exact `cached_tokens` match |
| **14.5K Needle-in-a-Haystack** | — | — | — | **427.4** prefill / **53.06** decode | **3/3 exact PASS** (Split-K FP8 attn) |
| **132K Needle-in-a-Haystack** | — | — | — | **233.6** prefill / **18.8** decode | **3/3 exact PASS**, **860.9 ms** warm (**656.9×**) |
| **Parallel Ladder (`c=1..8`)** | — | — | — | **68.81** (`c=1`) → **74.92 agg** (`c=8`) | Batched continuous decode |

---

## Architecture & Key Optimizations

- **Native Blackwell NVFP4 (`cuBLASLt` `CUDA_R_4F_E2M1`):** Swizzled `128×4` `VEC16_UE4M3` block scales hitting **195.6 TFLOPS** on GB10 for grouped MoE prefill with cached `fp4_plan_cache` descriptors — zero runtime weight dequantization.
- **Zero-Spill Small-Batch NVFP4 MoE:** GPU-deduplicated expert routing with templated `cnt = 1..4` register-resident PTX (`cvt.rn.f16x2.e2m1x2`) and fused `out_fp32` zeroing.
- **W8A16 Per-Row FP8 E4M3 Dense, LM Head & Drafter:** Quantized `w_qrgs`, `w_kv`, `wo`, 200K `unembed`, and all 3 `DSpark` draft layers + 32K frequency-pruned unembed + folded Markov head for 2× lower decode memory traffic.
- **Per-`(pos, head)` FP8 E4M3 Global KV Cache:** Cuts `8 × 136K` live KV memory from `46.5 → 30.03 GiB` (**−16.5 GiB saved**).
- **GQA=3 Shared-KV + Shared-T Split-K Decode Attention:** Cooperatively loads `BK=32` FP8 KV tiles into SMEM once across all `T` verify warps × 3 GQA heads — cutting 132K decode KV DRAM traffic by **24×** (`24.1 tok/s` at 132K, `65.2 tok/s` at 14.5K).
- **WMMA BF16 Tensor Core FlashAttention:** Custom GQA=3 `BK=64` Tensor Core prefill with exact relative-position logits, attention sinks, and sigmoid output gating.
- **Full-Step CUDA Graphs:** Captured per-`T` (`1..8`) single-slot verify, per-`B` (`1..8`) parallel decode, batched `B=2` speculative verify, and graphed drafting.
- **Exact Prefix Caching:** Per-slot end-of-prefill snapshots (local SWA ring + conv states) delivering **9.4×–246×** warm TTFT speedups and reporting `usage.prompt_tokens_details.cached_tokens`.

---

## Quick Start

### 1. Build (Auto-Detects macOS/Metal or Linux/CUDA)
- **macOS (Apple Silicon M3/M4/M5):** Requires Xcode Command Line Tools (`clang++`, Metal framework; optional `xcrun metal` toolchain for precompiled `.metallib` embedding).
- **Linux (NVIDIA Blackwell):** Requires CUDA Toolkit (12.8+ for `sm_121a`) and `g++` (C++20).

```bash
make -j
make test
```

This builds a single standalone binary: `./warpgrid` (and runs the portable C++20 host + GPU kernel unit test suite).

### 2. Serve (OpenAI-Compatible API)

```bash
./warpgrid \
  --model /path/to/thinkingmachines/Inkling-2-Mini \
  --port 8888 \
  --max-slots 8 \
  --max-seq-len 139264 \
  --max-chunk 4096 \
  --draft-k 7 \
  --draft-vocab-file ./draft_vocab_32k.bin
```

Endpoints:
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` (streaming SSE & non-streaming, tool calls, `reasoning_content`, `cached_tokens` usage stats)

### 3. CLI Options

| Flag | Default | Description |
|---|---:|---|
| `--model <dir>` | *(required)* | Hugging Face model snapshot directory |
| `--served-model-name <name>` | `Inkling-2-Mini` | Advertised model ID |
| `--port <int>` | `8888` | HTTP listen port |
| `--max-slots <int>` | `8` | Max concurrent continuous-batching slots (`1..8`) |
| `--max-seq-len <int>` | `32768` | Max context length per slot (`139264` for ~136K/slot) |
| `--max-chunk <int>` | `512` | Max prefill chunk size (`4096` recommended) |
| `--draft-k <int>` | `0` | Max speculative draft tokens (`1..7`; `7` recommended) |
| `--draft-vocab-file <path>` | *(empty)* | Optional pruned frequency-ranked draft vocab binary |
| `--moe-group-thresh <int>` | `64` | Token threshold `T` for grouped `cuBLASLt` NVFP4 MoE |
| `--bench` / `--bench-tokens <N>` | `false` / `128` | Run standalone single-request decode benchmark |
| `--bench-prefill <N>` | `0` | Run synthetic `N`-token prefill benchmark |
| `--profile` / `--no-graph` | `false` | Per-stage GPU timing breakdown |

---

## Roadmap

- **Apple Metal Backend (`metal-cpp`):** Clean backend abstraction so the existing portable C++20 host layer (`server.hpp`, `tokenizer.hpp`, `safetensors.hpp`, `json.hpp`) runs natively on Apple Silicon alongside CUDA.
- **Multi-Model Support:** Generalize beyond `Inkling-2-Mini` to additional NVFP4 / MoE / dense Transformer families.
- **Paged Radix Prefix Cache & Chunked Prefill Interleaving:** Cross-slot shared system prompts and zero-stall decode during 100K+ incoming prefills.

---

## License

MIT
