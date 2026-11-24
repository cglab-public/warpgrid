<p align="center">
  <img src="assets/warpgrid-logo.svg" alt="warpgrid" width="680" />
</p>

<h3 align="center">
  Bare-metal C++20/GPU LLM inference engine for unified-memory workstations &amp; local accelerators.
</h3>

<p align="center">
  <strong>Native CUDA &amp; Apple Metal · 1.18s Cold Boot · DFlash 2 · 1.11M Live KV Tokens</strong>
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

Measured end-to-end over the live OpenAI-compatible HTTP/SSE server (`./warpgrid` serving `thinkingmachines/Inkling-2-Mini` + native CUDA `DFlash 2` with **8 concurrent slots × 139,264 tokens = 1.11M live KV tokens**, **30.03 GiB KV**, **57 GiB RAM free**):

### Single-User Decode (`c=1`) vs. vLLM

| Workload | vLLM | **`warpgrid` (CUDA + DFlash 2)** | Delta |
|---|---:|---:|---|
| **`c=1` Ladder (per-user SSE)** | 59.8 tok/s (code) | **95.51 tok/s** | **+60%** vs vLLM |
| **`c=1` Code Panel (med / peak SSE)** | 59.8 tok/s | **60.63 / 74.94 tok/s** | Peak **+25.3%** vs vLLM |
| **`c=1` Prose Panel (med / peak SSE)** | 33.2 tok/s | **46.99 / 47.36 tok/s** | **+41.5% med** vs vLLM (full 200K vocab) |
| **`c=1` CLI DFlash 2 (`--dflash2`, `k=7`)** | — | **95.90** math / **76.66** code / **49.31** prose | **+79% / +42% / +36%** vs DSpark (`0.19 s` load, CUDA-graphed) |
| **Cold Server Load** | ~45–60 s | **3.81 s** (+0.19 s DFlash 2) | **~15×** faster startup |

### Parallel Continuous-Batching Ladder (`c = 1..8`)

| Concurrency | Per-User Median | Aggregate Throughput | Median TTFT |
|---:|---:|---:|---:|
| **`c = 1`** *(DFlash 2 `k=7`, full 200K vocab)* | **95.51 tok/s** | **87.56 tok/s** | 375 ms |
| **`c = 2`** *(Batched DFlash 2 `k=3`)* | **46.43 tok/s** | **81.08 tok/s** | 688 ms |
| **`c = 4`** *(Continuous batching)* | **25.05 tok/s** | **92.50 tok/s** | 1,314 ms |
| **`c = 6`** *(Continuous batching)* | **20.04 tok/s** | **109.36 tok/s** | 1,951 ms |
| **`c = 8`** *(Continuous batching)* | **17.45 tok/s** | **125.26 tok/s** | 2,578 ms |

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
| **Cold Server Load** | — | — | — | **1.18 s** (+0.07 s DFlash2) | Direct unified-memory safetensors load |
| **`c=1` Code SSE (med / peak)** | 41.0 tok/s | 59.5 tok/s | 83.3 tok/s | **82.10 / 111.36 tok/s** | **116.46 tok/s CLI** (+38% med / +87% peak vs oMLX) |
| **`c=1` Prose SSE (med / peak)** | 41.0 tok/s | 57.2 tok/s | 57.0 tok/s | **58.19 / 59.62 tok/s** | **Beats both `oMLX` & `MTPLX`** (`--dflash2`, `k=6`) |
| **4.1K Multi-Turn Prefix Cache** | — | — | — | **358.2 ms** warm (**27.5×**) | Exact `cached_tokens` match |
| **14.5K Needle-in-a-Haystack** | — | — | — | **443.3** prefill / **53.06** decode | **3/3 exact PASS** (Split-K FP8 attn) |
| **132K Needle-in-a-Haystack** | — | — | — | **233.6** prefill / **18.8** decode | **3/3 exact PASS**, **860.9 ms** warm (**656.9×**) |
| **Parallel Ladder (`c=1..8`)** | — | — | — | **68.81** (`c=1`) → **74.92 agg** (`c=8`) | Batched continuous decode |
| **Apple M4 Mac mini (16 GB, [`Q3`](https://huggingface.co/cglab-org/Inkling-2-Mini-Q3-warpgrid))** | 16.1 tok/s (`IQ2_XXS`) | — | — | **20.64** raw / **25.45** DFlash 2 code | **+58% vs `llama.cpp`**, **90.0% HumanEval** (93.8% retention), 17/17 tests |

---

## Architecture & Key Optimizations

- **Native Blackwell NVFP4 (`cuBLASLt` `CUDA_R_4F_E2M1`):** Swizzled `128×4` `VEC16_UE4M3` block scales hitting **195.6 TFLOPS** on GB10 for grouped MoE prefill with cached `fp4_plan_cache` descriptors — zero runtime weight dequantization.
- **Zero-Spill Small-Batch NVFP4 MoE:** GPU-deduplicated expert routing with templated `cnt = 1..4` register-resident PTX (`cvt.rn.f16x2.e2m1x2`) and fused `out_fp32` zeroing.
- **W8A16 Per-Row FP8 E4M3 Dense, LM Head & Drafter:** Quantized `w_qrgs`, `w_kv`, `wo`, 200K `unembed`, and all 3 `DSpark` draft layers + 32K frequency-pruned unembed + folded Markov head for 2× lower decode memory traffic.
- **Per-`(pos, head)` FP8 E4M3 Global KV Cache:** Cuts `8 × 136K` live KV memory from `46.5 → 30.03 GiB` (**−16.5 GiB saved**).
- **GQA=3 Shared-KV + Shared-T Split-K Decode Attention:** Cooperatively loads `BK=32` FP8 KV tiles into SMEM once across all `T` verify warps × 3 GQA heads — cutting 132K decode KV DRAM traffic by **24×** (`24.1 tok/s` at 132K, `65.2 tok/s` at 14.5K).
- **WMMA BF16 Tensor Core FlashAttention:** Custom GQA=3 `BK=64` Tensor Core prefill with exact relative-position logits, attention sinks, and sigmoid output gating.
- **Native DFlash 2 Block-Diffusion Drafter (CUDA & Metal):** Full 5-layer `incoai/Inkling-2-Mini-DFlash2` drafter (`--dflash2`) with 2-tap `GroupedDynamicCausalConv`, bidirectional block + SWA=2048 attention, W8A16 FP8 projections, top-16 candidate path selector, and 100% CUDA-graph capture on Blackwell (**+36% to +79%** vs DSpark; **116.5 tok/s CLI / 58.2 tok/s prose SSE** on M5 Max).
- **2-Row/SIMDgroup W8A16 FP8 & Deduplicated Small-Batch MoE (Apple Metal):** Hits **94%** of M5 Max's `465.69 GB/s` DRAM bandwidth on the 200K LM head (`1.05 ms`), with grouped `simdgroup_matrix` MoE prefill, FlashAttention-2 FP8 prefill, Split-K FP8 decode, and runtime MSL fallback for Macs without Xcode.
- **Compact `Q3_G64` / `Q2_G64` Native MoE Quantizer (`scripts/quantize_q2.py`):** Activation-`imatrix` Lloyd-Max bit-split 3-bit (`Q3_G64`, 3.5 bpw) and 2-bit (`Q2_G64`, 2.5 bpw) MoE weights (`18×Q3_G64 + 6×Q2_G64`) + automatic `low_mem` BF16-dense release (−1.2 GiB), delivering **90.0% HumanEval Pass@1 (93.8% retention vs 18 GiB NVFP4)** and **25.45 tok/s** coding with DFlash 2 on a **16 GB Apple M4 Mac mini** (**+58%** vs `llama.cpp` `IQ2_XXS`).
- **Full-Step CUDA Graphs:** Captured per-`T` (`1..8`) single-slot verify, per-`B` (`1..8`) parallel decode, batched `B=2` speculative verify, and graphed DSpark / DFlash 2 drafting.
- **Exact Prefix Caching:** Per-slot end-of-prefill snapshots (local SWA ring + conv states) delivering **9.4×–656.9×** warm TTFT speedups and reporting `usage.prompt_tokens_details.cached_tokens`.

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
# Download official weights + DFlash 2 drafter and serve (CUDA & Apple Metal)
hf download thinkingmachines/Inkling-2-Mini --local-dir ./Inkling-2-Mini
hf download incoai/Inkling-2-Mini-DFlash2 --local-dir ./Inkling-2-Mini-DFlash2
./warpgrid \
  --model ./Inkling-2-Mini \
  --dflash2 ./Inkling-2-Mini-DFlash2 \
  --draft-k 6 \
  --port 8888 \
  --max-slots 8 \
  --max-seq-len 139264 \
  --max-chunk 4096

# 16 GB Apple Silicon Macs (e.g., M4 Mac mini): pull the pre-quantized Q3 hybrid checkpoint (90% HumanEval)
hf download cglab-org/Inkling-2-Mini-Q3-warpgrid --local-dir ./Inkling-2-Mini-Q3
hf download incoai/Inkling-2-Mini-DFlash2 --local-dir ./Inkling-2-Mini-DFlash2
./warpgrid \
  --model ./Inkling-2-Mini-Q3 \
  --dflash2 ./Inkling-2-Mini-DFlash2 \
  --draft-k 6 \
  --max-slots 1 \
  --max-seq-len 2048 \
  --port 8888

# Or quantize your own Q3/Q2 checkpoint from official weights:
uv run --with numpy python3 scripts/quantize_q2.py \
  --src ./Inkling-2-Mini \
  --dst ./Inkling-2-Mini-Q3 \
  --keep-nvfp4-layers "" \
  --q3-layers 0,1,2,3,4,5,6,11,12,15,16,17,18,19,20,21,22,23 \
  --imatrix ./imatrix.bin \
  --drop-vision
```

Endpoints:
- `GET /health`
- `GET /status` (live JSON snapshot: active/pending requests, per-slot KV occupancy, prefix-cache hit ratio, decode `tok/s`, DFlash 2 `mean_a`)
- `GET /metrics` (Prometheus / Datadog OpenMetrics text format)
- `GET /v1/models`
- `POST /v1/chat/completions` (streaming SSE & non-streaming, tool calls, `reasoning_content`, `cached_tokens` usage stats)

### 3. CLI Options

| Flag | Default | Description |
|---|---:|---|
| `--model <dir>` | *(required)* | Hugging Face model snapshot directory (NVFP4/MXFP8 or `Q2_G64`) |
| `--served-model-name <name>` | `Inkling-2-Mini` | Advertised model ID |
| `--port <int>` | `8888` | HTTP listen port |
| `--max-slots <int>` | `8` | Max concurrent continuous-batching slots (`1..8`) |
| `--max-seq-len <int>` | `32768` | Max context length per slot (`139264` for ~136K/slot) |
| `--max-chunk <int>` | `512` | Max prefill chunk size (`4096` recommended) |
| `--dflash2 <dir>` | *(empty)* | Enable native DFlash 2 drafter (`incoai/Inkling-2-Mini-DFlash2`) |
| `--draft-k <int>` | `0` | Max speculative draft tokens (`1..7`; `6` on Metal, `7` on CUDA) |
| `--draft-vocab-file <path>` | *(empty)* | Optional pruned frequency-ranked draft vocab binary (DSpark & CUDA DFlash 2) |
| `--otlp-endpoint <url>` | *(off)* | Optional OTLP/HTTP JSON trace exporter (e.g. `http://127.0.0.1:4318` for Datadog / OTel Collector) |
| `--moe-group-thresh <int>` | `64` | Token threshold `T` for grouped `cuBLASLt` NVFP4 MoE |
| `--bench` / `--bench-tokens <N>` | `false` / `128` | Run standalone single-request decode benchmark |
| `--prompt <text>` | *(math canary)* | Custom prompt for `--bench` mode |
| `--bench-prefill <N>` | `0` | Run synthetic `N`-token prefill benchmark |
| `--profile` / `--no-graph` | `false` | Per-stage GPU timing breakdown |

---

## Roadmap

- ~~**Apple Metal Backend:**~~ Shipped (`src/metal/`, zero-Python/zero-MLX, M3/M4/M5 + 16 GB `Q2_G64`).
- ~~**DFlash 2 Block-Diffusion Speculative Drafting:**~~ Shipped on both Apple Metal and NVIDIA Blackwell CUDA.
- **Multi-Model Support:** Generalize beyond `Inkling-2-Mini` to additional NVFP4 / MoE / dense Transformer families.
- **Paged Radix Prefix Cache & Chunked Prefill Interleaving:** Cross-slot shared system prompts and zero-stall decode during 100K+ incoming prefills.

---

## License

MIT
