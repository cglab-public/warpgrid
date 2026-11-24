# Inkling-2-Mini Theoretical Ceilings on NVIDIA GB10 (spark-e068)

## 1. Measured Hardware (spark-e068, 2026-11-23)

Measured with `bench_hw/hw_probe.cu` (`nvcc -O3 -arch=sm_121`, idle GPU, 43°C, 113 GiB MemAvailable):

| Metric | Measured | Notes |
|---|---:|---|
| Device | NVIDIA GB10 (`sm_121`) | 48 SMs, 24.0 MB L2, 121.69 GiB unified memory |
| Memory bandwidth (D2D `float4` kernel, 512 MB × 20) | **222.30 GB/s** | 218.31 GB/s via `cudaMemcpy` |
| Compute (BF16 GEMM 4096³ × 50) | **85.58 TFLOPS** | 1.61 ms/iter |
| Compute (BF16 GEMM 8192³ × 10) | **95.74 TFLOPS** | 11.48 ms/iter (~47.9 TFLOPS practical @ 50%) |

## 2. Exact Model Footprint (`thinkingmachines/Inkling-2-Mini`, rev `ed40e7da`)

- **Architecture:** `Inkling2LoopForConditionalGeneration` (24 stored layers; loop core layers 6–17 run 3 passes → **48 effective layers** per token).
- **Attention:** 20 local SWA layers (`W = 1024`, 24 Q heads, 8 KV heads, `head_dim = 128`, `d_rel = 16`, static `k=4` sconv on K/V) + 4 global layers (L5, L11, L17, L23; dynamic `k=4` sconv on K/V, log-scaling `tau` with `n_floor = 8192, alpha = 0.1`). Across 3 loop passes: **40 local visits + 8 global visits**.
- **MoE:** 128 routed experts, top-8 (`hidden = 2304`, `intermediate = 1152`, interleaved gate/up, softcaps 4.0 / 25.0, `route_scale = 8.0`).
  - L00–L19: **NVFP4** (`U8` packed e2m1 + `F8_E4M3` per-16 scales + per-expert `amax` / `outer_scale_denominator`).
  - L20–L23: **MXFP8** (`F8_E4M3` + `U8` ue8m0 per-32 scales).
- **Dense weights:** BF16 (`wq_du`, `wk_dv`, `wv_dv`, `wo_ud`, `gate_attn`, `wr_du`, `attn_sink_proj`, `controllers`, `embed`, `unembed`).

### Bytes Read per Emitted Token (Short Context, No Speculation)

| Component | Per stored layer | Effective visits | Bytes / token |
|---|---:|---:|---:|
| Prelude dense (L00–L05) | 54.48–54.78 MB | 6 | 327.18 MB |
| Loop core dense (L06–L17, incl. controllers) | 59.20–59.50 MB | 36 (12 × 3) | 2,133.00 MB |
| Coda dense (L18–L23) | 54.48–54.78 MB | 6 | 327.18 MB |
| **Subtotal dense** | | **48** | **2,787.36 MB (2.787 GB)** |
| NVFP4 routed experts (L00–L19, 8/128 active) | 35.83 MB active | 44 (6 + 36 + 2) | 1,576.52 MB |
| MXFP8 routed experts (L20–L23, 8/128 active) | 65.69 MB active | 4 | 262.76 MB |
| **Subtotal active experts** | | **48** | **1,839.28 MB (1.839 GB)** |
| LM head (`unembed`, BF16 `201024 × 2304`) | — | 1 | **926.32 MB (0.926 GB)** |
| **Grand Total (short ctx, no spec)** | | | **5,552.96 MB (5.553 GB)** |

## 3. Theoretical Ceilings (1× GB10 @ 222.3 GB/s, 95.7 TFLOPS BF16)

### Single-User Decode (c1, Short Context)

| Mode | Bytes / emitted token | Ceiling (at 222.3 GB/s) | Notes |
|---|---:|---:|---|
| **Lossless, no speculation** | 5.553 GB | **40.0 tok/s** | 50.2% dense BF16, 33.1% MoE, 16.7% LM head |
| **Lossless + DSpark (k=3, a=2.5, 50% routing locality)** | ~3.45 GB | **~64.4 tok/s** | Dense + LM head amortized across 4 verify tokens |
| **Lossless + DSpark (k=5, a=3.8, code locality)** | ~2.75 GB | **~80.8 tok/s** | Upper bound on high-acceptance code |
| **vLLM baseline (observed, DSpark k=7)** | — | **33.2 prose / 59.8 code** | Target to beat |

### Prefill (Compute Bound, Practical = 50% of 95.74 TFLOPS = 47.9 TFLOPS)

- Active linear params per token ≈ 3.4B → `2 × 3.4e9 = 6.8 GFLOPs/token` (plus attention `4 × 8 × 24 × 128 × L/2` on 8 global passes).
- **4K prefill practical ceiling:** `47,900 / 6.8` ≈ **7,040 tok/s** (vLLM baseline: 1,526 tok/s).
- **64K prefill practical ceiling:** attention adds `~3.1 GFLOPs/tok` → `47,900 / 9.9` ≈ **4,840 tok/s** (vLLM baseline: 2,356 tok/s @ 77.9K).

### KV Cache Capacity (Memory Cap ≤ 100 GiB on 121.7 GiB Unified RAM)

- Model weights (text + DSpark, excluding vision): **~19.2 GiB**
- Global KV cache: **8 effective global passes** × 2 (K,V) × 8 kv_heads × 128 head_dim × 2 B (BF16) = **32,768 B/token (32 KiB/token)**.
- Local SWA KV cache: 40 local passes × 1,024 window × 2 × 8 × 128 × 2 B = **160 MiB per active stream** (constant!).
- At **262,144 tokens**: `32 KiB × 262,144 = 8.0 GiB` per full-length stream → up to **9–10 full 262K streams** fit in BF16 KV under a 100 GiB cap.
