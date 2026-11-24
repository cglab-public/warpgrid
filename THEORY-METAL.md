# Inkling-2-Mini Theoretical Ceilings on Apple M5 Max (128 GiB Unified Memory)

## 1. Measured Hardware (Apple M5 Max, 2026-11-24)

Measured with [`bench_hw/hw_probe_metal.mm`](file:///Users/daniel/warpgrid/bench_hw/hw_probe_metal.mm) (`clang++ -O3 -std=c++20`, AC power, Metal 4 / MSL 3.1+, Apple9 family):

| Metric | Measured (Median) | Peak | Notes |
|---|---:|---:|---|
| Device | **Apple M5 Max** | — | 40 GPU cores, 18 CPU cores (6P + 12E), 128 GiB unified RAM |
| Recommended Max Working Set | **107.52 GiB** | — | Max single buffer: 80.64 GiB; max threads/TG: 1024 |
| Memory bandwidth (D2D `float4` copy, 512 MiB × 20) | **477.17 GB/s** | 494.31 GB/s | **2.15×** NVIDIA GB10 (222.30 GB/s) |
| Memory bandwidth (Read-only `float4` stream, 512 MiB × 20) | **465.69 GB/s** | 505.92 GB/s | Closest to weight-streaming decode |
| Compute (`MPSMatrixMultiplication` FP16 `4096³` × 30) | **58.44 TFLOPS** | 60.09 TFLOPS | 2.35 ms/iter |
| Compute (`MPSMatrixMultiplication` FP16 `8192³` × 10) | **59.68 TFLOPS** | 60.15 TFLOPS | 18.42 ms/iter (~29.8 TFLOPS practical @ 50%) |
| Compute (naive untiled `simdgroup_matrix<bfloat,8,8>` `4096³` × 15) | **12.82 TFLOPS** | 12.97 TFLOPS | Proves native BF16 hardware MMA; TG-tiled target ~45–55 TFLOPS |

> **API Note:** `MPSMatrixMultiplication` does not support `MPSDataTypeBFloat16` (asserts on macOS). Prefill GEMMs will use TG-tiled `simdgroup_matrix<bfloat, 8, 8>` (lossless BF16) or FP16 MPS where range-safe.

## 2. Exact Model Footprint (`thinkingmachines/Inkling-2-Mini`)

Weights on disk: `/Users/daniel/models/mtplx/Youssofal--Inkling-2-Mini-MTPLX-Optimized-Quality/` (`model-00001-of-00001.safetensors` 18 GiB + `mtp.safetensors` 606 MiB).

| Component | Per stored layer | Effective visits | BF16 Bytes/tok | W8A16 FP8 Bytes/tok |
|---|---:|---:|---:|---:|
| Prelude dense (L00–L05) | 54.48–54.78 MB | 6 | 327.18 MB | 163.59 MB |
| Loop core dense (L06–L17, incl. controllers) | 59.20–59.50 MB | 36 (12 × 3) | 2,133.00 MB | 1,066.50 MB |
| Coda dense (L18–L23) | 54.48–54.78 MB | 6 | 327.18 MB | 163.59 MB |
| **Subtotal dense** | | **48** | **2,787.36 MB** | **1,393.68 MB** |
| NVFP4 routed experts (L00–L19, 8/128 active) | 35.83 MB active | 44 | 1,576.52 MB | 1,576.52 MB |
| MXFP8 routed experts (L20–L23, 8/128 active) | 65.69 MB active | 4 | 262.76 MB | 262.76 MB |
| **Subtotal active experts** | | **48** | **1,839.28 MB** | **1,839.28 MB** |
| LM head (`unembed`, `200058 × 2304`) | — | 1 | **921.87 MB** | **460.93 MB** |
| **Grand Total (short ctx, no spec)** | | | **5,548.51 MB (5.55 GB)** | **3,693.89 MB (3.69 GB)** |

## 3. Theoretical Ceilings (Apple M5 Max @ 477.17 GB/s, 59.68 TFLOPS FP16)

### Single-User Decode (`c=1`, Short Context)

| Mode | Bytes / emitted token | Ceiling (@ 477.17 GB/s) | Notes |
|---|---:|---:|---|
| **Lossless BF16 dense, no spec** | 5.55 GB | **86.0 tok/s** | 50.2% dense BF16, 33.1% MoE, 16.6% LM head |
| **W8A16 FP8 dense + LM head, no spec** | 3.69 GB | **129.2 tok/s** | 2× lower dense + LM head traffic |
| **Lossless + DSpark (`k=5`, `a=3.8`, code)** | ~2.75 GB | **~173.5 tok/s** | Dense + LM head amortized across 6 verify tokens |
| **W8A16 + DSpark (`k=5`, `a=3.8`, code)** | ~2.05 GB | **~232.8 tok/s** | Practical target ceiling on high-accept code |

### Local M5 Max Baselines to Beat

| Engine | c1 Code | c1 Prose | Prefill (~4K / ~18K / ~71K) |
|---|---:|---:|---|
| `llama.cpp` PR #32694 (`IQ4_NL`, no MTP) | 41.0 tok/s | 41.0 tok/s | 605 / 467 tok/s (3.3K / 25.5K) |
| `oMLX 0.8.3` (`oQ4e`, quiet, no MTP) | 59.5 tok/s | 57.2 tok/s | 1,807 / 1,783 / 1,349 tok/s |
| `MTPLX 2.13.2` (Optimized Quality, MTP) | **83.3 tok/s** | **57.0 tok/s** | **1,981 / 2,295 / 1,936 tok/s** |
| `MTPLX 2.13.2` (Optimized Speed, MTP) | **89.7 tok/s** | **63.7 tok/s** | 1,366 / 2,227 / 1,991 tok/s |

### Prefill (Compute Bound, Practical = 50% of 59.68 TFLOPS = 29.84 TFLOPS)

- Active linear params per token ≈ 3.4B → `6.8 GFLOPs/token` (+ ~3.1 GFLOPs/tok attention at 64K).
- **4K prefill practical ceiling:** `29,840 / 6.8` ≈ **4,388 tok/s**.
- **64K prefill practical ceiling:** `29,840 / 9.9` ≈ **3,014 tok/s**.

### KV Cache & Memory Cap (≤ 100 GiB on 128 GiB Unified RAM; Max WS 107.52 GiB)

- Text weights + DSpark: **~19.2 GiB** (+ ~2.0 GiB W8A16 FP8 copies if BF16 retained, or **~17.0 GiB** if BF16 dense freed).
- Global KV cache (FP8 E4M3): **16.25 KiB/token** (vs 32 KiB/tok BF16) → **2.21 GiB per 139K stream**.
- Local SWA KV cache (BF16): **160 MiB per active stream** (constant).
- **8 concurrent slots × 139,264 tokens (1.11M live KV tokens):** ~17.7 GiB global KV + 1.25 GiB local SWA + ~21 GiB weights ≈ **~40 GiB total**, well under the 100 GiB cap (~88 GiB headroom).
- **M4 Mac mini (16 GB) note:** Official `Inkling-2-Mini` text weights alone (~18.9 GiB) exceed a 16 GB Mac's ~10.7–12 GiB wired GPU limit; however, the Metal backend + `make test` suite targets Apple9+ (MSL 3.1) cleanly for M4/M5 parity and future ≤ 10 GB models.
