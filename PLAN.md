# Inkling-2-Mini From-Scratch Serving Engine — Experiment Plan & Log

## Targets & Constraints
- **Target hardware:** `spark-e068` (`10.1.1.59`), 1× NVIDIA GB10 (`sm_121a`, 48 SMs, 222.3 GB/s measured, 95.7 TFLOPS BF16, 121.7 GiB unified RAM).
- **Model:** `thinkingmachines/Inkling-2-Mini` (`ed40e7da`, NVFP4 L00–L19 + MXFP8 L20–L23, 24 stored / 48 effective looped layers, DSpark `mtp.safetensors`).
- **vLLM Baseline (1 Spark, `results-vllm-inkling2-mini-x2-20261121`):**
  - c1 decode: **59.8 tok/s code / 33.2 tok/s prose** (with DSpark k=7)
  - Prefill: **1,526 tok/s @ 4.1K**, **2,275 tok/s @ 20.3K**
  - MemAvailable idle: ~26 GiB
- **Constraints:** Standalone C++20 / CUDA `sm_121a` engine from scratch (no PyTorch/vLLM/AgEnFK), lossless numerics, memory cap ≤ 100 GiB, OpenAI `/v1/chat/completions` streaming + reasoning + tool calling.

## Hill-Climb Ledger

| Phase | Lever / Milestone | Measured Result | Verdict |
|---|---|---|---|
| 0 | Hardware probe + exact byte inventory (`THEORY.md`) | 222.3 GB/s D2D, 95.7 TFLOPS BF16, 5.553 GB/tok → 40.0 tok/s no-spec ceiling | **Locked** |
| 1 | `r01-baseline`: From-scratch C++20/CUDA `Inkling2LoopModel` + safetensors + BPE tokenizer + OpenAI HTTP/SSE server | Canary `323` PASS; c1 decode **24.01 tok/s** (41.65 ms/tok); prefill 46 tok/s; warm load 2.63 s | **Baseline** |
| 2 | Lever 1: Multi-SM `uint4` MoE router + Blackwell `cvt.rn.f16x2.e2m1x2` PTX (`sm_121a`) + `uint4` MXFP8 | Router 6.47→0.82 ms (7.9×); NVFP4 9.78→7.52 ms (94.5% BW); MXFP8 1.81→1.16 ms; c1 decode **29.98 tok/s** (+25%) | **Accepted** |
| 3 | Lever 2: Fused loop-controller `gain`/`forget`/`blend`/`RMSNorm` + full-step `T=1` CUDA Graph capture | Interleaved A/B: CTL1 29.02 → **CAND 32.25 tok/s** (+11.2%, 81% of no-spec ceiling) → CTL2 28.97 | **Accepted** |
| 4 | Lever 3: Grouped-expert prefill MoE (per-expert NVFP4/MXFP8 dequant + cuBLAS BF16 Tensor Cores) | 4K prefill 75 → **1,539.7 tok/s** (beats vLLM 1,526 @ 4.1K); canary bit-exact | **Accepted** |
| 5 | Lever 4: From-scratch DSpark drafter (`mtp.safetensors`: 3-layer DFlare + folded Markov head + conv-state rollback) | `k=7`: 72.8% accept (`mean_a = 6.04`), **45.66 tok/s** (broke past 40.0 tok/s no-spec ceiling) | **Accepted** |
| 6 | Lever 5: GPU-deduplicated small-batch (`T=2..8`) NVFP4/MXFP8 MoE + zero-sync GPU Markov loop | CLI Code `k=4`: **41.08 tok/s** (`mean_a=3.12`); CLI `k=0` (CUDA graph): **32.2 tok/s** code / **32.1 tok/s** prose | **Accepted (Code win)** |
| 7 | Qualification on live `spark-e068:8888` (`--max-seq-len 131072 --draft-k 5`) | Health/Models PASS, Canary `323` PASS, Tool `get_weather` PASS, SSE Code median **37.45 tok/s** (peak **42.84**), Prose median **25.67 tok/s**, 14.5K Needle **3/3 exact** (`420.4 tok/s` prefill, `22.76 tok/s` decode), **88 GiB RAM free** | **Qualified** |
| 8 | Lever 6 (naive prefix 32K/64K draft vocab) | Draft 9.0→4.4 ms, **but** accept crashed 48.2%→14.5% (o200k common tokens > 65K) | **Rejected** |
| 9 | Lever 7 (`T=k+1` Verify CUDA Graph) | `k=4` 37.99 → **39.14 tok/s** (`step_ms` 77.04→74.78 ms, identical 48.2% accept) | **Accepted** |
| 10 | Lever 8 (`bf16_small_gemm<M>` for `M=2..8` + zero-spill `cnt==1` NVFP4 MoE) | `T=5` verify **70.96 → 60.55 ms** (−14.7%); Dense BF16 15.61→13.66 ms; LM Head 5.23→4.03 ms; NVFP4 29.88→26.53 ms | **Accepted** |
| 11 | Lever 6b (Frequency-ranked 32K draft vocab `draft_vocab_32k.bin`, 99.65%+ coverage) | Interleaved A/B `k=4`: CTL1 38.04 → **CAND 41.63 tok/s** (+9.4%, draft 8.79→3.71 ms, zero accept drop 42.2%) → CTL2 38.14; `k=5` **42.71 tok/s** | **Accepted** |
| 12 | Phase 2: Parallel Continuous Batching (`8 slots × 65K ctx`, per-batch CUDA graphs `B=1..8`, DSpark at `c=1`) | `c=1` **40.91 tok/s**; `c=2` **40.53 agg** (22.58/u); `c=4` **58.13 agg** (16.57/u); `c=6` **68.09 agg** (13.10/u); `c=8` **74.88 agg** (11.05/u); 0 errors; **78 GiB RAM free** | **Accepted (Live)** |

## STATUS & Remaining Levers
- **Current best:** `inkling-serve` running live on `http://10.1.1.59:8888` (`Inkling-2-Mini`, `max_seq=131072`, `draft_k=5`, 2.94 s warm startup, 85 GiB free).
- **Ranked next levers (if continuing):**
  1. **CUDA Graph capture for `T=2..8` verify & DSpark draft** — eliminates remaining host-launch overhead in speculative steps (est. +10–15% c1).
  2. **Pruned Markov head vocabulary (top 32K tokens)** — cuts 102 MB `fold_w` reads by 6× during draft steps (est. +8–12% c1).
  3. **Continuous batching & prefix caching** — for multi-user concurrent ladder (2–8 devs).
  4. **Sampling-mode DSpark verification** — extend speculative rejection sampling to `temperature > 0`.
