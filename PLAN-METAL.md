# Inkling-2-Mini Apple Metal Backend — Experiment Plan & Log

## Targets & Constraints
- **Target hardware:** Apple **M5 Max** (40 GPU cores, 18 CPU cores, **477.17 GB/s** measured copy / **465.69 GB/s** read-only, **59.68 TFLOPS** FP16 MPS, 128 GiB unified RAM, 107.52 GiB recommended max WS).
- **ISA portability:** MSL 3.1+ (Apple9+ — compatible with M3/M4/M5).
- **Model:** `thinkingmachines/Inkling-2-Mini` official weights (`/Users/daniel/models/mtplx/Youssofal--Inkling-2-Mini-MTPLX-Optimized-Quality/`, NVFP4 L00–L19 + MXFP8 L20–L23 + DSpark `mtp.safetensors`).
- **Local M5 Max baselines:**
  - `llama.cpp` PR #32694 (`IQ4_NL`): 41.0 tok/s decode, 605 / 467 tok/s prefill.
  - `oMLX 0.8.3` (`oQ4e`, quiet): 59.5 code / 57.2 prose, 1,807 / 1,783 / 1,349 tok/s prefill.
  - `MTPLX 2.13.2` (Optimized Quality, MTP): 83.3 code / 57.0 prose, 1,981 / 2,295 / 1,936 tok/s prefill.
- **Constraints:** Zero Python / PyTorch / MLX runtime dependencies; single `./warpgrid` binary via `Makefile`; preserve CUDA backend (`src/model.cu`, `src/kernels.cu`, `src/dspark.cu`) untouched; memory cap ≤ 100 GiB; noise-controlled A/B/A (`control → candidate → control`).

## Hill-Climb Ledger (`ledger-metal.jsonl`)

| Run | Story / Lever | Measured Result | % of Ceiling | Verdict |
|---|---|---|---:|---|
| `m00-hw-probe` | Story 1: M5 Max hardware probe (`bench_hw/hw_probe_metal.mm`) | 477.17 GB/s copy, 465.69 GB/s read, 59.68 TFLOPS FP16, 12.82 TFLOPS untiled BF16 | — | **Locked** |
| `m01-baseline` | Story 3: Metal v1 full 48-visit forward (`MTLResourceStorageModeShared`, `-fno-fast-math`) | Load 1.35 s; Canary `323` PASS; Tool `get_weather` PASS; SSE **33.17** code / **33.31** prose tok/s (TTFT 211 ms) | 38.7% (BF16) | **Baseline** |
