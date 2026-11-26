# DeepSeek-V4.1-Flash-1017 Hill-Climb Plan & Log (Apple M5 Max)

Hardware: Apple M5 Max (128 GiB unified RAM, 107.52 GiB Metal working set, 465.69 GB/s Metal read-only BW).
Harness: `tests/bench_deepseek_metal --layers 4 --experts 16 --samples 7 --steps 16` (exact `deepseek41` layer widths: `D=5120`, `H=64`, `HD=512`, `q_lora=1280`, `o_groups=8`, `o_lora=1024`, `hc=4`, `top_k=6`, `Inter=2304`, `IQ1_S`/`IQ2_XXS`/`Q6_K`/`Q5_K`/`BF16`, zero-copy `mmap`, single-CB GPU MoE).

## Experiment Ledger
| ID | Change | Med ms/tok (4L) | Proj 40L tok/s | Delta vs Prev | RSS (GiB) | Greedy Parity | Decision |
|---|---|---:|---:|---:|---:|---|---|
| e00 | Control baseline (zero-copy mmap + single-CB GPU MoE, 4 SIMD/TG, 1 row/SIMD) | 7.323 | 13.66 | — | 0.300 | PASS (`tok=5`) | BASELINE |
| e01 | Small-N `gguf_gemv_f32_wide` (256 threads/row + `float4` for `N<=64`, fixes `hc_attn_fn`/`hc_ffn_fn`) | 5.954 | 16.80 | **-18.7%** (`+23.0%` tok/s) | 0.300 | PASS (`tok=5`) | **KEEP** |
| e02 | Naive 4-row-per-SIMD `gguf_gemv_q6_k_4r` (still 1 super-block/iter + inner half loop) | 5.919 | 16.89 | -0.6% (noise) | 0.301 | PASS (`tok=5`) | REVERT |
| e03 | `gguf_gemv_q6_k_fast` (2 super-blocks/SIMD, 2 rows/SIMD, `float4` X loads, 64t/TG) | 5.306 | 18.85 | **-10.9%** (`+12.2%` tok/s) | 0.300 | PASS (`tok=5`) | **KEEP** |
| e04 | `ib32 = lane` + `float4` X loads + `sumy` factorization in `IQ1_S`/`IQ2_XXS` | 5.182 | 19.30 | **-2.3%** (`+2.4%` tok/s) | 0.299 | PASS (`tok=5`) | **KEEP** |
| e05 | Fused `gguf_moe_gate_up_swiglu_{iq1_s,iq2_xxs}` + 4-row `down_accum_4r` (`+41.6%` microbench) | 5.159 | 19.38 | **-0.4%** (4L) / `+41.6%` MoE | 0.301 | PASS (`tok=5`) | **KEEP** |
| e06 | 256-thread `float4` `ds_rmsnorm_f32` & `ds_rmsnorm_unweighted_f32` (fixes 20,480-dim mHC norm) | 3.805 | 26.28 | **-26.2%** (`+35.6%` tok/s) | 0.300 | PASS (`tok=5`) | **KEEP** |
| e07 | Batched `gguf_grouped_gemv_q6_k_fast` for `wo_a` (`8*T` dispatches -> `1` per layer) | 3.761 | 26.59 | **-1.2%** (`+1.2%` tok/s) | 0.301 | PASS (`tok=5`) | **KEEP** |
| e08 | In-CB 256-thread GPU `ds_argmax_f32` in `forward_and_greedy` (eliminates 517 KB logits copy on 129K vocab) | 3.758 | 26.61 | **-0.1%** (1K vocab) / avoids 517 KB copy on 129K | 0.299 | PASS (`tok=5`, `argmax_parity=true`) | **KEEP** |

## Summary
- **Total speedup vs e00 baseline:** `7.323 ms -> 3.758 ms` (**1.95x faster**, `+94.8%` tok/s, `13.66 -> 26.61 tok/s` projected 40L; zero-copy `mmap` load in `~40 ms`, `0.30 GiB` RSS).
- **Exact parity:** 42/42 unit tests pass (`test_host` 15/15, `test_metal_kernels` 12/12 Inkling untouched, `test_metal_gguf_kernels` 15/15), `greedy_deterministic: true`, `argmax_parity: true`.
- **Real checkpoint note (COMPLETE — 217 GiB `unsloth/DeepSeek-V4.1-Flash-1017-GGUF:UD-IQ1_S`):** Downloaded all 5 shards to `/Users/daniel/models/DeepSeek-V4.1-Flash-1017-UD-IQ1_S/UD-IQ1_S` and completed live 748B serving, hill-climbing, `llama-cli` parity verification, and 50% (16K) / 100% (32K) needle-in-a-haystack stability gates via `scripts/bench_deepseek_live.py`.

## Live 748B Hill-Climb Ledger (`UD-IQ1_S`, 217 GiB, `--max-seq 32768 --max-slots 1`)
| ID | Change | Startup | Warm Decode Peak | 16K / 30K Decode | Peak RSS | Output Hash / Parity | Decision |
|---|---|---:|---:|---:|---:|---|---|
| live-e00 | Per-tensor `newBufferWithBytesNoCopy:` + on-demand top-6 `moe_stream=ON` (fixes 114 GiB IOGPU wiring stall) | 1.01 s | 8.66 tok/s (6.57 med e2e) | — | 74.11 GiB | `92e9647ad973216b` (`323` parity vs `llama-cli`) | BASELINE |
| live-e01 | Budget-aware hybrid MoE residency (`moe_res=23/40` direct GPU, tail streamed) | 1.01 s | 11.48 tok/s (7.15 med e2e) | — | 37.39 GiB | `92e9647ad973216b` (exact) | **KEEP** |
| live-e02 | Vectorized 2-row-per-SIMD `gguf_gemv_{q8_0,q5_k,bf16}_fast` Metal kernels | 1.02 s | 12.29 tok/s (`+7.1%` P5, `+19.9%` P6) | — | 37.23 GiB | `92e9647ad973216b` (exact) | **KEEP** |
| live-e03 | 1-pass online FlashAttention (`float4` register-cached Q/acc) for `ds_mqa_sink_{,comp_}attn_f32` + 256-thread indexer | 1.01 s | 12.56 tok/s | 0.44 / 0.40 tok/s (2048-barrier bottleneck) | 36.78 GiB | `92e9647ad973216b` (exact) | **KEEP** |
| live-e04 | Binary-search threshold top-K in `ds_indexer_score_topk_f32` (24 iterations, 0 device barriers) | 1.01 s | 12.92 tok/s | 0.42 tok/s (staging leak at T=64) | 25.25 GiB | `92e9647ad973216b` (exact) | **KEEP** |
| live-e05 | Live `host_statistics64` OOM-safe MoE budget (`moe_res=13/40`) + `@autoreleasepool` on worker thread + fixed `<=16`-expert (`124 MiB` max) chunked staging | 1.02 s | 11.39 tok/s (zero leak, swap -1.3 GiB) | **8.81 tok/s @ 16.4K** (`21.0x`), **7.54 tok/s @ 30.2K** (`18.9x`), `44.1 tok/s` prefill | 64.35 GiB (`<107.52 GiB`) | `92e9647ad973216b`, **16K & 30K needle PASS (`WARPGRID-NEEDLE-74819`)** | **KEEP** |
| live-e06 | `gguf_gemv_q6_k_smallk` (2.18x `wq_b`: `146.6 -> 319.4 GB/s`) + `half4` 2-row `IQ1_S` + tiled `simdgroup_matrix<half,8,8>` `gguf_gemm_{q6_k,q5_k,q8_0}_tiled` prefill (`+31%` to `57.7 tok/s`) + 0-byte in-place prefix cache (`cached=1984/2014`, **17.4x** Turn-2 TTFT `47.0s -> 2.70s` = **`744.9 tok/s`** effective) | 1.02 s | 11.24 tok/s | `57.7 tok/s` raw / **`744.9 tok/s`** cached prefill | ~64 GiB | 43/43 tests (`test_gguf_gemm_tiled_and_smallk_parity`), exact `323` parity | **KEEP** |
| live-e07 | `F_NOCACHE` `pread` + GPU-resident Expert-Level LRU Cache (`5694/15360` slots, `48.4 GiB`, zero `mmap` page-cache duplication for the 107 GiB MoE shards) | 1.01 s | **15.81 tok/s** (median e2e **12.12 tok/s**, **2.08x** vs `live-e06` `5.84`) | — | **46.69 GiB** (`-17.7 GiB` vs `live-e06`, 0 swap) | 43/43 tests, bit-exact `bca2765e13829096` + `323` parity | **KEEP** |
| live-e08 | DeepSeek-V4.1 DSpark speculative decoding (`dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf`, 3L MXFP4 + BF16 Markov head + sliding-window non-causal MQA + `gguf_gemv_{q6_k,q5_k,q8_0}_multi` register-resident multi-token GEMV) | 1.01 s | **17.44 tok/s** (`mean_a=2.56–2.91`, median e2e **13.04 tok/s**, **2.23x** vs `live-e06`) | — | **39.31 GiB** (`-25.0 GiB` vs `live-e06`, 0 swap) | 44/44 tests (`test_ds_dspark_and_multi_gemv_parity`), exact `323` parity | **KEEP** |
| live-e09 | DSpark Metal kernel fusion (`gguf_grouped_gemv_q8_0_fast` for 8-group `wo_a`, fused `gguf_moe_gate_up_swiglu_mxfp4` + `gguf_moe_down_accum_mxfp4_4r`, FP32 `IQ1_S` sub-block accumulation; reverted 1-row grouped `IQ1_S` after A/B regression `13.01 -> 11.94 tok/s`) | 2.01 s | **16.10 tok/s** (median e2e **13.01 tok/s**, Prompt 3 `+10.1%`, Prompt 6 `+7.1%`) | — | **39.55 GiB** (0 swap) | 45/45 tests (`test_ds_grouped_moe_and_mxfp4_fused_parity`) | **KEEP** |
| live-e10 | DeepSeek-V4.1 `--profile` stage profiler + `ExpertUsageProfile` pre-warm (`WARPGRID_DS_EXPERT_PROFILE`) + parallel Engram `dispatch_apply` (`14.41ms -> 0.19ms`, `75x`) + async shared-expert GPU / CPU `exp_pread` overlap | 1.01 s | **16.45 tok/s** (median e2e **13.75 tok/s**, **2.35x** vs `live-e06`; Prompt 1 `+31.1%` to `11.63 tok/s`) | — | **44.38 GiB** (0 swap) | 46/46 tests (`test_deepseek_stage_profile_and_expert_usage_roundtrip`), bit-exact `cc149606a28bfb16` | **KEEP** |
