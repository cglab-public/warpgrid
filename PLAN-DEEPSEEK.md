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
- **Real checkpoint note:** `unsloth/DeepSeek-V4.1-Flash-1017-GGUF:UD-IQ1_S` (217 GiB) is downloading to `/Users/daniel/models/DeepSeek-V4.1-Flash-1017-UD-IQ1_S`; once complete, re-run `./warpgrid --model /Users/daniel/models/DeepSeek-V4.1-Flash-1017-UD-IQ1_S/UD-IQ1_S --max-seq 32768` for live 748B serving and 32K needle-in-a-haystack verification.
