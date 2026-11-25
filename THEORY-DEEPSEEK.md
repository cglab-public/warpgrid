# DeepSeek-V4.1-Flash-1017 Theoretical Ceilings — Apple M5 Max (128 GiB)

## 1. Measured Hardware Bounds (Apple M5 Max, 128 GiB Unified Memory)
- **Metal recommended max GPU working set:** `107.52 GiB` (`115.45 GB`)
- **Unified memory bandwidth:** `465.69 GB/s` (Metal read-only microbenchmark, `THEORY-METAL.md`) / `527 GB/s` (MLX 2 GiB FP32 copy, median of 10)
- **Peak BF16 GEMM compute:** `59.6 TFLOPS` (`8192^3` measured)

## 2. Exact Checkpoint Profile (`unsloth/DeepSeek-V4.1-Flash-1017-GGUF:UD-IQ1_S`)
Parsed directly from the 5-shard GGUF v3 headers (`split.tensors.count = 1043`):

| Component | Shards | Quant Types | Size (GiB) | Placement |
|---|---|---|---:|---|
| Metadata + Tokenizer | Shard 1 (`00001-of-00005`) | — | 0.01 | Host RAM |
| Trunk + Shared Experts + Routed Experts + Embed + LM Head | Shards 2–4 (`00002`–`00004`) | `Q6_K` (220), `Q5_K` (65), `Q8_0` (39), `BF16` (27), `F32` (571), `IQ1_S` (88), `IQ2_XXS` (32) | **113.98** | Zero-copy `mmap` Metal (`newBufferWithBytesNoCopy`) |
| Engram Conditional Memory (`per_layer_token_embd.weight`) | Shard 5 (`00005-of-00005`) | `IQ4_NL` (1) | **103.00** | NVMe `mmap` (`MADV_RANDOM`) + hot-row LRU cache |
| **Total** | **5 shards** | — | **216.98** | — |

### Architecture Dimensions (`deepseek41`)
- Layers: `40` (`block_count = 40`)
- Hidden size: `D = 5120`, mHC streams: `hc_mult = 4` (`20480` flat), `sinkhorn_iterations = 20`
- Shared-KV MQA Attention: `n_heads = 64`, `n_kv_heads = 1`, `head_dim = 512`, `n_rot = 64`, `q_lora_rank = 1280`, `o_group_count = 8`, `o_lora_rank = 1024`, `sliding_window = 128`
- CSA/HCA + Lightning Indexer: `compress_ratios` = `[0,0, 2×18, 1×20]`, `kv_source_layers = [2,8,14,20]`, `indexer.source_layers = [2,8,14,20,24,28,32,36]`, `index_heads = 32`, `index_dim = 128`, `index_topk = 512`
- MoE FFN: `384` routed experts (`top-6`), `expert_feed_forward_length = 2304`, `1` shared expert (`2304`), `sqrtsoftplus` router (`expert_gating_func = 4`), `route_scale = 1.5`, `swiglu_clamp = 10.0`
- Engram: `ple.layers = [1, 14]`, `ngram_size = 4`, `heads_per_ngram = 8` (`24` heads/layer), `ple_head_dim = 256`
- Vocab: `129,280` (`token_embd.weight` & `output.weight` both `Q6_K`)

## 3. Bandwidth-Bound Decode Ceiling (`c = 1`)
Active bytes read per decoded token:
- **Dense trunk (40 layers):** attention (`wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`), CSA/HCA compressor + indexer, mHC projections, shared experts (`gate`, `up`, `down`), routers ≈ `3.95 GB`
- **Active routed experts (`6 / 384` of `~108 GiB` = `116 GB`):** `116 / 64 = 1.81 GB`
- **LM head (`129280 × 5120`, `Q6_K`):** `129280 × (5120/256) × 210 = 0.543 GB`
- **KV cache (shared-KV MQA, `n_kv_heads = 1`, `head_dim = 512`):** `40 × 512 × 4 B = 81.9 KB/tok` (FP32) — at 32K context, sliding window `128` + `512` selected compressed blocks reads only `~40 × 640 × 512 × 4 B = 52 MB/tok` (`< 1%` of total traffic).
- **Total active bytes/token:** `~6.35 GB/tok`

### Theoretical `c = 1` Decode Ceilings
| Regime | Formula | Ceiling |
|---|---|---:|
| Ideal fully-resident (Metal read-only `465.69 GB/s`) | `465.69 / 6.35` | **73.3 tok/s** (`13.6 ms/tok`) |
| Ideal fully-resident (MLX copy `527 GB/s`) | `527 / 6.35` | **83.0 tok/s** (`12.0 ms/tok`) |
| Realistic NVMe-streamed (`113.98 GiB > 107.52 GiB` working set + `48` Engram `IQ4_NL` random lookups/tok) | `+3–5 ms/tok` NVMe tail | **53–60 tok/s** (`16.6–18.8 ms/tok`) |

## 4. Compute-Bound Prefill Ceiling
- Active FLOPs/token ≈ `2 × 16B = 32 GFLOP/tok`
- At `59.6 TFLOPS` peak (`50%` MFU = `29.8 TFLOPS` practical): **`~931 tok/s`** practical prefill ceiling.

## 5. Primary Bottlenecks Identified in Story 4 Baseline
1. **Duplicate RAM copy on load:** `newBufferWithBytes:` copied 114 GiB into anonymous RAM -> fixed via zero-copy `newBufferWithBytesNoCopy:` on `mmap`'d shards.
2. **84 CPU-GPU syncs per token:** per-layer router CPU readback + Engram syncs -> fixed via single-command-buffer GPU-side `gguf_moe_indexed_*` / `gguf_moe_down_accum_*` + pre-fetched Engram (`84 -> 1` sync).
3. **Grouped `wo_a` dispatch overhead:** `8 × T` separate `encode_gemv` calls per layer (`320` dispatches/tok at `T=1`) -> candidate for fused `gguf_grouped_gemv_*`.
4. **Separate `gate` + `up` + `swiglu` routed MoE dispatches:** 3 dispatches + 2 full intermediate buffer round-trips per layer -> candidate for fused `gguf_moe_gate_up_swiglu_*`.
5. **Scalar `X` loads & single-row-per-SIMDgroup GEMV:** candidate for `float4` vectorized `X` loads and 2-rows-per-SIMDgroup register reuse in `IQ1_S`, `IQ2_XXS`, `Q6_K`, `Q5_K`.
6. **CPU logits `memcpy` + argmax in `forward_and_greedy`:** `517 KB` copy per token -> candidate for in-CB GPU `ds_argmax_f32`.
