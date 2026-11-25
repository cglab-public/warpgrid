# Observability (`warpgrid`)

`warpgrid` ships with built-in, zero-dependency observability across both CUDA and Apple Metal backends:

1. **Per-request structured logs (`stderr`)** — always on, one line per completed/cancelled/failed request.
2. **`GET /status`** — live JSON snapshot of active slots, KV occupancy, queue depth, cache hit ratio, throughput, and DFlash 2 acceptance.
3. **`GET /metrics`** — Prometheus / OpenMetrics text exposition (`text/plain; version=0.0.4; charset=utf-8`), directly scrapeable by **Prometheus** and **Datadog Agent (`openmetrics`)**.
4. **OpenTelemetry OTLP/HTTP Traces (`--otlp-endpoint`)** — togglable (off by default, zero overhead when unset), pushing standard OTLP/HTTP JSON spans to `/v1/traces` on a non-blocking background thread.

---

## 1. Per-Request `stderr` Logs

Every finished, cancelled, or errored request logs a single line to `stderr`:

```text
[warpgrid] req slot=0 prompt=27 cached=0 gen=46 ttft=397.0ms tps=67.21 spec=df2(mean_a=4.18) temp=0.00 finish=stop
[warpgrid] req slot=0 prompt=42 cached=27 gen=48 ttft=132.7ms tps=60.41 spec=df2(mean_a=3.62) temp=0.00 finish=length
```

| Field | Meaning |
| :--- | :--- |
| `slot` | KV slot index (`0..max_slots-1`) |
| `prompt` / `cached` / `gen` | Total prompt tokens, prefix-cache hit tokens, and generated completion tokens |
| `ttft` | Time to first token (`ms`) |
| `tps` | Per-request decode throughput (`(gen - 1) / (total - ttft)`) |
| `spec` | Speculative mode (`df2`, `dspark`, or `off`) and `mean_a` (mean tokens emitted per verify step) |
| `temp` | Request sampling temperature (`0.00` required for speculative decode) |
| `finish` | `stop`, `length`, `tool_calls`, or `cancelled` |

---

## 2. Live Status (`GET /status`)

```bash
curl -s http://localhost:8888/status | jq
# If --api-key / WARPGRID_API_KEY is enabled:
curl -s -H "Authorization: Bearer $WARPGRID_API_KEY" http://localhost:8888/status | jq
```

Returns live server metadata (`model`, `backend`, `uptime_s`, `max_slots`, `max_seq_len`, `dflash2_enabled`, `otlp_enabled`), request counters (`total`, `active`, `pending`, `cancelled`, `error`), token & prefix-cache totals, latency averages (`ttft_ms_avg`, `decode_tok_per_sec`), speculative stats (`steps`, `drafted`, `accepted`, `accept_ratio`, `mean_tokens_per_step`), and per-slot live state (`slot_id`, `active`, `pos`, `cached_tokens`, `generated`, `dflash2_active`).

---

## 3. Prometheus & Datadog (`GET /metrics`)

`GET /metrics` serves standard Prometheus / OpenMetrics text (`Content-Type: text/plain; version=0.0.4; charset=utf-8`):

| Metric | Type | Description |
| :--- | :--- | :--- |
| `warpgrid_uptime_seconds` | `gauge` | Server uptime in seconds |
| `warpgrid_requests_total` | `counter` | Total completed requests |
| `warpgrid_requests_active` | `gauge` | Currently executing requests |
| `warpgrid_pending_requests` | `gauge` | Queued requests waiting for a free slot |
| `warpgrid_requests_cancelled_total` | `counter` | Client-cancelled requests |
| `warpgrid_requests_errors_total` | `counter` | Failed / 400 requests |
| `warpgrid_active_slots` | `gauge` | Currently occupied KV slots |
| `warpgrid_max_slots` | `gauge` | Configured max concurrent KV slots |
| `warpgrid_kv_tokens_in_use` | `gauge` | Total KV tokens held by active slots (updated per prefill chunk & decode step) |
| `warpgrid_prompt_tokens_total` | `counter` | Cumulative prompt tokens processed |
| `warpgrid_prefix_cache_hit_tokens_total` | `counter` | Cumulative prompt tokens served from prefix cache |
| `warpgrid_prefix_cache_hit_ratio` | `gauge` | `cached / prompt` ratio |
| `warpgrid_generated_tokens_total` | `counter` | Cumulative completion tokens generated |
| `warpgrid_spec_steps_total` | `counter` | Speculative verify steps |
| `warpgrid_spec_drafted_tokens_total` | `counter` | Total drafted tokens |
| `warpgrid_spec_accepted_tokens_total` | `counter` | Total accepted draft tokens |
| `warpgrid_spec_accept_ratio` | `gauge` | `accepted / drafted` ratio |
| `warpgrid_spec_mean_tokens_per_step` | `gauge` | Mean tokens emitted per speculative step |
| `warpgrid_ttft_ms_avg` | `gauge` | Mean TTFT across completed requests (`ms`) |
| `warpgrid_decode_tok_per_sec` | `gauge` | Cumulative decode throughput (`(gen - 1) / decode_s`) |

### Prometheus Scrape Config

```yaml
scrape_configs:
  - job_name: warpgrid
    scrape_interval: 10s
    static_configs:
      - targets: ["10.1.1.59:8888"]
```

### Datadog Agent (`openmetrics.d/conf.yaml`)

```yaml
init_config:

instances:
  - openmetrics_endpoint: http://10.1.1.59:8888/metrics
    namespace: warpgrid
    metrics:
      - warpgrid_.*
    min_collection_interval: 15
```

---

## 4. OpenTelemetry Traces (`--otlp-endpoint`)

### How It Works

- **Togglable & Zero-Overhead When Off:** OTLP export is disabled by default. When `--otlp-endpoint` is omitted, no exporter thread is spawned and `emit()` is a single branch no-op.
- **Zero Dependencies, Non-Blocking:** Implemented in pure C++20 POSIX sockets ([src/telemetry.hpp](src/telemetry.hpp)). When enabled, a single background worker thread drains a bounded queue (256 spans; drops oldest on overflow) and sends `POST /v1/traces` with `Content-Type: application/json` and a 2-second socket timeout. Decode and HTTP responses never block on collector availability.
- **URL Parsing:** Supports `http://host:port`, `http://host:port/custom/path`, and bracketed IPv6 (`http://[::1]:4318`). If no path is given, `/v1/traces` is appended automatically.
- **Span Schema (`chat.completions`, `SPAN_KIND_SERVER`):**
  - Resource: `service.name = "warpgrid"`
  - Timing: `startTimeUnixNano`, `endTimeUnixNano`, `traceId` (128-bit hex), `spanId` (64-bit hex)
  - Attributes:
    - `gen_ai.system` (`"warpgrid"`)
    - `gen_ai.request.model`
    - `gen_ai.usage.input_tokens`
    - `gen_ai.usage.output_tokens`
    - `warpgrid.cached_tokens`
    - `warpgrid.ttft_ms`
    - `warpgrid.gen_tok_per_sec`
    - `warpgrid.spec_mean_a`
    - `warpgrid.dflash2_active`
    - `warpgrid.slot_id`
    - `warpgrid.finish_reason`
    - `warpgrid.cancelled`

### Enabling with Datadog Agent OTLP Ingest

1. Enable OTLP/HTTP in `datadog.yaml`:
   ```yaml
   otlp_config:
     receiver:
       protocols:
         http:
           endpoint: 0.0.0.0:4318
   ```
2. Launch `warpgrid` with `--otlp-endpoint`:
   ```bash
   ./warpgrid \
     --model ./Inkling-2-Mini \
     --dflash2 ./Inkling-2-Mini-DFlash2 \
     --draft-k 7 \
     --port 8888 \
     --otlp-endpoint http://127.0.0.1:4318
   ```

### Enabling with an OpenTelemetry Collector / Jaeger / Grafana Tempo

Point `--otlp-endpoint` at any OTLP/HTTP JSON receiver (port `4318` by default):

```bash
./warpgrid --model ./Inkling-2-Mini --otlp-endpoint http://otel-collector:4318
```
