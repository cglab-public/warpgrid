#!/usr/bin/env python3
import concurrent.futures, json, statistics, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8888"
PROMPTS = [
    "Write a Python function to implement quicksort with comments.",
    "Write a C++20 thread-safe ring buffer class with push and pop methods.",
    "Write a Python async HTTP client with exponential backoff retries.",
    "Explain in detail how memory bandwidth affects LLM inference speed.",
    "Write a Rust binary search tree with insert and lookup methods.",
    "Describe how CUDA graphs reduce kernel launch latency on GPUs.",
    "Write a Go worker pool pattern with context cancellation.",
    "Explain why speculative decoding speeds up autoregressive generation.",
]

def run_one(idx, max_tokens=160):
    payload = {
        "model": "Inkling-2-Mini",
        "reasoning_effort": "low",
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": [{"role": "user", "content": PROMPTS[idx % len(PROMPTS)]}],
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    usage = {}
    chunks = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if (delta.get("content") or delta.get("reasoning")):
                    chunks += 1
                    if ttft is None:
                        ttft = time.perf_counter() - t0
    t1 = time.perf_counter()
    comp = usage.get("completion_tokens", 0)
    dec_t = max(1e-6, (t1 - t0) - (ttft or 0.0))
    return {
        "ttft": ttft or (t1 - t0),
        "wall": t1 - t0,
        "comp": comp,
        "tok_s": (max(0, comp - 1) / dec_t) if comp > 1 else 0.0,
    }

print("=== PARALLEL CONTINUOUS-BATCHING LADDER (c = 1, 2, 4, 6, 8) ===")
print(f"{'c':>3} | {'per-user med tok/s':>18} | {'agg tok/s':>10} | {'TTFT med ms':>11} | {'tokens':>7} | {'wall s':>7}")
print("-" * 72)

for c in [1, 2, 4, 6, 8]:
    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
        futs = [ex.submit(run_one, i, 160) for i in range(c)]
        res = [f.result() for f in futs]
    wall = time.perf_counter() - t_start
    tot_toks = sum(r["comp"] for r in res)
    agg = tot_toks / wall
    med_user = statistics.median(r["tok_s"] for r in res)
    med_ttft = statistics.median(r["ttft"] for r in res) * 1000.0
    print(f"{c:>3} | {med_user:>18.2f} | {agg:>10.2f} | {med_ttft:>11.0f} | {tot_toks:>7} | {wall:>7.2f}")
