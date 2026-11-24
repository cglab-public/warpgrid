#!/usr/bin/env python3
import json, statistics, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8888"

def post_json(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())

def stream_chat(payload):
    payload = dict(payload, stream=True)
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    content = []
    reasoning = []
    usage = {}
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if "usage" in obj and obj["usage"]:
                usage = obj["usage"]
            choices = obj.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if (delta.get("content") or delta.get("reasoning")) and ttft is None:
                    ttft = time.perf_counter() - t0
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning"):
                    reasoning.append(delta["reasoning"])
    t1 = time.perf_counter()
    comp = usage.get("completion_tokens", 0)
    prompt = usage.get("prompt_tokens", 0)
    dec_time = max(1e-6, (t1 - t0) - (ttft or 0.0))
    tok_s = (max(0, comp - 1) / dec_time) if comp > 1 else 0.0
    return {
        "ttft": ttft or (t1 - t0),
        "total_s": t1 - t0,
        "prompt_tokens": prompt,
        "completion_tokens": comp,
        "decode_tok_s": tok_s,
        "content": "".join(content),
        "reasoning": "".join(reasoning),
    }

print("=== 1. HEALTH & MODELS ===")
with urllib.request.urlopen(BASE + "/health", timeout=10) as r:
    print("health:", r.read().decode())
with urllib.request.urlopen(BASE + "/v1/models", timeout=10) as r:
    print("models:", r.read().decode())

print("\n=== 2. CANARY (17 * 19 = 323) ===")
res = post_json("/v1/chat/completions", {
    "model": "Inkling-2-Mini",
    "reasoning_effort": "low",
    "temperature": 0.0,
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "What is 17 * 19? Reply with just the number."}],
})
msg = res["choices"][0]["message"]
print("content:", repr(msg.get("content")), "usage:", res.get("usage"))
assert "323" in (msg.get("content") or ""), "Canary failed!"
print("PASS: Canary 323")

print("\n=== 3. TOOL CALL (get_weather) ===")
res = post_json("/v1/chat/completions", {
    "model": "Inkling-2-Mini",
    "reasoning_effort": "low",
    "temperature": 0.0,
    "max_tokens": 256,
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
    "messages": [{"role": "user", "content": "What is the weather in Tokyo right now? Call get_weather."}],
})
msg = res["choices"][0]["message"]
print("tool_calls:", msg.get("tool_calls"))
assert msg.get("tool_calls") and msg["tool_calls"][0]["function"]["name"] == "get_weather", "Tool call failed!"
print("PASS: Tool call get_weather")

print("\n=== 4. DECODE PANEL (6 samples: 3 code, 3 prose, SSE usage stats) ===")
code_prompts = [
    "Write a Python function to implement quicksort with comments.",
    "Write a C++20 thread-safe ring buffer class with push and pop methods.",
    "Write a Python async HTTP client with exponential backoff retries.",
]
prose_prompts = [
    "Explain in detail how memory bandwidth affects LLM inference speed.",
    "Describe the history and architecture of modern GPU tensor cores.",
    "Explain why speculative decoding speeds up autoregressive generation.",
]
code_speeds, prose_speeds = [], []
for idx, p in enumerate(code_prompts):
    r = stream_chat({
        "model": "Inkling-2-Mini", "reasoning_effort": "low", "temperature": 0.0,
        "max_tokens": 200, "messages": [{"role": "user", "content": p}],
    })
    code_speeds.append(r["decode_tok_s"])
    print(f"  code[{idx}]: {r['decode_tok_s']:.2f} tok/s ({r['completion_tokens']} tok, TTFT {r['ttft']*1000:.0f} ms)")
for idx, p in enumerate(prose_prompts):
    r = stream_chat({
        "model": "Inkling-2-Mini", "reasoning_effort": "low", "temperature": 0.0,
        "max_tokens": 200, "messages": [{"role": "user", "content": p}],
    })
    prose_speeds.append(r["decode_tok_s"])
    print(f"  prose[{idx}]: {r['decode_tok_s']:.2f} tok/s ({r['completion_tokens']} tok, TTFT {r['ttft']*1000:.0f} ms)")
print(f"MEDIAN code: {statistics.median(code_speeds):.2f} tok/s | MEDIAN prose: {statistics.median(prose_speeds):.2f} tok/s")

print("\n=== 5. LONG-CONTEXT NEEDLE GATE (~16K tokens, 3 exact needles) ===")
filler = "The quick brown fox jumps over the lazy dog while servers process telemetry logs steadily. " * 120
doc = (
    "Document start.\n" + filler +
    "\nCRITICAL CODE ALPHA: 7491-ZETA-82.\n" + filler * 2 +
    "\nCRITICAL CODE BETA: 3820-ORION-55.\n" + filler * 2 +
    "\nCRITICAL CODE GAMMA: 9104-LYRA-19.\n" + filler +
    "\nQuestion: List the three CRITICAL CODE values (ALPHA, BETA, GAMMA) verbatim."
)
r = stream_chat({
    "model": "Inkling-2-Mini", "reasoning_effort": "low", "temperature": 0.0,
    "max_tokens": 160, "messages": [{"role": "user", "content": doc}],
})
print(f"  prompt_tokens={r['prompt_tokens']}, TTFT={r['ttft']:.2f} s ({r['prompt_tokens']/r['ttft']:.1f} tok/s), decode={r['decode_tok_s']:.2f} tok/s")
print("  answer:", repr(r["content"][:300]))
found = sum(1 for k in ("7491-ZETA-82", "3820-ORION-55", "9104-LYRA-19") if k in (r["content"] + r["reasoning"]))
print(f"  Needles found: {found}/3")
assert found == 3, f"Only found {found}/3 needles!"
print("PASS: Long-context needle gate (3/3 exact)")
