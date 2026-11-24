#!/usr/bin/env python3
"""130K+ context needle-in-a-haystack + warm prefix-cache benchmark."""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8888"


def post_chat(messages, max_tokens=256):
    body = json.dumps({
        "model": "Inkling-2-Mini",
        "messages": messages,
        "temperature": 0.0,
        "reasoning_effort": "low",
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    content = []
    reasoning = []
    usage = {}
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                c = delta.get("content") or ""
                r = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if (c or r) and ttft is None:
                    ttft = time.perf_counter() - t0
                if c:
                    content.append(c)
                if r:
                    reasoning.append(r)
    t_end = time.perf_counter()
    return {
        "ttft_s": ttft if ttft is not None else (t_end - t0),
        "total_s": t_end - t0,
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "usage": usage,
    }


def main():
    # ~132,000 tokens total with 3 exact needles at 5%, 50%, and 95% depth
    filler = (
        "Telemetry log entry: node spark-e068 unified LPDDR5X fabric nominal, "
        "thermal envelope steady, cache coherence verified across memory channels, "
        "scheduler queue balanced, tensor pipeline operational. "
    )
    # Each filler line is ~40 tokens; 3300 copies ~= 132,100 tokens (>130K, <139,264)
    n1 = "NEEDLE_ONE: The launch azimuth is 274.8 degrees.\n"
    n2 = "NEEDLE_TWO: The reactor coolant code is ZIRCON-772.\n"
    n3 = "NEEDLE_THREE: The orbital docking port is BAY-19.\n"
    doc = (
        (filler * 200) + "\n" + n1
        + (filler * 1450) + "\n" + n2
        + (filler * 1450) + "\n" + n3
        + (filler * 200)
    )
    sys_msg = {"role": "system", "content": "Extract exact facts from the log concisely.\n\n" + doc}

    m1 = [sys_msg, {"role": "user", "content": "List the launch azimuth, reactor coolant code, and orbital docking port."}]
    print("Running Turn 1 (cold 130K+ prefill)...", flush=True)
    r1 = post_chat(m1, max_tokens=256)
    pt1 = r1["usage"].get("prompt_tokens", 0)
    ct1 = r1["usage"].get("completion_tokens", 0)
    c1 = r1["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
    dec_s = max(r1["total_s"] - r1["ttft_s"], 1e-3)
    full1 = r1["content"] + " " + r1["reasoning"]
    hits = sum(k in full1 for k in ["274.8", "ZIRCON-772", "BAY-19"])
    print(f"[130K Turn 1 cold] prompt={pt1} cached={c1} TTFT={r1['ttft_s']:.2f}s "
          f"({pt1 / r1['ttft_s']:.1f} prefill tok/s) decode={ct1 / dec_s:.1f} tok/s "
          f"needles={hits}/3 ans={r1['content'].strip()!r}", flush=True)
    assert pt1 >= 130000, f"Prompt too short: {pt1}"
    assert hits == 3, f"Expected 3/3 needles at 130K, got {hits}/3: {r1['content']!r}"

    # Turn 2: warm prefix-cache continuation at 130K+!
    m2 = m1 + [
        {"role": "assistant", "content": r1["content"]},
        {"role": "user", "content": "Repeat only the reactor coolant code."},
    ]
    print("Running Turn 2 (warm 130K+ prefix cache hit)...", flush=True)
    r2 = post_chat(m2, max_tokens=128)
    pt2 = r2["usage"].get("prompt_tokens", 0)
    c2 = r2["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"[130K Turn 2 warm] prompt={pt2} cached={c2} TTFT={r2['ttft_s']*1000:.1f}ms "
          f"ans={r2['content'].strip()!r}", flush=True)
    assert c2 == pt1, f"Expected cached={pt1}, got {c2}"
    assert "ZIRCON-772" in (r2["content"] + " " + r2["reasoning"]), f"Turn 2 wrong: {r2['content']!r}"
    print(f"\nPASS: 130K+ context verified! Cold TTFT={r1['ttft_s']:.2f}s -> Warm TTFT={r2['ttft_s']*1000:.1f}ms "
          f"({r1['ttft_s']/max(r2['ttft_s'],1e-4):.1f}x faster)")


if __name__ == "__main__":
    main()
