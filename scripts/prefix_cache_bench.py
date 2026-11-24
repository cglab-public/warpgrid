#!/usr/bin/env python3
"""Verify exact prefix caching (multi-turn stripped-reasoning prefill snapshot + full continuation)."""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8888"


def post_chat(messages, max_tokens=512, stream=True):
    body = json.dumps({
        "model": "thinkingmachines/Inkling-2-Mini",
        "messages": messages,
        "temperature": 0.0,
        "reasoning_effort": "low",
        "max_tokens": max_tokens,
        "stream": stream,
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
    usage = None
    with urllib.request.urlopen(req, timeout=180) as resp:
        if not stream:
            data = json.loads(resp.read().decode())
            t_end = time.perf_counter()
            msg = data["choices"][0]["message"]
            return {
                "ttft_ms": (t_end - t0) * 1000.0,
                "total_ms": (t_end - t0) * 1000.0,
                "content": msg.get("content", ""),
                "reasoning": msg.get("reasoning") or msg.get("reasoning_content") or "",
                "usage": data.get("usage", {}),
            }
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
                    ttft = (time.perf_counter() - t0) * 1000.0
                if c:
                    content.append(c)
                if r:
                    reasoning.append(r)
    t_end = time.perf_counter()
    return {
        "ttft_ms": ttft if ttft is not None else (t_end - t0) * 1000.0,
        "total_ms": (t_end - t0) * 1000.0,
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "usage": usage or {},
    }


def main():
    # ~4,000-token shared system/context document with two distinct facts.
    filler = (
        "The Horizon archival ledger records telemetry across distributed inference clusters, "
        "noting memory bandwidth, cache residency, KV page alignment, and thermal margins. "
    )
    doc = (
        "FACT ALPHA: The vault passkey for sector 7 is COBALT-941.\n"
        + (filler * 90)
        + "\nFACT BETA: The backup relay frequency is 418.75 MHz.\n"
        + (filler * 40)
    )
    sys_msg = {"role": "system", "content": "Answer in 5 words or fewer using only the document.\n\n" + doc}

    # Turn 1 (cold prefill)
    m1 = [sys_msg, {"role": "user", "content": "What is the vault passkey for sector 7?"}]
    r1 = post_chat(m1, max_tokens=512, stream=True)
    p1 = r1["usage"].get("prompt_tokens", 0)
    c1 = r1["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"[Turn 1 cold]       prompt={p1} cached={c1} ttft={r1['ttft_ms']:.1f}ms ans={r1['content'].strip()!r}")
    assert c1 == 0, f"Expected cold cached_tokens==0, got {c1}"
    assert "COBALT-941" in r1["content"], f"Turn 1 wrong answer: {r1['content']!r}"

    # Turn 2: multi-turn continuation (assistant content without reasoning -> hits prefill_snap!)
    m2 = m1 + [
        {"role": "assistant", "content": r1["content"]},
        {"role": "user", "content": "Now what is the backup relay frequency?"},
    ]
    r2 = post_chat(m2, max_tokens=512, stream=True)
    p2 = r2["usage"].get("prompt_tokens", 0)
    c2 = r2["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"[Turn 2 warm snap]  prompt={p2} cached={c2} ttft={r2['ttft_ms']:.1f}ms ans={r2['content'].strip()!r}")
    assert c2 == p1, f"Expected Turn 2 cached_tokens=={p1}, got {c2}"
    assert "418.75" in r2["content"], f"Turn 2 wrong answer: {r2['content']!r}"

    # Turn 3: non-streaming multi-turn continuation (hits Turn 2 prefill_snap!)
    m3 = m2 + [
        {"role": "assistant", "content": r2["content"]},
        {"role": "user", "content": "Repeat both the passkey and the frequency."},
    ]
    r3 = post_chat(m3, max_tokens=512, stream=False)
    p3 = r3["usage"].get("prompt_tokens", 0)
    c3 = r3["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"[Turn 3 non-stream] prompt={p3} cached={c3} total={r3['total_ms']:.1f}ms ans={r3['content'].strip()!r}")
    assert c3 == p2, f"Expected Turn 3 cached_tokens=={p2}, got {c3}"
    assert "COBALT-941" in r3["content"] and "418.75" in r3["content"], f"Turn 3 wrong: {r3['content']!r}"

    speedup = r1["ttft_ms"] / max(r2["ttft_ms"], 1e-3)
    print(f"\nPASS: Prefix caching verified! Cold TTFT={r1['ttft_ms']:.1f}ms -> Warm TTFT={r2['ttft_ms']:.1f}ms ({speedup:.1f}x faster)")


if __name__ == "__main__":
    main()
