#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request


PROMPTS = [
    "Explain the difference between TCP and UDP in three concise bullet points.",
    "Write a Python function to compute the nth Fibonacci number iteratively.",
    "What is the capital of Japan and what river runs through Tokyo?",
    "Summarize the time complexity of merge sort, quicksort, and heapsort.",
    "List four key properties of ACID database transactions.",
    "Translate 'Good morning, how are you today?' into French and Spanish.",
]


def http_json(url, payload=None, timeout=600):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_ready(base_url, proc, timeout_s=600):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"warpgrid exited early with code {proc.returncode}")
        try:
            http_json(f"{base_url}/v1/models", timeout=5)
            return time.monotonic() - t0
        except Exception:
            time.sleep(1.0)
    raise TimeoutError("warpgrid server did not become ready in time")


def rss_gib(pid):
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
        return int(out) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def run_completion(base_url, prompt, max_tokens=128):
    t0 = time.monotonic()
    resp = http_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": "DeepSeek-V4.1-Flash",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
        timeout=1800,
    )
    dt = time.monotonic() - t0
    msg = resp["choices"][0]["message"]
    choice = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = resp.get("usage", {})
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    return {"text": choice, "reasoning": reasoning, "prompt_tokens": pt, "completion_tokens": ct, "elapsed_s": dt}


def run_needle(base_url, target_tokens):
    needle = "WARPGRID-NEEDLE-74819"
    filler = "The quick brown fox jumps over the lazy dog in a quiet valley. "
    approx_chars = max(1000, int(target_tokens * 4.5))
    half = approx_chars // 2
    chunk = (filler * ((half // len(filler)) + 1))[:half]
    prompt = (
        f"Read the following document carefully and find the secret code.\n\n"
        f"{chunk}\nThe secret verification code is {needle}.\n{chunk}\n\n"
        f"What is the exact secret verification code mentioned above? Reply with only the code."
    )
    res = run_completion(base_url, prompt, max_tokens=160)
    combined = res["text"] if res["text"] else res["reasoning"]
    passed = needle in res["text"] or needle in res["reasoning"]
    return {
        "target_tokens": target_tokens,
        "prompt_tokens": res["prompt_tokens"],
        "completion_tokens": res["completion_tokens"],
        "elapsed_s": round(res["elapsed_s"], 3),
        "passed": passed,
        "output": combined[:160],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="./warpgrid")
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--max-seq", type=int, default=32768)
    ap.add_argument("--max-slots", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--dspark", default="")
    ap.add_argument("--draft-k", type=int, default=0)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--needle", action="store_true")
    ap.add_argument("--needle-only", action="store_true")
    args = ap.parse_args()

    cmd = [
        args.binary,
        "--model", args.model,
        "--max-seq-len", str(args.max_seq),
        "--max-slots", str(args.max_slots),
        "--port", str(args.port),
    ]
    if args.dspark:
        cmd += ["--dspark", args.dspark]
    if args.draft_k > 0:
        cmd += ["--draft-k", str(args.draft_k)]
    base_url = f"http://127.0.0.1:{args.port}"
    log_f = open("/tmp/warpgrid_live_bench.log", "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    try:
        startup_s = wait_ready(base_url, proc, timeout_s=600)
        warm = run_completion(base_url, "Say hello.", max_tokens=16)

        tok_s_list = []
        outputs = []
        peak_rss = rss_gib(proc.pid)
        if not args.needle_only:
            for pass_i in range(max(1, args.passes)):
                pass_tok_s = []
                pass_outs = []
                for p in PROMPTS:
                    r = run_completion(base_url, p, max_tokens=args.max_tokens)
                    peak_rss = max(peak_rss, rss_gib(proc.pid))
                    tps = r["completion_tokens"] / max(r["elapsed_s"], 1e-6)
                    pass_tok_s.append(tps)
                    pass_outs.append(json.dumps({"text": r["text"], "reasoning": r["reasoning"]}, sort_keys=True))
                tok_s_list = pass_tok_s
                outputs = pass_outs

        out_hash = hashlib.sha256("\n---\n".join(outputs).encode("utf-8")).hexdigest()[:16] if outputs else ""
        sample_out = ""
        if outputs:
            first = json.loads(outputs[0])
            sample_out = (first["text"] or first["reasoning"])[:200]
        result = {
            "startup_s": round(startup_s, 2),
            "median_tok_s": round(statistics.median(tok_s_list), 2) if tok_s_list else 0.0,
            "min_tok_s": round(min(tok_s_list), 2) if tok_s_list else 0.0,
            "max_tok_s": round(max(tok_s_list), 2) if tok_s_list else 0.0,
            "peak_rss_gib": round(peak_rss, 2),
            "output_hash": out_hash,
            "sample_output": sample_out,
        }
        if args.needle or args.needle_only:
            result["needle_50pct"] = run_needle(base_url, args.max_seq // 2)
            peak_rss = max(peak_rss, rss_gib(proc.pid))
            result["needle_100pct"] = run_needle(base_url, int(args.max_seq * 0.92))
            peak_rss = max(peak_rss, rss_gib(proc.pid))
            result["peak_rss_gib"] = round(peak_rss, 2)
        print(json.dumps(result, indent=2))
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_f.close()


if __name__ == "__main__":
    main()
