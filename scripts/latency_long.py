"""Resident-model latency by state length: per-question against packed checkpoints.

A real test case's questions are kept and its state is padded with realistic text
(other test states, concatenated) to each target length, so the measurement covers
the served `system_one` path, tokenization included. Times are wall-clock with CUDA
synchronization, after warm-up.

    USE_TF=0 python3 scripts/latency_long.py --model laya=checkpoints/v1-laya/final \\
        --model packed=checkpoints/caldec-ultra-150m --out runs/latency-long.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from data import load_agent, load_split  # noqa: E402


def padded_state(base_state, filler, tok, target):
    """`base_state` with a `context` field of filler text grown to about `target` tokens."""
    state = dict(base_state) if isinstance(base_state, dict) else {"state": base_state}
    words = filler.split()
    lo, hi = 0, len(words)
    while lo < hi:                      # the longest prefix whose state fits the target
        mid = (lo + hi + 1) // 2
        state["context"] = " ".join(words[:mid])
        if len(tok(json.dumps(state, ensure_ascii=False))["input_ids"]) <= target:
            lo = mid
        else:
            hi = mid - 1
    state["context"] = " ".join(words[:lo])
    return state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=checkpoint")
    ap.add_argument("--lengths", default="512,1024,2048,4096,8192")
    ap.add_argument("--questions", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = load_split(Path("data"), "test")
    case = next(r for r in rows if len(r["questions"]) >= args.questions)
    questions = dict(list(case["questions"].items())[:args.questions])
    filler = " ".join(json.dumps(r["state"], ensure_ascii=False) for r in rows[:600])
    out = {"case_id": case["case_id"], "site": case["site"], "questions": len(questions), "models": {}}
    for spec in args.model:
        name, path = spec.split("=", 1)
        agent = load_agent(path)
        agent.cfg["max_len"] = max(int(x) for x in args.lengths.split(",")) + 1024
        out["models"][name] = {"checkpoint": path, "packed": hasattr(agent, "agent"), "by_length": {}}
        for L in (int(x) for x in args.lengths.split(",")):
            state = padded_state(case["state"], filler, agent.tok, L)
            for _ in range(args.warmup):
                agent.system_one(state, questions)
            times = []
            for _ in range(args.calls):
                torch.cuda.synchronize()
                t = time.perf_counter()
                agent.system_one(state, questions)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t) * 1000)
            times.sort()
            row = {"p50_ms": round(statistics.median(times), 1),
                   "p95_ms": round(times[min(len(times) - 1, int(0.95 * len(times)))], 1)}
            out["models"][name]["by_length"][str(L)] = row
            print(f"{name:12s} {L:5d} tokens  p50 {row['p50_ms']:7.1f} ms  p95 {row['p95_ms']:7.1f} ms", flush=True)
        del agent
        torch.cuda.empty_cache()
    out["gpu"] = torch.cuda.get_device_name(0)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
