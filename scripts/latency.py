"""Measure resident CalDec checkpoint latency on one assistant-test case.

    USE_TF=0 python3 scripts/latency.py laya --checkpoint checkpoints/v1-laya/final
    USE_TF=0 python3 scripts/latency.py gliner --checkpoint checkpoints/v1-gliner/final

The output is a single-case microbenchmark, not throughput or a deployment SLA.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import torch  # noqa: E402

from data import load_split  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("laya", "gliner"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--out", type=Path, help="save the measurement as JSON")
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1:
        parser.error("warmup must be nonnegative and samples must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")

    row = next(r for r in load_split(Path(__file__).resolve().parents[1] / "data", "test")
               if len(r["questions"]) >= 3)
    if args.model == "laya":
        import laya

        agent = laya.load(args.checkpoint, device=args.device)

        def call(count):
            return agent.system_one(row["state"], dict(list(row["questions"].items())[:count]))
    else:
        from gliner2_transfer import build_tasks, from_pretrained, render, schema_for

        model = from_pretrained(args.checkpoint)
        if args.device == "cuda":
            model.cuda()
        state, tasks = render(row["state"]), build_tasks(row)

        def call(count):
            return model.classify_text(state, schema_for(tasks[:count]),
                                       include_confidence=True, format_results=False)

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    results = {}
    for count in (1, 3):
        with torch.inference_mode():
            for _ in range(args.warmup):
                call(count)
            times = []
            for _ in range(args.samples):
                sync()
                start = time.perf_counter()
                call(count)
                sync()
                times.append((time.perf_counter() - start) * 1000)
        times.sort()
        results[f"{count}q"] = {
            "p50_ms": round(statistics.median(times), 1),
            "p95_ms": round(times[math.ceil(0.95 * len(times)) - 1], 1),
        }
    report = {"model": args.model, "checkpoint": args.checkpoint,
              "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
              "case_id": row["case_id"], "warmup": args.warmup,
              "samples": args.samples, "results": results}
    payload = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
