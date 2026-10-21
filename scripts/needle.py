"""A long-state probe built from existing labels: `tools.injection` pages at 4K and 8K tokens.

`injection` asks whether fetched text contains instructions addressed to an AI agent.
Text made of chunks the teacher rated clean, plus one test case's text, carries that
test case's label: clean filler adds no injection, and the case's own text decides.
So each untied `tools.injection` test case becomes a long page with its text placed at
the start, the middle or the end of filler drawn from confidently clean TRAIN cases
(teacher P(injection) < 0.05), and the case's own target is kept.

This is a probe of reading position and length, not a benchmark of real long pages:
real pages are not stitched from unrelated chunks. The long-context builds in
builds/long*.json are the real test.

    python3 scripts/needle.py --out data-needle      # writes data-needle/test.jsonl
    USE_TF=0 python3 scripts/evaluate.py --data data-needle --checkpoint <ckpt> --max-len 9000
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import encode, load_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHARS_PER_TOKEN = 3.9      # JSON states under the UltraBERT tokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lengths", default="4000,8000", help="target state lengths in tokens")
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    train = [r for r in load_split(ROOT / "data", "train") if r["site"] == "tools.injection"]
    clean = [r["state"]["text"] for r in train
             if r["gold"]["injection"]["probabilities"]["true"] < 0.05]
    test = [r for r in load_split(ROOT / "data", "test") if r["site"] == "tools.injection"]
    rows = []
    for length in (int(x) for x in args.lengths.split(",")):
        budget = int(length * CHARS_PER_TOKEN)
        for case in test:
            p = case["gold"]["injection"]["probabilities"]
            if abs(p["true"] - p["false"]) < 1e-9:
                continue
            for where in ("start", "middle", "end"):
                filler, size = [], 0
                while size < budget - len(case["state"]["text"]):
                    chunk = rng.choice(clean)
                    filler.append(chunk)
                    size += len(chunk) + 2
                cut = {"start": 0, "middle": len(filler) // 2, "end": len(filler)}[where]
                text = "\n\n".join(filler[:cut] + [case["state"]["text"]] + filler[cut:])
                rows.append(dict(case_id=f"needle-{length}-{where}-{case['case_id']}", site=case["site"],
                                 workflow=case["workflow"], state={**case["state"], "text": text},
                                 questions=case["questions"], gold=case["gold"]))
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("sites.json", "prompts.json", "train.jsonl", "validation.jsonl"):
        shutil.copy(ROOT / "data" / name, args.out / name)
    (args.out / "test.jsonl").write_text("".join(json.dumps(encode(r), ensure_ascii=False) + "\n" for r in rows),
                                         encoding="utf-8")
    print(f"{len(rows)} rows from {len(test)} test cases and {len(clean)} clean filler chunks -> {args.out / 'test.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
