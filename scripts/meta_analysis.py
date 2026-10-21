"""Where the assistant-test errors are: accuracy by teacher confidence, per-site
ceilings, model overlap and confidence calibration, from the committed predictions.

    python3 scripts/meta_analysis.py            # prints the tables
    python3 scripts/meta_analysis.py --write    # also writes runs/evidence/ultra/meta.json

Every model is scored on the same untied targets as RESULTS.md. "All wrong" counts
decisions no model gets right; "oracle" counts decisions at least one gets right.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "runs/evidence"
MODELS = {
    "CalDec Laya": E / "caldec-laya.test.predictions.json",
    "CalDec GLiNER": E / "caldec-gliner.test.predictions.json",
    "ModernBERT-large fresh head": E / "ultra/dev-modernbert-large-short.test.predictions.json",
    "Short-context Ultra 150M": E / "ultra/dev-ultra-150m-short.test.predictions.json",
    "Short-context Ultra 380M": E / "ultra/dev-ultra-380m-short.test.predictions.json",
}
BANDS = [(0, .6), (.6, .7), (.7, .8), (.8, .9), (.9, .95), (.95, 1.01)]


def load(path):
    return {(d["case_id"], d["question"]): d for d in json.loads(path.read_text())["decisions"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    preds = {name: load(path) for name, path in MODELS.items()}
    ref = next(iter(preds.values()))
    keys = sorted(k for k, d in ref.items() if sorted(d["t"])[-1] > sorted(d["t"])[-2] + 1e-9)
    target = {k: int(np.argmax(ref[k]["t"])) for k in keys}
    correct = {n: np.array([int(np.argmax(p[k]["p"]) == target[k]) for k in keys]) for n, p in preds.items()}
    hits = np.stack(list(correct.values())).sum(0)
    tmax = np.array([max(ref[k]["t"]) for k in keys])
    site = np.array([ref[k]["site"] for k in keys])
    out = {"untied": len(keys), "all_correct": float((hits == len(preds)).mean()),
           "all_wrong": float((hits == 0).mean()), "oracle": float((hits > 0).mean()),
           "by_teacher_confidence": [], "by_site": {}, "disagreement": {}, "confidence": {}}

    print(f"{len(keys)} untied decisions; all correct {out['all_correct']:.3f}, "
          f"all wrong {out['all_wrong']:.3f}, oracle {out['oracle']:.3f}\n")
    print("Teacher max probability band: share and accuracy")
    for lo, hi in BANDS:
        m = (tmax >= lo) & (tmax < hi)
        row = {"band": [lo, min(hi, 1.0)], "n": int(m.sum()), "share": float(m.mean()),
               "accuracy": {n: float(c[m].mean()) for n, c in correct.items()},
               "all_wrong": float((hits[m] == 0).mean())}
        out["by_teacher_confidence"].append(row)
        print(f"  [{lo:.2f}, {min(hi, 1):.2f}) n={row['n']:4d} share={row['share']:.2f} "
              + " ".join(f"{v:.3f}" for v in row["accuracy"].values()) + f"  all-wrong {row['all_wrong']:.3f}")
    print("\nPer site: teacher confidence, soft-target share, all-wrong, oracle")
    for s in sorted(set(site)):
        m = site == s
        row = {"n": int(m.sum()), "teacher_max_mean": float(tmax[m].mean()),
               "share_teacher_below_0.7": float((tmax[m] < .7).mean()),
               "all_wrong": float((hits[m] == 0).mean()), "oracle": float((hits[m] > 0).mean())}
        out["by_site"][s] = row
        print(f"  {s:22s} n={row['n']:4d} tmax={row['teacher_max_mean']:.3f} "
              f"<0.7={row['share_teacher_below_0.7']:.2f} all-wrong={row['all_wrong']:.3f} oracle={row['oracle']:.3f}")
    names = list(correct)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out["disagreement"][f"{a} | {b}"] = float((correct[a] != correct[b]).mean())
    print("\nModel confidence (max predicted probability) against accuracy")
    for n, p in preds.items():
        conf = np.array([max(p[k]["p"]) for k in keys])
        rows = []
        for lo, hi in [(0, .6), (.6, .8), (.8, .9), (.9, 1.01)]:
            m = (conf >= lo) & (conf < hi)
            if m.sum():
                rows.append({"band": [lo, min(hi, 1.0)], "n": int(m.sum()),
                             "confidence": float(conf[m].mean()), "accuracy": float(correct[n][m].mean())})
        out["confidence"][n] = rows
        print(f"  {n:28s} " + "  ".join(f"conf {r['confidence']:.2f} -> acc {r['accuracy']:.2f}" for r in rows))
    if args.write:
        path = E / "ultra/meta.json"
        path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
