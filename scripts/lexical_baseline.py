"""A surface-feature baseline: hashed bag-of-words logistic regression, one head per question.

The point is to show how much of the encoder's score comes from vocabulary alone.
No tuning, one configuration, CPU only.

    python scripts/lexical_baseline.py --data data --out runs/lexical.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_split                                                        # noqa: E402
from evaluate import QTYPE_INDEX, baselines, data_fingerprint, metrics, serialise, target_vector  # noqa: E402

BUCKETS = 2 ** 16
TOKEN = re.compile(r"[a-z0-9]+")


def render(value) -> str:
    if isinstance(value, dict):
        return " ".join(f"{k} {render(v)}" for k, v in value.items())
    if isinstance(value, list):
        return " ".join(render(v) for v in value)
    return "" if value is None else str(value)


def features(text: str) -> List[int]:
    """Hashed unigram and bigram ids. Kept sparse until a head is fitted."""
    tokens = TOKEN.findall(text.lower())
    grams = tokens + [a + "_" + b for a, b in zip(tokens, tokens[1:])]
    return [zlib.crc32(gram.encode()) % BUCKETS for gram in grams]


def dense(items: List[Dict]) -> torch.Tensor:
    x = torch.zeros(len(items), BUCKETS)
    for row, item in enumerate(items):
        for bucket in item["x"]:
            x[row, bucket] += 1.0
    norm = x.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return x / norm


def decisions(rows: List[Dict]) -> Dict[tuple, List[Dict]]:
    out: Dict[tuple, List[Dict]] = defaultdict(list)
    for row in rows:
        text = render(row["state"])
        for qid, question in row["questions"].items():
            gold = (row.get("gold") or {}).get(qid)
            if not gold or "probabilities" not in gold:
                continue
            target = target_vector(question, gold)
            if target is None:
                continue
            out[(row["site"], qid)].append({
                "case_id": row["case_id"], "question": qid, "site": row["site"],
                "qtype": QTYPE_INDEX[question["type"]], "x": features(text), "t": target,
            })
    return out


def fit(items: List[Dict], k: int, epochs: int = 200) -> torch.nn.Linear:
    x = dense(items)
    t = torch.tensor(np.stack([it["t"] for it in items]), dtype=torch.float)
    head = torch.nn.Linear(BUCKETS, k)
    optimiser = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        optimiser.zero_grad()
        loss = -(t * torch.log_softmax(head(x), -1)).sum(-1).mean()   # soft cross-entropy
        loss.backward()
        optimiser.step()
    return head


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--build", default=None, help="keep one build's rows (original, a generate.py build's name, or a comma list)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--predictions")
    args = parser.parse_args(argv)
    torch.manual_seed(0)

    root = Path(args.data)
    load = lambda name: load_split(root, name, args.build)
    train, test = load("train"), load("test")
    train_by, test_by = decisions(train), decisions(test)

    scored = []
    for key, items in sorted(test_by.items()):
        fitted = train_by.get(key)
        if not fitted:
            continue
        head = fit(fitted, len(items[0]["t"]))
        with torch.no_grad():
            probs = torch.softmax(head(dense(items)), -1).numpy()
        for item, p in zip(items, probs):
            scored.append({**{k: item[k] for k in ("case_id", "question", "site", "qtype", "t")},
                           "p": p.astype(float)})

    overall = metrics(scored)
    by_site: Dict[str, List[Dict]] = defaultdict(list)
    for row in scored:
        by_site[row["site"]].append(row)
    result = {"overall": overall, "baselines": baselines(train, scored),
              "per_site": {s: metrics(g) for s, g in sorted(by_site.items())}}
    print(json.dumps({"overall": overall, "baselines": result["baselines"]}, indent=1))
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    if args.predictions:
        Path(args.predictions).write_text(json.dumps(serialise(
            scored, checkpoint="lexical", split="test", dataset_sha256=data_fingerprint(test))),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
