"""General decision data to mix into training, so a fresh-head model keeps broad skills.

CalDec Ultra starts from a masked-LM encoder and a fresh head, and learns decisions only
from the assistant data, so on outside decision benchmarks (ANLI, NLI4CT, VAST) it falls
to chance while CalDec Laya, fine-tuned from a broadly trained base, does not (ULTRA.md).
This writes a small, deterministic sample of two public train splits in the dataset's
row format, with the datasets' own gold labels as targets:

- MNLI (nyu-mll/glue, mnli): premise and hypothesis, a three-way `choice`;
- BoolQ (google/boolq): passage and question, a `noul` (P(yes)).

Neither is on the Decision Index panel this repository checks (ULTRA.md), so the panel
measures transfer, not memorisation. Targets put 0.9 on the gold label and spread 0.1
over the rest: gold labels are single human judgments, not certainties.

    python3 scripts/general_data.py --out data-general     # train.jsonl and validation.jsonl
    python3 scripts/train.py ... --extra-train data-general/train.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import encode  # noqa: E402

REVISIONS = {"nyu-mll/glue": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
             "google/boolq": "35b264d03638db9f4ce671b711558bf7ff0f80d5"}
GOLD = 0.9

MNLI_Q = {"relation": {"type": "choice", "instructions": "How does the hypothesis relate to the premise?",
                       "criteria": {"entailment": "the premise shows the hypothesis is true",
                                    "neutral": "the premise neither shows it true nor false",
                                    "contradiction": "the premise shows the hypothesis is false"}}}
BOOLQ_Q = {"yes": {"type": "noul", "instructions": "According to the passage, the answer to the question is yes."}}


def smooth(keys, gold):
    rest = (1 - GOLD) / (len(keys) - 1)
    return {"probabilities": {k: GOLD if k == gold else rest for k in keys}}


def main() -> int:
    from datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mnli", type=int, default=2000)
    ap.add_argument("--boolq", type=int, default=1000)
    ap.add_argument("--val-share", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = []
    mnli = load_dataset("nyu-mll/glue", "mnli", split="train", revision=REVISIONS["nyu-mll/glue"])
    names = ["entailment", "neutral", "contradiction"]
    for i in rng.sample(range(len(mnli)), args.mnli):
        r = mnli[i]
        rows.append(dict(case_id=f"general-mnli-{r['idx']}", site="general.nli", workflow="general",
                         state={"premise": r["premise"], "hypothesis": r["hypothesis"]},
                         questions=MNLI_Q, gold={"relation": smooth(names, names[r["label"]])}))
    boolq = load_dataset("google/boolq", split="train", revision=REVISIONS["google/boolq"])
    for i in rng.sample(range(len(boolq)), args.boolq):
        r = boolq[i]
        rows.append(dict(case_id=f"general-boolq-{i}", site="general.boolq", workflow="general",
                         state={"passage": r["passage"], "question": r["question"]},
                         questions=BOOLQ_Q, gold={"yes": smooth(["false", "true"], "true" if r["answer"] else "false")}))
    rng.shuffle(rows)
    n_val = int(len(rows) * args.val_share)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, part in (("validation", rows[:n_val]), ("train", rows[n_val:])):
        (args.out / f"{name}.jsonl").write_text("".join(json.dumps(encode(r), ensure_ascii=False) + "\n" for r in part),
                                                 encoding="utf-8")
        print(f"{len(part)} rows -> {args.out / (name + '.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
