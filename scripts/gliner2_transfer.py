"""Train and score this dataset on a GLiNER2 backbone instead of Laya.

The question this answers: does the recipe travel, or was the result a property
of one base model? The data, the split, the option order, the tie policy and the
metric definitions are `evaluate.py`'s, so the numbers land in the same table.
Noul labels are ordered `false, true`; tied labels are excluded from hard accuracy.

Three runs:

    zeroshot   the published checkpoint, no training
    train      fine-tune on hard labels (the argmax of the teacher distribution)
    train --soft   fine-tune on the teacher distribution itself

The third is the point. GLiNER2's public training path takes one true label per
question, so the distribution is discarded before the loss sees it. The loss is
`binary_cross_entropy_with_logits` against a float tensor, which already accepts
probabilities, so two small patches carry them through:

1. `label_probs` rides along on each classification, through `to_dict` and
   `InputExample.from_dict`, which otherwise rebuild the object from a fixed
   field list.
2. `SchemaTransformer._build_outputs` writes that vector as the target instead
   of the one-hot.

The label augmentation in `SamplingConfig` is switched off for both fine-tunes.
Dropping or renaming a label deletes part of a distribution, and with it off the
two runs differ only in the target. That also means the hard-label run is not a
replication of Fastino's recipe. It is their objective without their robustness
step.

Install:  pip install -r requirements.txt   (gliner2 and peft, which its trainer imports)

The base model is loaded at the commit pinned in `data.REVISIONS`; pass
`--model <id>@<revision>` to use another.

Usage:
    python scripts/gliner2_transfer.py zeroshot --model fastino/GLiNER2.5-Decide
    python3 scripts/gliner2_transfer.py train --soft --out checkpoints/my-gliner-run
    python3 scripts/gliner2_transfer.py eval --model checkpoints/v1-gliner/final
    python3 scripts/gliner2_transfer.py latency --model checkpoints/v1-gliner/final
    python scripts/gliner2_transfer.py compare runs/soft.pred.json runs/hard.pred.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BUILD = None  # keep one build's rows (data.build_of)
BASE_MODEL = "fastino/GLiNER2.5-Decide"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import REVISIONS, split_revision                               # noqa: E402
from evaluate import (QTYPE_INDEX, data_fingerprint, is_tied, load_predictions, mcnemar,   # noqa: E402
                      metrics, option_keys, serialise)


# --------------------------------------------------------------------------- #
# State rendering — one deterministic form, used by every run
# --------------------------------------------------------------------------- #

def render(value: Any, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            if isinstance(item, (dict, list)) and item:
                out.append(f"{pad}{key}:")
                out.append(render(item, indent + 1))
            else:
                out.append(f"{pad}{key}: {scalar(item)}")
        return "\n".join(out)
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, (dict, list)):
                out.append(f"{pad}-")
                out.append(render(item, indent + 1))
            else:
                out.append(f"{pad}- {scalar(item)}")
        return "\n".join(out)
    return f"{pad}{scalar(value)}"


def scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


# --------------------------------------------------------------------------- #
# Noul, Choice and Score as GLiNER2 label sets
# --------------------------------------------------------------------------- #

def build_tasks(row: Dict) -> List[Dict]:
    """One dict per question: labels in gold order, descriptions, prompt, probs."""
    tasks: List[Dict] = []
    names: List[str] = []
    for qid, question in row["questions"].items():
        qtype = question["type"]
        criteria = question.get("criteria")
        gold = ((row.get("gold") or {}).get(qid) or {}).get("probabilities")
        if not gold:                                   # no label: skipped, as evaluate.score does
            continue

        if qtype == "noul":
            labels = ["false", "true"]                 # the order evaluate.py scores in
            descriptions = ({k: v for k, v in (criteria or {}).items() if v}
                            if isinstance(criteria, dict) else {})
            keys = labels
        elif qtype == "choice":
            # A list names the options without descriptions, as laya serving reads it.
            labels = list(criteria)
            descriptions = {k: v for k, v in criteria.items() if v} if isinstance(criteria, dict) else {}
            keys = labels
        elif qtype == "score":
            labels, seen = [], set()
            for index, level in enumerate(criteria):
                label = str(level).strip()[:80] or f"level {index}"
                while label in seen:
                    label = f"{label} ({index})"
                seen.add(label)
                labels.append(label)
            descriptions = {}
            keys = [str(i) for i in range(len(labels))]   # gold is keyed by index
        else:
            raise ValueError(f"unknown question type {qtype}")

        probs = {label: float(gold.get(key, 0.0)) for label, key in zip(labels, keys)}
        total = sum(probs.values())
        if total <= 0:
            continue
        probs = {k: v / total for k, v in probs.items()}

        name = distinct(qid, names)
        names.append(name)
        tasks.append({"task": name, "qid": qid, "qtype": qtype, "labels": labels,
                      "label_descriptions": descriptions,
                      "prompt": question["instructions"], "label_probs": probs})
    return tasks


def distinct(qid: str, taken: List[str]) -> str:
    """The processor matches a task by prefix, so no name may prefix another."""
    name = qid
    while any(name.startswith(t) or t.startswith(name) for t in taken):
        name += "_x"
    return name


def to_example(row: Dict, soft: bool):
    from gliner2.training import Classification, InputExample

    classifications = []
    for task in build_tasks(row):
        probs = task["label_probs"]
        item = Classification(
            task=task["task"], labels=task["labels"],
            true_label=max(probs, key=probs.get), prompt=task["prompt"],
            label_descriptions=task["label_descriptions"] or None,
        )
        if soft:
            item.label_probs = probs
        classifications.append(item)
    return InputExample(text=render(row["state"]), classifications=classifications)


def load(split: str) -> List[Dict]:
    if split in ("public", "public-test"):
        from benchmark import load_public
        return load_public("test")
    from data import load_split
    return load_split(DATA, split, BUILD)


def schema_for(tasks: List[Dict]) -> Dict:
    """Use descriptions only when every label has one; never drop an option."""
    schema = {}
    for task in tasks:
        described = {label: task["label_descriptions"][label] for label in task["labels"]
                     if label in task["label_descriptions"]}
        schema[task["task"]] = {
            "labels": described if len(described) == len(task["labels"]) else task["labels"],
            "multi_label": True,        # returns every label, not only the winner
            "cls_threshold": 0.0,
            "prompt": task["prompt"],
        }
    return schema


# --------------------------------------------------------------------------- #
# Patches
# --------------------------------------------------------------------------- #

def default_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def local_snapshot(model: str) -> str:
    """A local directory for a model named by path, Hub id or `id@revision`. gliner2's
    loaders take a name but resolve `main` themselves, so a Hub id is downloaded here at
    its pinned (or given) commit and loaded from disk."""
    repo, revision = split_revision(model)
    if revision is None:
        return repo
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=revision, ignore_patterns=["*.png", "*.md", ".gitattributes"])


def from_pretrained(model: str):
    from gliner2 import AutoExtractor
    return AutoExtractor.from_pretrained(local_snapshot(model))


def apply_patches(soft: bool) -> None:
    from gliner2.processor import SamplingConfig, SchemaTransformer

    if not getattr(SamplingConfig, "_transfer_patched", False):
        original_init = SamplingConfig.__init__

        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.remove_classification_prob = 0.0
            self.remove_classification_label_prob = 0.0
            self.synthetic_label_prob = 0.0
            self.include_true_label_prob = 1.0
            self.shuffle_classification_labels = False

        SamplingConfig.__init__ = init
        SamplingConfig._transfer_patched = True

    if not soft:
        return

    from gliner2.training import Classification, InputExample

    original_to_dict = Classification.to_dict

    def to_dict(self):
        payload = original_to_dict(self)
        probs = getattr(self, "label_probs", None)
        if probs:
            payload["label_probs"] = probs
        return payload

    Classification.to_dict = to_dict

    original_from_dict = InputExample.from_dict.__func__

    def from_dict(cls, data):
        example = original_from_dict(cls, data)
        raw = (data.get("output") or {}).get("classifications") or []
        for obj, item in zip(example.classifications or [], raw):
            if isinstance(item, dict) and item.get("label_probs"):
                obj.label_probs = item["label_probs"]
        return example

    InputExample.from_dict = classmethod(from_dict)

    original_build = SchemaTransformer._build_outputs

    def build_outputs(self, processed, schema, text_tokens, len_prefix):
        results = original_build(self, processed, schema, text_tokens, len_prefix)
        for result in results:
            if result.get("task_type") != "classifications":
                continue
            tokens = result["schema_tokens"]
            item = next((c for c in schema.get("classifications", [])
                         if tokens[2].startswith(c["task"])), None)
            if item is None or not item.get("label_probs"):
                raise ValueError(f"soft classification target missing for {tokens[2]!r}")
            probs = item["label_probs"]
            vector = [float(probs.get(label, 0.0)) for label in item["labels"]]
            total = sum(vector)
            if total > 0:
                vector = [v / total for v in vector]
            result["output"] = vector
        return results

    SchemaTransformer._build_outputs = build_outputs


# --------------------------------------------------------------------------- #
# Evaluation — metrics, tie policy and the paired test are evaluate.py's
# --------------------------------------------------------------------------- #

def evaluate(model_path: str, split: str, device: str) -> Dict:
    model = from_pretrained(model_path)
    if device == "cuda":
        model.cuda()

    scored = []
    rows = load(split)
    for row in rows:
        tasks = build_tasks(row)
        if not tasks:
            continue
        answers = model.classify_text(render(row["state"]), schema_for(tasks),
                                      include_confidence=True, format_results=False)
        for task in tasks:
            by_label = {label: float(score) for label, score in (answers.get(task["task"]) or [])}
            raw = np.array([by_label.get(label, 0.0) for label in task["labels"]], dtype=float)
            total = raw.sum()
            probabilities = raw / total if total > 0 else np.full(len(raw), 1.0 / len(raw))
            target = np.array([task["label_probs"][label] for label in task["labels"]], dtype=float)
            scored.append({"site": row["site"], "case_id": row["case_id"],
                           "question": task["qid"], "qtype": QTYPE_INDEX[task["qtype"]],
                           "p": probabilities, "t": target})

    by_site: Dict[str, List[Dict]] = {}
    for item in scored:
        by_site.setdefault(item["site"], []).append(item)
    public = split in ("public", "public-test")
    return {"overall": metrics(scored),
            "per_site": {site: metrics(items) for site, items in sorted(by_site.items())},
            "predictions": serialise(scored, checkpoint=model_path,
                                      split="public-test" if public else split,
                                      dataset_sha256=data_fingerprint(rows),
                                      **({"public_revision": REVISIONS["LocalLLaMA/typed-decisions"]}
                                         if public else {}))}


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def report(result: Dict, out: str | None, predictions: str | None) -> None:
    print(json.dumps(result["overall"], indent=1))
    if out:
        Path(out).write_text(json.dumps({k: v for k, v in result.items() if k != "predictions"},
                                        indent=1), encoding="utf-8")
        print("wrote", out)
    if predictions:
        Path(predictions).write_text(json.dumps(result["predictions"]), encoding="utf-8")
        print("predictions ->", predictions)


def cmd_compare(args) -> None:
    a, b = (load_predictions(Path(p)) for p in args.predictions)
    print(json.dumps(mcnemar(a, b), indent=1))


def cmd_zeroshot(args) -> None:
    apply_patches(soft=False)
    report(evaluate(args.model, args.split, args.device), args.out, args.predictions)


def cmd_eval(args) -> None:
    apply_patches(soft=False)
    report(evaluate(args.model, args.split, args.device), args.out, args.predictions)


def cmd_train(args) -> None:
    if Path(args.out).exists():
        raise SystemExit(f"refusing to overwrite {args.out}; choose a new --out directory")
    apply_patches(soft=args.soft)
    from gliner2.training import TrainingDataset, train_gliner2

    train = [to_example(row, args.soft) for row in load("train")]
    validation = [to_example(row, args.soft) for row in load("validation")]

    if args.soft:
        # The soft target is written by a patch to the collate function. With worker
        # processes that patch only exists in a worker if the worker was forked; under
        # spawn or forkserver (Windows, and Linux from Python 3.14) a worker re-imports
        # an unpatched gliner2 and the run silently becomes hard-label training. So a
        # soft run collates in the main process, and one example is pushed through the
        # patched path first to prove a distribution reaches the target.
        check_soft_targets(train)
    started = time.perf_counter()
    train_gliner2(
        local_snapshot(args.model), TrainingDataset(train), output_dir=args.out,
        eval_data=TrainingDataset(validation),
        num_epochs=args.epochs, batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum,
        encoder_lr=args.encoder_lr, task_lr=args.task_lr,
        bf16=True, fp16=False, seed=args.seed,
        eval_strategy="epoch", save_total_limit=1, logging_steps=50,
        num_workers=0 if args.soft else 2,
    )
    print(f"train seconds: {time.perf_counter() - started:.0f}")
    failed = Path(args.out) / "failed_batches.jsonl"
    if failed.exists():
        count = len(failed.read_text(encoding="utf-8").splitlines())
        print(f"WARNING: {count} batches failed. "
              f"The trainer skips them silently — lower --batch-size and run again.")


def check_soft_targets(examples) -> None:
    """Prove a non-one-hot target survives serialization and target construction."""
    from gliner2.training import InputExample
    from gliner2.processor import SchemaTransformer

    for example in examples:
        rebuilt = InputExample.from_dict(example.to_dict())
        for item in rebuilt.classifications or []:
            probs = getattr(item, "label_probs", None)
            if probs and 0.0 < max(probs.values()) < 1.0:
                raw = rebuilt.to_dict()["output"]["classifications"]
                processed = {"schemas": [["[C]", "", item.task]],
                             "task_types": ["classifications"], "structure_labels": [None]}
                output = SchemaTransformer._build_outputs(
                    SchemaTransformer.__new__(SchemaTransformer), processed,
                    {"classifications": raw}, [], 0,
                )[0]["output"]
                expected = [float(probs[label]) for label in item.labels]
                if len(output) != len(expected) or any(abs(a - b) > 1e-8 for a, b in zip(output, expected)):
                    raise SystemExit("--soft: processed training targets differ from teacher probabilities")
                return
    raise SystemExit("--soft: no non-one-hot target survived training preprocessing")


def cmd_latency(args) -> None:
    """40 calls after 10 warm-up, one question and three, as RESULTS.md measures."""
    model = from_pretrained(args.model)
    if args.device == "cuda":
        model.cuda()
    row = next(r for r in load("test") if len(r["questions"]) >= 3)
    text, tasks = render(row["state"]), build_tasks(row)

    out = {}
    for count in (1, 3):
        schema = schema_for(tasks[:count])
        for _ in range(10):
            model.classify_text(text, schema, include_confidence=True, format_results=False)
        times = []
        for _ in range(40):
            start = time.perf_counter()
            model.classify_text(text, schema, include_confidence=True, format_results=False)
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        out[f"{count}q"] = {"p50": round(times[len(times) // 2], 1),
                            "p95": round(times[int(len(times) * 0.95)], 1)}
    print(json.dumps(out, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser(prog="gliner2_transfer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn in (("zeroshot", cmd_zeroshot), ("eval", cmd_eval)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--model", default=BASE_MODEL)
        p.add_argument("--split", default="test", help="Assistant Decisions split name, or public for the pinned LocalLLaMA/typed-decisions test")
        p.add_argument("--device", default=default_device())
        p.add_argument("--out")
        p.add_argument("--predictions", help="save per-decision predictions for --compare")
        p.add_argument("--data", default=None, help="dataset directory (default: data/)")
        p.add_argument("--build", default=None, help="score one build's rows only (data.build_of)")

    p = sub.add_parser("compare")
    p.set_defaults(fn=cmd_compare)
    p.add_argument("predictions", nargs=2, help="two prediction files; exact McNemar test")

    p = sub.add_parser("train")
    p.set_defaults(fn=cmd_train)
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--soft", action="store_true", help="train on the teacher distribution")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2, help="8 runs out of memory on 16 GB")
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--encoder-lr", type=float, default=1e-5)
    p.add_argument("--task-lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data", default=None, help="dataset directory with train.jsonl and validation.jsonl (default: data/)")
    p.add_argument("--build", default=None, help="train on one build's rows only (data.build_of)")

    p = sub.add_parser("latency")
    p.set_defaults(fn=cmd_latency)
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--device", default=default_device())

    args = parser.parse_args()
    global DATA, BUILD
    if getattr(args, "data", None):
        DATA = Path(args.data)
    BUILD = getattr(args, "build", None)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
