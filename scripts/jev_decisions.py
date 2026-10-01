"""Measure a hosted decision model — TypeSafe's Jev — on the same test sets, under the
same evaluation contract, as every checkpoint in RESULTS.md. It calls OpenRouter's
Decisions endpoint, which proxies the TypeSafe `/v1/systemone` contract:

    POST https://openrouter.ai/api/alpha/decisions
    {"model": "typesafe/jev-1.13", "state": {...}, "questions": {...}}
    -> {"answers": {qid: {"type": "noul", "noul": 0.02}
                    | {"type": "choice", "choice": ..., "probabilities": {...}, "confidence": ...}
                    | {"type": "score", "score": ..., "probabilities": {"0": ...}, ...}},
        "usage": {"input_tokens": 436, "output_tokens": 70, "cost": 1.8e-05}}

One request per case carries every question of that case, which is the pattern
TypeSafe documents and the way Laya batches. Every answer goes through
`evaluate.answer_vector`, `target_vector` and `metrics` unchanged, so the numbers land
in the same tables as the local checkpoints.

Supply `LLM_OPENROUTER_API_KEYS` through the process environment. The script never
opens a credential file and never prints the key. Every call is journalled under the
output directory keyed by a request hash, so a rerun re-pays nothing. A hard budget
cap stops the run before a request that would exceed it.

    USE_TF=0 python scripts/jev_decisions.py run --model typesafe/jev-1.13 \\
        --out runs/jev --sets internal,public
    python scripts/jev_decisions.py compare --out runs/jev \\
        --internal laya=runs/laya.internal.json --public laya=runs/laya.public.json

`--out` is required: name the run directory yourself, and pass the same one to
`check_numbers.py --collect --summary ...` afterwards.

Two facts about the measurement that the report must carry:

- Jev returns probabilities at two decimals. A `0.00` on the label's top option costs
  about 27 nats of soft NLL under the contract's 1e-12 clip, so the calibration columns
  are reported twice: as measured, and with every probability floored at 0.005 (half a
  quantisation step) and renormalised. The floor is Jev's output precision, not a
  scorer choice.
- Latency here is a network round trip from this machine through OpenRouter to
  TypeSafe. It is not comparable to the local served-path figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (QTYPE_INDEX, answer_vector, data_fingerprint, is_tied, load_predictions, mcnemar,  # noqa: E402
                      metrics, option_keys, serialise, target_vector)
from data import REVISIONS  # noqa: E402

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
PRICE_PER_TOKEN = 0.042 / 1e6           # fallback reservation estimate; check current pricing
FLOOR = 0.005                            # half of Jev's two-decimal quantisation step
LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# The three test sets, in the shape evaluate.py scores
# --------------------------------------------------------------------------- #

def read_lines(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_set(name: str, data: Path, build: Optional[str]) -> List[Dict[str, Any]]:
    if name == "internal":                # this dataset's test split
        from data import load_split
        return load_split(data, "test", build)
    if name == "public":
        from benchmark import load_public
        return load_public("test")
    raise SystemExit(f"unknown set {name!r}; use internal or public")


def git_revision() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None                       # not a git checkout (a release tarball, a Hub download)


# --------------------------------------------------------------------------- #
# The client
# --------------------------------------------------------------------------- #

def digest(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def journal_spend(path: Path) -> float:
    """Completed charges plus reservations for calls without a completed response."""
    if not path.exists():
        return 0.0
    pending: Dict[str, List[float]] = defaultdict(list)
    completed = 0.0
    for row in read_lines(path):
        tag = row.get("tag", "")
        if row.get("event") == "start":
            pending[tag].append(float(row["reservation"]))
        elif row.get("event") == "end":
            completed += float(row.get("cost") or 0.0)
            if pending[tag]:
                pending[tag].pop(0)
    return completed + sum(sum(values) for values in pending.values())


class Client:
    def __init__(self, out: Path, name: str, budget_usd: float):
        self.key = next((k.strip() for k in os.environ.get("LLM_OPENROUTER_API_KEYS", "").split(",")
                         if k.strip()), None)
        if not self.key:
            raise RuntimeError("No configured OpenRouter credential in the process environment")
        # One cache and one journal per run name, so two models never share a cached call.
        self.responses = out / "responses" / name
        self.responses.mkdir(parents=True, exist_ok=True)
        self.journal = out / f"{name}.calls.jsonl"
        self.budget = budget_usd
        self.spent = journal_spend(self.journal)
        self.reserved = 0.0
        self.latencies: List[float] = []

    def _log(self, row: Dict[str, Any]) -> None:
        with LOCK:
            with self.journal.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def decide(self, tag: str, model: str, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"model": model, "state": state, "questions": questions}
        request_hash = digest(payload)
        cached = self.responses / f"{tag}.json"
        if cached.exists():
            saved = json.loads(cached.read_text(encoding="utf-8"))
            if saved["request_hash"] != request_hash:
                raise RuntimeError(f"request changed for cached call {tag}")
            return saved["response"]

        body = json.dumps(payload, ensure_ascii=False).encode()
        estimate = (len(body) / 3.5 + 64) * PRICE_PER_TOKEN * 1.5     # conservative token guess
        with LOCK:
            if self.spent + self.reserved + estimate > self.budget:
                raise RuntimeError(f"budget cap ${self.budget:.2f} would be exceeded")
            self.reserved += estimate
        self._log({"event": "start", "tag": tag, "request_hash": request_hash, "reservation": estimate})

        request = urllib.request.Request(
            ENDPOINT, data=body,
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        delay = 2.0
        for attempt in range(6):
            started = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=90) as raw:
                    response = json.load(raw)
                elapsed = time.monotonic() - started
                break
            except urllib.error.HTTPError as exc:
                text = exc.read().decode(errors="replace")[:400]
                if exc.code in (429, 500, 502, 503, 529) and attempt < 5:
                    time.sleep(delay)
                    delay *= 2
                    continue
                with LOCK:
                    self.reserved -= estimate
                self._log({"event": "error", "tag": tag, "status": exc.code, "body": text})
                raise
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < 5:
                    time.sleep(delay)
                    delay *= 2
                    continue
                with LOCK:
                    self.reserved -= estimate
                self._log({"event": "error", "tag": tag, "error": repr(exc)})
                raise
        usage = response.get("usage") or {}
        cost = float(usage.get("cost") or 0.0)
        if not usage.get("cost"):
            cost = float(usage.get("input_tokens") or 0) * PRICE_PER_TOKEN
        with LOCK:
            self.reserved -= estimate
            self.spent += cost
            self.latencies.append(elapsed)
        cached.write_text(json.dumps({"tag": tag, "request_hash": request_hash, "response": response,
                                      "latency_s": round(elapsed, 4)}, indent=1, sort_keys=True), encoding="utf-8")
        self._log({"event": "end", "tag": tag, "request_hash": request_hash, "cost": cost,
                   "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                   "model": response.get("model"), "id": response.get("id"), "latency_s": round(elapsed, 4)})
        return response


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def questions_of(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The questions the contract scores: those with a usable gold distribution."""
    return {qid: q for qid, q in row["questions"].items()
            if (row.get("gold") or {}).get(qid, {}).get("probabilities")}


def to_request(question: Dict[str, Any]) -> Dict[str, Any]:
    """The dataset's question shape is already TypeSafe's. Pass it through, minus keys
    the endpoint does not know."""
    out = {"type": question["type"], "instructions": question["instructions"]}
    if question.get("criteria") is not None:
        out["criteria"] = question["criteria"]
    return out


def score_set(rows: List[Dict[str, Any]], answers: Dict[str, Dict[str, Any]], site_field: str = "site"):
    scored, missing = [], 0
    for index, row in enumerate(rows):
        case_id = row.get("case_id") or f"row-{index}"
        got = answers.get(case_id)
        if got is None:
            missing += len(questions_of(row))
            continue
        for qid, question in questions_of(row).items():
            target = target_vector(question, row["gold"][qid])
            if target is None or qid not in got:
                missing += 1
                continue
            scored.append({
                "case_id": case_id, "question": qid, "site": row.get(site_field, ""),
                "qtype": QTYPE_INDEX[question["type"]],
                "p": answer_vector(question, got[qid]), "t": target,
            })
    return scored, missing


def floored(scored: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The same decisions with every probability floored at FLOOR and renormalised."""
    out = []
    for row in scored:
        p = np.maximum(row["p"], FLOOR)
        out.append({**row, "p": p / p.sum()})
    return out


def accuracy_all_first_option(scored: List[Dict[str, Any]]) -> float:
    """Accuracy over every decision, ties broken by option order — the old rule, kept so
    a published all-decisions figure can be reproduced."""
    hits = [int(np.argmax(r["p"]) == np.argmax(r["t"])) for r in scored]
    return round(float(np.mean(hits)), 4) if hits else float("nan")


def by_site(scored: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scored:
        groups[row["site"]].append(row)
    return {site: metrics(rows) for site, rows in sorted(groups.items())}


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def run(args: argparse.Namespace) -> int:
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    client = Client(out, args.name, args.budget_usd)
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    print(f"\n  model     {args.model}")
    print(f"  sets      {', '.join(sets)}")
    print(f"  budget    ${args.budget_usd:.2f}, spent so far ${client.spent:.4f}\n")

    summary: Dict[str, Any] = {
        "model": args.model, "endpoint": ENDPOINT, "date": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(), "build": args.build,
        "sets": {},
    }
    for name in sets:
        rows = load_set(name, ROOT / args.data, args.build)
        tags = [(f"{name}-{row.get('case_id') or i}".replace("/", "_"), row.get("case_id") or f"row-{i}", row)
                for i, row in enumerate(rows)]

        def job(item):
            tag, case_id, row = item
            questions = {qid: to_request(q) for qid, q in questions_of(row).items()}
            if not questions:
                return case_id, None
            try:
                response = client.decide(tag, args.model, row["state"], questions)
            except Exception as exc:                                  # noqa: BLE001
                print(f"    {case_id}: failed ({type(exc).__name__}: {exc})", flush=True)
                return case_id, None
            return case_id, response.get("answers") or {}

        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(job, tags))
        answers = {case_id: ans for case_id, ans in results if ans is not None}
        wall = time.monotonic() - started

        scored, missing = score_set(rows, answers)
        if (missing or len(answers) < len(rows)) and not args.allow_partial:
            raise SystemExit(f"{name}: {len(rows) - len(answers)} case(s) unanswered, {missing} decision(s) "
                             "missing. Rerun (answered cases are cached) or pass --allow-partial.")
        resolved = {r.get("model") for r in
                    (json.loads(p.read_text(encoding="utf-8"))["response"] for p in client.responses.glob(f"{name}-*.json"))}
        result = {
            "cases": len(rows), "answered_cases": len(answers), "missing_decisions": missing,
            "dataset_sha256": data_fingerprint(rows),
            "allow_partial": args.allow_partial,
            "resolved_model": sorted(m for m in resolved if m),
            "metrics": metrics(scored),
            "metrics_floored": metrics(floored(scored)),
            "accuracy_all_decisions_first_option": accuracy_all_first_option(scored),
            "by_site": by_site(scored),
            "wall_clock_s": round(wall, 1),
        }
        summary["sets"][name] = result
        (out / f"{args.name}.{name}.predictions.json").write_text(json.dumps(
            serialise(scored, checkpoint=args.model,
                      split="public-test" if name == "public" else "test",
                      dataset_sha256=data_fingerprint(rows),
                      **({"public_revision": REVISIONS["LocalLLaMA/typed-decisions"]}
                         if name == "public" else {})), indent=1), encoding="utf-8")
        print(f"  {name:<9} {len(rows):,} cases, {len(scored):,} decisions scored, {missing} missing")
        print(f"            {json.dumps(result['metrics'])}")
        print(f"            floored {json.dumps(result['metrics_floored'])}\n")

    if client.latencies:
        lat = sorted(client.latencies)
        summary["latency_s"] = {
            "note": "network round trip from this machine through OpenRouter to TypeSafe, one case per call; "
                    "not comparable to the local served-path figures",
            "calls": len(lat), "p50": round(statistics.median(lat), 3),
            "p95": round(lat[int(0.95 * (len(lat) - 1))], 3), "mean": round(statistics.fmean(lat), 3),
        }
    summary["spend_usd"] = round(client.spent, 4)
    summary["input_tokens"] = sum(int(r.get("input_tokens") or 0) for r in read_lines(client.journal)
                                  if r.get("event") == "end")
    (out / f"{args.name}.summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"  spent     ${client.spent:.4f}")
    print(f"  written   {out}/{args.name}.summary.json\n")
    return 0


def compare(args: argparse.Namespace) -> int:
    out = ROOT / args.out
    report: Dict[str, Any] = {}
    for name, entries in (("internal", args.internal), ("public", args.public)):
        path = out / f"{args.name}.{name}.predictions.json"
        if not entries or not path.exists():
            continue
        ours = load_predictions(path)
        report[name] = {}
        for entry in entries:
            label, _, other_path = entry.partition("=")
            theirs = load_predictions(Path(other_path) if Path(other_path).is_absolute() else ROOT / other_path)
            result = mcnemar(theirs, ours)
            result = {**result, "a": label, "b": args.name}
            report[name][label] = result
            print(f"  {name:<9} {label:<14} {label} only {result['a_only_correct']:>4}  "
                  f"{args.name} only {result['b_only_correct']:>4}  p = {result['p_two_sided']}")
    (out / f"{args.name}.paired.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n  written   {out}/{args.name}.paired.json\n")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="jev_decisions", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="call the endpoint on the chosen sets and score the answers")
    r.add_argument("--model", default="typesafe/jev-1.13")
    r.add_argument("--name", default="jev")
    r.add_argument("--out", required=True, help="the run directory, e.g. runs/jev")
    r.add_argument("--data", default="data", help="the dataset directory")
    r.add_argument("--build", default=None,
                   help="score one build's rows of the test split only (data.build_of); default: every row")
    r.add_argument("--sets", default="internal,public")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--budget-usd", type=float, default=1.0)
    r.add_argument("--allow-partial", action="store_true",
                   help="write predictions and metrics even if some cases failed (recorded in the summary)")
    c = sub.add_parser("compare", help="paired tests against saved prediction files")
    c.add_argument("--name", default="jev")
    c.add_argument("--out", required=True, help="the run directory `run` wrote")
    c.add_argument("--internal", action="append", default=[], metavar="LABEL=PATH")
    c.add_argument("--public", action="append", default=[], metavar="LABEL=PATH")
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
