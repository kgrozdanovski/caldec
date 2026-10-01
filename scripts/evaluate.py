"""Score a fine-tuned checkpoint through the path that serves it, and propose thresholds.

One evaluation contract, applied to every model in this repository:

* **Scoring goes through `laya.Agent.system_one`**, the function that serves the
  checkpoint. It applies the fitted temperatures, the temperature clamp and the
  bf16 autocast.
* **"Accuracy" means agreement with the argmax of the target distribution.**
  Targets are synthetic GLM 5.3 judgments.
  Agreement with them is not independent correctness.
* **A tied label has no argmax.** When two options share the top probability,
  hard accuracy is decided by option order, which is a convention and not a
  measurement. Tied labels are excluded from accuracy and from ECE, and kept in
  every distribution metric. Every table reports `n_untied` beside `n`.
* **The majority baseline is fitted on the training split** and weighted by
  decision, which is how a deployed majority predictor would score.
* **Thresholds are proposed on the validation split**, never on test. A question
  with no threshold that reaches the requested precision is reported as unsupported
  rather than defaulted to a number that accepts everything.
* **Per-decision predictions are saved**, so two models can be compared with a
  paired test (`--compare a.json b.json`) instead of by eye. The test refuses two
  files whose shared decisions carry different labels: they were scored on
  different data.

    USE_TF=0 python scripts/evaluate.py \\
        --data data --checkpoint checkpoints/v1-laya/final --predictions runs/test.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("USE_TF", "0")

import numpy as np                                                     # noqa: E402

from data import load_split, resolve_laya                              # noqa: E402

TIE_TOLERANCE = 1e-9
QTYPE_INDEX = {"choice": 0, "score": 1, "noul": 2}


# --------------------------------------------------------------------------- #
# Targets and predictions in one key order
# --------------------------------------------------------------------------- #

def option_keys(question: Dict[str, Any]) -> List[str]:
    """The option order every model is scored in. Noul is `false, true`."""
    qtype = question["type"]
    criteria = question.get("criteria")
    if qtype == "noul":
        return ["false", "true"]
    if qtype == "choice":
        return list(criteria or {})
    return [str(i) for i in range(len(criteria or []))]


def target_vector(question: Dict[str, Any], gold: Dict[str, Any]) -> Optional[np.ndarray]:
    keys = option_keys(question)
    values = np.array([float(gold["probabilities"].get(k, 0.0)) for k in keys], dtype=float)
    total = values.sum()
    if total <= 0:
        return None
    return values / total


def is_tied(target: np.ndarray) -> bool:
    if len(target) < 2:
        return False
    top = np.sort(target)[-2:]
    return bool(top[1] - top[0] < TIE_TOLERANCE)


def answer_vector(question: Dict[str, Any], answer: Dict[str, Any]) -> np.ndarray:
    """A served answer, back in `option_keys` order."""
    qtype = question["type"]
    if qtype == "noul":
        yes = float(answer["noul"])
        return np.array([1.0 - yes, yes], dtype=float)
    probabilities = answer["probabilities"]
    keys = option_keys(question)
    values = np.array([float(probabilities.get(k, 0.0)) for k in keys], dtype=float)
    total = values.sum()
    return values / total if total > 0 else np.full(len(keys), 1.0 / len(keys))


def score(agent, rows: List[Dict[str, Any]], site_field: str = "site") -> List[Dict[str, Any]]:
    """Score every decision of every row through the served path."""
    scored, expected = [], 0
    for index, row in enumerate(rows):
        questions = {qid: q for qid, q in row["questions"].items()
                     if (row.get("gold") or {}).get(qid, {}).get("probabilities")}
        if not questions:
            continue
        expected += len(questions)
        try:
            answers = agent.system_one(row["state"], questions)["answers"]
        except ValueError as exc:
            raise ValueError(f"{row.get('case_id', index)}: inference failed for "
                             f"{list(questions)}: {exc}") from exc
        case_id = row.get("case_id") or f"row-{index}"
        for qid, question in questions.items():
            target = target_vector(question, row["gold"][qid])
            if target is None:
                raise ValueError(f"{case_id}/{qid}: invalid target distribution")
            scored.append({
                "case_id": case_id, "question": qid, "site": row.get(site_field, ""),
                "qtype": QTYPE_INDEX[question["type"]],
                "p": answer_vector(question, answers[qid]), "t": target,
            })
    if len(scored) != expected:
        raise RuntimeError(f"scored {len(scored)} of {expected} eligible decisions")
    return scored


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def hit(row: Dict[str, Any]) -> Optional[int]:
    """1 or 0 on an untied label, None on a tied one."""
    if is_tied(row["t"]):
        return None
    return int(np.argmax(row["p"]) == np.argmax(row["t"]))


def metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    hits, confidences = [], []
    soft, brier, nll, maes = [], [], [], []
    for row in rows:
        p, t = row["p"], row["t"]
        h = hit(row)
        if h is not None:
            hits.append(h)
            confidences.append((float(np.max(p)), h))
        soft.append(float(np.sum(np.minimum(p, t))))          # distribution overlap
        brier.append(float(np.sum((p - t) ** 2) / len(p)))
        nll.append(float(-np.sum(t * np.log(np.clip(p, 1e-12, 1.0)))))
        if row["qtype"] == QTYPE_INDEX["score"]:
            levels = np.arange(len(p))
            maes.append(abs(float(p @ levels) - float(t @ levels)))

    # Expected calibration error, 15 equal-width bins, the last one closed at 1.0.
    ece, total = 0.0, len(confidences)
    edges = np.linspace(0, 1, 16)
    for i in range(15):
        lower, upper = edges[i], edges[i + 1]
        bucket = [(c, h) for c, h in confidences
                  if (lower <= c < upper) or (i == 14 and c == upper)]
        if not bucket:
            continue
        mean_conf = sum(c for c, _ in bucket) / len(bucket)
        accuracy = sum(h for _, h in bucket) / len(bucket)
        ece += len(bucket) / total * abs(mean_conf - accuracy)

    out = {
        "n": len(rows),
        "n_untied": len(hits),
        "accuracy": round(float(np.mean(hits)), 4) if hits else None,
        "soft_accuracy": round(float(np.mean(soft)), 4),
        "brier": round(float(np.mean(brier)), 4),
        "soft_nll": round(float(np.mean(nll)), 4),
        "ece": round(float(ece), 4) if hits else None,
    }
    if maes:
        out["score_mae"] = round(float(np.mean(maes)), 4)
    return out


def baselines(train_rows: List[Dict[str, Any]], scored: List[Dict[str, Any]]) -> Dict[str, float]:
    """Majority class fitted on train, and a random guess, on the untied test decisions."""
    counts: Dict[tuple, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for row in train_rows:
        for qid, question in row["questions"].items():
            gold = (row.get("gold") or {}).get(qid)
            if not gold or "probabilities" not in gold:
                continue
            target = target_vector(question, gold)
            if target is None or is_tied(target):
                continue
            counts[(row["site"], qid)][int(np.argmax(target))] += 1

    majority_hits, random_guess = [], []
    for row in scored:
        if is_tied(row["t"]):
            continue
        fitted = counts.get((row["site"], row["question"]))
        if not fitted:
            continue
        predicted = max(fitted, key=fitted.get)
        majority_hits.append(int(predicted == int(np.argmax(row["t"]))))
        random_guess.append(1.0 / len(row["t"]))
    return {
        "majority_class": round(float(np.mean(majority_hits)), 4) if majority_hits else None,
        "random_guess": round(float(np.mean(random_guess)), 4) if random_guess else None,
        "n_untied": len(majority_hits),
    }


# --------------------------------------------------------------------------- #
# Paired comparison
# --------------------------------------------------------------------------- #

class PredictionRows(dict):
    def __init__(self, *args, header: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.header = header or {}


def data_fingerprint(rows: List[Dict[str, Any]]) -> str:
    """Bind a prediction file to ordered case contents, including states and questions."""
    digest = hashlib.sha256()
    for row in rows:
        selected = {key: row.get(key) for key in ("case_id", "site", "state", "questions", "gold")}
        digest.update(json.dumps(selected, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8") + b"\n")
    return digest.hexdigest()


def load_predictions(path: Path) -> PredictionRows:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = document["decisions"]
    out = PredictionRows(header={k: v for k, v in document.items() if k != "decisions"})
    for row in rows:
        row["p"], row["t"] = np.asarray(row["p"], dtype=float), np.asarray(row["t"], dtype=float)
        key = (row["case_id"], row["question"])
        if key in out:
            raise SystemExit(f"{path}: duplicate prediction {key}")
        out[key] = row
    return out


def mcnemar(a: Dict[tuple, Dict[str, Any]], b: Dict[tuple, Dict[str, Any]]) -> Dict[str, Any]:
    """Exact two-sided McNemar on the untied decisions both files hold.

    Decisions are paired by (case id, question). LocalLLaMA/typed-decisions ids are positional,
    so two files scored on different revisions of that dataset would pair unrelated
    rows; a shared key whose label differs between the files is therefore an error."""
    if set(a) != set(b):
        raise SystemExit(f"prediction coverage differs: {len(a)} vs {len(b)} decisions")
    header_a, header_b = getattr(a, "header", {}), getattr(b, "header", {})
    for field in ("split", "dataset_sha256", "public_revision"):
        value_a, value_b = header_a.get(field), header_b.get(field)
        if field == "split":
            value_a = {"internal": "test", "public": "public-test"}.get(value_a, value_a)
            value_b = {"internal": "test", "public": "public-test"}.get(value_b, value_b)
        if (value_a or value_b) and value_a != value_b:
            raise SystemExit(f"prediction {field} differs: {value_a!r} vs {value_b!r}")
    keys = sorted(a)
    if not keys:
        raise SystemExit("the two prediction files share no decision")
    mismatched = [k for k in keys if a[k]["t"].shape != b[k]["t"].shape
                  or not np.allclose(a[k]["t"], b[k]["t"], atol=1e-4)]
    if mismatched:
        raise SystemExit(f"{len(mismatched)} of {len(keys)} shared decisions carry different labels in the "
                         f"two files (first: {mismatched[0]}); they were not scored on the same data")
    a_only = b_only = both = neither = 0
    for key in keys:
        ha, hb = hit(a[key]), hit(b[key])
        if ha is None or hb is None:
            continue
        if ha and not hb:
            a_only += 1
        elif hb and not ha:
            b_only += 1
        elif ha and hb:
            both += 1
        else:
            neither += 1
    n = a_only + b_only
    if n == 0:
        p_value = 1.0
    else:
        k = min(a_only, b_only)
        p_value = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"paired": a_only + b_only + both + neither, "a_only_correct": a_only,
            "b_only_correct": b_only, "discordant": n, "p_two_sided": round(p_value, 4)}


def serialise(scored: List[Dict[str, Any]], **header: Any) -> Dict[str, Any]:
    return {**header, "decisions": [
        {"case_id": r["case_id"], "question": r["question"], "site": r["site"],
         "qtype": r["qtype"], "p": [round(float(v), 6) for v in r["p"]],
         "t": [round(float(v), 6) for v in r["t"]]} for r in scored]}


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #

def propose_bands(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A starting threshold per question, fitted for precision on the given rows.

    For a noul: the loosest `yes_at` and `no_at` that keep the decided side at
    least 95% precise, so the band between them is what gets escalated.
    For a choice or score: the lowest confidence floor that reaches 90% precision.
    A question with no threshold that reaches the target is reported as such.
    """
    proposals: Dict[str, Any] = {}
    by_question = defaultdict(list)
    for row in rows:
        if not is_tied(row["t"]):
            by_question[(row["site"], row["question"])].append(row)

    for (site, question), group in sorted(by_question.items()):
        qtype = group[0]["qtype"]
        name = f"{site}.{question}"
        if qtype == QTYPE_INDEX["noul"]:
            pairs = [(float(r["p"][1]), int(np.argmax(r["t"]) == 1)) for r in group]
            yes_at = _threshold(pairs, positive=True, precision=0.95)
            no_at = _threshold(pairs, positive=False, precision=0.95)
            if yes_at is None or no_at is None:
                proposals[name] = {"kind": "Band", "supported": False,
                                   "reason": "no threshold reaches 95% precision on 10+ decisions"}
                continue
            decided = sum(1 for p, _ in pairs if p >= yes_at or p <= no_at)
            proposals[name] = {
                "kind": "Band", "supported": True,
                "yes_at": round(yes_at, 3), "no_at": round(no_at, 3),
                "decided_share": round(decided / len(pairs), 3),
            }
        else:
            pairs = [(float(np.max(r["p"])), int(np.argmax(r["p"]) == np.argmax(r["t"])))
                     for r in group]
            floor = _threshold(pairs, positive=True, precision=0.90)
            if floor is None:
                proposals[name] = {"kind": "ChoiceBand", "supported": False,
                                   "reason": "no floor reaches 90% precision on 10+ decisions"}
                continue
            decided = sum(1 for c, _ in pairs if c >= floor)
            proposals[name] = {
                "kind": "ChoiceBand", "supported": True,
                "min_confidence": round(floor, 3),
                "decided_share": round(decided / len(pairs), 3),
            }
    return proposals


def _threshold(pairs, *, positive: bool, precision: float) -> Optional[float]:
    """The loosest threshold that reaches ``precision`` on at least 10 decided
    decisions, or None when no candidate does."""
    for candidate in np.linspace(0.50, 0.99, 50):
        if positive:
            decided = [h for value, h in pairs if value >= candidate]
        else:
            candidate = 1.0 - candidate
            decided = [1 - h for value, h in pairs if value <= candidate]
        if len(decided) >= 10 and sum(decided) / len(decided) >= precision:
            return float(candidate)
    return None


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def fmt(value: Optional[float]) -> str:
    return "     —" if value is None else f"{value:>7.3f}"


def print_table(title: str, groups: Dict[str, Dict[str, Any]]) -> None:
    print(f"\n  {title}")
    print(f"  {'':<30} {'n':>6} {'untied':>6} {'acc':>7} {'soft':>7} {'brier':>7} "
          f"{'nll':>7} {'ece':>7}")
    for name, m in groups.items():
        print(f"  {name:<30} {m['n']:>6} {m['n_untied']:>6} {fmt(m['accuracy'])} "
              f"{fmt(m['soft_accuracy'])} {fmt(m['brier'])} {fmt(m['soft_nll'])} {fmt(m['ece'])}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evaluate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", nargs=2, metavar="PREDICTIONS",
                        help="two prediction files; run the paired test and exit")
    parser.add_argument("--data", default="data")
    parser.add_argument("--build", default=None, help="keep one build's rows (original, a generate.py build's name, or a comma list)")
    parser.add_argument("--checkpoint", help="a local directory, a Hub id or `id@revision`")
    parser.add_argument("--split", default="test")
    parser.add_argument("--bands-split", default="validation",
                        help="the split thresholds are fitted on; never the one being reported")
    parser.add_argument("--device", default=None, help="default: cuda when available, else cpu")
    parser.add_argument("--predictions", help="where to save per-decision predictions")
    args = parser.parse_args(argv)

    if args.compare:
        a, b = (load_predictions(Path(p)) for p in args.compare)
        result = mcnemar(a, b)
        print(json.dumps(result, indent=1))
        return 0
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --compare is given")

    import laya
    agent = laya.load(resolve_laya(args.checkpoint), device=args.device)

    rows = load_split(Path(args.data), args.split, args.build)
    if not rows:
        raise SystemExit(f"\n  No '{args.split}' split at {args.data}.\n")
    train_rows = load_split(Path(args.data), "train", args.build)

    print(f"\n  checkpoint  {args.checkpoint}")
    print(f"  split       {args.split}: {len(rows):,} cases")
    scored = score(agent, rows)
    overall = metrics(scored)
    base = baselines(train_rows, scored)

    print_table("Accuracy = agreement with the teacher's argmax, untied labels only.",
                {"OVERALL": overall})
    print(f"  {'  majority class (train-fitted)':<30} {'':>6} {base['n_untied']:>6} "
          f"{fmt(base['majority_class'])}")
    print(f"  {'  random guess':<30} {'':>6} {base['n_untied']:>6} {fmt(base['random_guess'])}")

    by_site = defaultdict(list)
    for row in scored:
        by_site[row["site"]].append(row)
    per_site = {site: metrics(group) for site, group in sorted(by_site.items())}
    print_table("Per site", per_site)

    band_rows = load_split(Path(args.data), args.bands_split, args.build)
    bands: Dict[str, Any] = {}
    if band_rows and args.bands_split != args.split:
        band_scored = score(agent, band_rows)
        bands = propose_bands(band_scored)
        supported = {k: v for k, v in bands.items() if v["supported"]}
        unsupported = [k for k, v in bands.items() if not v["supported"]]
        print(f"\n  Proposed thresholds, fitted on '{args.bands_split}'. A starting point to review,")
        print(f"  not a setting to ship:\n")
        for name, proposal in sorted(supported.items()):
            rule = (f"yes at >= {proposal['yes_at']:.2f}, no at <= {proposal['no_at']:.2f}"
                    if proposal["kind"] == "Band" else f"accept at confidence >= {proposal['min_confidence']:.2f}")
            print(f"    {name:<44} {rule:<36} decides {proposal['decided_share']:.0%}")
        if unsupported:
            print(f"\n  No supported threshold for: {', '.join(unsupported)}")
        print()

    report = {
        "checkpoint": str(args.checkpoint), "split": args.split,
        "contract": "served path; accuracy on untied labels; majority fitted on train",
        "overall": overall, "baselines": base, "per_site": per_site,
        "proposed_bands": bands, "bands_split": args.bands_split,
    }
    # Next to a local checkpoint; a Hub id has no directory to write beside.
    where = Path(args.checkpoint).parent if Path(args.checkpoint).exists() else Path("runs")
    where.mkdir(parents=True, exist_ok=True)
    out = where / f"evaluation-{args.split}.json"
    suffix = 1
    while out.exists():
        out = where / f"evaluation-{args.split}.{suffix}.json"
        suffix += 1
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"  report -> {out}")

    if args.predictions:
        Path(args.predictions).parent.mkdir(parents=True, exist_ok=True)
        Path(args.predictions).write_text(json.dumps(
            serialise(scored, checkpoint=str(args.checkpoint), split=args.split,
                      dataset_sha256=data_fingerprint(rows))), encoding="utf-8")
        print(f"  predictions -> {args.predictions}")

    if overall["accuracy"] is not None and base["majority_class"] is not None \
            and overall["accuracy"] <= base["majority_class"]:
        print("\n  The checkpoint does not beat always answering the majority class.")
        print("  More data or better labels, not more epochs.\n")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
