"""Verify the CalDec Ultra evaluation evidence without loading any model.

    python3 scripts/check_evidence_ultra.py            # verify
    python3 scripts/check_evidence_ultra.py --write    # regenerate manifest and numbers

`runs/evidence/ultra/` holds the per-decision prediction vectors behind ULTRA.md:
assistant-test and public-test predictions at its top level, the long test in `long/`,
the long-state probe in `probe/`, Decision Index subset summaries in `index/` and
latency measurements in `latency/`. The check confirms every file's hash; that test
files were scored against the released test split, long files against
`data-long/test.jsonl` and public files against the pinned `LocalLLaMA/typed-decisions`
revision, each with matching targets; and that every number in `numbers.json`
recalculates from the vectors, including the scores against the relabelled test
targets in `data-relabelled/test.jsonl`. CalDec Laya, CalDec GLiNER and the upstream
baselines are read from `runs/evidence/` (verified by `check_evidence.py`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/evidence"
ULTRA = BASE / "ultra"
sys.path.insert(0, str(ROOT / "scripts"))
from data import REVISIONS, load_split  # noqa: E402
from evaluate import data_fingerprint, hit, is_tied, load_predictions, mcnemar, metrics, target_vector  # noqa: E402

# Models scored on the assistant and public tests, in the order of ULTRA.md's tables.
RELEASED = ["caldec-ultra-380m", "caldec-ultra-150m"]
SEEDS = ["caldec-ultra-380m-seed2", "caldec-ultra-150m-seed2"]
DEVELOPMENT = [
    "dev-ultra-380m-short", "dev-ultra-150m-short", "dev-modernbert-large-short",
    "dev-ultra-150m-packed512", "dev-ultra-150m-packed1k", "dev-ultra-380m-packed1k", "dev-ultra-380m-packed1k-seed2",
    "dev-ultra-150m-packed1k-general", "dev-laya-packed1k", "dev-laya-packed-long", "dev-laya-packed-long-sitebalance",
    "ultrabert-380m-zeroshot", "ultrabert-150m-zeroshot",
]
REFERENCE = ["caldec-laya", "caldec-gliner", "gliner2.5-decide", "laya-specialist", "laya-base"]
# Paired tests ULTRA.md reports, (a, b, split). "relabelled" scores the test predictions
# against data-relabelled/test.jsonl.
PAIRS = [
    ("caldec-ultra-380m", "caldec-laya", "test"), ("caldec-ultra-380m", "caldec-gliner", "test"),
    ("caldec-ultra-380m", "caldec-ultra-150m", "test"), ("caldec-ultra-380m", "dev-ultra-380m-short", "test"),
    ("caldec-ultra-150m", "dev-ultra-150m-short", "test"), ("caldec-ultra-150m", "caldec-laya", "test"),
    ("caldec-ultra-150m", "caldec-gliner", "test"),
    ("dev-ultra-380m-short", "caldec-laya", "test"), ("dev-ultra-380m-short", "caldec-gliner", "test"),
    ("dev-ultra-380m-short", "dev-modernbert-large-short", "test"), ("dev-modernbert-large-short", "caldec-laya", "test"),
    ("dev-ultra-150m-packed1k", "dev-ultra-150m-short", "test"), ("dev-ultra-380m-packed1k", "dev-ultra-380m-short", "test"),
    ("dev-ultra-380m-packed1k-seed2", "dev-ultra-380m-short", "test"),
    ("dev-ultra-380m-packed1k", "dev-ultra-380m-packed1k-seed2", "test"),
    ("dev-ultra-150m-packed1k-general", "dev-ultra-150m-packed1k", "test"),
    ("dev-laya-packed-long", "caldec-laya", "test"), ("dev-laya-packed-long-sitebalance", "caldec-laya", "test"),
    ("caldec-ultra-380m", "dev-laya-packed-long", "test"),
    ("caldec-ultra-380m", "caldec-laya", "public"), ("caldec-ultra-150m", "caldec-laya", "public"),
    ("caldec-ultra-380m", "dev-ultra-380m-short", "public"), ("caldec-ultra-150m", "dev-ultra-150m-short", "public"),
    ("caldec-ultra-380m", "caldec-laya", "relabelled"), ("caldec-ultra-380m", "caldec-gliner", "relabelled"),
    ("caldec-ultra-380m", "caldec-ultra-150m", "relabelled"), ("caldec-ultra-380m", "dev-ultra-380m-short", "relabelled"),
    ("caldec-ultra-380m", "dev-laya-packed-long", "relabelled"),
    ("caldec-ultra-380m", "caldec-ultra-380m-seed2", "test"), ("caldec-ultra-150m", "caldec-ultra-150m-seed2", "test"),
    ("caldec-ultra-380m", "caldec-ultra-380m-seed2", "long"), ("caldec-ultra-150m", "caldec-ultra-150m-seed2", "long"),
    ("caldec-ultra-380m", "caldec-laya", "long"), ("caldec-ultra-150m", "caldec-laya", "long"),
    ("caldec-ultra-380m", "caldec-ultra-150m", "long"), ("caldec-ultra-380m", "dev-ultra-380m-short", "long"),
    ("caldec-ultra-150m", "dev-ultra-150m-short", "long"), ("caldec-ultra-380m", "caldec-gliner", "long"),
]
METRIC_KEYS = ("n", "n_untied", "accuracy", "soft_accuracy", "brier", "soft_nll", "ece")
POSITIONS = [(L, w) for L in ("4000", "8000") for w in ("start", "middle", "end")]


def _short_path(slug: str, split: str) -> Path:
    return (BASE if slug in REFERENCE else ULTRA) / f"{slug}.{split}.predictions.json"


def _long_rows(slug: str):
    """A model's long-test predictions; CalDec GLiNER's are scored one build per file."""
    path = ULTRA / "long" / f"{slug}.predictions.json"
    if path.is_file():
        return load_predictions(path)
    parts = sorted((ULTRA / "long").glob(f"{slug}.long*.predictions.json"))
    if not parts:
        return None
    rows = load_predictions(parts[0])
    for extra in parts[1:]:
        rows.update(load_predictions(extra))
    rows.header = {**rows.header, "dataset_sha256": "per-build"}
    return rows


def _accuracy(rows) -> float:
    hits = [h for h in (hit(r) for r in rows) if h is not None]
    return round(float(np.mean(hits)), 4) if hits else None


def _relabel(preds, relabelled):
    """The same predictions with the relabelled targets in place of the originals."""
    out = type(preds)(header={**preds.header, "dataset_sha256": "relabelled"})
    for key, row in preds.items():
        out[key] = {**row, "t": relabelled[key]}
    return out


def compute():
    out = {"metrics": {}, "long": {}, "probe": {}, "relabelled": {}, "paired": []}
    rows = {}
    for slug in RELEASED + SEEDS + DEVELOPMENT + REFERENCE:
        for split in ("test", "public"):
            path = _short_path(slug, split)
            if path.is_file():
                rows[(slug, split)] = load_predictions(path)
                m = metrics(list(rows[(slug, split)].values()))
                out["metrics"].setdefault(slug, {})[split] = {k: m[k] for k in METRIC_KEYS}

    relabelled_rows = load_split(ROOT / "data-relabelled", "test")
    relabelled = {(r["case_id"], q): target_vector(question, r["gold"][q])
                  for r in relabelled_rows for q, question in r["questions"].items()}
    original = rows[("caldec-laya", "test")]
    both = [k for k in original if not is_tied(original[k]["t"]) and not is_tied(relabelled[k])]
    out["relabelled"]["_targets"] = {
        "decisions": len(relabelled),
        "untied": sum(not is_tied(t) for t in relabelled.values()),
        "argmax_agreement_with_original": round(float(np.mean(
            [np.argmax(original[k]["t"]) == np.argmax(relabelled[k]) for k in both])), 4),
        "both_untied": len(both)}
    for (slug, split), preds in list(rows.items()):
        if split == "test":
            rows[(slug, "relabelled")] = _relabel(preds, relabelled)
            out["relabelled"][slug] = _accuracy(rows[(slug, "relabelled")].values())

    for path in sorted((ULTRA / "long").glob("*.predictions.json")):
        slug = path.name.split(".")[0]
        if (slug, "long") in rows:
            continue
        rows[(slug, "long")] = preds = _long_rows(slug)
        by = lambda prefix: [r for k, r in preds.items() if k[0].startswith(prefix)]  # noqa: E731
        out["long"][slug] = {"accuracy": _accuracy(preds.values()), "long8k": _accuracy(by("long8k")),
                             "long4k": _accuracy(by("long4k")),
                             "n_untied": sum(hit(r) is not None for r in preds.values())}

    for path in sorted((ULTRA / "probe").glob("*.predictions.json")):
        slug = path.name.split(".")[0]
        preds = load_predictions(path)
        entry = {}
        for L, where in POSITIONS:
            entry[f"{L}/{where}"] = _accuracy(r for k, r in preds.items() if k[0].startswith(f"needle-{L}-{where}-"))
        for L in ("4000", "8000"):
            page = [r for k, r in preds.items() if k[0].startswith(f"needle-{L}-") and hit(r) is not None]
            clean = [r for r in page if np.argmax(r["t"]) == 0]
            injected = [r for r in page if np.argmax(r["t"]) == 1]
            entry[f"{L}/clean_flagged"] = round(float(np.mean([np.argmax(r["p"]) == 1 for r in clean])), 4)
            entry[f"{L}/injected_caught"] = round(float(np.mean([np.argmax(r["p"]) == 1 for r in injected])), 4)
        out["probe"][slug] = entry

    # Decision Index subset: the kit's own scores, with iSarcasmEval as the board reports it
    # (sarcasm F1 on track A, English) from the kept confusion counts; board rows for reference.
    out["index"] = {}
    for path in sorted((ULTRA / "index").glob("*.benchmark-summary.json")):
        slug = path.name.split(".")[0]
        summary = json.loads(path.read_text(encoding="utf-8"))
        scores = {str(b["catalog_id"]): round(b["score"], 4) for b in summary["benchmarks"] if isinstance(b, dict) and b.get("score") is not None}
        counts = ULTRA / "index" / f"{slug}.isarcasm-a-en.json"
        if counts.is_file():
            c = json.loads(counts.read_text(encoding="utf-8"))
            scores["40"] = round(2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"]), 4) if c["tp"] else 0.0
        else:
            scores.pop("40", None)
        out["index"][slug] = scores
    board = json.loads((ULTRA / "index/board.json").read_text(encoding="utf-8"))
    out["index_board"] = {name: {cid: r["score"] for cid, r in results.items()} for name, results in board["models"].items()}

    # Latency: median ms by state length for each model in each measurement.
    out["latency"] = {}
    for path in sorted((ULTRA / "latency").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if "models" in doc:
            out["latency"][path.stem] = {m: {L: v["p50_ms"] for L, v in d["by_length"].items()} for m, d in doc["models"].items()}

    for a, b, split in PAIRS:
        if (a, split) in rows and (b, split) in rows:
            ra, rb = rows[(a, split)], rows[(b, split)]
            if split == "long" and "per-build" in (ra.header.get("dataset_sha256"), rb.header.get("dataset_sha256")):
                ra = type(ra)(ra, header={**ra.header, "dataset_sha256": "per-build"})
                rb = type(rb)(rb, header={**rb.header, "dataset_sha256": "per-build"})
            out["paired"].append({"a": a, "b": b, "split": split, **mcnemar(ra, rb)})
    return rows, out


def verify() -> list[str]:
    problems = []
    manifest = json.loads((ULTRA / "manifest.json").read_text(encoding="utf-8"))["files"]
    on_disk = {str(p.relative_to(ULTRA)) for p in ULTRA.rglob("*") if p.is_file()} - {"manifest.json", "numbers.json"}
    for name in sorted(on_disk - set(manifest)):
        problems.append(f"evidence file not in the manifest: {name}")
    for name, expected in manifest.items():
        path = ULTRA / name
        if not path.is_file():
            problems.append(f"missing evidence file: {name}")
        elif path.stat().st_size != expected["size"] or \
                hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            problems.append(f"evidence changed: {name}")
    if problems:
        return problems

    references = {}
    for name, directory in (("test", ROOT / "data"), ("long", ROOT / "data-long")):
        split = load_split(directory, "test")
        references[name] = (data_fingerprint(split), {(r["case_id"], q): target_vector(question, r["gold"][q])
                                                      for r in split for q, question in r["questions"].items()})
    v1_numbers = json.loads((ROOT / "runs/numbers.json").read_text(encoding="utf-8"))
    rows, recomputed = compute()
    for (slug, split), preds in rows.items():
        header = preds.header
        if split in ("test", "long"):
            fingerprint, targets = references[split]
            if header.get("dataset_sha256") not in (fingerprint, "per-build"):
                problems.append(f"{slug}: {split} predictions scored against a different file")
            bad = [k for k, r in preds.items() if k not in targets or not np.allclose(r["t"], targets[k], atol=1e-4)]
            if bad or len(preds) != len(targets):
                problems.append(f"{slug}: {split} targets do not match the released split ({len(bad)} differ)")
        elif split == "public" and (header.get("dataset_sha256") != v1_numbers["public_dataset_sha256"] or
                                    header.get("public_revision") != REVISIONS["LocalLLaMA/typed-decisions"]):
            problems.append(f"{slug}: public predictions have a different dataset revision")

    claimed = json.loads((ULTRA / "numbers.json").read_text(encoding="utf-8"))
    if json.dumps(claimed, sort_keys=True) != json.dumps(recomputed, sort_keys=True):
        problems.append("numbers.json does not match the metrics recalculated from the predictions")
    return problems


def write() -> None:
    _rows, numbers = compute()
    (ULTRA / "numbers.json").write_text(json.dumps(numbers, indent=1) + "\n", encoding="utf-8")
    entries = {str(p.relative_to(ULTRA)): {"size": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
               for p in sorted(ULTRA.rglob("*")) if p.is_file() and p.name not in ("manifest.json", "numbers.json")}
    (ULTRA / "manifest.json").write_text(json.dumps({"files": entries}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote numbers.json ({len(numbers['metrics'])} models, {len(numbers['paired'])} paired tests) "
          f"and manifest.json ({len(entries)} files)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="regenerate numbers.json and manifest.json")
    if parser.parse_args().write:
        write()
        return 0
    problems = verify()
    for problem in problems:
        print(problem)
    if problems:
        return 1
    print("CalDec Ultra evaluation evidence verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
