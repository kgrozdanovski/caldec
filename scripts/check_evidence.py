"""Verify the compact, public evaluation evidence without loading either model.

    python3 scripts/check_evidence.py

The prediction vectors are copies of the original scored runs. Their manifest hashes
make accidental edits visible; recalculation checks the reported metrics and paired
test against those vectors and the released assistant test targets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "runs/evidence"
sys.path.insert(0, str(ROOT / "scripts"))
from data import REVISIONS, load_split  # noqa: E402
from evaluate import data_fingerprint, is_tied, load_predictions, mcnemar, metrics, target_vector  # noqa: E402


def verify() -> list[str]:
    problems = []
    manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))["files"]
    for name, expected in manifest.items():
        path = EVIDENCE / name
        if not path.is_file():
            problems.append(f"missing evidence file: {name}")
            continue
        if path.stat().st_size != expected["size"] or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            problems.append(f"evidence changed: {name}")
    if problems:
        return problems

    numbers = json.loads((ROOT / "runs/numbers.json").read_text(encoding="utf-8"))
    benchmark = json.loads((EVIDENCE / "benchmark.json").read_text(encoding="utf-8"))
    jev_full_report = json.loads((EVIDENCE / "jev-full.summary.json").read_text(encoding="utf-8"))

    test = load_split(ROOT / "data", "test")
    targets = {(row["case_id"], qid): target_vector(question, row["gold"][qid])
               for row in test for qid, question in row["questions"].items()}
    test_fingerprint = data_fingerprint(test)

    def check_header(slug: str, split: str, rows) -> None:
        if split == "test" and rows.header.get("dataset_sha256") != test_fingerprint:
            problems.append(f"{slug} predictions were scored against a different test file")
        if split == "public" and (
                rows.header.get("dataset_sha256") != numbers["public_dataset_sha256"] or
                rows.header.get("public_revision") != REVISIONS["LocalLLaMA/typed-decisions"]):
            problems.append(f"{slug} public predictions have a different dataset revision")

    predictions = {}
    for model, slug in (("caldec-laya", "caldec-laya"), ("laya", "laya-base"),
                        ("laya-td", "laya-specialist")):
        for split in ("test", "public"):
            path = EVIDENCE / f"{slug}.{split}.predictions.json"
            rows = load_predictions(path)
            predictions[(slug, split)] = rows
            check_header(slug, split, rows)
            actual = metrics(list(rows.values()))
            expected = benchmark["models"][model]["internal" if split == "test" else "public"]
            if actual != expected:
                problems.append(f"{path.name}: metrics differ from benchmark report")
            if actual != numbers["models"][model]["internal" if split == "test" else "public"]["overall"]:
                problems.append(f"{path.name}: metrics differ from runs/numbers.json")
    gliner = {}
    for slug, report_key in (("caldec-gliner", "caldec_gliner"),
                             ("gliner2.5-decide", "gliner2_5_decide")):
        for split in ("test", "public"):
            key = report_key if split == "test" else report_key + "_public"
            report = json.loads((EVIDENCE / f"{slug}.{split}.eval.json").read_text(encoding="utf-8"))
            rows = load_predictions(EVIDENCE / f"{slug}.{split}.predictions.json")
            gliner[(slug, split)] = rows
            if metrics(list(rows.values())) != report["overall"] or report != numbers[key]:
                problems.append(f"{slug} {split} predictions, report and numbers differ")
            check_header(slug, split, rows)
    if (jev_full_report["sets"]["internal"]["metrics"] != numbers["assistant_jev"] or
            jev_full_report["sets"]["internal"]["cases"] != len(test) or
            jev_full_report["sets"]["internal"]["missing_decisions"] != 0 or
            jev_full_report["sets"]["internal"].get("dataset_sha256") != test_fingerprint):
        problems.append("full-test Jev summary and numbers differ")

    for name, rows in [(slug, predictions[(slug, "test")]) for slug in
                       ("caldec-laya", "laya-base", "laya-specialist")] + [
                           ("caldec-gliner", gliner[("caldec-gliner", "test")]),
                           ("gliner2.5-decide", gliner[("gliner2.5-decide", "test")])]:
        if set(rows) != set(targets):
            problems.append(f"{name}: assistant test prediction coverage differs from data/test.jsonl")
            continue
        if any(rows[key]["t"].shape != target.shape or
               not np.allclose(rows[key]["t"], target, atol=1e-5)
               for key, target in targets.items()):
            problems.append(f"{name}: assistant test targets differ from data/test.jsonl")

    paired = mcnemar(predictions[("caldec-laya", "test")], gliner[("caldec-gliner", "test")])
    reported = numbers["paired_caldec_laya_gliner"]
    if (paired["paired"] != reported["n"] or
            paired["a_only_correct"] != reported["laya_only"] or
            paired["b_only_correct"] != reported["gliner_only"] or
            paired["p_two_sided"] != reported["p_two_sided"]):
        problems.append("Laya/GLiNER paired result differs from runs/numbers.json")
    laya_rows = predictions[("caldec-laya", "test")]
    gliner_rows = gliner[("caldec-gliner", "test")]
    different_answers = sum(
        np.argmax(row["p"]) != np.argmax(gliner_rows[key]["p"])
        for key, row in laya_rows.items() if not is_tied(row["t"])
    )
    if different_answers != reported["predicted_labels_differ"]:
        problems.append("CalDec models' differing predictions do not match runs/numbers.json")

    public_reference = predictions[("caldec-laya", "public")]
    for name in ("laya-base", "laya-specialist"):
        mcnemar(public_reference, predictions[(name, "public")])
    # The comparisons the docs report as point gaps must stay pairable by --compare.
    mcnemar(gliner[("caldec-gliner", "test")], gliner[("gliner2.5-decide", "test")])
    mcnemar(predictions[("caldec-laya", "test")], predictions[("laya-base", "test")])
    mcnemar(gliner[("caldec-gliner", "public")], gliner[("gliner2.5-decide", "public")])
    mcnemar(public_reference, gliner[("caldec-gliner", "public")])
    for (slug, split), rows in gliner.items():
        if split == "public":
            if set(rows) != set(public_reference):
                problems.append(f"{slug}: public prediction coverage differs from CalDec Laya")
            elif any(rows[key]["t"].shape != public_reference[key]["t"].shape or
                     not np.allclose(rows[key]["t"], public_reference[key]["t"], atol=1e-5)
                     for key in public_reference):
                problems.append(f"{slug}: public targets differ from CalDec Laya")
    return problems


def main() -> int:
    problems = verify()
    for problem in problems:
        print(problem)
    print("evaluation evidence verified" if not problems else f"{len(problems)} evidence problem(s)")
    return bool(problems)


if __name__ == "__main__":
    raise SystemExit(main())
