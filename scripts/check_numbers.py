"""Check the release summary, headline prose, Hub cards and SVG figures.

    python3 scripts/check_numbers.py

This checks the committed summary reports and charts. Run check_evidence.py to
recalculate metrics from the committed per-decision predictions.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import charts  # noqa: E402


def totals_from_data() -> dict:
    splits = {}
    for split in ("train", "validation", "test"):
        cases = decisions = tied = 0
        for line in (ROOT / "data" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cases += 1
            row = json.loads(line)
            for q in json.loads(row["gold"]).values():
                probs = sorted(q["probabilities"].values())
                decisions += 1
                tied += len(probs) > 1 and abs(probs[-1] - probs[-2]) < 1e-9
        splits[split] = {"cases": cases, "decisions": decisions, "tied": tied,
                         "untied": decisions - tied}
    return {"cases": sum(x["cases"] for x in splits.values()),
            "decisions": sum(x["decisions"] for x in splits.values()), "splits": splits}


def main() -> int:
    numbers = json.loads((ROOT / "runs/numbers.json").read_text(encoding="utf-8"))
    problems = []
    if numbers["totals"] != totals_from_data():
        problems.append("runs/numbers.json split totals differ from data/")
    benchmark_path = ROOT / numbers["source"]["laya_benchmark"]
    if not benchmark_path.exists():
        problems.append(f"missing source report: {benchmark_path.relative_to(ROOT)}")
    else:
        report = json.loads(benchmark_path.read_text(encoding="utf-8"))
        for name in ("caldec-laya", "laya", "laya-td"):
            for split in ("internal", "public"):
                if numbers["models"][name][split]["overall"] != report["models"][name][split]:
                    problems.append(f"{name} {split} metrics differ from saved benchmark")
    for source, report_key in (
        ("caldec_gliner_test", "caldec_gliner"),
        ("caldec_gliner_public", "caldec_gliner_public"),
        ("gliner2_5_decide_test", "gliner2_5_decide"),
        ("gliner2_5_decide_public", "gliner2_5_decide_public"),
    ):
        path = ROOT / numbers["source"][source]
        if not path.exists():
            problems.append(f"missing source report: {path.relative_to(ROOT)}")
        elif numbers[report_key] != json.loads(path.read_text(encoding="utf-8")):
            problems.append(f"{report_key} metrics differ from saved evaluation")
    jev_path = ROOT / numbers["source"]["public_jev"]
    if not jev_path.exists():
        problems.append(f"missing source report: {jev_path.relative_to(ROOT)}")
    elif numbers["public_jev"] != json.loads(jev_path.read_text(encoding="utf-8"))["sets"]["public"]["metrics"]:
        problems.append("Jev public metrics differ from saved evaluation")
    assistant_jev_path = ROOT / numbers["source"]["assistant_jev"]
    if not assistant_jev_path.exists():
        problems.append(f"missing source report: {assistant_jev_path.relative_to(ROOT)}")
    elif numbers["assistant_jev"] != json.loads(assistant_jev_path.read_text(encoding="utf-8"))["sets"]["internal"]["metrics"]:
        problems.append("Jev assistant metrics differ from saved evaluation")
    laya = numbers["models"]["caldec-laya"]["internal"]["overall"]
    gliner = numbers["caldec_gliner"]["overall"]
    decide = numbers["gliner2_5_decide"]["overall"]
    assistant_jev = numbers["assistant_jev"]
    if len({(m["n"], m["n_untied"]) for m in (laya, gliner, decide, assistant_jev)}) != 1:
        problems.append("full-test scores cover different decision counts")
    if numbers["accuracy"]["assistant"]["CalDec Laya"] != laya["accuracy"]:
        problems.append("CalDec Laya chart accuracy differs from metrics")
    if numbers["accuracy"]["assistant"]["CalDec GLiNER"] != gliner["accuracy"]:
        problems.append("CalDec GLiNER chart accuracy differs from metrics")
    if numbers["accuracy"]["assistant"]["GLiNER2.5-Decide"] != decide["accuracy"]:
        problems.append("GLiNER2.5-Decide chart accuracy differs from metrics")
    if numbers["accuracy"]["assistant"]["Jev 1.13"] != assistant_jev["accuracy"]:
        problems.append("Jev chart accuracy differs from metrics")
    for label, score in (
        ("CalDec Laya", numbers["models"]["caldec-laya"]["public"]["overall"]["accuracy"]),
        ("CalDec GLiNER", numbers["caldec_gliner_public"]["overall"]["accuracy"]),
        ("GLiNER2.5-Decide", numbers["gliner2_5_decide_public"]["overall"]["accuracy"]),
    ):
        if numbers["accuracy"]["public"][label] != score:
            problems.append(f"public chart accuracy differs from {label} metrics")
    latency = numbers["latency_ms"]
    for slug, key in (("caldec-laya", "caldec_laya"), ("caldec-gliner", "caldec_gliner")):
        report = json.loads((ROOT / "runs/evidence" / f"{slug}.latency.json").read_text(encoding="utf-8"))
        if (report["device"] != latency["device"] or report["case_id"] != latency["case_id"] or
                report["warmup"] != latency["warmup"] or report["samples"] != latency["samples"]):
            problems.append(f"{slug} latency setup differs from runs/numbers.json")
        for count in (1, 3):
            for percentile in (50, 95):
                name = f"{count}q_p{percentile}"
                if report["results"][f"{count}q"][f"p{percentile}_ms"] != latency[key][name]:
                    problems.append(f"{slug} {name} differs from runs/numbers.json")
    expected_pages = {
        "README.md": ("0.823", "0.843", "0.836", "0.650", "0.582", "0.540"),
        "RESULTS.md": ("0.823", "0.843", "0.836", "0.650", "0.582", "0.540"),
        "huggingface/dataset-card.md": ("0.823", "0.843", "0.836", "0.650", "0.582", "0.540"),
        "huggingface/model-card.md": ("0.823",),
        "huggingface/gliner2-model-card.md": ("0.843", "0.823", "0.582"),
    }
    for page, expected in expected_pages.items():
        content = (ROOT / page).read_text(encoding="utf-8")
        if "LocalLLaMA/typed-decisions" not in content:
            problems.append(f"{page} does not name the external benchmark")
        for value in expected:
            if value not in content:
                problems.append(f"{page} does not contain current accuracy {value}")
        if page in ("README.md", "huggingface/dataset-card.md"):
            for value in ("4,984", "805", "1,647"):
                if value not in content:
                    problems.append(f"{page} does not contain split size {value}")
    for theme in charts.THEMES:
        for name, renderer in (("accuracy", charts.accuracy_chart),
                               ("calibration", charts.calibration_chart),
                               ("per-site", charts.per_site_chart)):
            path = ROOT / "assets" / f"{name}-{theme}.svg"
            if path.read_text(encoding="utf-8") != renderer(numbers, theme):
                problems.append(f"{path.relative_to(ROOT)} is stale; run scripts/charts.py")
    for problem in problems:
        print(problem)
    print("numbers and figures consistent" if not problems else f"{len(problems)} inconsistency(s)")
    return bool(problems)


if __name__ == "__main__":
    raise SystemExit(main())
