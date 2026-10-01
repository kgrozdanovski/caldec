"""Score several checkpoints side by side, on both test sets.

Two questions, and they are different:

**How does the checkpoint perform on an external dataset?** The
`LocalLLaMA/typed-decisions` test split has 400 cases, 2,000 decisions and
four synthetic workflows.

**Is it good at the assistant's own decisions?** Only this dataset's test split
answers that. Performance on `LocalLLaMA/typed-decisions` need not transfer
to these assistant decisions.

Every model is scored by `evaluate.score`, through the path that serves it, under
one option order and one tie policy. Per-decision predictions are saved per
model, so any two rows in the table can be compared with `evaluate.py --compare`.

**Jev is not measured by this script.** Its reference column is a published
figure, marked as unmeasured here. `jev_decisions.py` makes a separate measured
run through the same scorer (JEV.md).

A `--model` is a local directory, a Hub id or `id@revision`. A bare Hub id that
`data.REVISIONS` knows is loaded at that pinned commit, and `LocalLLaMA/typed-decisions` is
read at its pinned commit too (`--public-revision` overrides it), so a rerun scores
the same weights on the same rows. A model that fails to load stops the run.

    USE_TF=0 python3 scripts/benchmark.py \\
        --model ours=checkpoints/v1-laya/final \\
        --model laya=convaiinnovations/laya \\
        --model laya-td=convaiinnovations/laya-typed-decisions \\
        --data data --out runs/my-benchmark
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("USE_TF", "0")

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

from data import REVISIONS, build_items, load_split, resolve_laya      # noqa: E402
from evaluate import baselines, data_fingerprint, metrics, score, serialise  # noqa: E402

PUBLIC_REPO = "LocalLLaMA/typed-decisions"

#: Published on the public split by Convai and by
#: TypeSafe. NOT measured here. The measured endpoint path is jev_decisions.py.
PUBLISHED: Dict[str, Dict[str, Any]] = {
    "TypeSafe Jev 1.13.0": {
        "accuracy": 0.727, "soft_accuracy": 0.580, "brier": 0.148,
        "ece": 0.144, "score_mae": 0.391, "source": "published by TypeSafe, via the Laya card",
    },
    "teacher self-agreement": {"accuracy": 0.735, "source": "Laya card: teacher self-agreement"},
    "ModernBERT-base specialist": {"accuracy": 0.646, "source": "published, via the Laya card"},
    "laya-typed-decisions": {
        "accuracy": 0.766, "soft_accuracy": 0.471, "brier": 0.062,
        "ece": 0.213, "score_mae": 0.242, "source": "Convai's own measurement",
    },
    "laya (base)": {
        "accuracy": 0.362, "soft_accuracy": 0.332, "brier": 0.316,
        "ece": 0.175, "score_mae": 0.694, "source": "Convai's own measurement",
    },
}


# --------------------------------------------------------------------------- #
# The public split
# --------------------------------------------------------------------------- #

def load_public(split: str = "test", config: str = "all",
                revision: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read the benchmark's parquet straight from the Hub with pyarrow, at a pinned
    commit. Case ids are positional (`<repo>/<split>/<row>`), so two prediction files
    pair correctly only when both were scored on the same revision; pinning it is what
    keeps `evaluate.py --compare` meaningful across reruns."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=PUBLIC_REPO, repo_type="dataset", revision=revision or REVISIONS[PUBLIC_REPO],
        filename=f"{config}/{split}-00000-of-00001.parquet",
    )
    table = pq.read_table(path).to_pylist()
    rows = []
    for index, record in enumerate(table):
        rows.append({
            "case_id": f"{PUBLIC_REPO}/{split}/{index}",
            "state": json.loads(record["state"]),
            "questions": json.loads(record["questions"]),
            "gold": json.loads(record["gold"]),
            "site": record.get("workflow") or config,
        })
    return rows


def build_public_items(rows: List[Dict[str, Any]], tok, cfg) -> List[Dict[str, Any]]:
    """Tokenized items for the public rows. `load_public` returns them in this
    dataset's row shape, so this is `data.build_items`."""
    return build_items(rows, tok, cfg)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _fmt(value: Optional[float], width: int = 7) -> str:
    return f"{'—':>{width}}" if value is None else f"{value:>{width}.3f}"


def table(title: str, rows: List[tuple], note: str = "") -> str:
    lines = [f"\n## {title}\n",
             "| model | n | untied | acc | soft acc | Brier | soft NLL | ECE | score MAE | measured |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, m, measured in rows:
        lines.append(
            f"| {name} | {m.get('n', '')} | {m.get('n_untied', '')} | {_fmt(m.get('accuracy')).strip()} | "
            f"{_fmt(m.get('soft_accuracy')).strip()} | {_fmt(m.get('brier')).strip()} | "
            f"{_fmt(m.get('soft_nll')).strip()} | {_fmt(m.get('ece')).strip()} | "
            f"{_fmt(m.get('score_mae')).strip()} | {'yes' if measured else '**no — published**'} |"
        )
    if note:
        lines.append(f"\n{note}")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmark", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", default=[], metavar="NAME=PATH",
                        help="a checkpoint to score; repeat for each")
    parser.add_argument("--data", default="data", help="the dataset directory")
    parser.add_argument("--build", default=None, help="keep one build's rows (original, a generate.py build's name, or a comma list)")
    parser.add_argument("--out", default="runs/benchmark",
                        help="where the report and predictions are written (runs/ is ignored by git)")
    parser.add_argument("--public-revision", default=None,
                        help=f"commit of {PUBLIC_REPO} to score on (default: the one pinned in data.REVISIONS)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-public", action="store_true")
    parser.add_argument("--skip-internal", action="store_true")
    args = parser.parse_args(argv)

    if not args.model:
        raise SystemExit("\n  Pass at least one --model NAME=PATH.\n")
    models = []
    for entry in args.model:
        name, _, path = entry.partition("=")
        if not path:
            raise SystemExit(f"  --model wants NAME=PATH, not {entry!r}")
        models.append((name.strip(), path.strip()))

    import laya

    public_revision = args.public_revision or REVISIONS[PUBLIC_REPO]
    public_rows = None if args.skip_public else load_public(revision=public_revision)
    internal_rows = None if args.skip_internal else load_split(Path(args.data), "test", args.build)
    train_rows = load_split(Path(args.data), "train", args.build) if internal_rows else []
    if public_rows:
        print(f"\n  public   {PUBLIC_REPO} test: {len(public_rows):,} cases")
    if internal_rows:
        print(f"  internal {args.data} test: {len(internal_rows):,} cases")
    if not public_rows and not internal_rows:
        raise SystemExit("\n  Nothing to score.\n")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "contract": "served path (laya.Agent.system_one); accuracy and ECE on untied labels; "
                    "majority class fitted on train, weighted by decision",
        "public_benchmark": PUBLIC_REPO,
        "public_revision": None if args.skip_public else public_revision,
        "cases": {"public": len(public_rows or []), "internal": len(internal_rows or [])},
        "models": {},
        "published_reference": PUBLISHED,
    }
    public_rows_out: List[tuple] = []
    internal_rows_out: List[tuple] = []
    per_site: Dict[str, Dict[str, Any]] = {}

    for name, path in models:
        print(f"\n  loading {name}  <- {path}")
        # A model that does not load stops the run: a report silently missing a row
        # reads as a finished comparison.
        try:
            agent = laya.load(resolve_laya(path), device=args.device)
        except Exception as exc:                                # noqa: BLE001
            raise SystemExit(f"\n  could not load {name} from {path}: {exc}\n") from exc
        entry: Dict[str, Any] = {"path": path, "predictions": {}}

        if public_rows:
            scored = score(agent, public_rows)
            m = metrics(scored)
            entry["public"] = m
            public_rows_out.append((name, m, True))
            print(f"    public   acc {_fmt(m['accuracy'])} soft {_fmt(m['soft_accuracy'])} "
                  f"brier {_fmt(m['brier'])} nll {_fmt(m['soft_nll'])} ece {_fmt(m['ece'])} "
                  f"(n={m['n']}, untied={m['n_untied']})")
            by_wf = defaultdict(list)
            for row in scored:
                by_wf[row["site"]].append(row)
            entry["public_by_workflow"] = {w: metrics(g) for w, g in by_wf.items()}
            pred = out_dir / f"predictions-public-{name}-{stamp}.json"
            pred.write_text(json.dumps(serialise(scored, checkpoint=path, split="public-test",
                                                public_revision=public_revision,
                                                dataset_sha256=data_fingerprint(public_rows))), encoding="utf-8")
            entry["predictions"]["public"] = str(pred)

        if internal_rows:
            scored = score(agent, internal_rows)
            m = metrics(scored)
            entry["internal"] = m
            internal_rows_out.append((name, m, True))
            print(f"    internal acc {_fmt(m['accuracy'])} soft {_fmt(m['soft_accuracy'])} "
                  f"brier {_fmt(m['brier'])} nll {_fmt(m['soft_nll'])} ece {_fmt(m['ece'])} "
                  f"(n={m['n']}, untied={m['n_untied']})")
            by_site = defaultdict(list)
            for row in scored:
                by_site[row["site"]].append(row)
            entry["internal_by_site"] = {s: metrics(g) for s, g in by_site.items()}
            for site, group in by_site.items():
                per_site.setdefault(site, {})[name] = metrics(group)
            entry["internal_baselines"] = baselines(train_rows, scored)
            pred = out_dir / f"predictions-internal-{name}-{stamp}.json"
            pred.write_text(json.dumps(serialise(scored, checkpoint=path, split="test",
                                                dataset_sha256=data_fingerprint(internal_rows))), encoding="utf-8")
            entry["predictions"]["internal"] = str(pred)

        report["models"][name] = entry
        del agent
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # --- the report --------------------------------------------------------- #
    lines = [
        "# Decision model: side-by-side",
        "",
        f"Generated {report['generated_at']}, on {args.device}.",
        "",
        "Every measured row is scored through `laya.Agent.system_one`, the path that",
        "serves the checkpoint. **Accuracy is agreement with the argmax of the teacher's",
        "label, on untied labels only.** ECE is computed on the same decisions. The",
        "distribution metrics use every decision.",
        "",
        "**Jev is not measured by this script.** Its row is the figure published by",
        "TypeSafe and quoted on the Laya model card, on the same public split. The",
        "measured Jev rows come from `jev_decisions.py` (JEV.md).",
    ]

    if public_rows_out:
        rows = list(public_rows_out)
        for name in ("TypeSafe Jev 1.13.0", "teacher self-agreement", "ModernBERT-base specialist"):
            rows.append((name, PUBLISHED[name], False))
        lines.append(table(
            f"{PUBLIC_REPO} test ({len(public_rows)} cases)", rows,
            "Teacher self-agreement is the label source agreeing with itself on a second\n"
            "pass. It is not a ceiling on a student: a student that learns the rule can\n"
            "exceed it. Read it as the noise level of the labels.",
        ))

    if internal_rows_out:
        lines.append(table(
            f"The assistant's own decisions — this dataset's test split ({len(internal_rows)} cases)",
            internal_rows_out,
            "No published figure exists for this split, by construction: these are\n"
            "questions only this dataset asks.",
        ))
        first = models[0][0] if models else ""
        base = report["models"].get(first, {}).get("internal_baselines")
        if base:
            lines.append(
                f"\nBaselines on the internal split, fitted on train and weighted by decision: "
                f"majority class **{_fmt(base['majority_class']).strip()}**, random guess "
                f"**{_fmt(base['random_guess']).strip()}** (n untied = {base['n_untied']}). "
                f"A model below majority class learned nothing."
            )

    if per_site:
        lines.append("\n### Internal split, per site (accuracy on untied labels)\n")
        names = [n for n, _ in models if n in report["models"]]
        lines.append("| site | n | untied | " + " | ".join(names) + " |")
        lines.append("|---|---:|---:|" + "|".join(["---:"] * len(names)) + "|")
        for site in sorted(per_site):
            row = per_site[site]
            n = next((m["n"] for m in row.values()), "")
            untied = next((m["n_untied"] for m in row.values()), "")
            cells = [_fmt(row[n2]["accuracy"]).strip() if n2 in row else "" for n2 in names]
            lines.append(f"| `{site}` | {n} | {untied} | " + " | ".join(cells) + " |")

    md_path = out_dir / f"benchmark-{stamp}.md"
    json_path = out_dir / f"benchmark-{stamp}.json"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\n  report -> {md_path}\n            {json_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
