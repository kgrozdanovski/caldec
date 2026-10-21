"""Relabel an existing split with other annotators, for a second set of test targets.

The released test targets are single GLM 5.3 judgments, and the test informed model
development. Relabelling the same states with independent model families measures
how much of a model's accuracy is agreement with GLM in particular. The relabel is a
generate.py build whose candidates are the existing rows (generator recorded as the
original labeller, so its family is excluded from the new labels), labelled and
frozen by generate.py's own `label` and `freeze`. It is never folded into a split:
`export` writes the relabelled rows as a separate dataset directory instead.

    python3 scripts/relabel.py init   --config builds/us_relabel_test.json --source data/test.jsonl
    python3 scripts/generate.py label  --build us_relabel_test
    python3 scripts/generate.py freeze --build us_relabel_test
    python3 scripts/relabel.py export --build us_relabel_test --out data-uslabels

`export` writes `<out>/test.jsonl` (the split named by the build) plus the
validation split and site files copied from data/, so `evaluate.py --data <out>`
scores against the new targets.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate  # noqa: E402
import llm  # noqa: E402
from data import decode  # noqa: E402

ORIGINAL_LABELLER = "z-ai/glm-5.3"


def init(config: Path, source: Path, per_call: int) -> None:
    generate.initialise(config)
    protocol = generate.load_protocol(json.loads(config.read_text(encoding="utf-8"))["build"])
    out = generate.build_dir(protocol["build"])
    rows = [decode(json.loads(line)) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_site = defaultdict(list)
    for row in rows:
        by_site[row["site"]].append(row)
    candidates = []
    for site, group in sorted(by_site.items()):
        for start in range(0, len(group), per_call):
            tag = f"{protocol['build']}-{site}-{start // per_call}"
            for row in group[start:start + per_call]:
                candidates.append(dict(case_id=row["case_id"], site=site, state=row["state"],
                                       questions=row["questions"], generator=ORIGINAL_LABELLER, cluster_id=tag))
    llm.write_json(out / "candidates.json", candidates)
    llm.write_json(out / "generate.complete.json", {
        "relabel_of": str(source), "accepted_cases": len(candidates),
        "sha256": {"candidates.json": generate.file_sha256(out / "candidates.json")}})
    print(f"{len(candidates)} rows from {source} staged for labelling in {out}")


def export(build: str, out_dir: Path) -> None:
    protocol = generate.load_protocol(build)
    src = generate.build_dir(build)
    if not (src / "freeze.json").exists():
        raise SystemExit("freeze the build first")
    out_dir.mkdir(parents=True, exist_ok=True)
    root = generate.ROOT / "data"
    for name in ("sites.json", "prompts.json", "train.jsonl", "validation.jsonl", "test.jsonl"):
        shutil.copy(root / name, out_dir / name)
    shutil.copy(src / "cases.jsonl", out_dir / f"{protocol['split']}.jsonl")
    print(f"wrote {out_dir / (protocol['split'] + '.jsonl')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["init", "export"])
    ap.add_argument("--config", type=Path)
    ap.add_argument("--source", type=Path)
    ap.add_argument("--per-call", type=int, default=6)
    ap.add_argument("--build")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.command == "init":
        init(args.config, args.source, args.per_call)
    else:
        export(args.build, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
