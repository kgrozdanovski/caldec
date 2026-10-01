"""Fold a finished build into a working dataset directory's split files.

A build (scripts/generate.py) keeps its
working files under data/provenance/<build>/, which git ignores. Its finished cases
end up appended to <data_dir>/<split>.jsonl in the dataset's one row schema:

    {"case_id", "site", "workflow", "state", "questions", "gold"}

with `state`, `questions` and `gold` stored as JSON strings (scripts/data.py).

Build-only fields (generator, cluster, prompt family, annotator agreement) stay in
the provenance files. The case id carries the build (scripts/data.py `build_of`), so
nothing else marks where a row came from.

    python scripts/fold.py data/provenance/<build>/cases.jsonl --split train --data data-next

The v1 release files in data/ are never a target: their rows, the reported scores and
the committed evidence depend on them. Fold into a working copy instead, and release
it as a new dataset version (CONTRIBUTING.md).

Refuses a case id that already exists in any split, a state that already exists in
any split, a case whose build prefix is the original set's, and a row with a finding
of the publishing scan (scripts/check_data.py). It checks identical states only:
near-duplicates are rejected earlier, by `generate.py generate`.

Run directly, it also refuses a cases file that has no `freeze.json` beside it, whose
hash differs from the frozen one, or whose frozen split is not `--split`. Cases
produced some other way have no freeze record; pass `--unfrozen` to fold them anyway.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_data import scan_row  # noqa: E402
from data import build_of, decode, to_line  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data"
SPLITS = ("train", "validation", "test")
FIELDS = ("case_id", "site", "workflow", "state", "questions", "gold")


def dump(row: dict) -> str:
    """A row as plain nested JSON: the build records' format, and the split files' format
    before the nested fields were stored as strings (the recorded original hashes)."""
    return json.dumps(row, separators=(",", ":"), ensure_ascii=False)


def read(path: Path) -> list:
    return [decode(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def normalise(row: dict, workflows: dict) -> dict:
    row = {**row, "workflow": row.get("workflow") or workflows[row["site"]]}
    return {k: row[k] for k in FIELDS}


def fold(source: Path, split: str, data: Path, dry_run: bool = False) -> int:
    if data.resolve() == RELEASE.resolve():
        raise SystemExit("data/ holds the v1 release files, which stay unchanged; fold into a working "
                         "copy of it instead (README, 'Generate your own data')")
    workflows = {s["site"]: s["workflow"] for s in json.loads((data / "sites.json").read_text(encoding="utf-8"))}
    existing = {name: read(data / f"{name}.jsonl") for name in SPLITS}
    ids = {r["case_id"] for rows in existing.values() for r in rows}
    states = {json.dumps(r["state"], sort_keys=True) for rows in existing.values() for r in rows}
    rows = [normalise(r, workflows) for r in read(source)]
    for r in rows:
        if build_of(r["case_id"]) == "original":
            raise SystemExit(f"{r['case_id']}: a new build needs a `<build>-` id prefix")
        if r["case_id"] in ids:
            raise SystemExit(f"{r['case_id']}: already in a split")
        state = json.dumps(r["state"], sort_keys=True)
        if state in states:
            raise SystemExit(f"{r['case_id']}: identical state already in a split")
        findings = scan_row(r["state"], r["gold"])
        if findings:
            raise SystemExit(f"{r['case_id']}: data scrub finding(s) {findings[:3]}; nothing was folded")
        ids.add(r["case_id"])
        states.add(state)
    if not dry_run:
        with (data / f"{split}.jsonl").open("a", encoding="utf-8") as f:
            f.writelines(to_line(r) + "\n" for r in rows)
    builds = sorted({build_of(r["case_id"]) for r in rows})
    print(f"{'would fold' if dry_run else 'folded'} {len(rows)} case(s) of build {', '.join(builds)} into {split}.jsonl")
    return len(rows)


def check_frozen(source: Path, split: str) -> None:
    """The build's freeze record must exist, match the cases file, and name this split."""
    record = source.parent / "freeze.json"
    if not record.exists():
        raise SystemExit(f"{source}: no freeze.json beside it. Freeze the build first "
                         f"(generate.py freeze), or pass --unfrozen for cases built some other way.")
    frozen = json.loads(record.read_text(encoding="utf-8"))
    if frozen.get("sha256", {}).get(source.name) != hashlib.sha256(source.read_bytes()).hexdigest():
        raise SystemExit(f"{source} changed after freeze")
    if frozen.get("split") != split:
        raise SystemExit(f"{source} was frozen for the {frozen.get('split')!r} split, not {split!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="a finished build's cases, one JSON row per line")
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--data", required=True, type=Path,
                        help="the working dataset directory to fold into (not data/, the v1 release)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--unfrozen", action="store_true",
                        help="fold cases that have no freeze record (not built by generate.py)")
    args = parser.parse_args()
    if not args.unfrozen:
        check_frozen(args.source, args.split)
    fold(args.source, args.split, args.data, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
