"""Turn a decision dataset into tokenized training items.

One labelled case becomes one item **per question**, because the model answers
each question at its own set of option markers. A case with four questions is
four training decisions, which is why the dataset counts both.

The tokenization itself is `laya.common.build_sequence`, the same function the
package uses at inference. Reimplementing it would be the surest way to train a
checkpoint that answers differently from the one that serves.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# The case id says which build a row came from: the original cases use 16-hex hashes,
# and every build added with scripts/generate.py uses its name as the prefix,
# `<build>-<site>-...`. Map a prefix here only if a build's ids use a different one.
PREFIX_BUILD: Dict[str, str] = {}

# The split files have one flat schema: case_id, site, workflow, and three nested
# fields stored as JSON strings. `state` differs in shape from site to site, so as a
# nested column no single type fits it (Arrow, and so `datasets`, refuses the file);
# as a string column it loads anywhere. load_split decodes them.
ENCODED = ("state", "questions", "gold")


def training_batch_plan(item_count: int, micro_batch: int, grad_accum: int,
                        legacy_drop_tail: bool = False) -> List[tuple[int, int, float, bool]]:
    """(start, stop, loss scale, optimizer step) for each training microbatch."""
    if micro_batch < 1 or grad_accum < 1:
        raise ValueError("micro_batch and grad_accum must be positive")
    usable = (item_count // micro_batch) * micro_batch if legacy_drop_tail else item_count
    effective = micro_batch * grad_accum
    plan = []
    for start in range(0, usable, micro_batch):
        stop = min(start + micro_batch, usable)
        if legacy_drop_tail:
            scale = 1 / grad_accum
        else:
            group_start = (start // effective) * effective
            scale = (stop - start) / min(effective, item_count - group_start)
        step = ((start // micro_batch + 1) % grad_accum == 0 or
                (not legacy_drop_tail and stop == usable))
        plan.append((start, stop, scale, step))
    return plan

# Upstream revisions the scripts download by default. A Hub `main` moves; a pinned
# commit does not, so a rerun reads the same weights and the same benchmark rows.
# Override one with `<repo>@<revision>` wherever a checkpoint is named, or with
# `--public-revision` for the benchmark.
REVISIONS: Dict[str, str] = {
    "convaiinnovations/laya": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
    "convaiinnovations/laya-typed-decisions": "1a793eb568e6718f15941d08f85432581df534e3",
    "fastino/GLiNER2.5-Decide": "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6",
    "LocalLLaMA/typed-decisions": "f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8",
}


def split_revision(spec: str) -> tuple:
    """`repo@revision` -> (repo, revision); a bare Hub id gets its pinned revision,
    if it has one. A local path is returned unchanged with no revision."""
    if Path(spec).exists():
        return spec, None
    repo, _, revision = spec.partition("@")
    return repo, revision or REVISIONS.get(repo)


def resolve_laya(spec: str) -> str:
    """A local directory for a Laya checkpoint named by path, Hub id or `id@revision`.

    `laya.load` takes no revision, so a Hub id is downloaded here at the pinned (or
    given) commit and the local snapshot directory is what gets loaded."""
    repo, revision = split_revision(spec)
    if revision is None:
        return repo
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, revision=revision, allow_patterns=[
        "rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])


def decode(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: json.loads(v) if k in ENCODED and isinstance(v, str) else v for k, v in row.items()}


def encode(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: json.dumps(v, ensure_ascii=False, separators=(",", ":")) if k in ENCODED else v
            for k, v in row.items()}


def to_line(row: Dict[str, Any]) -> str:
    """One row as it is stored in a split file."""
    return json.dumps(encode(row), ensure_ascii=False, separators=(",", ":"))


def build_of(case_id: str) -> str:
    if re.fullmatch(r"[0-9a-f]{16}", case_id):
        return "original"
    prefix = case_id.split("-", 1)[0]
    return PREFIX_BUILD.get(prefix, prefix)


def all_cases(root: Path, build: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every case in the three public split files, optionally filtered by build."""
    return [r for split in ("train", "validation", "test") for r in load_split(root, split, build)]


def load_split(root: Path, split: str, build: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read one split. Each row carries its own `questions`, so there is no
    registry to import and no way for a spec to drift from the data.

    `build` keeps only that build's rows ("original", a generate.py build's name, or a
    comma-separated list), for a split that holds more than one build."""
    # "validation" is the Hugging Face convention and what the files are called;
    # "val" is what the trainer asks for. Accept both.
    names = {"val": "validation", "validation": "validation"}.get(split, split)
    path = root / f"{names}.jsonl"
    if not path.exists():
        return []
    rows, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = decode(json.loads(line))
        if r["case_id"] in seen:          # a shard appended twice is not two cases
            continue
        seen.add(r["case_id"])
        rows.append(r)
    if build:
        wanted = set(build.split(","))
        rows = [r for r in rows if build_of(r["case_id"]) in wanted]
    return rows


def build_items(rows: List[Dict[str, Any]], tok, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One tokenized item per (case, question).

    Tokenization is `laya.common.build_sequence`, the same function the package
    uses at inference — reimplementing the marker layout is the surest way to
    train a checkpoint that answers differently from the one that serves. An item
    whose marker count does not match its option count is DROPPED, not patched.

    Each question goes through `laya.agent.Agent._to_internal`, the normalisation
    `system_one` applies, so list-form choice criteria and non-string instructions
    train exactly as they serve."""
    from laya.agent import Agent
    from laya.common import QTYPES, build_sequence, render_options

    items: List[Dict[str, Any]] = []
    dropped = 0
    for row in rows:
        for qid, question in row["questions"].items():
            gold = (row.get("gold") or {}).get(qid)
            if not gold or "probabilities" not in gold:
                continue
            packed = Agent._to_internal(question)
            qtype, crit = packed["t"], packed["crit"]
            if qtype == "noul":
                keys = ["false", "true"]
            elif qtype == "choice":
                keys = list(crit or {})
            else:
                keys = [str(i) for i in range(len(crit or []))]

            target = [float(gold["probabilities"].get(k, 0.0)) for k in keys]
            total = sum(target)
            if total <= 0:
                dropped += 1
                continue
            target = [v / total for v in target]

            expected = len(render_options(packed))
            seq, markers = build_sequence(
                tok, row["state"], packed, cfg["max_len"], cfg["head_max_len"],
            )
            if len(markers) != expected or expected != len(target):
                dropped += 1
                continue
            items.append({
                "ids": seq, "markers": markers, "qtype": QTYPES[qtype], "target": target,
                "site": row["site"], "question": qid, "case_id": row["case_id"],
            })
    if dropped:
        print(f"  note: dropped {dropped} item(s) whose markers did not match their options.")
    return items


def collate(batch: List[Dict[str, Any]], device, pad_id: int) -> Dict[str, torch.Tensor]:
    """Pad a batch into the tensors ``DecisionModel.forward`` takes.

    Delegates to ``laya.common.collate_items`` rather than padding by hand: the
    marker positions and the mask have to line up exactly with what the model
    reads at inference, and reimplementing that is the surest way to train a
    checkpoint that answers differently from the one that serves."""
    import torch
    from laya.common import collate_items

    packed = collate_items([[item] for item in batch], pad_id)
    if packed is None:
        raise ValueError("empty batch")
    out = {}
    for key, value in packed.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out
