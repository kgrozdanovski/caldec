"""Grow the dataset: generate, label and fold a new build of cases into any split.

A build is one budgeted run with its own models and briefs. It reads and extends a
working dataset directory (`data_dir`), never the v1 release files in data/, so the
release and its reported scores stay unchanged. Its working files go to
data/provenance/<build>/ (ignored by git); its finished cases are appended to
<data_dir>/<split>.jsonl with ids `<build>-<site>-<generator>-<batch>-<n>`, so every
row still says which build it came from (scripts/data.py `build_of`).

    mkdir data-next && cp data/*.json data/*.jsonl data-next/   # the working copy, once
    cp scripts/generate.example.json my-build.json      # edit: build, split, models, budget, briefs
    python scripts/generate.py init     --config my-build.json
    python scripts/generate.py pilot    --build <build>  # 2 cases from each of up to 4 generators, never kept
    python scripts/generate.py generate --build <build>
    python scripts/generate.py label    --build <build>
    python scripts/generate.py freeze   --build <build>
    python scripts/generate.py fold     --build <build>  # appends to <data_dir>/<split>.jsonl
    python scripts/generate.py cost     --build <build>

Config fields (scripts/generate.example.json):
- `build`: the build's name and case-id prefix (lowercase letters, digits, `_`). It
  must not be a prefix that existing case ids already use (`exp`, `gen`, `fill1`, ...).
- `data_dir`: the dataset directory the build reads sites, prompts and existing cases
  from and folds into, relative to the repository root. It holds `sites.json`,
  `prompts.json` and the three split files; start it as a copy of data/. It may not be
  data/ itself.
- `split`: `train`, `validation` or `test`. Cases built for `validation` or `test` must
  never be trained on.
- `models`: ids with `price_per_million` [input, output], optional
  `needs_reasoning` (the model rejects reasoning disabled), `extra_output_tokens` (billed
  reasoning beyond max_tokens), `roles` (default both `generate` and `label`),
  `provider` (default `openrouter`) and `family` (default the id prefix).
- `providers`: named OpenAI-compatible chat-completions endpoints. Each has
  `endpoint` and `api_key_env`; OpenRouter is built in. Keep keys in the process
  environment, never in the config. An endpoint must support JSON-object replies.
- `sites`: a list of sites, or null for every site in `<data_dir>/sites.json`.
- `cases_per_batch`, `batches_per_generator_per_site`: the size of the build.
- `annotators_per_case`: labels per case, each from a family other than the generator's.
- `allow_same_family_labels`: permit a single model to generate and label in
  separate, blind calls. Use this for the one-model GLM example; leave false when
  you configure independent model families.
- `briefs`: `"original"` (each site's brief in data/prompts.json) or a file mapping each
  site to a new brief. New briefs move the scenarios away from the existing data:
  A custom brief file can vary scenario domains without changing the site schema.
- `max_jaccard`: the near-duplicate threshold. `budget_usd`: the hard spending cap.

What it enforces, and what it leaves to you:
- **Equal requested shares per generator.** Every configured generator is asked for the
  same number of batches per site. Rejections and dropped cases can leave the kept
  shares unequal; `freeze.json` records the kept count per generator, so check it.
  Using several generator families is your choice of config, not a rule: one generator
  writes in one style, and a model trained on it can learn that style along with the task.
- **Blind labels.** Labelling calls do not receive the generation brief. By default,
  each case is labelled by `annotators_per_case` models from families other than
  the generator's, rotating which ones; `family` can override the id prefix.
  `allow_same_family_labels` permits the single-model recipe to use a separate
  call to the generator model. The target is the mean of the label distributions.
- **A content scan, twice.** Generated hosts on public top-level domains are rewritten
  to reserved `.example` names, and a state with any remaining finding of the
  publishing scan (scripts/check_data.py: hosts, emails, public IPs, key-shaped
  strings, card numbers) is rejected in `generate`, before labelling is paid for.
  `fold` scans every row again and refuses the build on any finding.
- **No duplicates, at generation time.** A state identical to, or with token-set Jaccard
  >= `max_jaccard` against, any case in any split (or earlier in the build) within its
  site is rejected. The check runs in `generate`, against the splits as they are then:
  if another build is folded between this build's `generate` and its `fold`, only an
  identical state is caught (by `fold`). Generate and fold one build at a time.
- **A hard budget.** Every call is journalled before dispatch; a request that could take
  the build past `budget_usd` is not sent. Responses are cached, so reruns re-pay nothing.
  A batch lost to the cap or to an API error is reported and makes `generate` and
  `label` exit non-zero, so a short build is not mistaken for a finished one; rerun the
  command, after raising `budget_usd` in the build's protocol.json if the cap was the cause.
- **Frozen before use.** `freeze` writes the cases and their hashes once; `fold` refuses
  to run on an unfrozen or changed build.
- **Not enforced: that a `test` or `validation` build is never trained on.** Nothing in
  a split file stops it. Fold such a build before scoring anything on it, and never pass
  its rows to a training script.

Check that every model's terms allow what you do with its output: training on it, and
distributing models trained on it. Supply LLM_OPENROUTER_API_KEYS through the environment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm  # noqa: E402
import fold  # noqa: E402
from collections import defaultdict  # noqa: E402

import check_data  # noqa: E402
from data import PREFIX_BUILD, all_cases, build_of  # noqa: E402

ROOT = llm.ROOT
# Names a build cannot take: "original" is the original cases' build, "pilot" prefixes
# pilot cases, and mapped prefixes are already in use.
RESERVED = {"original", "pilot", *PREFIX_BUILD}


def family(model_id: str, protocol: dict | None = None) -> str:
    if protocol:
        for model in protocol["models"]:
            if model["id"] == model_id:
                return model.get("family") or model_id.split("/", 1)[0]
    return model_id.split("/", 1)[0]


def reference_states(data: Path) -> dict:
    """One reference state per site, to check a generated state's keys and types against:
    a state of the site's most common key set, so a row with a misspelled key (there are
    a few) cannot become the reference and reject every well-formed case."""
    by_site = defaultdict(list)
    for row in all_cases(data):
        by_site[row["site"]].append(row["state"])
    references = {}
    for site, states in by_site.items():
        common = Counter(frozenset(s) for s in states).most_common(1)[0][0]
        references[site] = next(s for s in states if frozenset(s) == common)
    return references


def build_dir(build: str) -> Path:
    return ROOT / "data" / "provenance" / build


def data_root(protocol: dict) -> Path:
    """The working dataset directory a build reads from and folds into."""
    path = Path(protocol["data_dir"])
    return path if path.is_absolute() else ROOT / path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_complete(out: Path, stage: str, files: tuple[str, ...]) -> None:
    marker = out / f"{stage}.complete.json"
    if not marker.exists():
        raise SystemExit(f"{stage} did not finish; rerun it before continuing")
    recorded = json.loads(marker.read_text(encoding="utf-8"))
    for name in files:
        path = out / name
        if not path.is_file() or recorded.get("sha256", {}).get(name) != file_sha256(path):
            raise SystemExit(f"{path} changed after {stage} completed; rerun {stage}")


def load_protocol(build: str) -> dict:
    path = build_dir(build) / "protocol.json"
    if not path.exists():
        raise SystemExit(f"No build {build!r}: run init first")
    return json.loads(path.read_text(encoding="utf-8"))


def client_for(protocol: dict) -> llm.Client:
    models = protocol["models"]
    providers = protocol.get("providers", {})
    connections = {}
    for model in models:
        provider = model.get("provider", "openrouter")
        if provider == "openrouter":
            connections[model["id"]] = {"openrouter": True}
        else:
            info = providers[provider]
            connections[model["id"]] = {"endpoint": info["endpoint"],
                                         "api_key_env": info["api_key_env"], "openrouter": False}
    return llm.Client(out=build_dir(protocol["build"]), cap_usd=protocol["budget_usd"],
                     models=llm.model_table([m["id"] for m in models], [m["price_per_million"] for m in models],
                                           {m["id"] for m in models if m.get("needs_reasoning")},
                                           {m["id"]: m.get("extra_output_tokens", 0) for m in models},
                                           connections))


def initialise(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    build = config["build"]
    if not re.fullmatch(r"[a-z][a-z0-9_]*", build) or build in RESERVED:
        raise SystemExit(f"build {build!r}: lowercase letters, digits and _, starting with a letter, "
                         f"and not one of {sorted(RESERVED)}")
    if not isinstance(config.get("data_dir"), str) or not config["data_dir"]:
        raise SystemExit("data_dir must name the working dataset directory this build reads and folds into "
                         "(README, 'Generate your own data')")
    data = data_root(config)
    if data.resolve() == (ROOT / "data").resolve():
        raise SystemExit("data_dir must not be data/: the v1 release files stay unchanged. Copy them into a "
                         "working directory first: mkdir data-next && cp data/*.json data/*.jsonl data-next/")
    absent = [name for name in ("sites.json", "prompts.json", *(f"{s}.jsonl" for s in fold.SPLITS))
              if not (data / name).is_file()]
    if absent:
        raise SystemExit(f"{data}: missing {absent}. Start the working directory as a copy of data/: "
                         f"mkdir {config['data_dir']} && cp data/*.json data/*.jsonl {config['data_dir']}/")
    # A build's name is its rows' case-id prefix, so it must not already be in use: a
    # collision would otherwise surface only at fold, after the build was paid for.
    used = {build_of(r["case_id"]) for root in {data, ROOT / "data"} for r in all_cases(root)}
    if build in used:
        raise SystemExit(f"build {build!r}: case ids with that prefix already exist in the data; choose another name")
    if config["split"] not in fold.SPLITS:
        raise SystemExit(f"split must be one of {fold.SPLITS}")
    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise SystemExit("models must be a nonempty list")
    for key in ("cases_per_batch", "batches_per_generator_per_site", "annotators_per_case"):
        if not isinstance(config.get(key), int) or isinstance(config[key], bool) or config[key] < 1:
            raise SystemExit(f"{key} must be a positive integer")
    if not isinstance(config.get("budget_usd"), (int, float)) or isinstance(config["budget_usd"], bool) or not math.isfinite(config["budget_usd"]) or config["budget_usd"] <= 0:
        raise SystemExit("budget_usd must be positive and finite")
    if any(not isinstance(m, dict) or not isinstance(m.get("id"), str) or not m["id"] for m in models):
        raise SystemExit("each model needs a nonempty string id")
    if len({m["id"] for m in config["models"]}) != len(config["models"]):
        raise SystemExit("model ids must be unique")
    for model in config["models"]:
        prices = model.get("price_per_million")
        if not isinstance(model.get("id"), str) or not model["id"] or not isinstance(prices, list) or len(prices) != 2 or any(
                not isinstance(p, (int, float)) or isinstance(p, bool) or not math.isfinite(p) or p < 0 for p in prices):
            raise SystemExit("each model needs an id and two nonnegative price estimates")
        roles = model.get("roles", ["generate", "label"])
        if not isinstance(roles, list) or not roles or not set(roles) <= {"generate", "label"}:
            raise SystemExit(f"{model['id']}: roles must contain generate and/or label")
        provider = model.get("provider", "openrouter")
        if provider != "openrouter":
            info = config.get("providers", {}).get(provider)
            if not info or not isinstance(info.get("endpoint"), str) or not isinstance(info.get("api_key_env"), str):
                raise SystemExit(f"{model['id']}: provider {provider!r} needs endpoint and api_key_env")
            if not info["endpoint"].startswith("https://"):
                raise SystemExit(f"{provider}: endpoint must use https")
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", info["api_key_env"]):
                raise SystemExit(f"{provider}: invalid api_key_env name")
    generators = [m for m in config["models"] if "generate" in m.get("roles", ["generate", "label"])]
    labellers = [m for m in config["models"] if "label" in m.get("roles", ["generate", "label"])]
    if not generators or not labellers:
        raise SystemExit("at least one generator and one labeller are required")
    for g in generators:
        others = {family(m["id"], config) for m in labellers if family(m["id"], config) != family(g["id"], config)}
        if len(others) < config["annotators_per_case"] and not (
                config.get("allow_same_family_labels", False) and config["annotators_per_case"] == 1
                and any(m["id"] == g["id"] for m in labellers)):
            raise SystemExit(f"{g['id']}: needs {config['annotators_per_case']} labelling families other than "
                             f"its own, has {len(others)}")
    out = build_dir(build)
    if (out / "protocol.json").exists():
        raise SystemExit(f"{out}/protocol.json exists; a build's protocol is written once")
    sites = json.loads((data / "sites.json").read_text(encoding="utf-8"))
    problems = check_data.site_problems(sites)
    if problems:
        raise SystemExit("sites.json: " + "; ".join(problems))
    if config.get("sites"):
        sites = [s for s in sites if s["site"] in config["sites"]]
        unknown = set(config["sites"]) - {s["site"] for s in sites}
        if unknown:
            raise SystemExit(f"unknown sites: {sorted(unknown)}")
    if not sites:
        raise SystemExit("no sites selected")
    briefs = config.get("briefs") or "original"
    if briefs != "original":
        if isinstance(briefs, str):   # a path, next to the config or from the repository root
            path = config_path.parent / briefs
            briefs = json.loads((path if path.exists() else ROOT / briefs).read_text(encoding="utf-8"))
        missing = {s["site"] for s in sites} - set(briefs)
        if missing:
            raise SystemExit(f"briefs missing for {sorted(missing)}")
    # Everything a site needs, checked before any call is paid for.
    prompts = json.loads((data / "prompts.json").read_text(encoding="utf-8"))
    references = reference_states(data)
    for s in sites:
        entry = prompts["per_site"].get(s["site"]) or {}
        absent = [k for k in ("generation_brief", "diversity_axes", "state_shape", "label_brief") if k not in entry]
        if absent:
            raise SystemExit(f"{s['site']}: data/prompts.json per_site entry lacks {absent}")
        if s["site"] not in references:
            raise SystemExit(f"{s['site']}: no existing case to take the state's keys and types from. A new "
                             f"site needs at least one hand-written case in a split file of {data} first "
                             f"(README, 'Add a decision site').")
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(config, created_at=datetime.now(timezone.utc).isoformat(), sites=sites,
                    prompts=prompts,
                    new_briefs=briefs if briefs != "original" else {},
                    brief_family="original" if briefs == "original" else "new",
                    split_files={name: hashlib.sha256((data / f"{name}.jsonl").read_bytes()).hexdigest()
                                 for name in fold.SPLITS})
    llm.write_json(out / "protocol.json", protocol)
    print(f"initialised build {build!r} -> {out}; {len(sites)} site(s), {len(generators)} generator(s), "
          f"{config['batches_per_generator_per_site'] * config['cases_per_batch'] * len(generators) * len(sites)} cases planned")


def generate(client, protocol, pilot=False) -> int:
    """Returns the number of batches lost to the budget cap or an API error."""
    out = build_dir(protocol["build"])
    generators = [m["id"] for m in protocol["models"] if "generate" in m.get("roles", ["generate", "label"])]
    sites = protocol["sites"]
    if pilot:
        jobs = [(g, sites[(3 * i) % len(sites)], 2, 0) for i, g in enumerate(generators[:4])]
    else:
        jobs = [(g, s, protocol["cases_per_batch"], b) for s in sites for g in generators
                for b in range(protocol["batches_per_generator_per_site"])]
    prefix = "pilot" if pilot else protocol["build"]
    tagged = [(g, s, n, f"{prefix}-{s['site']}-{generators.index(g)}-{b}") for g, s, n, b in jobs]

    def job(item):
        g, s, n, tag = item
        system, user = llm.generation_request(protocol, s["site"], protocol["brief_family"], n, tag)

        def validate(content):
            states = llm.parse_json(content)["states"]
            if not isinstance(states, list) or len(states) != n or not all(isinstance(x, dict) for x in states):
                raise ValueError(f"Wrong state count: {tag}")
            return states
        try:
            states = llm.request_validated(client, tag, g, system, user, 3600, .9, validate)
        except (RuntimeError, ValueError) as exc:
            print(f"{tag}: skipped ({type(exc).__name__})", flush=True)
            return None
        return [dict(case_id=f"{tag}-{j}", site=s["site"], state=llm.reserve_hosts(state), questions=s["questions"],
                     generator=g, cluster_id=tag) for j, state in enumerate(states)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        batches = list(pool.map(job, tagged))
    skipped = sum(b is None for b in batches)
    rows = [r for batch in batches if batch for r in batch]
    if pilot:
        llm.write_json(out / "pilot.json", rows)
        print(f"pilot: {len(rows)} case(s) in {out / 'pilot.json'}; they are never kept")
        return skipped
    pool_rows = all_cases(data_root(protocol))
    references = reference_states(data_root(protocol))
    exact = {llm.normal(r["state"]): r["case_id"] for r in pool_rows}
    accepted, rejected = [], []
    for row in rows:
        reason = None
        findings = check_data.scan_row(row["state"])
        if not llm.valid_state(row["state"], references[row["site"]]):
            reason = "state_schema_or_length"
        elif findings:
            reason = "scrub:" + ",".join(sorted({kind for kind, _ in findings}))
        elif llm.normal(row["state"]) in exact:
            reason = f"duplicate:{exact[llm.normal(row['state'])]}"
        else:
            tok = llm.tokens(row["state"])
            for other in pool_rows:
                if other["site"] == row["site"]:
                    ot = llm.tokens(other["state"])
                    if tok | ot and len(tok & ot) / len(tok | ot) >= protocol["max_jaccard"]:
                        reason = f"near_duplicate:{other['case_id']}"
                        break
        if reason:
            rejected.append(dict(row=row, reason=reason))
        else:
            accepted.append(row)
            pool_rows.append(row)
            exact[llm.normal(row["state"])] = row["case_id"]
    llm.write_json(out / "rejections.json", rejected)
    llm.write_json(out / "candidates.json", accepted)
    marker = out / "generate.complete.json"
    if marker.exists():
        marker.unlink()
    if not skipped and accepted:
        llm.write_json(marker, {"planned_batches": len(tagged), "completed_batches": len(batches),
                                "accepted_cases": len(accepted),
                                "sha256": {"candidates.json": file_sha256(out / "candidates.json")}})
    print(f"accepted {len(accepted)}, rejected {len(rejected)}", flush=True)
    if not accepted and not skipped:
        raise SystemExit("no cases accepted; revise the build before labelling")
    return skipped


def label(client, protocol) -> int:
    """Returns the number of labelling calls lost to the budget cap or an API error."""
    out = build_dir(protocol["build"])
    require_complete(out, "generate", ("candidates.json",))
    rows = json.loads((out / "candidates.json").read_text(encoding="utf-8"))
    labellers = [m["id"] for m in protocol["models"] if "label" in m.get("roles", ["generate", "label"])]
    batches = {}
    for row in rows:
        batches.setdefault(row["cluster_id"], []).append(row)
    jobs = []
    for n, (tag, batch) in enumerate(sorted(batches.items())):
        # Distinct families other than the generator's, rotated so each takes an equal share;
        # within a family, its models take turns.
        families = sorted({family(m, protocol) for m in labellers} - {family(batch[0]["generator"], protocol)})
        if not families and protocol.get("allow_same_family_labels"):
            families = [family(batch[0]["generator"], protocol)]
        picked = [families[(n + k) % len(families)] for k in range(protocol["annotators_per_case"])]
        chosen = [[m for m in labellers if family(m, protocol) == f][n % sum(family(m, protocol) == f for m in labellers)] for f in picked]
        jobs += [(tag, batch, m) for m in chosen]

    def job(item):
        tag, batch, model = item
        system, user = llm.label_request(protocol, batch)
        try:
            labels = llm.request_validated(client, f"annotate-{tag}-{labellers.index(model)}", model, system, user,
                                          6000, .2, lambda content: llm.parse_labels(content, batch), reasoning="medium")
        except (RuntimeError, ValueError) as exc:
            print(f"annotate-{tag}: {model} skipped ({type(exc).__name__})", flush=True)
            return None
        return [dict(case_id=r["case_id"], annotator=model, probabilities=p) for r, p in zip(batch, labels)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(job, jobs))
    skipped = sum(r is None for r in results)
    annotations = [a for batch in results if batch for a in batch]
    llm.write_json(out / "annotations.json", annotations)
    marker = out / "label.complete.json"
    if marker.exists():
        marker.unlink()
    if not skipped and len(annotations) == len(rows) * protocol["annotators_per_case"]:
        llm.write_json(marker, {"cases": len(rows), "annotations": len(annotations),
                                "sha256": {name: file_sha256(out / name)
                                           for name in ("candidates.json", "annotations.json")}})
    print(f"annotations: {len(annotations)} for {len(rows)} cases", flush=True)
    return skipped


def freeze(protocol) -> None:
    """Keep cases with every label present and from families other than the generator;
    the target is the mean of the labels. Writes cases.jsonl and freeze.json once."""
    out = build_dir(protocol["build"])
    require_complete(out, "generate", ("candidates.json",))
    require_complete(out, "label", ("candidates.json", "annotations.json"))
    if (out / "freeze.json").exists():
        raise SystemExit(f"{out}/freeze.json exists; a build is frozen once")
    rows = json.loads((out / "candidates.json").read_text(encoding="utf-8"))
    by_case = {}
    for a in json.loads((out / "annotations.json").read_text(encoding="utf-8")):
        by_case.setdefault(a["case_id"], []).append(a)
    need = protocol["annotators_per_case"]
    kept, dropped, agree, total = [], 0, 0, 0
    for row in rows:
        labels = by_case.get(row["case_id"], [])
        if len(labels) != need or len({family(a["annotator"], protocol) for a in labels}) != need \
                or (not protocol.get("allow_same_family_labels") and any(
                    family(a["annotator"], protocol) == family(row["generator"], protocol) for a in labels)):
            dropped += 1
            continue
        row["gold"] = {}
        for qid, q in row["questions"].items():
            vectors = [a["probabilities"][qid] for a in labels]
            row["gold"][qid] = {"probabilities": {k: sum(v[k] for v in vectors) / need for k in llm.keys(q)}}
            tops = [{k for k in v if abs(v[k] - max(v.values())) < 1e-9} for v in vectors]
            agree += all(len(t) == 1 and t == tops[0] for t in tops)
            total += 1
        kept.append(row)
    if dropped or not kept:
        raise SystemExit(f"freeze refused: {dropped} case(s) lack the required independent labels, "
                         f"{len(kept)} complete; inspect annotations and rerun label")
    path = out / "cases.jsonl"
    path.write_text("".join(fold.dump(r) + "\n" for r in kept), encoding="utf-8")
    llm.write_json(out / "freeze.json", dict(
        frozen_at=datetime.now(timezone.utc).isoformat(), split=protocol["split"], cases=len(kept), dropped=dropped,
        decisions=total, annotator_agreement=round(agree / max(total, 1), 4),
        by_site=Counter(r["site"] for r in kept), by_generator=Counter(r["generator"] for r in kept),
        sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / "protocol.json", out / "annotations.json", path)}))
    print(f"kept {len(kept)} cases ({total} decisions), dropped {dropped}; annotator agreement {agree / max(total, 1):.3f}")


def fold_build(protocol) -> None:
    out = build_dir(protocol["build"])
    frozen = json.loads((out / "freeze.json").read_text(encoding="utf-8"))
    for name, digest in frozen["sha256"].items():
        if hashlib.sha256((out / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"{out / name} changed after freeze")
    if (out / "folded.json").exists():
        raise SystemExit(f"build {protocol['build']!r} was already folded")
    count = fold.fold(out / "cases.jsonl", protocol["split"], data_root(protocol))
    llm.write_json(out / "folded.json", dict(folded_at=datetime.now(timezone.utc).isoformat(),
                                            split=protocol["split"], cases=count))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["init", "pilot", "generate", "label", "freeze", "fold", "cost"])
    parser.add_argument("--config", type=Path, help="init only: the build's configuration")
    parser.add_argument("--build", help="every other command: the build's name")
    args = parser.parse_args()
    if args.command == "init":
        if not args.config:
            parser.error("init needs --config")
        initialise(args.config)
        return 0
    if not args.build:
        parser.error(f"{args.command} needs --build")
    protocol = load_protocol(args.build)
    llm.OUT = build_dir(args.build)  # request_validated journals invalid responses here
    if args.command == "freeze":
        freeze(protocol)
    elif args.command == "fold":
        fold_build(protocol)
    elif args.command == "cost":          # reads the journal; needs no credential
        print(f"{llm.spent(build_dir(args.build) / 'calls.jsonl'):.4f}")
    else:
        client = client_for(protocol)
        if args.command == "pilot":
            skipped = generate(client, protocol, pilot=True)
        elif args.command == "generate":
            skipped = generate(client, protocol)
        else:
            skipped = label(client, protocol)
        if skipped:
            print(f"{skipped} call(s) were skipped (budget cap or API error; see above). The build is "
                  f"incomplete: rerun `{args.command}` — answered calls are cached and cost nothing.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
