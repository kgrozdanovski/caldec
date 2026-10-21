"""CalDec Ultra: Laya's typed-decision head on an UltraBERT encoder.

UltraBERT (Park, Lee & Kang, arXiv 2610.15028) ships as remote code for
transformers 5.x before 5.20. This module does three things with it:

* **register** — teach `transformers.AutoConfig`/`AutoModel` the `ultrabert`
  model type, so `laya.load`, `scripts/train.py` and `scripts/evaluate.py` load a
  CalDec Ultra checkpoint unchanged. `data.resolve_laya` calls it for any
  checkpoint whose encoder is UltraBERT.
* **init** — write an *untrained* Laya-format base checkpoint: the pretrained
  UltraBERT encoder, a freshly initialized decision head, the v1 sequence budget
  and a tokenizer whose `cls_token` is UltraBERT's own sequence-start token.
  `train.py --base <that directory>` then fine-tunes it with the CalDec recipe.
* **zeroshot** — score the raw pretrained encoder on the Assistant Decisions
  contract with no training at all, reading each answer from the MLM head at a
  single mask token, as the paper's Appendix B.3 does for multiple-choice QA.

    USE_TF=0 python3 scripts/ultra.py zeroshot --model dmis-lab/UltraBERT-380M \\
        --predictions runs/ultra-380m-zeroshot.test.predictions.json
    USE_TF=0 python3 scripts/ultra.py init --encoder dmis-lab/UltraBERT-380M \\
        --out checkpoints/ultra-380m-init
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("USE_TF", "0")

# The encoders are pinned with every other upstream, in data.REVISIONS.
from data import REVISIONS as ULTRA_REVISIONS                          # noqa: E402
# Both sizes ship byte-identical modeling code; one repo is enough to register it.
CODE_REPO = "dmis-lab/UltraBERT-150M"
CODE_FILES = ("configuration_ultrabert.py", "modeling_ultrabert.py")

# UltraBERT's tokenizer starts every sequence with <|endoftext|> (id 0) and has no
# cls_token. Laya's build_sequence needs one, so the CalDec Ultra tokenizer names
# that same sequence-start token, keeping inputs in the pre-training format.
CLS_TOKEN = "<|endoftext|>"


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

def is_ultra_checkpoint(path: Path) -> bool:
    cfg = Path(path) / "encoder" / "config.json"
    if not cfg.exists():
        return False
    try:
        return json.loads(cfg.read_text(encoding="utf-8")).get("model_type") == "ultrabert"
    except (OSError, ValueError):
        return False


def register(code_dir: Optional[Path] = None) -> None:
    """Register `ultrabert` with the transformers Auto classes, once.

    `code_dir` is a checkpoint's `encoder/` directory carrying the modeling files;
    without one the code comes from the pinned Hub revision. A transformers that
    already knows the type natively (5.20+) is left alone."""
    from transformers import AutoConfig, AutoModel
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if "ultrabert" in CONFIG_MAPPING:
        return
    if code_dir is not None and all((Path(code_dir) / f).exists() for f in CODE_FILES):
        import importlib.util

        mods = {}
        pkg = "caldec_ultrabert_code"
        # The modeling file imports its configuration relatively, so load both
        # under one synthetic package.
        import types
        sys.modules.setdefault(pkg, types.ModuleType(pkg)).__path__ = [str(code_dir)]
        for name in ("configuration_ultrabert", "modeling_ultrabert"):
            spec = importlib.util.spec_from_file_location(f"{pkg}.{name}", Path(code_dir) / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"{pkg}.{name}"] = module
            spec.loader.exec_module(module)
            mods[name] = module
        config_cls = mods["configuration_ultrabert"].UltrabertConfig
        model_cls = mods["modeling_ultrabert"].UltrabertModel
    else:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        rev = ULTRA_REVISIONS[CODE_REPO]
        config_cls = get_class_from_dynamic_module(
            "configuration_ultrabert.UltrabertConfig", CODE_REPO, revision=rev)
        model_cls = get_class_from_dynamic_module(
            "modeling_ultrabert.UltrabertModel", CODE_REPO, revision=rev)
    AutoConfig.register("ultrabert", config_cls, exist_ok=True)
    AutoModel.register(config_cls, model_cls, exist_ok=True)


def _split(spec: str):
    repo, _, revision = spec.partition("@")
    return repo, revision or ULTRA_REVISIONS.get(repo)


def load_tokenizer(spec: str):
    from transformers import AutoTokenizer

    repo, revision = _split(spec)
    tok = AutoTokenizer.from_pretrained(repo, revision=revision, trust_remote_code=False)
    if tok.cls_token is None:
        tok.cls_token = CLS_TOKEN
    return tok


# --------------------------------------------------------------------------- #
# init: an untrained Laya-format base checkpoint on UltraBERT
# --------------------------------------------------------------------------- #

def init_checkpoint(encoder: str, out: Path, template: Path, seed: int,
                    max_len: int, head_max_len: int) -> None:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file
    from transformers import AutoModel

    from laya.common import DecisionModel

    if out.exists():
        raise SystemExit(f"  {out} already exists; choose a new --out")
    repo, revision = _split(encoder)
    register()
    torch.manual_seed(seed)
    enc = AutoModel.from_pretrained(repo, revision=revision, attn_implementation="sdpa",
                                    dtype=torch.float32)
    tok = load_tokenizer(encoder)
    # Any encoder works here; UltraBERT alone needs its modeling code shipped. A
    # ModernBERT base built this way is the fresh-head control for the backbone.
    is_ultra = enc.config.model_type == "ultrabert"
    # Same head shape as Laya: 2 transformer layers, act head sized from act_costs.
    cfg = json.loads((template / "rl_agent_config.json").read_text(encoding="utf-8"))
    cfg = {k: cfg[k] for k in ("head_layers", "max_prefixes", "act_costs", "cost_wrong_act") if k in cfg}
    cfg.update({
        "encoder": repo, "encoder_revision": revision, "max_len": max_len,
        "head_max_len": head_max_len, "amp_dtype": "bf16",
        "model_name": out.name, "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
        "training": {"initialized_from": repo, "encoder_revision": revision,
                     "head": "random init", "seed": seed},
    })
    model = DecisionModel(enc, cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1)

    out.mkdir(parents=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(out / "model.safetensors"))
    (out / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    tok.save_pretrained(str(out / "tokenizer"))
    enc.config.save_pretrained(str(out / "encoder"))
    # Ship the modeling code beside the config, so the checkpoint loads offline.
    for name in CODE_FILES if is_ultra else ():
        shutil.copy(hf_hub_download(repo, name, revision=revision), out / "encoder" / name)
    n_enc = sum(p.numel() for p in enc.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    print(f"  {out}: {n_enc / 1e6:.1f}M encoder + {(n_all - n_enc) / 1e6:.1f}M head")


# --------------------------------------------------------------------------- #
# zeroshot: the pretrained MLM head read at one mask token
# --------------------------------------------------------------------------- #

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
YES = (" yes", " Yes", " true", " True")
NO = (" no", " No", " false", " False")


class ZeroShotAgent:
    """`system_one`-compatible scorer over a raw masked-LM checkpoint.

    The prompt follows the paper's MCQA protocol (App. B.3): a `Context:` prefix,
    the question, lettered options, and `Answer:` followed by one mask token whose
    logits are compared over the option letters. A yes/no question compares the
    summed probabilities of the yes and no variants, as the paper does for BoolQ.
    The state is truncated from the left to fit `max_len`."""

    def __init__(self, spec: str, device: str, max_len: int = 2048):
        import torch
        from transformers import AutoModelForMaskedLM

        repo, revision = _split(spec)
        self.tok = load_tokenizer(spec)
        self.model = AutoModelForMaskedLM.from_pretrained(
            repo, revision=revision, trust_remote_code=True, attn_implementation="sdpa",
            dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        ).to(device).eval()
        self.device, self.max_len = device, max_len
        ids = lambda words: [self.tok(w, add_special_tokens=False)["input_ids"] for w in words]
        for name, words in (("yes", YES), ("no", NO)):
            toks = ids(words)
            if any(len(t) != 1 for t in toks):
                raise SystemExit(f"  a {name} variant is not one token: {words}")
            setattr(self, f"{name}_ids", [t[0] for t in toks])
        letters = ids([" " + c for c in LETTERS])
        if any(len(t) != 1 for t in letters):
            raise SystemExit("  an option letter is not one token")
        self.letter_ids = [t[0] for t in letters]

    def _prompt(self, state, question):
        from laya.common import render_criterion, serialize_state

        qtype, crit = question["type"], question.get("criteria")
        lines = ["Question: " + str(question["instructions"])]
        if qtype == "noul":
            lines.append("Answer yes or no.")
            keys = None
        else:
            if qtype == "choice":
                items = list(crit.items()) if isinstance(crit, dict) else [(c, None) for c in crit]
                opts = [k if v in (None, "") else f"{k}: {render_criterion(v)}" for k, v in items]
                keys = [k for k, _ in items]
            else:
                opts = [f"level {i}: {render_criterion(c)}" for i, c in enumerate(crit)]
                keys = [str(i) for i in range(len(crit))]
            lines += [f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)]
        tail = "\n" + "\n".join(lines) + "\nAnswer:" + self.tok.mask_token
        tail_ids = self.tok(tail, add_special_tokens=False)["input_ids"]
        ctx_ids = self.tok("Context: " + serialize_state(state).replace(self.tok.mask_token, " "),
                           add_special_tokens=False)["input_ids"]
        room = max(0, self.max_len - len(tail_ids) - 1)
        ids = [self.tok.cls_token_id] + ctx_ids[-room:] + tail_ids if room else \
            [self.tok.cls_token_id] + tail_ids[-(self.max_len - 1):]
        return ids, keys

    def system_one(self, state, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        answers = {}
        for qid, q in questions.items():
            ids, keys = self._prompt(state, q)
            x = torch.tensor([ids], device=self.device)
            with torch.no_grad():
                logits = self.model(input_ids=x, attention_mask=torch.ones_like(x)).logits
            pos = ids.index(self.tok.mask_token_id, len(ids) - 4)
            p = torch.softmax(logits[0, pos].float(), -1)
            if keys is None:
                yes, no = float(p[self.yes_ids].sum()), float(p[self.no_ids].sum())
                answers[qid] = {"noul": yes / (yes + no)}
            else:
                lp = logits[0, pos, self.letter_ids[:len(keys)]].float()
                pr = torch.softmax(lp, -1).tolist()
                answers[qid] = {"probabilities": dict(zip(keys, pr))}
        return {"answers": answers}


def zeroshot(args) -> int:
    import torch

    from data import load_split
    from evaluate import baselines, data_fingerprint, metrics, print_table, score, serialise

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    agent = ZeroShotAgent(args.model, device, args.max_len)
    rows = load_split(Path(args.data), args.split)
    train_rows = load_split(Path(args.data), "train")
    print(f"\n  model   {args.model} (zero-shot MLM read-out)\n  split   {args.split}: {len(rows):,} cases")
    scored = score(agent, rows)
    overall = metrics(scored)
    print_table("Accuracy = agreement with the teacher's argmax, untied labels only.", {"OVERALL": overall})
    by_site: Dict[str, List] = {}
    for r in scored:
        by_site.setdefault(r["site"], []).append(r)
    per_site = {s: metrics(g) for s, g in sorted(by_site.items())}
    print_table("Per site", per_site)
    report = {"model": args.model, "revision": _split(args.model)[1], "split": args.split,
              "contract": "zero-shot MLM read-out at one mask token; accuracy on untied labels",
              "max_len": args.max_len, "overall": overall,
              "baselines": baselines(train_rows, scored), "per_site": per_site}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
        print(f"  report -> {args.out}")
    if args.predictions:
        Path(args.predictions).parent.mkdir(parents=True, exist_ok=True)
        Path(args.predictions).write_text(json.dumps(serialise(
            scored, checkpoint=args.model, split=args.split,
            dataset_sha256=data_fingerprint(rows))), encoding="utf-8")
        print(f"  predictions -> {args.predictions}")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ultra", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    z = sub.add_parser("zeroshot", help="score a raw UltraBERT checkpoint with no training")
    z.add_argument("--model", required=True, help="Hub id or id@revision")
    z.add_argument("--data", default="data")
    z.add_argument("--split", default="test")
    z.add_argument("--max-len", type=int, default=2048)
    z.add_argument("--device", default=None)
    z.add_argument("--out", help="where to write the report")
    z.add_argument("--predictions", help="where to write per-decision predictions")
    i = sub.add_parser("init", help="write an untrained Laya-format base on UltraBERT")
    i.add_argument("--encoder", required=True, help="Hub id or id@revision")
    i.add_argument("--out", required=True)
    i.add_argument("--template", default="checkpoints/v1-laya/final",
                   help="a Laya checkpoint whose head shape and act config to copy")
    i.add_argument("--max-len", type=int, default=512)
    i.add_argument("--head-max-len", type=int, default=192)
    i.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args(argv)
    if args.cmd == "zeroshot":
        return zeroshot(args)
    init_checkpoint(args.encoder, Path(args.out), Path(args.template), args.seed,
                    args.max_len, args.head_max_len)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
