"""Fine-tune Laya on an assistant decision dataset, single GPU.

A port of the recipe Convai publishes as `laya_finetune_typed_decisions_2xT4_kaggle.ipynb`,
reading this repository's dataset format instead of the Hugging Face one, and
running on one GPU rather than two. The published notebook uses two T4s for DDP
speed; one 16 GB card reaches the same effective batch of 32 with micro-batch 4 and
twice the gradient accumulation.

**The CalDec Laya release objective is soft cross-entropy
against the teacher's distributions.** That term is why the dataset stores
probabilities rather than answers.

The script also carries a policy-gradient term ported from Convai's RLCD recipe:
Gaussian noise on the logits, a strictly proper scoring rule as the reward, and
REINFORCE against a group-mean baseline. It is off by default (weight 0) and has
not been measured, so treat `--rl-weight` above 0 as an experiment, not a recipe.

Then a temperature is fitted per (question type, option count) on the validation
split, so the probabilities the checkpoint serves are calibrated on held-out data.
The fitted values are saved with the checkpoint.

The release recipe uses these flags; the defaults alone use four epochs and do
not add the public split or site balancing.

    USE_TF=0 python3 scripts/train.py --data data --out checkpoints/my-laya-run \
        --epochs 6 --balance-sites --with-public --legacy-drop-tail

Nothing on disk is overwritten: the output directory must not already exist.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("USE_TF", "0")
# A 421M encoder backpropagating at sequence length 512 fragments the
# allocator badly; without this a 16 GB card runs out with over a gigabyte
# reserved-but-unallocated.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

from data import load_split, build_items, collate, resolve_laya, split_revision, training_batch_plan  # noqa: E402


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="train", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", default="data", help="the dataset directory (data/ in this repository)")
    parser.add_argument("--build", default=None,
                        help="train on one build's rows only (data.build_of); default: every row of train.jsonl")
    parser.add_argument("--out", required=True, help="checkpoint directory; must NOT exist")
    parser.add_argument("--base", default="convaiinnovations/laya",
                        help="a local checkpoint, a Hub id (loaded at the revision pinned in "
                             "data.REVISIONS) or `id@revision`")
    parser.add_argument("--epochs", type=int, default=4)
    # The notebook uses micro-batch 8 across two T4s. One 16 GB card holding the
    # optimiser state as well runs out there, so the same effective batch of 32
    # is reached with half the micro-batch and twice the accumulation.
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--legacy-drop-tail", action="store_true",
                        help="replay the v1 training loop, which discarded incomplete batches and accumulation groups")
    parser.add_argument("--lr-encoder", type=float, default=2.5e-5)
    parser.add_argument("--lr-head", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--group-size", type=int, default=4, help="GRPO baseline samples")
    parser.add_argument("--sigma-start", type=float, default=0.4)
    parser.add_argument("--sigma-end", type=float, default=0.1)
    parser.add_argument("--ce-weight", type=float, default=1.0, help="soft cross-entropy against the teacher")
    parser.add_argument("--rl-weight", type=float, default=0.0,
                        help="weight on the policy-gradient term. 0, the default, is pure "
                             "distillation, which is what every released checkpoint is. "
                             "The term is unmeasured at any weight.")
    parser.add_argument("--with-public", action="store_true",
                        help="also train on the public LocalLLaMA/typed-decisions TRAIN split "
                             "(free, 1,200 cases); its test split stays untouched")
    parser.add_argument("--balance-sites", action="store_true",
                        help="weight each site's loss by 1/sqrt(n), so a large site cannot "
                             "impose its answer prior on a small one")
    parser.add_argument("--max-len", type=int, default=0, help="0 = the checkpoint's own")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--dry-run", action="store_true", help="build the data, print shapes, train nothing")
    parser.add_argument("--save-epochs", action="store_true",
                        help="keep a checkpoint per epoch (~1.6 GiB each); off by default")
    parser.add_argument("--min-free-gib", type=float, default=4.0,
                        help="refuse to start a save below this much free space")
    parser.add_argument("--no-amp", action="store_true",
                        help="disable bfloat16 autocast (roughly doubles activation memory)")
    args = parser.parse_args(argv)

    if args.epochs < 1 or args.micro_batch < 1 or args.grad_accum < 1:
        parser.error("--epochs, --micro-batch and --grad-accum must be positive")
    if args.rl_weight > 0 and min(args.sigma_start, args.sigma_end) <= 0:
        parser.error("--rl-weight above 0 needs --sigma-start and --sigma-end above 0")
    out = Path(args.out)
    if out.exists():
        raise SystemExit(
            f"\n  {out} already exists. Choose a new --out; this script never "
            f"overwrites a checkpoint.\n"
        )

    _seed_everything(args.seed)
    data_root = Path(args.data)

    # -- load ---------------------------------------------------------------- #
    from laya.common import QTYPES, build_model, clamp_temperature, proper_reward, temp_bucket

    print(f"\n  base      {args.base}")
    print(f"  device    {args.device}")
    model, cfg, tok = _load_base(args.base, args.device)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    if args.max_len:
        cfg["max_len"] = args.max_len

    train_rows = load_split(data_root, "train", args.build)
    val_rows = load_split(data_root, "val", args.build)
    if not train_rows:
        raise SystemExit(
            f"\n  No training rows at {data_root}: expected train.jsonl, validation.jsonl "
            f"and test.jsonl there.\n"
        )
    train_items = build_items(train_rows, tok, cfg)
    val_items = build_items(val_rows, tok, cfg)

    if args.with_public:
        # The LocalLLaMA/typed-decisions train split. Its test split is never touched,
        # so the benchmark stays honest. This answers one question directly: is our
        # low score there a matter of distribution, or of capacity? If the model
        # holds its internal accuracy while gaining on the public split, the answer
        # is distribution.
        from benchmark import load_public

        pub = load_public("train")
        pub_items = build_items(pub, tok, cfg)
        for item in pub_items:
            item["site"] = "public:" + item["site"]
        print(f"  public    {len(pub):,} cases -> {len(pub_items):,} decisions mixed in")
        train_items += pub_items
    print(f"  train     {len(train_rows):,} cases -> {len(train_items):,} decisions")
    print(f"  val       {len(val_rows):,} cases -> {len(val_items):,} decisions")
    if len(train_items) < 500:
        print("\n  warning: fewer than 500 training decisions. This is small;")
        print("           expect the result to move little.\n")
    if args.dry_run:
        print("\n  --dry-run: nothing trained.\n")
        return 0

    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    # -- optimiser ----------------------------------------------------------- #
    head_params, enc_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head_params if _is_head(name) else enc_params).append(param)
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr_encoder},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    print(f"  params    {sum(p.numel() for p in enc_params):,} encoder  "
          f"{sum(p.numel() for p in head_params):,} head")

    # --- interference control -------------------------------------------------- #
    # A shared decision head sees every site's questions through one set of option
    # markers, so a site with many items and a lopsided answer distribution teaches
    # a global prior that a small, weak-signal site then inherits. That is measured,
    # not theoretical: `voice.addressee` collapsed to always-"false" with 130 items
    # while larger sites skewed false, and `memory.preference` lost 13 points when
    # OTHER sites gained data. Weighting each item by 1/sqrt(n) of its site flattens
    # the pull without silencing the large sites.
    if args.balance_sites:
        per_site = Counter(item["site"] for item in train_items)
        mean_w = sum((1.0 / math.sqrt(per_site[s])) for s in per_site) / len(per_site)
        for item in train_items:
            item["w"] = (1.0 / math.sqrt(per_site[item["site"]])) / mean_w
        top = sorted(per_site.items(), key=lambda kv: -kv[1])
        print(f"  balance   on — {len(per_site)} sites, "
              f"largest {top[0][0]} x{top[0][1]}, smallest {top[-1][0]} x{top[-1][1]}")
    else:
        for item in train_items:
            item["w"] = 1.0

    batch_plan = training_batch_plan(len(train_items), args.micro_batch, args.grad_accum,
                                     args.legacy_drop_tail)
    steps_per_epoch = sum(step for _, _, _, step in batch_plan)
    if steps_per_epoch == 0:
        raise SystemExit("No optimizer updates: add training decisions or disable --legacy-drop-tail")
    print(f"  schedule  {args.epochs} epochs x {steps_per_epoch} updates\n")

    device = torch.device(args.device)
    # bfloat16 autocast: the encoder's activations dominate memory, and Ada
    # handles bf16 natively so no loss scaler is needed. The decision head already
    # casts its logits back to float32, so the scoring rule stays in full
    # precision where the small probability differences actually matter.
    use_amp = (not args.no_amp) and device.type == "cuda"
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp else contextlib.nullcontext()
    )
    print(f"  amp       {'bfloat16' if use_amp else 'off'}")
    model.train()
    started = time.monotonic()
    history = []

    for epoch in range(args.epochs):
        random.shuffle(train_items)
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * (
            epoch / max(1, args.epochs - 1)
        )
        running, seen, updates = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)

        for start, stop, scale, do_step in batch_plan:
            chunk = train_items[start:stop]
            batch = collate(chunk, device, pad_id)
            with autocast:
                # forward returns (option logits, act/escalate logits). Only the
                # option logits carry the decision this trains.
                logits, _act = model(
                    batch["input_ids"], batch["attention_mask"],
                    batch["marker_pos"], batch["marker_mask"], batch["qtype"],
                )
            logits = logits.float()
            mask = batch["marker_mask"].float()

            w = torch.tensor(
                [it["w"] for it in chunk],
                device=device, dtype=torch.float,
            )

            # --- soft cross-entropy against the teacher's distribution ------- #
            log_p = torch.log_softmax(logits.masked_fill(mask == 0, -1e9), dim=-1)
            loss = args.ce_weight * (-(batch["target"] * log_p * mask).sum(-1) * w).mean()

            # --- optional: REINFORCE with a group-mean baseline, against a proper
            # scoring rule. Off at the default weight of 0, and then not computed. --- #
            if args.rl_weight > 0:
                eps = torch.randn((args.group_size,) + logits.shape, device=device) * sigma * mask
                sampled = logits.unsqueeze(0) + eps
                q = torch.softmax(sampled.masked_fill(mask.unsqueeze(0) == 0, -1e9), dim=-1)
                reward = proper_reward(
                    q.reshape(-1, q.shape[-1]),
                    batch["target"].repeat(args.group_size, 1),
                    batch["qtype"].repeat(args.group_size),
                    mask.repeat(args.group_size, 1),
                ).reshape(args.group_size, -1)
                advantage = reward - reward.mean(dim=0, keepdim=True)
                # The sample must be detached: `sampled - logits` is the noise itself
                # and has no gradient.
                logp = -(((sampled.detach() - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
                loss = loss + args.rl_weight * -((advantage.detach() * logp) * w.unsqueeze(0)).mean()

            (loss * scale).backward()
            running += float(loss.detach()) * len(chunk)
            seen += len(chunk)

            if do_step:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt.param_groups for p in g["params"]], 1.0,
                )
                opt.step()
                opt.zero_grad(set_to_none=True)
                updates += 1
                print(f"    epoch {epoch + 1}/{args.epochs}  step {updates}"
                      f"/{steps_per_epoch}  loss {running / max(1, seen):.4f}  sigma {sigma:.2f}",
                      end="\r", flush=True)

        if device.type == "cuda":
            torch.cuda.empty_cache()
        epoch_loss = running / max(1, seen)
        history.append({"epoch": epoch + 1, "loss": epoch_loss, "sigma": sigma,
                        "decisions_seen": seen, "updates": updates})
        print(f"\n  epoch {epoch + 1}: loss {epoch_loss:.4f}")
        if args.save_epochs:
            _save(model, cfg, tok, out / f"epoch-{epoch + 1}", args.min_free_gib)

    seconds = time.monotonic() - started
    print(f"\n  trained in {seconds:.0f}s")
    # The base checkpoint's config describes the base's own training run. Replace that
    # block with this run's, so the saved file does not claim someone else's history.
    base_repo, base_revision = split_revision(args.base)
    cfg["model_name"] = out.name
    cfg["training"] = {
        "fine_tuned_from": base_repo, "base_revision": base_revision,
        "epochs_completed": args.epochs, "updates": sum(row["updates"] for row in history),
        "decisions": len(train_items), "seconds": round(seconds),
        "legacy_drop_tail": args.legacy_drop_tail,
        "objective": "soft cross-entropy" + (f" + {args.rl_weight} policy gradient" if args.rl_weight > 0 else ""),
        "with_public": args.with_public, "balance_sites": args.balance_sites, "seed": args.seed,
    }

    # -- temperature refit --------------------------------------------------- #
    if val_items:
        print("\n  fitting temperatures on the validation split")
        temps = _fit_temperatures(
            model, val_items, cfg, device, clamp_temperature, temp_bucket, pad_id,
        )
        cfg["temperature_by_options"] = temps
        # A question shape with fewer than 12 validation decisions gets no bucket and
        # is served at the base checkpoint's per-type `temperature`, which this run
        # does not refit.
        print("    any other question shape is served at the base checkpoint's per-type temperature")
        print(f"    {len(temps)} bucket(s): " +
              ", ".join(f"{k}={v:.2f}" for k, v in sorted(temps.items())[:8]))
    else:
        print("\n  no validation split; temperatures NOT refitted. The checkpoint stays")
        print("  over-confident, and no band fitted on it should be trusted.")

    _save(model, cfg, tok, out / "final", args.min_free_gib)
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\n  checkpoint -> {out / 'final'}")
    print(f"\n  Next:")
    print(f"    python scripts/evaluate.py \\")
    print(f"        --data {args.data} --checkpoint {out / 'final'}\n")
    return 0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _is_head(name: str) -> bool:
    return any(part in name for part in ("head", "scorer", "decision", "act_", "option"))


def _load_base(base: str, device: str):
    """Load the base checkpoint's weights, config and tokenizer through laya."""
    import laya

    agent = laya.load(resolve_laya(base), device=device)
    model = getattr(agent, "model", None)
    cfg = dict(getattr(agent, "cfg", {}) or {})
    tok = getattr(agent, "tok", None) or getattr(agent, "tokenizer", None)
    if model is None or tok is None:
        raise SystemExit(
            "  Could not reach the model and tokenizer on the loaded agent. The laya "
            "package changed shape; check `laya.agent.Agent` and update _load_base."
        )
    cfg.setdefault("max_len", 512)
    cfg.setdefault("head_max_len", 192)
    return model, cfg, tok


def _save(model, cfg: Dict[str, Any], tok, path: Path, min_free_gib: float = 4.0) -> None:
    """Write the layout ``laya.load`` reads, the same one the base checkpoint ships:
    `rl_agent_config.json`, `model.safetensors`, `tokenizer/` and `encoder/config.json`.

    `tokenizer/` and `encoder/` are the only places `laya.load` looks for them. Without
    them it downloads the encoder's tokenizer and its full weights from the Hub, at
    whatever `main` is that day, before overwriting the weights with these. With them
    the checkpoint loads offline and depends on no third repository."""
    from safetensors.torch import save_file

    if path.exists():
        raise SystemExit(f"  refusing to overwrite {path}")
    # A checkpoint is ~1.7 GiB. Checking first turns "the disk filled up halfway
    # through serializing" into a clean stop with the training still in memory.
    free = shutil.disk_usage(path.parent).free
    if free < min_free_gib * 1024 ** 3:
        raise SystemExit(
            f"\n  Only {free / 1024 ** 3:.1f} GiB free at {path.parent}; a checkpoint "
            f"needs about 1.7 GiB and the floor is {min_free_gib:.1f} GiB. Nothing was "
            f"written. Free some space and re-run.\n"
        )
    path.mkdir(parents=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(path / "model.safetensors"))
    (path / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    tok.save_pretrained(str(path / "tokenizer"))
    model.encoder.config.save_pretrained(str(path / "encoder"))


@torch.no_grad()
def _fit_temperatures(model, items, cfg, device, clamp_temperature, temp_bucket, pad_id) -> Dict[str, float]:
    """One temperature per (question type, option count), by LBFGS on NLL.

    Fit on validation decisions, never on the test split."""
    model.eval()
    buckets: Dict[str, List] = {}
    for start in range(0, len(items), 16):
        batch = collate(items[start:start + 16], device, pad_id)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" \
                else contextlib.nullcontext():
            logits, _act = model(
                batch["input_ids"], batch["attention_mask"],
                batch["marker_pos"], batch["marker_mask"], batch["qtype"],
            )
        logits = logits.float()
        mask = batch["marker_mask"].float()
        for row in range(logits.shape[0]):
            item = items[start + row]
            key = str(temp_bucket(item["qtype"], int(mask[row].sum().item())))
            buckets.setdefault(key, []).append((
                logits[row].detach().cpu(),
                mask[row].detach().cpu(),
                batch["target"][row].detach().cpu(),
            ))

    fitted: Dict[str, float] = {}
    for key, rows in buckets.items():
        if len(rows) < 12:            # too few to fit anything meaningful
            continue
        # A bucket groups option counts by RANGE ("3-5"), so its rows are not the
        # same width. Pad to the widest; the mask already excludes the padding
        # from both the softmax and the loss.
        width = max(r[0].shape[0] for r in rows)

        def _pad(tensor, size=width):
            if tensor.shape[0] == size:
                return tensor
            out = torch.zeros(size, dtype=tensor.dtype)
            out[: tensor.shape[0]] = tensor
            return out

        logits = torch.stack([_pad(r[0]) for r in rows])
        mask = torch.stack([_pad(r[1]) for r in rows])
        target = torch.stack([_pad(r[2]) for r in rows])
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            scaled = logits / torch.exp(log_t).clamp(0.05, 20.0)
            log_p = torch.log_softmax(scaled.masked_fill(mask == 0, -1e9), dim=-1)
            loss = -(target * log_p * mask).sum(-1).mean()
            loss.backward()
            return loss

        opt.step(closure)
        fitted[key] = float(clamp_temperature(float(torch.exp(log_t).item())))
    model.train()
    return fitted


if __name__ == "__main__":
    raise SystemExit(main())
