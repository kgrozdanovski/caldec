"""Stage a CalDec Ultra checkpoint as a Hugging Face model repository.

    python3 scripts/stage_ultra.py --checkpoint <local final/ dir> --model caldec-ultra-380m \\
        --out dist/hub/caldec-ultra-380m

The staged directory holds exactly what `data.load_agent` reads: `model.safetensors`
(the trained weights, byte-identical to the scored checkpoint), `tokenizer/`,
`encoder/` (UltraBERT's config and its pinned modeling code) and
`rl_agent_config.json`, plus the model card rendered from
`huggingface/ultra-model-card.md` with the numbers in `runs/evidence/ultra/numbers.json`.
Only the config's descriptive fields change: `model_name` and the local paths under
`training`. Every staged file's sha256 is written to `SHA256SUMS`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMBERS = ROOT / "runs/evidence/ultra/numbers.json"
CARD = ROOT / "huggingface/ultra-model-card.md"

MODELS = {
    "caldec-ultra-380m": {"encoder": "dmis-lab/UltraBERT-380M", "size": "380M", "other": "caldec-ultra-150m"},
    "caldec-ultra-150m": {"encoder": "dmis-lab/UltraBERT-150M", "size": "150M", "other": "caldec-ultra-380m"},
}
LAYOUT = ("model.safetensors", "rl_agent_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
          "encoder/config.json", "encoder/configuration_ultrabert.py", "encoder/modeling_ultrabert.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_config(cfg: dict, model: str) -> dict:
    spec = MODELS[model]
    if cfg.get("encoder") != spec["encoder"] or not cfg.get("packed") or cfg.get("max_len") != 8192:
        raise SystemExit(f"checkpoint config does not match {model}: encoder {cfg.get('encoder')}, "
                         f"packed {cfg.get('packed')}, max_len {cfg.get('max_len')}")
    cfg = json.loads(json.dumps(cfg))
    cfg["model_name"] = model
    training = cfg.get("training", {})
    training["fine_tuned_from"] = f"{spec['encoder']}@{cfg['encoder_revision']} with a freshly initialized decision head"
    training["base_revision"] = cfg["encoder_revision"]
    rename = {"data-general/train.jsonl": "general mix (scripts/general_data.py: MNLI, BoolQ)",
              "data-general/validation.jsonl": "general mix validation (scripts/general_data.py)",
              "data-long/train.jsonl": "kgrozdanovski/assistant-decisions long/train.jsonl",
              "data-long/validation.jsonl": "kgrozdanovski/assistant-decisions long/validation.jsonl"}
    for key in ("extra_train", "extra_val"):
        training[key] = [rename.get(v, v) for v in training.get(key, [])]
    leftovers = re.findall(r"checkpoints/[^\"]+|\bv[23]-[\w-]+", json.dumps(cfg))
    if leftovers:
        raise SystemExit(f"local names left in the config: {leftovers}")
    return cfg


def render_card(model: str, numbers: dict | None = None) -> str:
    numbers = numbers or json.loads(NUMBERS.read_text(encoding="utf-8"))
    text = CARD.read_text(encoding="utf-8")
    spec = MODELS[model]
    values = {"model": model, "size": spec["size"], "encoder": spec["encoder"], "other": spec["other"],
              "bibkey": model.replace("-", ""), "other_size": MODELS[spec["other"]]["size"]}

    def lookup(path: str):
        if path.startswith("pair|"):            # pair|a|b|split|field
            _, a, b, split, field = path.split("|")
            hits = [p for p in numbers["paired"] if (p["a"], p["b"], p["split"]) == (a, b, split)]
            if len(hits) != 1:
                raise SystemExit(f"no paired test {a} vs {b} on {split}")
            return hits[0][field]
        node = numbers
        for part in path.split("|"):
            node = node[part]
        return node

    # {{name}} is a value above; {{path|to|number:.3f}} reads numbers.json, where @M is
    # this model and @O the other size.
    def fill(match):
        key, fmt = match.group(1), match.group(2) or ""
        if key in values:
            return str(values[key])
        value = lookup(key.replace("@M", model).replace("@O", spec["other"]).replace("@S", spec["size"].lower()))
        if fmt == "p":                          # a p-value as the docs print it
            return "< 0.0001" if value < 1e-4 else f"= {value:.2g}"
        if fmt == "ms":
            return f"{value:.0f}"
        return format(value, fmt) if fmt else str(value)

    out = re.sub(r"\{\{([^}:]+)(?::([^}]+))?\}\}", fill, text)
    if "{{" in out:
        raise SystemExit(f"unfilled placeholder in the card: {out[out.index('{{'):out.index('{{') + 60]}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; stage into a new directory")
    for name in LAYOUT:
        if not (args.checkpoint / name).is_file():
            raise SystemExit(f"{args.checkpoint} has no {name}")
    cfg = public_config(json.loads((args.checkpoint / "rl_agent_config.json").read_text(encoding="utf-8")), args.model)
    args.out.mkdir(parents=True)
    for name in LAYOUT:
        dst = args.out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if name != "rl_agent_config.json":
            shutil.copyfile(args.checkpoint / name, dst)
    (args.out / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (args.out / "README.md").write_text(render_card(args.model), encoding="utf-8")
    if sha256(args.out / "model.safetensors") != sha256(args.checkpoint / "model.safetensors"):
        raise SystemExit("staged weights differ from the checkpoint")
    files = sorted(p for p in args.out.rglob("*") if p.is_file())
    sums = "".join(f"{sha256(p)}  {p.relative_to(args.out)}\n" for p in files)
    (args.out / "SHA256SUMS").write_text(sums, encoding="utf-8")
    print(sums, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
