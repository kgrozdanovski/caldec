"""Export CalDec Ultra 150M to a fixed-shape ONNX graph for the RK3566 NPU benchmark.

Run with the caldec repo's venv, from anywhere:
  CALDEC_DENSE_ATTENTION=1 $CALDEC_PY export_caldec.py --seq 512

The graph is the full per-question decision model: UltraBERT encoder (24 layers) + the 2-layer head +
the option scorer. It outputs the option logits (one row per question). The act head (escalation) is
left out: it needs topk, and its share of the compute is negligible.

Inputs (static): input_ids [1,S] int64, attention_mask [1,S] int64, marker_pos [1,6] int64,
marker_mask [1,6] float32 (1 = real option), qtype [1] int64.
Also writes test cases built from the assistant test split (real questions) as .npz, with fp32 PyTorch
reference logits, for the accuracy check on the PC simulator and on the device.
"""
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("CALDEC_DENSE_ATTENTION", "1")
CALDEC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CALDEC / "scripts"))
# the checkpoint: a Hub id (resolved at the revision scripts/data.py pins), id@revision, or a local directory
CKPT = os.environ.get("CALDEC_CKPT", "kgrozdanovski/caldec-ultra-150m")

import numpy as np
import torch
from torch import nn

from data import load_agent, load_split, resolve_laya  # caldec scripts/data.py
from laya.common import QTYPES, build_sequence, render_options

K = 6  # max_prefixes in rl_agent_config.json


class Core(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        m = self.m
        h = m.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + m.type_emb(qtype)[:, None, :]
        pad = attention_mask == 0
        for layer in m.head.layers:
            h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos[:, :, None].expand(-1, -1, h.size(-1))
        g = torch.gather(h, 1, idx)
        logits = m.scorer(g).squeeze(-1)
        return logits + (marker_mask - 1.0) * 1e4


TAPS = (0, 2, 12, 23)   # encoder layer outputs to expose with --taps (layer 2 is the first global-attention layer)


class CoreMasks(nn.Module):
    """Same computation as Core, but the attention masks come in as float inputs (0 = attend, -1e4 = blocked):
    local_mask / global_mask [1,1,S,S] for the encoder (sliding window 128 / full), head_mask [S,S] for the
    2-layer head. Removes the int64->bool->where mask subgraph that stalls the RKNN graph fusion."""
    def __init__(self, m, taps=False, sel=False):
        super().__init__()
        self.m = m; self.taps = taps; self.sel = sel; self.nlayers = None; self.embeds = False

    def forward(self, input_ids, local_mask, global_mask, head_mask, marker_pos, marker_mask, qtype):
        m, enc = self.m, self.m.encoder
        cfg = enc.config
        h = input_ids if self.embeds else enc.embed_tokens(input_ids); tap = [h]   # embeds: input is [1,S,H] float
        pos = torch.arange(h.size(1)).unsqueeze(0)
        pe = {b: enc.rotary_emb[str(b).replace(".", "_")](h, pos) for b in set(cfg.rope_theta)}
        for i, layer in enumerate(enc.layers[: (self.nlayers or cfg.num_hidden_layers)]):
            w = cfg.attention_windows[i]
            h = layer(h, attention_mask=local_mask if w is not None else global_mask, position_ids=pos,
                      position_embeddings=pe[cfg.rope_theta[i]], sliding_window=w)
            if i in TAPS: tap.append(h)
        h = enc.norm(h); tap.append(h)
        h = h + m.type_emb(qtype)[:, None, :]
        for layer in m.head.layers:
            h = layer(h, src_mask=head_mask)
        if self.sel:   # marker_pos is a one-hot [1,K,S] float matrix: exact row pick as a MatMul (no GatherElements)
            g = torch.matmul(marker_pos, h)
        else:
            g = torch.gather(h, 1, marker_pos[:, :, None].expand(-1, -1, h.size(-1)))
        logits = m.scorer(g).squeeze(-1)
        logits = logits + (marker_mask - 1.0) * 1e4
        return (logits, *tap) if self.taps else logits


def fp16_safe_rmsnorm(enc):
    """Patch every RMSNorm of the encoder to x/64 before squaring (and eps/64^2): the same function exactly,
    but the mean of squares stays inside fp16 range (residual activations reach ~5e3, so x^2 ~ 3e7 > 65504)."""
    import types
    def fwd(self, h):
        x = h / 64.0
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon / 4096.0))
    n = 0
    for m in enc.modules():
        if type(m).__name__.endswith("RMSNorm"):
            m.forward = types.MethodType(fwd, m); n += 1
    return n


def patch_npu_softmax():
    """RK3566 (RKNPU lite, fp16) runs Softmax and Exp only on the CPU, and the toolkit folds any hand-written
    softmax back into exSoftmax13 (CPU). This form avoids both: row max via MaxPool, exp(z) = elu(z) + 1 (exact for
    z <= 0, which holds after subtracting the max), row sum via AvgPool * n. All of these run on the NPU."""
    import torch.nn.functional as F
    orig = F.softmax
    def npu_softmax(x, dim=None, _stacklevel=3, dtype=None):
        if x.dim() in (3, 4) and dim in (-1, x.dim() - 1):
            y = x if x.dim() == 4 else x.unsqueeze(0)
            n = y.shape[-1]
            e = F.elu(y - F.max_pool2d(y, (1, n))) + 1.0
            out = e / (F.avg_pool2d(e, (1, n)) * n)
            out = out if x.dim() == 4 else out.squeeze(0)
            return out.to(dtype) if dtype is not None else out
        return orig(x, dim=dim, dtype=dtype)
    F.softmax = npu_softmax
    return orig


def make_masks(att, window=128):
    S = att.shape[-1]; valid = att[0].astype(bool)
    i = np.arange(S)[:, None]; j = np.arange(S)[None, :]
    key = valid[None, :]
    loc = np.where((np.abs(i - j) <= window) & key, 0.0, -1e4).astype(np.float32)[None, None]
    glo = np.where(np.broadcast_to(key, (S, S)), 0.0, -1e4).astype(np.float32)[None, None]
    head = glo[0, 0].copy()
    return loc, glo, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--cases", type=int, default=120)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[0]) + "/out")
    ap.add_argument("--mask-inputs", action="store_true")
    ap.add_argument("--safe-norm", action="store_true")
    ap.add_argument("--taps", action="store_true")
    ap.add_argument("--embeds", action="store_true", help="token-embedding lookup outside the graph: input inputs_embeds [1,S,H]")
    ap.add_argument("--npu-softmax", action="store_true", help="softmax as MaxPool/ELU/AvgPool so it runs on the RK3566 NPU")
    ap.add_argument("--layers", type=int, default=0, help="profiling only: keep the first N encoder layers (output is NOT the real model)")
    ap.add_argument("--sel", action="store_true", help="one-hot marker_sel [1,K,S] input + MatMul instead of GatherElements")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.backends.mha.set_fastpath_enabled(False)
    agent = load_agent(a.ckpt, device="cpu")
    agent.model.encoder.config._attn_implementation = "eager"
    core = Core(agent.model).eval().float()
    tok = agent.tok
    max_len, head_max_len = agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192)

    # real questions from the assistant test split
    rows = load_split(CALDEC / "data", "test")
    cases = []
    for r in rows:
        for qid, qdef in r["questions"].items():
            q = agent._to_internal(qdef)
            seq, markers = build_sequence(tok, r["state"], q, min(max_len, a.seq), head_max_len)
            if not (2 <= len(markers) <= K) or len(markers) != len(render_options(q)):
                continue
            cases.append((seq, markers, QTYPES[q["t"]]))
        if len(cases) >= a.cases:
            break
    cases = cases[: a.cases]

    def tensors(seq, markers, qt):
        S = a.seq
        ids = np.full((1, S), tok.pad_token_id, np.int64); att = np.zeros((1, S), np.int64)
        ids[0, : len(seq)] = seq; att[0, : len(seq)] = 1
        mp = np.zeros((1, K), np.int64); mm = np.zeros((1, K), np.float32)
        mp[0, : len(markers)] = markers; mm[0, : len(markers)] = 1
        return ids, att, mp, mm, np.array([qt], np.int64)

    arrays = {k: [] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype", "ref", "ntok")}
    with torch.inference_mode():
        for seq, markers, qt in cases:
            t = tensors(seq, markers, qt)
            ref = core(*[torch.from_numpy(x) for x in t]).numpy()
            for k, v in zip(("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"), t):
                arrays[k].append(v)
            arrays["ref"].append(ref); arrays["ntok"].append(len(seq))
    np.savez(out / f"cases_s{a.seq}.npz", **{k: np.stack(v) for k, v in arrays.items()})
    print(f"{len(cases)} cases, real token counts: min {min(arrays['ntok'])} median {int(np.median(arrays['ntok']))} max {max(arrays['ntok'])}")

    # sanity: our Core equals the agent's own model forward (logits) on the first case
    with torch.inference_mode():
        t = [torch.from_numpy(x) for x in tensors(*cases[0])]
        own, _ = agent.model(t[0], t[1], t[2], t[3].bool(), t[4])
        k = int(t[3].sum())
        print("core vs agent.model max |diff| on real options:", float((own[0, :k] - torch.from_numpy(arrays['ref'][0])[0, :k]).abs().max()))

    if a.mask_inputs:
        cm = CoreMasks(agent.model, taps=a.taps, sel=a.sel).eval().float(); cm.nlayers = a.layers or None; cm.embeds = a.embeds
        if a.npu_softmax:
            patch_npu_softmax(); print("softmax patched: MaxPool / ELU+1 / AvgPool form")
        if a.safe_norm:
            print("fp16-safe RMSNorm patched into", fp16_safe_rmsnorm(agent.model.encoder), "modules")
        def mtensors(seq, markers, qt):
            ids, att, mp, mm, q = tensors(seq, markers, qt); loc, glo, head = make_masks(att)
            if a.embeds:
                with torch.inference_mode():
                    ids = agent.model.encoder.embed_tokens(torch.from_numpy(ids)).float().numpy()
            if a.sel:
                oh = np.zeros((1, K, a.seq), np.float32); oh[0, np.arange(K), mp[0]] = 1.0; mp = oh
            return ids, loc, glo, head, mp, mm, q
        worst = 0.0
        with torch.inference_mode():
            for j, (seq, markers, qt) in enumerate(cases):
                y = cm(*[torch.from_numpy(x) for x in mtensors(seq, markers, qt)])
                if a.taps:
                    if j < 3:
                        np.savez(out / f"taps_ref_{j}.npz", *[t.numpy() for t in y])
                    y = y[0]
                y = y.numpy()
                k = len(markers); worst = max(worst, float(np.abs(y[0, :k] - arrays["ref"][j][0, :k]).max()))
        print("CoreMasks vs reference logits, max |diff| over all cases:", worst)
        d = out / f"cases_raw_m{'s' if a.sel else ''}{'e' if a.embeds else ''}_s{a.seq}"; d.mkdir(exist_ok=True)
        for j, (seq, markers, qt) in enumerate(cases):
            ids, loc, glo, head, mp, mm, q = mtensors(seq, markers, qt)
            for nm, v in (("inputs_embeds" if a.embeds else "input_ids", ids), ("local_mask", loc), ("global_mask", glo), ("head_mask", head),
                          ("marker_sel" if a.sel else "marker_pos", mp), ("marker_mask", mm), ("qtype", q)):
                v.tofile(d / f"{j}_{nm}.bin")
        onnx_path = out / (f"caldec_ultra150m_m{'n' if a.safe_norm else ''}{'t' if a.taps else ''}{'s' if a.sel else ''}{'x' if a.npu_softmax else ''}{'e' if a.embeds else ''}{f'_L{a.layers}' if a.layers else ''}_s{a.seq}.onnx")
        t = tuple(torch.from_numpy(x) for x in mtensors(*cases[0]))
        torch.onnx.export(cm, t, str(onnx_path), opset_version=17, dynamo=False,
                          input_names=["inputs_embeds" if a.embeds else "input_ids", "local_mask", "global_mask", "head_mask", "marker_sel" if a.sel else "marker_pos", "marker_mask", "qtype"],
                          output_names=["logits"] + ([f"tap{i}" for i in range(len(TAPS) + 2)] if a.taps else []), do_constant_folding=True)
        print("wrote", onnx_path, onnx_path.stat().st_size // 2**20, "MiB")
        return

    onnx_path = out / f"caldec_ultra150m_s{a.seq}.onnx"
    t = tuple(torch.from_numpy(x) for x in tensors(*cases[0]))
    torch.onnx.export(core, t, str(onnx_path), opset_version=17, dynamo=False,
                      input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
                      output_names=["logits"], do_constant_folding=True)
    print("wrote", onnx_path, onnx_path.stat().st_size // 2**20, "MiB")
    json.dump({"ckpt": a.ckpt, "seq": a.seq, "K": K, "cases": len(cases)}, open(out / f"meta_s{a.seq}.json", "w"))


if __name__ == "__main__":
    main()
