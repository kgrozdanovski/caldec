"""Export CalDec Ultra 150M with every attention written for the RK3566 NPU (no CPU ops inside the encoder/head).

Same weights and same math as export_caldec.py --mask-inputs --sel, but:
- attention as batched 3-D matmuls [H,S,D] with [S,S] masks, so the scores stay in the exMatMul layout
  (no CPU Transposes around the mask Add / softmax);
- softmax as  m = MaxPool(1,S) in the native (N=H, C=Sq, H=1, W=Sk) layout,  e = ELU(s - m) + 1  (= exp(s - m),
  exact for s - m <= 0),  e / sum(e)  -> all NPU ops on RK3566 (Softmax, Exp, ReduceMax, Max run on the CPU there);
- rotary cos/sin as constants (positions are always 0..S-1);
- the head's packed nn.MultiheadAttention split into q/k/v with the same formulation.

Usage (caldec venv):  CALDEC_DENSE_ATTENTION=1 PYTHONPATH=pylib $CALDEC_PY \
    export_npu.py --seq 256 [--layers N] [--softmax npu|std] [--check-only]
Writes out/caldec_npu_<tag>_s<S>.onnx and out/cases_raw_npu_s<S>/ (inputs per case).
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_caldec as ex  # noqa: E402  (load_agent, cases, masks, K)


FUSE = False
PRUNE_HEAD = False
EXP_POW = 1   # exp(z) = (elu(z/p) + 1)^p: elu(z)+1 alone cancels in fp16 (abs noise ~5e-4 on small weights)


def npu_exp_nonpos(z):
    """exp(z) for z <= 0 from the NPU ELU LUT with range reduction (p = EXP_POW, a power of 2)."""
    e = F.elu(z * (1.0 / EXP_POW)) + 1.0
    p = EXP_POW
    while p > 1:
        e = e * e; p //= 2
    return e


OPT2 = False  # fold q scaling and 1/EXP_POW into the q projection; normalise after P.V


def npu_attn(q, k, v, mask, S, scale, mode):
    """softmax((q k^T) * scale + mask) @ v with only NPU ops. q, k, v [H,S,D]."""
    if mode == "std" or not OPT2:
        sc = torch.bmm(q * scale, k.transpose(1, 2)) + mask
        return torch.bmm(npu_softmax_rows(sc, S, mode), v)
    H = q.shape[0]
    z = torch.bmm(q, k.transpose(1, 2)) + mask              # q already carries scale / EXP_POW
    m = F.max_pool2d(z.view(H, S, 1, S), (1, S)).view(H, S, 1)
    e = F.elu(z - m) + 1.0
    p = EXP_POW
    while p > 1:
        e = e * e; p //= 2
    ssum = F.avg_pool2d(e.view(H, S, 1, S), (1, S)).view(H, S, 1) * S
    return torch.bmm(e, v) / ssum


def npu_softmax_rows(sc, S, mode):
    """Row softmax over the last axis of sc [H,S,S] using only RK3566 NPU ops."""
    if mode == "std":
        return torch.softmax(sc, -1)
    H = sc.shape[0]
    m = F.max_pool2d(sc.view(H, S, 1, S), (1, S)).view(H, S, 1)
    e = npu_exp_nonpos(sc - m)
    if mode == "npu-sum":
        return e / e.sum(-1, keepdim=True)
    ssum = F.avg_pool2d(e.view(H, S, 1, S), (1, S)).view(H, S, 1) * S   # row sum in the native layout
    return e / ssum


class NpuCalDec(nn.Module):
    def __init__(self, model, S, nlayers=None, softmax="npu"):
        super().__init__()
        self.m, self.S, self.softmax = model, S, softmax
        enc = model.encoder
        cfg = enc.config
        self.nl = nlayers or cfg.num_hidden_layers
        self.windows = cfg.attention_windows
        self.H, self.D = cfg.num_attention_heads, getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        h0 = torch.zeros(1, S, cfg.hidden_size)
        pos = torch.arange(S).unsqueeze(0)
        thetas = cfg.rope_theta if isinstance(cfg.rope_theta, (list, tuple)) else [cfg.rope_theta] * self.nl
        self.thetas = list(thetas)
        with torch.no_grad():
            for t in set(self.thetas):
                cos, sin = enc.rotary_emb[str(t).replace(".", "_")](h0, pos)     # [1,S,D]
                self.register_buffer(f"cos_{str(t).replace('.', '_')}", cos[0].float().clone())
                self.register_buffer(f"sin_{str(t).replace('.', '_')}", sin[0].float().clone())

    def rope(self, x, t):          # x [H,S,D]
        key = str(t).replace(".", "_")
        cos, sin = getattr(self, f"cos_{key}"), getattr(self, f"sin_{key}")
        half = x.shape[-1] // 2
        rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * cos + rot * sin

    def enc_attn(self, a, h, mask, t):          # h [S,E] normed hidden, mask [S,S]
        S, H, D = self.S, self.H, self.D
        if FUSE:   # one 768 -> 3*768 projection, then split
            w = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0)
            b = None if a.q_proj.bias is None else torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias], 0)
            q, k, v = F.linear(h, w, b).split(H * D, -1)
            q, k, v = (x.reshape(S, H, D).transpose(0, 1) for x in (q, k, v))
        else:
            q = a.q_proj(h).view(S, H, D).transpose(0, 1)
            k = a.k_proj(h).view(S, H, D).transpose(0, 1)
            v = a.v_proj(h).view(S, H, D).transpose(0, 1)
        if OPT2 and self.softmax != "std":
            q = q * (a.scaling / EXP_POW)        # folded into the projection by the compiler (const Mul on a Conv output)
        abl = os.environ.get("ABL", "")
        if "rope" not in abl:
            q, k = self.rope(q, t), self.rope(k, t)
        if "sm" in abl:
            o = torch.bmm(torch.bmm(q, k.transpose(1, 2)) + mask, v)
        else:
            o = npu_attn(q, k, v, mask, S, a.scaling, self.softmax)
        return a.o_proj(o.transpose(0, 1).reshape(S, H * D))

    def mlp(self, mlp, h):
        if not FUSE:
            return mlp(h)
        w = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0)
        b = None if mlp.gate_proj.bias is None else torch.cat([mlp.gate_proj.bias, mlp.up_proj.bias], 0)
        g, u = F.linear(h, w, b).split(mlp.intermediate_size, -1)
        return mlp.down_proj(mlp.act_fn(g) * u)

    def head_last_pruned(self, layer, x, sel, mask_row):
        """Last head layer, evaluated only at the K marker rows (sel [K,S] one-hot). Exact: norms, MLP and residuals
        are per row; attention needs all keys/values but only the marker queries."""
        S = self.S
        mha = layer.self_attn
        E, Hh = mha.embed_dim, mha.num_heads
        Dh = E // Hh
        Kq = sel.shape[0]
        h = layer.norm1(x)                                        # [S,E] (keys/values from all rows)
        xs = torch.matmul(sel, x)                                 # [K,E] residual stream at the markers
        hs = torch.matmul(sel, h)
        wq, wk, wv = mha.in_proj_weight.chunk(3, 0)
        bq, bk, bv = mha.in_proj_bias.chunk(3, 0)
        scale = Dh ** -0.5 / (EXP_POW if (OPT2 and self.softmax != "std") else 1)
        q = (F.linear(hs, wq, bq) * scale).view(Kq, Hh, Dh).transpose(0, 1)      # [Hh,K,Dh]
        k = F.linear(h, wk, bk).view(S, Hh, Dh).transpose(0, 1)                   # [Hh,S,Dh]
        v = F.linear(h, wv, bv).view(S, Hh, Dh).transpose(0, 1)
        z = torch.bmm(q, k.transpose(1, 2)) + mask_row                             # [Hh,K,S]
        if self.softmax == "std":
            p = torch.softmax(z if OPT2 else z, -1)
            o = torch.bmm(p, v)
        else:
            m = F.max_pool2d(z.view(Hh, Kq, 1, S), (1, S)).view(Hh, Kq, 1)
            e = F.elu(z - m) + 1.0
            pw = EXP_POW if OPT2 else 1
            if not OPT2:
                e = npu_exp_nonpos(z - m); pw = 1
            while pw > 1:
                e = e * e; pw //= 2
            ssum = F.avg_pool2d(e.view(Hh, Kq, 1, S), (1, S)).view(Hh, Kq, 1) * S
            o = torch.bmm(e, v) / ssum
        xs = xs + mha.out_proj(o.transpose(0, 1).reshape(Kq, E))
        return xs + layer.linear2(layer.activation(layer.linear1(layer.norm2(xs))))

    def head_layer(self, layer, x, mask):         # nn.TransformerEncoderLayer, norm_first=True, batch_first=True
        S = self.S
        mha = layer.self_attn
        E, Hh = mha.embed_dim, mha.num_heads
        Dh = E // Hh
        h = layer.norm1(x)
        wq, wk, wv = mha.in_proj_weight.chunk(3, 0)
        bq, bk, bv = mha.in_proj_bias.chunk(3, 0)
        q = F.linear(h, wq, bq).view(S, Hh, Dh).transpose(0, 1)
        k = F.linear(h, wk, bk).view(S, Hh, Dh).transpose(0, 1)
        v = F.linear(h, wv, bv).view(S, Hh, Dh).transpose(0, 1)
        if OPT2 and self.softmax != "std":
            q = q * (Dh ** -0.5 / EXP_POW)
        o = npu_attn(q, k, v, mask, S, Dh ** -0.5, self.softmax).transpose(0, 1).reshape(S, E)
        x = x + mha.out_proj(o)
        if "hmlp" in os.environ.get("ABL", ""):
            return x
        return x + layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))

    def forward(self, input_ids, local_mask, global_mask, head_mask, marker_sel, marker_mask, qtype):
        m, enc = self.m, self.m.encoder
        x = enc.embed_tokens(input_ids)[0]                       # [S,E]
        for i, layer in enumerate(enc.layers[: self.nl]):
            mask = local_mask if self.windows[i] is not None else global_mask
            abl = os.environ.get("ABL", "")
            if "attn" not in abl:
                x = x + self.enc_attn(layer.self_attn, layer.input_layernorm(x), mask, self.thetas[i])
            if "mlp" not in abl:
                x = x + self.mlp(layer.mlp, layer.post_attention_layernorm(x))
        x = enc.norm(x)
        x = x + m.type_emb(qtype)                                # [1,E] broadcast
        hl = list(m.head.layers)
        if PRUNE_HEAD and "head" not in os.environ.get("ABL", ""):
            for layer in hl[:-1]:
                x = self.head_layer(layer, x, head_mask)
            g = self.head_last_pruned(hl[-1], x, marker_sel[0], head_mask[0:1])    # [K,E]
        else:
            for layer in hl:
                if "head" in os.environ.get("ABL", ""):
                    break
                x = self.head_layer(layer, x, head_mask)
            g = torch.matmul(marker_sel[0], x)                   # [K,E]
        logits = m.scorer(g).squeeze(-1).unsqueeze(0)            # [1,K]
        return logits + (marker_mask - 1.0) * 1e4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=ex.CKPT)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--softmax", default="npu", choices=["npu", "npu-sum", "std"])
    ap.add_argument("--cases", type=int, default=120)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[0]) + "/out")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--npz", default=None, help="cases file (e.g. out/cases_eval400_s256.npz); inputs truncated to --seq")
    ap.add_argument("--prune-head", action="store_true", help="last head layer only at the marker rows (exact)")
    ap.add_argument("--opt2", action="store_true", help="fold scaling into q, normalise after P.V")
    ap.add_argument("--fuse", action="store_true", help="fused QKV and gate|up projections")
    ap.add_argument("--exp-pow", type=int, default=1, help="range reduction for exp: 1, 2, 4, 8")
    a = ap.parse_args()
    global EXP_POW, FUSE, OPT2, PRUNE_HEAD; EXP_POW = a.exp_pow; FUSE = a.fuse; OPT2 = a.opt2; PRUNE_HEAD = a.prune_head
    out = Path(a.out); S = a.seq
    torch.backends.mha.set_fastpath_enabled(False)
    agent = ex.load_agent(a.ckpt, device="cpu")
    agent.model.encoder.config._attn_implementation = "eager"
    model = agent.model.eval().float()
    c = dict(np.load(a.npz or (out / f"cases_s{S}.npz")))         # questions + fp32 references
    fit = [i for i in range(c["input_ids"].shape[0]) if int(c["attention_mask"][i].sum()) <= S]
    for k in ("input_ids", "attention_mask"):
        c[k] = c[k][:, :, :S]                                     # buckets: truncate the padding only
    n = min(a.cases, len(fit))
    net = NpuCalDec(model, S, a.layers or None, a.softmax).eval()

    def inputs(i):
        att = c["attention_mask"][i]
        loc, glo, head = ex.make_masks(att)
        mp = c["marker_pos"][i]
        sel = np.zeros((1, ex.K, S), np.float32); sel[0, np.arange(ex.K), mp[0]] = 1.0
        return (c["input_ids"][i], loc[0, 0].astype(np.float32), glo[0, 0].astype(np.float32), head.astype(np.float32),
                sel, c["marker_mask"][i].astype(np.float32), c["qtype"][i])

    worst, agree = 0.0, 0
    with torch.inference_mode():
        for i in fit[:n]:
            y = net(*[torch.from_numpy(v) for v in inputs(i)]).numpy()[0]
            k = int(c["marker_mask"][i].sum()); r = c["ref"][i][0]
            worst = max(worst, float(np.abs(y[:k] - r[:k]).max())); agree += int(y[:k].argmax() == r[:k].argmax())
    print(f"rewrite vs reference fp32 logits ({'full' if not a.layers else f'{a.layers} layers'}, softmax={a.softmax}): "
          f"max |diff| {worst:.3g}, argmax {agree}/{n}")
    if a.check_only:
        return
    names = ["input_ids", "local_mask", "global_mask", "head_mask", "marker_sel", "marker_mask", "qtype"]
    if not a.npz:
        d = out / f"cases_raw_npu_s{S}"; d.mkdir(exist_ok=True)
        for i in range(c["input_ids"].shape[0]):
            for nm, v in zip(names, inputs(i)):
                v.tofile(d / f"{i}_{nm}.bin")
    tag = ("L%d_" % a.layers if a.layers else "") + (("A" + os.environ["ABL"].replace(",", "") + "_") if os.environ.get("ABL") else "") + a.softmax + (f"e{a.exp_pow}" if a.exp_pow > 1 else "") + ("f" if a.fuse else "") + ("o" if a.opt2 else "") + ("p" if a.prune_head else "")
    path = out / f"caldec_npu_{tag}_s{S}.onnx"
    torch.onnx.export(net, tuple(torch.from_numpy(v) for v in inputs(fit[0])), str(path), opset_version=17, dynamo=False,
                      input_names=names, output_names=["logits"], do_constant_folding=True)
    print("wrote", path, path.stat().st_size // 2**20, "MiB")


if __name__ == "__main__":
    main()
