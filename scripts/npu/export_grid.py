"""CalDec Ultra 150M for the RK3566 NPU with the hidden state kept as a token grid [1, E, Hs, Ws] (S = Hs*Ws).

Why: a chain of 768x768 linears runs 36% faster as 1x1 convs on (1,768,16,16) than on (1,768,1,256) and 56% faster
than as [S,E] matmuls (lay/lay.py on the board: 13.8 / 17.5 / 21.5 ms). Same weights and math as export_npu.py:
attention with the NPU softmax (MaxPool max, ELU-based exp with range reduction, AvgPool sum), rotary constants,
one-hot option select. RMSNorm / LayerNorm over the channel axis.

Usage (caldec venv): CALDEC_DENSE_ATTENTION=1 PYTHONPATH=pylib .../python export_grid.py --seq 256 [--layers N]
       [--npz out/cases_eval400_s256.npz] [--check-only] [--head-grid]
Writes out/caldec_grid_<tag>_s<S>.onnx (inputs identical to export_npu.py: input_ids, local/global/head masks [S,S],
marker_sel [1,K,S], marker_mask, qtype).
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_caldec as ex  # noqa: E402

EXP_POW = 2
PRUNE = False


def grid_hw(S):
    w = 16
    assert S % w == 0, S
    return S // w, w


CHUNK_HALVES = os.environ.get("CHUNK_HALVES") == "1"


def conv(x, lin_w, lin_b=None):
    """1x1 conv of a token grid with a Linear's weight [out,in].
    CHUNK_HALVES=1: above 128 tokens, run the conv on the two halves of the token grid and concatenate (exact: a 1x1
    conv is per token). Measured on an 8-conv chain: 160 tok 10.4 -> 8.3 ms, 192 11.2 -> 9.8, 224 12.5 -> 11.5,
    256 13.8 -> 12.8 (chunk/chunk.py)."""
    w = lin_w[:, :, None, None]
    if CHUNK_HALVES and x.shape[2] * x.shape[3] > 128 and x.shape[2] % 2 == 0:
        return torch.cat([F.conv2d(p, w, lin_b) for p in x.split(x.shape[2] // 2, 2)], 2)
    return F.conv2d(x, w, lin_b)


def rms_c(x, weight, eps):                       # RMSNorm over channels of [1,E,h,w]
    v = x.pow(2).mean(1, keepdim=True)
    return x * torch.rsqrt(v + eps) * weight[None, :, None, None]


def ln_c(x, ln):                                 # LayerNorm over channels of [1,E,h,w]
    mu = x.mean(1, keepdim=True)
    xc = x - mu
    v = xc.pow(2).mean(1, keepdim=True)
    return xc * torch.rsqrt(v + ln.eps) * ln.weight[None, :, None, None] + ln.bias[None, :, None, None]


SSCALE = None   # set per forward when --tables-input: a [1] input tensor holding S


def attn_core(q, k, v, mask, S):
    """q, k, v [H,D,S] (q already scaled by scale/EXP_POW); mask [S,S]; returns [H,D,S]."""
    H = q.shape[0]
    z = torch.bmm(q.transpose(1, 2), k) + mask                    # [H,Sq,Sk]
    m = F.max_pool2d(z.view(H, S, 1, S), (1, S)).view(H, S, 1)
    e = F.elu(z - m) + 1.0
    p = EXP_POW
    while p > 1:
        e = e * e; p //= 2
    avg = F.avg_pool2d(e.view(H, S, 1, S), (1, S)).view(H, S, 1)
    ssum = avg * SSCALE if SSCALE is not None else avg * S
    o = torch.bmm(e, v.transpose(1, 2)) / ssum                    # [H,Sq,D]
    return o.transpose(1, 2)                                      # [H,D,Sq]


class GridCalDec(nn.Module):
    def __init__(self, model, S, nlayers=None):
        super().__init__()
        self.m, self.S = model, S
        self.hh, self.ww = grid_hw(S)
        enc = model.encoder; cfg = enc.config
        self.nl = nlayers or cfg.num_hidden_layers
        self.windows = cfg.attention_windows
        self.H, self.D = cfg.num_attention_heads, getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        thetas = cfg.rope_theta if isinstance(cfg.rope_theta, (list, tuple)) else [cfg.rope_theta] * self.nl
        self.thetas = list(thetas)
        self.tabs = None; self.tkeys = sorted({str(t).replace(".", "_") for t in self.thetas})
        h0 = torch.zeros(1, S, cfg.hidden_size); pos = torch.arange(S).unsqueeze(0)
        with torch.no_grad():
            for t in set(self.thetas):
                cos, sin = enc.rotary_emb[str(t).replace(".", "_")](h0, pos)              # [1,S,D]
                key = str(t).replace(".", "_")
                self.register_buffer(f"cos_{key}", cos[0].t().contiguous().float())       # [D,S]
                self.register_buffer(f"sin_{key}", sin[0].t().contiguous().float())

    def rope(self, x, t):            # x [H,D,S]
        key = str(t).replace(".", "_")
        if self.tabs is not None:   # tables as an input: [2*nthetas, D, S] in the order of self.tkeys
            j = self.tkeys.index(key); cos, sin = self.tabs[2 * j], self.tabs[2 * j + 1]
        else:
            cos, sin = getattr(self, f"cos_{key}"), getattr(self, f"sin_{key}")
        half = x.shape[1] // 2
        if self.tabs is not None:     # input tables carry sin' = cat(sin[half:], -sin[:half]): rope = x*cos + swap(x*sin')
            y = x * sin
            H, D, S = y.shape
            if os.environ.get("ROPE_SWAP", "mm") == "flip":
                sw = y.view(H, 2, half, S).flip(1).reshape(H, D, S)
            else:                     # explicit half-swap permutation as a matmul
                P = torch.roll(torch.eye(D), half, 1)
                sw = torch.matmul(P, y)
            return x * cos + sw
        rot = torch.cat((-x[:, half:], x[:, :half]), dim=1)
        return x * cos + rot * sin

    def g2heads(self, y, H, D):      # [1,H*D,h,w] -> [H,D,S]
        return y.reshape(H, D, self.S)

    def heads2g(self, o):            # [H,D,S] -> [1,H*D,h,w]
        return o.reshape(1, -1, self.hh, self.ww)

    def enc_attn(self, a, h, mask, t):
        H, D = self.H, self.D
        w = torch.cat([a.q_proj.weight * (a.scaling / EXP_POW), a.k_proj.weight, a.v_proj.weight], 0)
        b = None if a.q_proj.bias is None else torch.cat([a.q_proj.bias * (a.scaling / EXP_POW), a.k_proj.bias, a.v_proj.bias], 0)
        qkv = conv(h, w, b)
        q, k, v = (self.g2heads(y, H, D) for y in qkv.split(H * D, 1))
        abl = os.environ.get("ABL", "")
        if "rope" not in abl:
            q, k = self.rope(q, t), self.rope(k, t)
        if "sm" in abl:      # timing only: no softmax chain
            o = torch.bmm(torch.bmm(q.transpose(1, 2), k) + mask, v.transpose(1, 2)).transpose(1, 2)
        elif "core" in abl:  # timing only: no attention core at all (q*k*v elementwise stand-in)
            o = q * k + v
        else:
            o = attn_core(q, k, v, mask, self.S)
        return conv(self.heads2g(o), a.o_proj.weight, a.o_proj.bias)

    def mlp(self, mlp, h):
        if os.environ.get("CHUNK_MLP") == "1" and self.S > 128 and not getattr(self, "_in_chunk", False):
            self._in_chunk = True     # whole MLP block per token half, one Concat (exact: the MLP is per token)
            try:
                return torch.cat([self.mlp(mlp, p) for p in h.split(h.shape[2] // 2, 2)], 2)
            finally:
                self._in_chunk = False
        w = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0)
        b = None if mlp.gate_proj.bias is None else torch.cat([mlp.gate_proj.bias, mlp.up_proj.bias], 0)
        g, u = conv(h, w, b).split(mlp.intermediate_size, 1)
        return conv(mlp.act_fn(g) * u, mlp.down_proj.weight, mlp.down_proj.bias)

    def head_layer(self, layer, x, mask):    # nn.TransformerEncoderLayer (norm_first) on the grid
        mha = layer.self_attn; E, Hh = mha.embed_dim, mha.num_heads; Dh = E // Hh
        h = ln_c(x, layer.norm1)
        wq, wk, wv = mha.in_proj_weight.chunk(3, 0); bq, bk, bv = mha.in_proj_bias.chunk(3, 0)
        sc = Dh ** -0.5 / EXP_POW
        qkv = conv(h, torch.cat([wq * sc, wk, wv], 0), torch.cat([bq * sc, bk, bv], 0))
        q, k, v = (self.g2heads(y, Hh, Dh) for y in qkv.split(E, 1))
        x = x + conv(self.heads2g(attn_core(q, k, v, mask, self.S)), mha.out_proj.weight, mha.out_proj.bias)
        f = conv(layer.activation(conv(ln_c(x, layer.norm2), layer.linear1.weight, layer.linear1.bias)),
                 layer.linear2.weight, layer.linear2.bias)
        return x + f

    def head_last_pruned(self, layer, x, sel):
        """Last head layer only at the K marker rows (exact). x grid [1,E,h,w]; sel [K,S]; returns [K,E]."""
        mha = layer.self_attn; E, Hh = mha.embed_dim, mha.num_heads; Dh = E // Hh; S = self.S; K = sel.shape[0]
        h = ln_c(x, layer.norm1)                                              # grid
        wq, wk, wv = mha.in_proj_weight.chunk(3, 0); bq, bk, bv = mha.in_proj_bias.chunk(3, 0)
        kv = conv(h, torch.cat([wk, wv], 0), torch.cat([bk, bv], 0))
        k, v = (self.g2heads(y, Hh, Dh) for y in kv.split(E, 1))             # [Hh,Dh,S]
        xs = torch.matmul(sel, x.reshape(-1, S).t())                          # [K,E] residual at the markers
        hs = torch.matmul(sel, h.reshape(-1, S).t())
        sc = Dh ** -0.5 / EXP_POW
        q = F.linear(hs, wq * sc, bq * sc).view(K, Hh, Dh).permute(1, 2, 0)  # [Hh,Dh,K]
        z = torch.bmm(q.transpose(1, 2), k) + self.mask_row                   # [Hh,K,S]
        m = F.max_pool2d(z.view(Hh, K, 1, S), (1, S)).view(Hh, K, 1)
        e = F.elu(z - m) + 1.0
        p = EXP_POW
        while p > 1:
            e = e * e; p //= 2
        ssum = F.avg_pool2d(e.view(Hh, K, 1, S), (1, S)).view(Hh, K, 1) * S
        o = (torch.bmm(e, v.transpose(1, 2)) / ssum).transpose(0, 1).reshape(K, E)
        xs = xs + mha.out_proj(o)
        return xs + layer.linear2(layer.activation(layer.linear1(layer.norm2(xs))))

    def rope_table(self):   # [2n, D, S]: cos, sin' (sign-folded, half-swapped sin) per theta
        out = []
        for k in self.tkeys:
            cos, sin = getattr(self, f"cos_{k}"), getattr(self, f"sin_{k}"); h = sin.shape[0] // 2
            out += [cos, torch.cat((sin[h:], -sin[:h]), 0)]
        return torch.stack(out)

    def forward(self, input_ids, local_mask, global_mask, head_mask, marker_sel, marker_mask, qtype, rope_tab=None, s_scale=None):
        global SSCALE
        self.tabs = rope_tab; SSCALE = s_scale
        m, enc = self.m, self.m.encoder
        x = enc.embed_tokens(input_ids)[0]                                     # [S,E]
        x = x.t().reshape(1, -1, self.hh, self.ww)                             # grid [1,E,h,w]
        for i, layer in enumerate(enc.layers[: self.nl]):
            mask = local_mask if self.windows[i] is not None else global_mask
            ln1, ln2 = layer.input_layernorm, layer.post_attention_layernorm
            abl = os.environ.get("ABL", "")
            if "attn" not in abl:
                x = x + self.enc_attn(layer.self_attn, rms_c(x, ln1.weight, ln1.variance_epsilon), mask, self.thetas[i])
            if "mlp" not in abl:
                x = x + self.mlp(layer.mlp, rms_c(x, ln2.weight, ln2.variance_epsilon))
        x = rms_c(x, enc.norm.weight, enc.norm.variance_epsilon)
        x = x + m.type_emb(qtype)[0][None, :, None, None]
        hl = list(m.head.layers)
        if PRUNE:
            for layer in hl[:-1]:
                x = self.head_layer(layer, x, head_mask)
            self.mask_row = head_mask[0:1]
            g = self.head_last_pruned(hl[-1], x, marker_sel[0])
        else:
            for layer in hl:
                if "head" in os.environ.get("ABL", ""):
                    break
                x = self.head_layer(layer, x, head_mask)
            xs = x.reshape(1, -1, self.S)[0]                                   # [E,S]
            g = torch.matmul(marker_sel[0], xs.t())                            # [K,E]
        logits = m.scorer(g).squeeze(-1).unsqueeze(0)
        return logits + (marker_mask - 1.0) * 1e4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=ex.CKPT)
    ap.add_argument("--seq", type=int, default=256); ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--cases", type=int, default=120); ap.add_argument("--npz", default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[0]) + "/out"); ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--prune-head", action="store_true")
    ap.add_argument("--tables-input", action="store_true", help="RoPE tables and the softmax sum scale as inputs (weight sharing across buckets)")
    a = ap.parse_args(); out = Path(a.out); S = a.seq
    global PRUNE; PRUNE = a.prune_head
    torch.backends.mha.set_fastpath_enabled(False)
    agent = ex.load_agent(a.ckpt, device="cpu")
    agent.model.encoder.config._attn_implementation = "eager"
    model = agent.model.eval().float()
    c = dict(np.load(a.npz or (out / f"cases_s{S}.npz")))
    fit = [i for i in range(c["input_ids"].shape[0]) if int(c["attention_mask"][i].sum()) <= S]
    for kk in ("input_ids", "attention_mask"):
        c[kk] = c[kk][:, :, :S]
    net = GridCalDec(model, S, a.layers or None).eval()

    def inputs(i):
        loc, glo, head = ex.make_masks(c["attention_mask"][i])
        sel = np.zeros((1, ex.K, S), np.float32); sel[0, np.arange(ex.K), c["marker_pos"][i][0]] = 1.0
        return (c["input_ids"][i], loc[0, 0].astype(np.float32), glo[0, 0].astype(np.float32), head.astype(np.float32),
                sel, c["marker_mask"][i].astype(np.float32), c["qtype"][i])

    worst, agree, n = 0.0, 0, 0
    with torch.inference_mode():
        for i in fit[: a.cases]:
            y = net(*[torch.from_numpy(v) for v in inputs(i)]).numpy()[0]
            k = int(c["marker_mask"][i].sum()); r = c["ref"][i][0]; n += 1
            worst = max(worst, float(np.abs(y[:k] - r[:k]).max())); agree += int(y[:k].argmax() == r[:k].argmax())
    print(f"grid rewrite vs reference fp32 logits ({'full' if not a.layers else f'{a.layers} layers'}): "
          f"max |diff| {worst:.3g}, argmax {agree}/{n}")
    if a.check_only:
        return
    names = ["input_ids", "local_mask", "global_mask", "head_mask", "marker_sel", "marker_mask", "qtype"]
    extra = ()
    if a.tables_input:
        names += ["rope_tab", "s_scale"]
        tab = net.rope_table().numpy().astype(np.float32); sc = np.array([float(S)], np.float32)
        extra = (torch.from_numpy(tab), torch.from_numpy(sc))
        td = out / f"tables_s{S}"; td.mkdir(exist_ok=True); tab.tofile(td / "rope_tab.bin"); sc.tofile(td / "s_scale.bin")
        with torch.inference_mode():   # the input-table graph must equal the constant-table graph
            i0 = fit[0]; y0 = net(*[torch.from_numpy(v) for v in inputs(i0)]).numpy()
            y1 = net(*[torch.from_numpy(v) for v in inputs(i0)], *extra).numpy()
        print("tables-input vs constants max |diff|:", float(np.abs(y0 - y1).max()))
    tag = ("L%d_" % a.layers if a.layers else "") + (("A" + os.environ["ABL"].replace(",", "") + "_") if os.environ.get("ABL") else "") + "g" + ("p" if a.prune_head else "") + ("t" if a.tables_input else "")
    path = out / f"caldec_grid_{tag}_s{S}.onnx"
    torch.onnx.export(net, tuple(torch.from_numpy(v) for v in inputs(fit[0])) + extra, str(path), opset_version=17,
                      dynamo=False, input_names=names, output_names=["logits"], do_constant_folding=True)
    print("wrote", path, path.stat().st_size // 2**20, "MiB")


if __name__ == "__main__":
    main()
