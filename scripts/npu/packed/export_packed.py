"""Packed CalDec Ultra 150M (packed, `CALDEC_CKPT`) for the RK3566 NPU: one encoder pass per case.

The packed sequence holds every question of a case ahead of one copy of the state (caldec scripts/packed.py). The
encoder runs once over it; the decision head then runs per question over [CLS] + that question's own segment. On the
NPU all questions' views are gathered into one head grid of Sh rows (a one-hot matmul) and run in a single head pass
under a block-diagonal mask, which is exactly per-question attention: a view's rows see only that view's rows.

Graph inputs (bucket S, head rows Sh = min(S, 192), Kt = 12 marker slots over all questions):
  input_ids [1,S] int64; local_mask, global_mask [S,S]; view_sel [S,Sh] (column r one-hot at the source token of head
  row r); row_type [T,Sh] (one-hot qtype of head row r); head_mask [Sh,Sh] block-diagonal; marker_sel [Kt,Sh];
  marker_mask [1,Kt]; rope_tab, s_scale (encoder, as in export_grid --tables-input); h_scale [1] (= Sh).
Output: logits [1,Kt] (marker k of the case, questions in order; masked slots -1e4).

Usage (caldec venv, PYTHONPATH=pylib CALDEC_DENSE_ATTENTION=1):
  export_packed.py prep  --n 400 --max-len 384          -> packed/cases.pkl (ids, views, markers, qtype, gold, refs)
  export_packed.py export --seq S                         -> packed/onnx/pk_s<S>.onnx (+ fp32 rewrite check)
  export_packed.py cases --buckets 96,128,...             -> packed/cases_b<S>/<j>_<input>.bin, packed/buckets.json
"""
import argparse, json, os, pickle, sys   # pickle: only our own packed/cases.pkl, written by "prep"
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import export_caldec as ex  # noqa: E402
import export_grid as eg    # noqa: E402

CKPT = Path(ex.resolve_laya(ex.CKPT))   # local directory (tokenizer, config)
KT, SH_MAX, W = 12, int(os.environ.get("SH_MAX", "192")), 128   # SH_MAX: head rows cap (experiments)
ACT = os.environ.get("ACT") == "1"   # ACT=1: + input cls_sel [Kt,Sh], + output hcls [Kt,E] (the act head's input rows), files pk2_*


def sh_for(S):
    return min(S, SH_MAX)


class GridPacked(eg.GridCalDec):
    def forward(self, input_ids, local_mask, global_mask, view_sel, row_type, head_mask, marker_sel, marker_mask,
                cls_sel=None, rope_tab=None, s_scale=None, h_scale=None):
        self.tabs = rope_tab; eg.SSCALE = s_scale
        m, enc = self.m, self.m.encoder
        S, E = self.S, enc.config.hidden_size
        x = enc.embed_tokens(input_ids)[0].t().reshape(1, -1, self.hh, self.ww)
        for i, layer in enumerate(enc.layers[: self.nl]):
            mask = local_mask if self.windows[i] is not None else global_mask
            ln1, ln2 = layer.input_layernorm, layer.post_attention_layernorm
            x = x + self.enc_attn(layer.self_attn, eg.rms_c(x, ln1.weight, ln1.variance_epsilon), mask, self.thetas[i])
            x = x + self.mlp(layer.mlp, eg.rms_c(x, ln2.weight, ln2.variance_epsilon))
        x = eg.rms_c(x, enc.norm.weight, enc.norm.variance_epsilon)
        Sh = view_sel.shape[1]
        hq = torch.matmul(x.reshape(E, S), view_sel) + torch.matmul(m.type_emb.weight.t(), row_type)   # [E,Sh]
        # head on the Sh grid: swap the grid geometry and the softmax-sum scale for the head pass
        S0, hh0, ww0 = self.S, self.hh, self.ww
        self.S, (self.hh, self.ww) = Sh, eg.grid_hw(Sh)
        eg.SSCALE = h_scale
        xh = hq.reshape(1, E, self.hh, self.ww)
        for layer in m.head.layers:
            xh = self.head_layer(layer, xh, head_mask)
        self.S, self.hh, self.ww = S0, hh0, ww0
        eg.SSCALE = None
        g = torch.matmul(marker_sel, xh.reshape(E, Sh).t())                                            # [Kt,E]
        logits = m.scorer(g).squeeze(-1).unsqueeze(0) + (marker_mask - 1.0) * 1e4
        if cls_sel is None:
            return logits
        return logits, torch.matmul(cls_sel, xh.reshape(E, Sh).t())                                     # [Kt,E]


def load():
    torch.backends.mha.set_fastpath_enabled(False)
    agent = ex.load_agent(str(CKPT), device="cpu")
    agent.model.encoder.config._attn_implementation = "eager"
    agent.model.eval().float()
    return agent


def ref_logits(agent, case):
    import packed
    batch = packed.collate_cases([case], agent.tok.pad_token_id, torch.device("cpu"))
    with torch.inference_mode():
        lg, _ = packed.forward(agent.model, batch)
    return [lg[j, : len(case["markers"][j])].numpy() for j in range(len(case["markers"]))]


def cmd_prep(a):
    import packed
    from data import load_split
    from laya.common import QTYPES
    agent = load(); tok = agent.tok
    items = packed.case_items(load_split(ex.CALDEC / "data", "test"), tok, dict(max_len=a.max_len, head_max_len=192))
    full = {it["case_id"]: it for it in packed.case_items(load_split(ex.CALDEC / "data", "test"), tok, dict(max_len=1024, head_max_len=192))}
    out = []
    for it in items[: a.n]:
        c = dict(ids=it["ids"], views=it["views"], markers=it["markers"], qtype=it["qtype"], target=it["target"],
                 case_id=it["case_id"], site=it["site"])
        c["ref"] = ref_logits(agent, c)                       # fp32 at this max_len (the NPU's input)
        f = full[it["case_id"]]
        c["ref1024"] = ref_logits(agent, dict(ids=f["ids"], views=f["views"], markers=f["markers"], qtype=f["qtype"]))
        c["len1024"] = len(f["ids"])
        out.append(c)
    pickle.dump(dict(cases=out, max_len=a.max_len, pad=tok.pad_token_id, ntypes=agent.model.type_emb.num_embeddings),
                open(HERE / "cases.pkl", "wb"))
    nq = sum(len(c["markers"]) for c in out)
    cut = sum(c["len1024"] > a.max_len for c in out)
    print(f"cases {len(out)}, questions {nq}, cut at {a.max_len}: {cut}")


def inputs_for(c, S, pad, ntypes):
    """Device inputs of one packed case in bucket S (None if it does not fit)."""
    n = len(c["ids"]); Sh = sh_for(S)
    rows = [(j, p) for j, v in enumerate(c["views"]) for p in v]
    nm = sum(len(mk) for mk in c["markers"])
    if n > S or len(rows) > Sh or nm > KT:
        return None
    ids = np.full((1, S), pad, np.int64); ids[0, :n] = c["ids"]
    valid = np.zeros(S, bool); valid[:n] = True
    i = np.arange(S)
    glo = np.broadcast_to(np.where(valid[None, :], 0.0, -1e4), (S, S)).astype(np.float32).copy()
    loc = np.where(np.abs(i[:, None] - i[None, :]) <= W, glo, -1e4).astype(np.float32)
    vsel = np.zeros((S, Sh), np.float32); rtype = np.zeros((ntypes, Sh), np.float32)
    hm = np.full((Sh, Sh), -1e4, np.float32); msel = np.zeros((KT, Sh), np.float32); mmask = np.zeros((1, KT), np.float32)
    csel = np.zeros((KT, Sh), np.float32)
    r0, k = 0, 0
    for j, v in enumerate(c["views"]):
        L = len(v); csel[j, r0] = 1.0
        for t, p in enumerate(v):
            vsel[p, r0 + t] = 1.0; rtype[c["qtype"][j], r0 + t] = 1.0
        hm[r0:r0 + L, r0:r0 + L] = 0.0
        for off in c["markers"][j]:
            msel[k, r0 + off] = 1.0; mmask[0, k] = 1.0; k += 1
        r0 += L
    for r in range(r0, Sh):           # padding rows: copy token 0 with type 0 (no all-zero row reaching LayerNorm in
        vsel[0, r] = 1.0; rtype[0, r] = 1.0   # fp16) and attend only to themselves (no all-masked row)
        hm[r, r] = 0.0
    d = dict(input_ids=ids, local_mask=loc, global_mask=glo, view_sel=vsel, row_type=rtype, head_mask=hm,
             marker_sel=msel, marker_mask=mmask)
    if ACT:
        d["cls_sel"] = csel
    return d


NAMES = ["input_ids", "local_mask", "global_mask", "view_sel", "row_type", "head_mask", "marker_sel", "marker_mask"] + (["cls_sel"] if ACT else [])


def cmd_export(a):
    agent = load(); S = a.seq; Sh = sh_for(S)
    D = pickle.load(open(HERE / "cases.pkl", "rb"))
    net = GridPacked(agent.model, S).eval()
    tab = net.rope_table().numpy().astype(np.float32); sc = np.array([float(S)], np.float32); hs = np.array([float(Sh)], np.float32)
    extra = tuple(torch.from_numpy(v) for v in (tab, sc, hs))
    worst, agree, nq, first = 0.0, 0, 0, None
    with torch.inference_mode():
        for c in D["cases"]:
            inp = inputs_for(c, S, D["pad"], D["ntypes"])
            if inp is None:
                continue
            if first is None:
                first = inp
            y = net(*[torch.from_numpy(inp[k]) for k in NAMES], *extra)
            if ACT:
                y, hc = y
                import packed
                b = packed.collate_cases([c], D["pad"], torch.device("cpu"))
                h = agent.model.encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
                hq = h.reshape(-1, h.size(-1))[b["view"]] + agent.model.type_emb(b["qtype"])[:, None, :]
                for layer in agent.model.head.layers:
                    hq = layer(hq, src_key_padding_mask=~b["view_mask"])
                nqc = len(c["views"]); worst_h = max(globals().get("worst_h", 0.0), float((hc[:nqc] - hq[:, 0]).abs().max()))
                globals()["worst_h"] = worst_h
            y = y.numpy()[0]
            k = 0
            for r in c["ref"]:
                worst = max(worst, float(np.abs(y[k:k + len(r)] - r).max())); agree += int(y[k:k + len(r)].argmax() == r.argmax())
                k += len(r); nq += 1
            if nq >= a.check:
                break
    print(f"S={S} Sh={Sh}: packed grid rewrite vs packed.forward fp32: max |diff| {worst:.3g}, argmax {agree}/{nq}"
          + (f"; hcls vs head row 0: max |diff| {globals().get('worst_h', 0.0):.3g}" if ACT else ""))
    od = HERE / "onnx"; od.mkdir(exist_ok=True)
    td = HERE / f"tables_s{S}"; td.mkdir(exist_ok=True)
    tab.tofile(td / "rope_tab.bin"); sc.tofile(td / "s_scale.bin"); hs.tofile(td / "h_scale.bin")
    path = od / (f"pk2_s{S}.onnx" if ACT else f"pk_s{S}.onnx")
    torch.onnx.export(net, tuple(torch.from_numpy(first[k]) for k in NAMES) + extra, str(path), opset_version=17, dynamo=False,
                      input_names=NAMES + ["rope_tab", "s_scale", "h_scale"], output_names=["logits"] + (["hcls"] if ACT else []),
                      do_constant_folding=True)
    print("wrote", path, path.stat().st_size // 2**20, "MiB")


def cmd_cases(a):
    D = pickle.load(open(HERE / "cases.pkl", "rb")); bks = [int(b) for b in a.buckets.split(",")]
    assign = {b: [] for b in bks}; skipped = []
    for ci, c in enumerate(D["cases"]):
        for b in bks:
            inp = inputs_for(c, b, D["pad"], D["ntypes"])
            if inp is not None:
                d = HERE / f"cases_b{b}"; d.mkdir(exist_ok=True); j = len(assign[b])
                for k in NAMES:
                    np.ascontiguousarray(inp[k]).tofile(d / f"{j}_{k}.bin")
                assign[b].append(ci); break
        else:
            skipped.append(ci)
    json.dump({str(b): v for b, v in assign.items()}, open(HERE / "buckets.json", "w"))
    print({b: len(v) for b, v in assign.items()}, "skipped", skipped)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd")
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--seq", type=int, default=256); ap.add_argument("--check", type=int, default=60)
    ap.add_argument("--buckets", default="96,128,160,192,224,256,320,384")
    a = ap.parse_args()
    {"prep": cmd_prep, "export": cmd_export, "cases": cmd_cases}[a.cmd](a)
