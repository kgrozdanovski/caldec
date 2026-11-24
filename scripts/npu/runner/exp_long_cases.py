"""PC, fp32: where does the long-case accuracy gap of the NPU runner come from? Public-test cases (5 questions,
~600-token states). Accuracy (argmax vs untied gold) of PackedAgent-style passes under:
  one pass at max_len L (no head limit), and the runner's split (groups with head rows <= 192, markers <= 12)
  at max_len L, for L in 384, 512, 640, 1024."""
from pathlib import Path
import json, sys
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import export_caldec as ex, packed
from benchmark import load_public
from laya.common import QTYPES
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
agent = ex.load_agent(ex.CKPT, device="cpu"); agent.model.float().eval()
tok, hml = agent.tok, 192

def run(state, qs, L):
    ids, views, marks = packed.build_case(tok, state, qs, L, hml)
    case = dict(ids=ids, views=views, markers=marks, qtype=[QTYPES[q["t"]] for q in qs])
    b = packed.collate_cases([case], tok.pad_token_id, torch.device("cpu"))
    with torch.inference_mode():
        lg, _ = packed.forward(agent.model, b)
    return [lg[j, : len(marks[j])].numpy() for j in range(len(qs))], views, marks

def groups(views, marks):
    gs, cur, rows, mk = [], [], 0, 0
    for j, (v, m) in enumerate(zip(views, marks)):
        r, k = len(v), len(m)
        if cur and (rows + r > 192 or mk + k > 12):
            gs.append(cur); cur, rows, mk = [], 0, 0
        cur.append(j); rows += r; mk += k
    return gs + [cur]

rows = load_public("test")[:N]
cfgs = [("one", L) for L in (384, 512, 640, 1024)] + [("split", L) for L in (384, 512, 640, 1024)]
hit = {c: 0 for c in cfgs}; n = 0; npass = []
for i, r in enumerate(rows):
    qids = list(r["questions"]); qs = [agent._to_internal(r["questions"][k]) for k in qids]
    gold = [(r.get("gold") or {}).get(k, {}).get("probabilities", {}) for k in qids]
    res = {}
    for mode, L in cfgs:
        if mode == "one":
            lg, views, marks = run(r["state"], qs, L)
        else:
            _, views, marks = packed.build_case(tok, r["state"], qs, L, hml)
            lg = [None] * len(qs); gs = groups(views, marks)
            if L == 384: npass.append(len(gs))
            for g in gs:
                sub, _, _ = run(r["state"], [qs[j] for j in g], L)
                for j, z in zip(g, sub): lg[j] = z
        res[(mode, L)] = lg
    for j, (k, q) in enumerate(zip(qids, qs)):
        keys = (["false", "true"] if q["t"] == "noul" else list(q["crit"]) if q["t"] == "choice" else [str(x) for x in range(len(q["crit"]))])
        gv = np.array([float(gold[j].get(kk, 0.0)) for kk in keys])
        if gv.max() <= 0 or (gv == gv.max()).sum() > 1: continue
        n += 1
        for c in cfgs: hit[c] += int(res[c][j].argmax() == gv.argmax())
    if (i + 1) % 50 == 0: print("cases", i + 1, flush=True)
print(f"cases {len(rows)}, untied questions {n}, runner passes at 384: mean {np.mean(npass):.2f}")
for c in cfgs: print(f"{c[0]:5s} max_len {c[1]:4d}: accuracy {hit[c] / n:.4f}")
