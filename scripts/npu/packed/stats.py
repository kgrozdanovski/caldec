"""Packed-case shape stats on the assistant test: encoder length, head rows (sum of views), total markers, questions."""
from pathlib import Path
import sys, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_caldec as ex
from data import load_split
import packed
agent = ex.load_agent(ex.CKPT, device="cpu")
print(type(agent).__name__)
tok = agent.tok
L, R, M, Q = [], [], [], []
for r in load_split(ex.CALDEC / "data", "test"):
    qs = [agent._to_internal(q) for q in r["questions"].values()]
    ids, views, markers = packed.build_case(tok, r["state"], qs, 100000, 192)
    L.append(len(ids)); R.append(sum(len(v) for v in views)); M.append(sum(len(m) for m in markers)); Q.append(len(qs))
L, R, M, Q = map(np.array, (L, R, M, Q))
print("cases", len(L), "questions", Q.sum())
for nm, a in (("enc len", L), ("head rows", R), ("markers", M), ("questions", Q)):
    print(nm, "p50/p90/p99/max", *[int(np.percentile(a, p)) for p in (50, 90, 99)], a.max())
for s in (64, 96, 128, 160, 192): print("head rows <=", s, f"{np.mean(R <= s):.1%}")
for k in (8, 12, 16, 24): print("markers <=", k, f"{np.mean(M <= k):.1%}")
for s in (128, 160, 192, 256, 320, 384, 512): print("enc <=", s, f"{np.mean(L <= s):.1%}")
