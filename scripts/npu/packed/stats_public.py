"""Packed shape statistics on the public test (LocalLLaMA/typed-decisions, caldec scripts/benchmark.load_public)."""
from pathlib import Path
import sys, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_caldec as ex
from benchmark import load_public
import packed
agent = ex.load_agent(ex.CKPT, device="cpu"); tok = agent.tok
rows = load_public("test"); print("rows", len(rows))
L, R, M, Q = [], [], [], []
for r in rows:
    qs = [agent._to_internal(q) for q in r["questions"].values()]
    ids, views, markers = packed.build_case(tok, r["state"], qs, 100000, 192)
    L.append(len(ids)); R.append(sum(len(v) for v in views)); M.append(sum(len(m) for m in markers)); Q.append(len(qs))
L, R, M, Q = map(np.array, (L, R, M, Q))
for nm, a in (("enc len", L), ("head rows", R), ("markers", M), ("questions", Q)):
    print(nm, "p50/p90/p99/max", *[int(np.percentile(a, p)) for p in (50, 90, 99)], a.max())
print(f"fit all limits (enc<=384, rows<=192, markers<=12): {np.mean((L <= 384) & (R <= 192) & (M <= 12)):.1%}; "
      f"rows<=192 {np.mean(R <= 192):.1%}; markers<=12 {np.mean(M <= 12):.1%}; enc<=384 {np.mean(L <= 384):.1%}")
