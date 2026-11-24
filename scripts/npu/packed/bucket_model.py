"""Expected packed NPU time per case for bucket sets, from the measured per-bucket p50 (multi_bench2, 11-22 evening)
and the packed lengths of all 1,647 assistant-test cases (linear interpolation between measured buckets)."""
from pathlib import Path
import sys, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_caldec as ex
from data import load_split
import packed
agent = ex.load_agent(ex.CKPT, device="cpu"); tok = agent.tok
L, Q = [], []
for r in load_split(ex.CALDEC / "data", "test"):
    qs = [agent._to_internal(q) for q in r["questions"].values()]
    ids, views, _ = packed.build_case(tok, r["state"], qs, 384, 192); L.append(len(ids)); Q.append(len(qs))
L, Q = np.array(L), np.array(Q)
meas = {96: 299.3, 128: 406.2, 160: 567.3, 192: 659.4, 224: 752.3, 256: 862.0, 320: 1123.6, 384: 1356.0}
xs = sorted(meas); t = lambda S: float(np.interp(S, xs, [meas[x] for x in xs]))
for name, bks in (("current 96..384 (8)", [96, 128, 160, 192, 224, 256, 320, 384]),
                  ("+64", [64, 96, 128, 160, 192, 224, 256, 320, 384]),
                  ("+288,352", [96, 128, 160, 192, 224, 256, 288, 320, 352, 384]),
                  ("every 32 from 64 (11)", list(range(64, 385, 32))),
                  ("every 16 from 64 (21)", list(range(64, 385, 16)))):
    tot = sum(t(min(b for b in bks if b >= l)) for l in L)
    print(f"{name:28s} mean/case {tot / len(L):6.1f} ms   per question {tot / Q.sum():6.1f} ms")
print("ideal (no padding)            per question %.1f ms" % (sum(t(max(l, 96)) for l in L) / Q.sum()))
