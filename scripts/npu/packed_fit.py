"""How many assistant-test cases fit one packed sequence of 256/384/512 tokens without cutting the state?"""
from pathlib import Path
import sys, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import export_caldec as ex
from data import load_split
from laya.common import build_sequence
import packed
agent = ex.load_agent(ex.CKPT, device="cpu"); tok = agent.tok
full, single, nq = [], [], []
for r in load_split(ex.CALDEC / "data", "test"):
    qs = [agent._to_internal(q) for q in r["questions"].values()]
    seq, _ = build_sequence(tok, r["state"], qs[0], 100000, 192)          # one-question length, no cut
    segs = [packed.question_segment(tok, q, 192)[0] for q in qs]
    state_len = len(seq) - len(segs[0])                                   # approx: total minus the question part
    full.append(state_len + sum(len(s) for s in segs)); single.append(len(seq)); nq.append(len(qs))
full, single, nq = map(np.array, (full, single, nq))
print(f"cases {len(full)}, questions {nq.sum()}, mean q/case {nq.mean():.2f}")
for L in (256, 384, 512):
    print(f"<= {L}: packed cases fit {np.mean(full <= L):.1%} (questions in them {nq[full <= L].sum() / nq.sum():.1%}); "
          f"single-question sequences fit {np.mean(single <= L):.1%}")
print("packed length p50/p90:", int(np.median(full)), int(np.percentile(full, 90)), " single p50/p90:", int(np.median(single)), int(np.percentile(single, 90)))
