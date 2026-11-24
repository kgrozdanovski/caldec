"""PC: the act head of the packed checkpoint as numpy arrays (runner/act_head.npz), for the runner's CPU part.
act_head = Linear(d+4, 256) -> GELU (erf) -> Linear(256, n_act); input [head row 0 of the question, 4 features]."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import export_caldec as ex
agent = ex.load_agent(ex.CKPT, device="cpu")
h = agent.model.act_head
assert len(h) == 3 and h[1].__class__.__name__ == "GELU" and h[1].approximate == "none", h
np.savez("runner/act_head.npz", w1=h[0].weight.detach().float().numpy(), b1=h[0].bias.detach().float().numpy(),
         w2=h[2].weight.detach().float().numpy(), b2=h[2].bias.detach().float().numpy())
print({k: v.shape for k, v in np.load("runner/act_head.npz").items()})
