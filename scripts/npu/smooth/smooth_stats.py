"""Per-channel activation abs-max at the MLP inputs of every encoder layer (gate/up input after the RMSNorm, and
down_proj input), over calibration cases, from the original fp32 model. Writes smooth/stats.npz.
Usage (caldec venv): PYTHONPATH=pylib python smooth/smooth_stats.py out/cases_calib100_s256.npz [n]"""
from pathlib import Path
import sys, os, numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_caldec as ex
npz = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
torch.backends.mha.set_fastpath_enabled(False)
agent = ex.load_agent(ex.CKPT, device="cpu")
agent.model.encoder.config._attn_implementation = "eager"
model = agent.model.eval().float(); core = ex.Core(model).eval()
L = model.encoder.layers; nl = len(L)
xin = [None] * nl; din = [None] * nl
def upd(store, i, t):
    a = t.detach().abs().reshape(-1, t.shape[-1]).amax(0).numpy()
    store[i] = a if store[i] is None else np.maximum(store[i], a)
hooks = []
for i, layer in enumerate(L):
    hooks.append(layer.mlp.register_forward_pre_hook(lambda m, a, i=i: upd(xin, i, a[0])))
    hooks.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, i=i: upd(din, i, a[0])))
c = np.load(npz)
with torch.inference_mode():
    for k in range(min(n, c["input_ids"].shape[0])):
        S = int(c["attention_mask"][k].sum())   # real tokens only: crop the padding so it does not enter the stats
        args = [torch.from_numpy(c[x][k]) for x in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
        args[0] = args[0][:, :S]; args[1] = args[1][:, :S]
        core(*args)
xin, din = np.stack(xin), np.stack(din)
np.savez(str(Path(__file__).resolve().parents[1]) + "/smooth/stats.npz", xin=xin, din=din)
r = lambda a: (a.max(1) / np.median(a, 1))
print("gate/up input max/median per layer:", np.round(r(xin), 1).tolist())
print("down input  max/median per layer:", np.round(r(din), 1).tolist())
