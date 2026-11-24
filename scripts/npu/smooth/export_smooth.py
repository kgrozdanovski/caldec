"""export_grid.py with exact SmoothQuant-style scale migration in every encoder MLP (no change in fp32 math):
  down_proj input a = act(gate) * up:  up row c /= s_c, down column c *= s_c
  gate/up input x = RMSNorm(h) * w:     w_c /= t_c,  gate/up column c *= t_c
s = amax_act^alpha / amax_w^(1-alpha) from smooth/stats.npz. Env SMOOTH_ALPHA (default 0.5), SMOOTH_PARTS (down,in).
Usage: same arguments as export_grid.py."""
from pathlib import Path
import os, sys, numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_grid as eg
A = float(os.environ.get("SMOOTH_ALPHA", "0.5")); parts = os.environ.get("SMOOTH_PARTS", "down,in").split(",")
st = np.load(str(Path(__file__).resolve().parents[1]) + "/smooth/stats.npz")
orig = eg.ex.load_agent
def scale(act, wmax):
    s = np.power(np.maximum(act, 1e-5), A) / np.power(np.maximum(wmax, 1e-5), 1 - A)
    return torch.from_numpy(np.clip(s, 1e-2, 1e4).astype(np.float32))
def load_agent(*a, **k):
    ag = orig(*a, **k); m = ag.model.float()
    with torch.no_grad():
        for i, layer in enumerate(m.encoder.layers):
            mlp, ln = layer.mlp, layer.post_attention_layernorm
            if "down" in parts:
                s = scale(st["din"][i], mlp.down_proj.weight.abs().amax(0).numpy())
                mlp.up_proj.weight.div_(s[:, None])
                if mlp.up_proj.bias is not None: mlp.up_proj.bias.div_(s)
                mlp.down_proj.weight.mul_(s[None, :])
            if "in" in parts:
                w = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0)
                t = scale(st["xin"][i], w.abs().amax(0).numpy())
                ln.weight.div_(t); mlp.gate_proj.weight.mul_(t[None, :]); mlp.up_proj.weight.mul_(t[None, :])
    print(f"smoothed MLPs: alpha {A}, parts {parts}")
    return ag
eg.ex.load_agent = load_agent
eg.main()
