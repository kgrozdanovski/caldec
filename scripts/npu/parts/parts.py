"""Component timing models (S=256, H=6, D=128, E=768). Usage: parts.py <name>"""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from rknn.api import RKNN
S, H, D, E = 256, 6, 128, 768; torch.manual_seed(0)
name = sys.argv[1]
class P(nn.Module):
    def __init__(s):
        super().__init__(); s.l = nn.ModuleList([nn.Linear(E, E) for _ in range(8)]); s.u = nn.Linear(E, 1024); s.dn = nn.Linear(1024, E)
    def forward(s, *a):
        if name == "lin4":     x = a[0][0]; return s.l[3](s.l[2](s.l[1](s.l[0](x))))          # 4 chained 768x768 linears on [S,E]
        if name == "lin4c":    return torch.stack([0])  # placeholder
        if name == "qk":       return torch.bmm(a[0], a[1].transpose(1, 2))                     # [H,S,D]x[H,D,S]
        if name == "pv":       return torch.bmm(a[0], a[1])                                     # [H,S,S]x[H,S,D]
        if name == "ew8":      # 8 elementwise ops on the score tensor
            z = a[0]
            for i in range(8): z = z * 1.001 + 0.001
            return z
        if name == "mlp":      x = a[0][0]; return s.dn(F.silu(s.u(x)) * s.u(x))
        if name == "rms":      x = a[0][0]; return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
ins = {"lin4": [(1, S, E)], "qk": [(H, S, D), (H, S, D)], "pv": [(H, S, S), (H, S, D)], "ew8": [(H, S, S)], "mlp": [(1, S, E)], "rms": [(1, S, E)]}[name]
xs = tuple(torch.randn(*sh) for sh in ins)
torch.onnx.export(P().eval(), xs, f"{name}.onnx", opset_version=17, dynamo=False, input_names=[f"i{k}" for k in range(len(xs))], output_names=["y"])
r = RKNN(); r.config(target_platform="rk3566"); r.load_onnx(f"{name}.onnx"); r.build(do_quantization=False); r.export_rknn(f"{name}.rknn"); r.release()
for k, x in enumerate(xs): x.numpy().tofile(f"0_i{k}.bin") if name == "lin4" else x.numpy().tofile(f"{name}_0_i{k}.bin")
