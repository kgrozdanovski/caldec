"""Micro attention block shaped like a CalDec encoder layer (S=256, H=6, D=128, hidden 768, additive mask).
Usage: attn_variants.py <variant> ; prints nothing itself: run via ../ops/targets.sh-like wrapper to list op targets."""
import sys, numpy as np, torch, torch.nn as nn
from rknn.api import RKNN
S, H, D, E = 256, 6, 128, 768
class Attn(nn.Module):
    def __init__(s, kt=False):
        super().__init__(); torch.manual_seed(0)
        s.q = nn.Linear(E, E); s.k = nn.Linear(E, E); s.v = nn.Linear(E, E); s.o = nn.Linear(E, E); s.kt = kt
    def forward(s, x, mask):                      # x [1,S,E], mask [1,1,S,S]
        q = s.q(x).view(1, S, H, D).transpose(1, 2)               # [1,H,S,D]
        k = s.k(x).view(1, S, H, D).transpose(1, 2)
        v = s.v(x).view(1, S, H, D).transpose(1, 2)
        if s.kt:   # K^T produced directly: [1,H,D,S] via permute of the projection output
            kt = s.k(x).view(1, S, H, D).permute(0, 2, 3, 1)
            sc = torch.matmul(q, kt) / D ** 0.5 + mask
        else:
            sc = torch.matmul(q, k.transpose(-1, -2)) / D ** 0.5 + mask
        p = torch.softmax(sc, -1)
        return s.o(torch.matmul(p, v).transpose(1, 2).reshape(1, S, E))
var = sys.argv[1]
m = Attn(kt=("kt" in var)).eval()
x = torch.randn(1, S, E); mask = torch.zeros(1, 1, S, S)
torch.onnx.export(m, (x, mask), f"{var}.onnx", opset_version=17, dynamo=False, input_names=["x", "mask"], output_names=["y"])
cfg = dict(target_platform="rk3566")
if "flash" in var: cfg["enable_flash_attention"] = True
if "force" in var: cfg["op_target"] = {"/Softmax": "npu"}
r = RKNN(verbose=True); r.config(**cfg); r.load_onnx(f"{var}.onnx")
q8 = "int8" in var
if q8:
    np.save("x.npy", x.numpy()); np.save("m.npy", mask.numpy()); open("ds.txt", "w").write("x.npy m.npy\n")
r.build(do_quantization=q8, dataset="ds.txt" if q8 else None); r.export_rknn(f"{var}.rknn"); r.release()
