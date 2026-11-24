"""Accuracy of NPU activation LUTs: input x in [-20, 0] (plus [0,8] for sigmoid/tanh), compare on device."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from rknn.api import RKNN
kind = sys.argv[1]
class M(nn.Module):
    def forward(s, x):
        if kind == "elu": return F.elu(x) + 1.0
        if kind == "sig2": return torch.sigmoid(x) / torch.sigmoid(-x)          # = e^x
        if kind == "tanh": t = torch.tanh(x * 0.5); return (1 + t) / (1 - t)   # = e^x
        if kind == "elu_half": e = F.elu(x * 0.5) + 1.0; return e * e          # = e^x, LUT used on x/2
        if kind == "elu_q": e = F.elu(x * 0.25) + 1.0; e = e * e; return e * e # = e^x via (e^{x/4})^4
x = torch.linspace(-20, 0, 1024 * 64).view(1, 64, 1, 1024)
torch.onnx.export(M(), (x,), f"{kind}.onnx", opset_version=17, dynamo=False, input_names=["x"], output_names=["y"])
r = RKNN(); r.config(target_platform="rk3566"); r.load_onnx(f"{kind}.onnx"); r.build(do_quantization=False); r.export_rknn(f"{kind}.rknn"); r.release()
x.numpy().tofile("0_x.bin")
