"""Which softmax formulations does the RK3566 NPU run (vs CPU fallback)? Builds tiny graphs, prints op targets."""
import os, re, sys, io, contextlib, numpy as np, torch, torch.nn as nn
from rknn.api import RKNN
H, L, S = 6, 256, 256
class V(nn.Module):
    def __init__(s, kind): super().__init__(); s.kind = kind
    def forward(s, x):                       # x: [1,H,L,S] attention scores (+mask already added)
        k = s.kind
        if k == "softmax_last": return torch.softmax(x, -1)
        if k == "softmax_dim1":              # put keys on the channel axis: [1,S,H,L], softmax over dim 1
            return torch.softmax(x.permute(0, 3, 1, 2), 1)
        if k == "manual_last":
            m = x.amax(-1, keepdim=True); e = torch.exp(x - m); return e / e.sum(-1, keepdim=True)
        if k == "manual_nomax":
            e = torch.exp(x); return e / e.sum(-1, keepdim=True)
        if k == "manual_dim1":
            y = x.permute(0, 3, 1, 2); m = y.amax(1, keepdim=True); e = torch.exp(y - m); return e / e.sum(1, keepdim=True)
        if k == "softmax_3d": return torch.softmax(x.reshape(H, L, S), -1)
        if k == "softmax_2d": return torch.softmax(x.reshape(H * L, S), -1)
        if k == "logsoftmax_exp": return torch.exp(torch.log_softmax(x, -1))
for kind in sys.argv[1:]:
    x = torch.randn(1, H, L, S)
    torch.onnx.export(V(kind), (x,), f"{kind}.onnx", opset_version=17, dynamo=False, input_names=["x"], output_names=["y"])
    r = RKNN(verbose=True); r.config(target_platform="rk3566")
    r.load_onnx(f"{kind}.onnx"); r.build(do_quantization=False); r.export_rknn(f"{kind}.rknn"); r.release()
