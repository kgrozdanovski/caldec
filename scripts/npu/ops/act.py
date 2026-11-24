import sys, torch, torch.nn as nn, torch.nn.functional as F
from rknn.api import RKNN
H, L, S = 6, 256, 256
class V(nn.Module):
    def __init__(s, kind): super().__init__(); s.kind = kind
    def forward(s, x):
        k = s.kind
        if k == "sigmoid": return torch.sigmoid(x)
        if k == "elu": return F.elu(x)
        if k == "tanh": return torch.tanh(x)
        if k == "softplus": return F.softplus(x)
        if k == "silu": return F.silu(x)
        if k == "exp_elu":            # exp(z) for z<=0 as elu(z)+1, with row max from MaxPool
            m = F.max_pool2d(x, (1, S)); e = F.elu(x - m) + 1.0; return e / e.sum(-1, keepdim=True)
        if k == "exp_sig":
            m = F.max_pool2d(x, (1, S)); sg = torch.sigmoid(x - m); e = sg / (1.0 - sg); return e / e.sum(-1, keepdim=True)
k = sys.argv[1]; x = torch.randn(1, H, L, S)
torch.onnx.export(V(k), (x,), f"{k}.onnx", opset_version=17, dynamo=False, input_names=["x"], output_names=["y"])
r = RKNN(verbose=True); r.config(target_platform="rk3566"); r.load_onnx(f"{k}.onnx"); r.build(do_quantization=False); r.export_rknn(f"{k}.rknn"); r.release()
