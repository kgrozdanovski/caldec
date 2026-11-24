import sys, torch, torch.nn as nn, torch.nn.functional as F
from rknn.api import RKNN
H, L, S = 6, 256, 256
class V(nn.Module):
    def __init__(s, kind): super().__init__(); s.kind = kind; s.register_buffer("ones", torch.ones(S, 1))
    def forward(s, x):
        k = s.kind
        if k == "exp_only": return torch.exp(x)
        if k == "maxpool": return F.max_pool2d(x, (1, S))
        if k == "npu_sm":
            m = F.max_pool2d(x, (1, S)); e = torch.exp(x - m); return e * torch.reciprocal(e @ s.ones)
        if k == "npu_sm_div":
            m = F.max_pool2d(x, (1, S)); e = torch.exp(x - m); return e / (e @ s.ones)
        if k == "npu_sm_sum":
            m = F.max_pool2d(x, (1, S)); e = torch.exp(x - m); return e * torch.reciprocal(e.sum(-1, keepdim=True))
k = sys.argv[1]; x = torch.randn(1, H, L, S)
torch.onnx.export(V(k), (x,), f"{k}.onnx", opset_version=17, dynamo=False, input_names=["x"], output_names=["y"])
r = RKNN(verbose=True); r.config(target_platform="rk3566"); r.load_onnx(f"{k}.onnx"); r.build(do_quantization=False); r.export_rknn(f"{k}.rknn"); r.release()
