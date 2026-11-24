import sys, torch, torch.nn as nn
from rknn.api import RKNN
kind = sys.argv[1]; S, E = 256, 768; torch.manual_seed(0)
class M(nn.Module):
    def __init__(s):
        super().__init__(); s.c = nn.ModuleList([nn.Conv2d(E, E, 1) for _ in range(8)])
    def forward(s, x):
        if kind == "lin":      # [S,E] matmuls (what the exporter emits)
            for c in s.c: x = torch.nn.functional.linear(x, c.weight[:, :, 0, 0], c.bias)
            return x
        for c in s.c: x = c(x)
        return x
shape = {"lin": (S, E), "c1x256": (1, E, 1, S), "c16x16": (1, E, 16, 16), "c256x1": (1, E, S, 1)}[kind]
x = torch.randn(*shape)
torch.onnx.export(M().eval(), (x,), f"{kind}.onnx", opset_version=17, dynamo=False, input_names=["x"], output_names=["y"])
r = RKNN(); r.config(target_platform="rk3566"); r.load_onnx(f"{kind}.onnx"); r.build(do_quantization=False); r.export_rknn(f"{kind}.rknn"); r.release()
import os; os.makedirs(f"c_{kind}", exist_ok=True); x.numpy().tofile(f"c_{kind}/0_x.bin")
