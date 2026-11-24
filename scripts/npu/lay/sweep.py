import sys, os, torch, torch.nn as nn
from rknn.api import RKNN
h, w = int(sys.argv[1]), int(sys.argv[2]); E = 768; torch.manual_seed(0)
class M(nn.Module):
    def __init__(s): super().__init__(); s.c = nn.ModuleList([nn.Conv2d(E, E, 1) for _ in range(8)])
    def forward(s, x):
        for c in s.c: x = c(x)
        return x
x = torch.randn(1, E, h, w); n = f"g{h}x{w}"
torch.onnx.export(M().eval(), (x,), f"{n}.onnx", opset_version=17, dynamo=False, input_names=["x"], output_names=["y"])
r = RKNN(); r.config(target_platform="rk3566"); r.load_onnx(f"{n}.onnx"); r.build(do_quantization=False); r.export_rknn(f"{n}.rknn"); r.release()
os.makedirs(f"c_{n}", exist_ok=True); x.numpy().tofile(f"c_{n}/0_x.bin"); os.remove(f"{n}.onnx")
