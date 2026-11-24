"""Real CalDec conv shapes on a token grid: 4 x (Cin->Cout, Cout->Cin) 1x1 conv pairs; plain vs split into token
halves (exact). Usage: chunk2.py S cout mode(plain|half)"""
import sys, os, torch, torch.nn as nn
from rknn.api import RKNN
S, Co, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; E = 768; torch.manual_seed(0)
class M(nn.Module):
    def __init__(s):
        super().__init__(); s.c = nn.ModuleList([nn.Conv2d(E, Co, 1) if i % 2 == 0 else nn.Conv2d(Co, E, 1) for i in range(8)])
    def forward(s, x):
        for c in s.c:
            x = c(x) if mode == "plain" else torch.cat([c(p) for p in x.split(x.shape[2] // 2, 2)], 2)
        return x
x = torch.randn(1, E, S // 16, 16); n = f"w{S}_{Co}_{mode}"
torch.onnx.export(M().eval(), (x,), f"{n}.onnx", opset_version=17, dynamo=False, input_names=["x"], output_names=["y"])
r = RKNN(); r.config(target_platform="rk3566"); r.load_onnx(f"{n}.onnx"); r.build(do_quantization=False); r.export_rknn(f"{n}.rknn"); r.release()
os.makedirs(f"c_{n}", exist_ok=True); x.numpy().tofile(f"c_{n}/0_x.bin"); os.remove(f"{n}.onnx")
