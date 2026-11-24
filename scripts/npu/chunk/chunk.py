"""Is the conv-cost cliff above 128 tokens avoidable by splitting the token axis (exact for 1x1 convs)?
8 chained 768x768 1x1 convs on a token grid [1,E,S/16,16]; mode plain | c<N> (split into N-token chunks, conv each,
concat) | m<N> (same but split/cat once around the whole chain). Usage: chunk.py S mode"""
import sys, os, torch, torch.nn as nn, torch.nn.functional as F
from rknn.api import RKNN
S, mode = int(sys.argv[1]), sys.argv[2]; E = 768; torch.manual_seed(0)
class M(nn.Module):
    def __init__(s): super().__init__(); s.c = nn.ModuleList([nn.Conv2d(E, E, 1) for _ in range(8)])
    def forward(s, x):
        if mode == "plain":
            for c in s.c: x = c(x)
            return x
        n = int(mode[1:]) // 16                       # rows per chunk
        if mode[0] == "c":                            # split/cat around every conv
            for c in s.c: x = torch.cat([c(p) for p in x.split(n, 2)], 2)
            return x
        ps = list(x.split(n, 2))                      # split once, chain per chunk, cat once
        for c in s.c: ps = [c(p) for p in ps]
        return torch.cat(ps, 2)
x = torch.randn(1, E, S // 16, 16); n = f"k{S}_{mode}"
m = M().eval()
with torch.no_grad():
    ref = M.forward.__wrapped__ if hasattr(M.forward, "__wrapped__") else None
torch.onnx.export(m, (x,), f"{n}.onnx", opset_version=17, dynamo=False, input_names=["x"], output_names=["y"])
r = RKNN(verbose=True); r.config(target_platform="rk3566"); r.load_onnx(f"{n}.onnx"); r.build(do_quantization=False); r.export_rknn(f"{n}.rknn"); r.release()
os.makedirs(f"c_{n}", exist_ok=True); x.numpy().tofile(f"c_{n}/0_x.bin"); os.remove(f"{n}.onnx")
