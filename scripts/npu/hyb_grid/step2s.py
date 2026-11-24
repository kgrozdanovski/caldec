"""Hybrid step 2 for the grid model: int8 for Conv nodes whose weight shape matches the MLP (gate|up 2048x768,
down 768x1024, head linear1 3072x768, head linear2 768x3072); everything else fp16 (tensor names as keys).
Usage: step2s.py <prefix> <out.rknn> [mlp|mlp_enc|mlp_head|all_lin]"""
import sys, re, onnx, os
from onnx import numpy_helper
from rknn.api import RKNN
pre, dst = sys.argv[1], sys.argv[2]; mode = sys.argv[3] if len(sys.argv) > 3 else "mlp"
m = onnx.load(f"{pre}.model", load_external_data=False)
shapes = {i.name: tuple(i.dims) for i in m.graph.initializer}
want = {"mlp": {(2048, 768), (1024, 768), (768, 1024), (3072, 768), (768, 3072)},
        "mlp_enc": {(2048, 768), (1024, 768), (768, 1024)}, "mlp_head": {(3072, 768), (768, 3072)},
        "all_lin": {(2048, 768), (1024, 768), (768, 1024), (3072, 768), (768, 3072), (2304, 768), (768, 768)},
        "gate_up": {(2048, 768), (1024, 768)}, "down": {(768, 1024)}}[mode]
cfgp = f"{pre}.quantization.cfg"
if not os.path.exists(cfgp + ".orig"): open(cfgp + ".orig", "w").write(open(cfgp).read())
qp = open(cfgp + ".orig").read(); body = qp.split("\n", 1)[1]
fp16 = {}; n8 = 0; seen = set()
for n in m.graph.node:
    ws = [shapes.get(x) for x in n.input[1:2]]
    is8 = n.op_type == "Conv" and ws and ws[0] is not None and tuple(ws[0][:2]) in want
    if is8: n8 += 1; seen.add(tuple(ws[0][:2])); continue
    for o in n.output:
        if f"\n    {o}:" in qp: fp16[o] = "float16"
print(f"mode {mode}: int8 convs {n8} {sorted(seen)}, fp16 tensors {len(fp16)}")
open(cfgp, "w").write("custom_quantize_layers:\n" + "".join(f"    {k!r}: float16\n" for k in fp16) + body)
r = RKNN()
assert r.hybrid_quantization_step2(model_input=f"{pre}.model", data_input=f"{pre}.data", model_quantization_cfg=cfgp) == 0
assert r.export_rknn(dst) == 0; r.release(); print("wrote", dst)
