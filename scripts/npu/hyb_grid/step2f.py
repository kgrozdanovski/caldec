"""Hybrid step 2 (fixed): int8 ONLY for the outputs of the selected Conv nodes; EVERY other tensor in the quantization
cfg - including graph inputs (masks, marker_sel) and constants (RoPE tables, norm scales, embedding) - stays fp16.
(step2s.py marked only node outputs fp16, so ~190 inputs/constants were silently int8.)
Usage: step2f.py <prefix> <out.rknn> [mlp_enc|mlp|all_lin|qkv_up]"""
import sys, re, os, onnx
from rknn.api import RKNN
pre, dst = sys.argv[1], sys.argv[2]; mode = sys.argv[3] if len(sys.argv) > 3 else "mlp_enc"
m = onnx.load(f"{pre}.model", load_external_data=False)
shapes = {i.name: tuple(i.dims) for i in m.graph.initializer}
want = {"mlp_enc": {(2048, 768), (1024, 768), (768, 1024)},
        "mlp": {(2048, 768), (1024, 768), (768, 1024), (3072, 768), (768, 3072)},
        "gate_up": {(2048, 768), (1024, 768)},
        "all_lin": {(2048, 768), (1024, 768), (768, 1024), (3072, 768), (768, 3072), (2304, 768), (768, 768)}}[mode]
cfgp = f"{pre}.quantization.cfg"
if not os.path.exists(cfgp + ".orig"): open(cfgp + ".orig", "w").write(open(cfgp).read())
qp = open(cfgp + ".orig").read(); body = qp.split("\n", 1)[1]
tens = re.findall(r"^    (\S.*?):\s*$", body, re.M)
int8 = set()
for n in m.graph.node:
    ws = [shapes.get(x) for x in n.input[1:2]]
    if n.op_type == "Conv" and ws and ws[0] is not None and tuple(ws[0][:2]) in want:
        int8.update(n.output)
fp16 = [t for t in tens if t not in int8]
print(f"mode {mode}: int8 tensors {len(int8 & set(tens))}, fp16 tensors {len(fp16)} of {len(tens)}")
open(cfgp, "w").write("custom_quantize_layers:\n" + "".join(f"    {k!r}: float16\n" for k in fp16) + body)
r = RKNN()
assert r.hybrid_quantization_step2(model_input=f"{pre}.model", data_input=f"{pre}.data", model_quantization_cfg=cfgp) == 0
assert r.export_rknn(dst) == 0; r.release(); print("wrote", dst)
