"""Hybrid quantization step 2: int8 only for projection convs (names with '_mm' = MatMul->Conv), all else fp16.
Usage: step2.py <prefix> <out.rknn> [--int8-filter substr,...] [--fp16-extra substr,...]"""
import sys, re, onnx
from rknn.api import RKNN
pre, dst = sys.argv[1], sys.argv[2]; args = sys.argv[3:]
keep = [s for s in (args[args.index("--int8-filter") + 1].split(",") if "--int8-filter" in args else ["_mm"])]
extra = args[args.index("--fp16-extra") + 1].split(",") if "--fp16-extra" in args else []
m = onnx.load(f"{pre}.model", load_external_data=False)
USE_TENSORS = "--node-names" not in args
qp = open(f"{pre}.quantization.cfg.orig" if __import__("os").path.exists(f"{pre}.quantization.cfg.orig") else f"{pre}.quantization.cfg").read()
fp16 = {}; n8 = 0
for n in m.graph.node:
    is8 = n.op_type == "Conv" and any(k in n.name for k in keep) and not any(e and e in n.name for e in extra)
    if is8: n8 += 1; continue
    if USE_TENSORS:
        for o in n.output:
            if f"\n    {o}:" in qp: fp16[o] = "float16"
    else: fp16[n.name] = "float16"
cfgp = f"{pre}.quantization.cfg"; orig = open(cfgp).read()
if not orig.startswith("custom_quantize_layers: {}"):
    orig = open(cfgp + ".orig").read()
else:
    open(cfgp + ".orig", "w").write(orig)
body = orig.split("\n", 1)[1]
for attempt in range(400):
    open(cfgp, "w").write("custom_quantize_layers:\n" + "".join(f"    {k!r}: float16\n" for k in fp16) + body)
    r = RKNN(verbose=True)
    try:
        ok = r.hybrid_quantization_step2(model_input=f"{pre}.model", data_input=f"{pre}.data", model_quantization_cfg=cfgp)
    except Exception as e:
        ok = -1; msg = str(e)
    else:
        msg = ""
    if ok == 0:
        break
    mm = re.search(r"Invalid operands name '([^']+)'", msg or open("/dev/null").read())
    if not mm:
        import traceback; print("step2 failed:", msg[:500]); sys.exit(1)
    fp16.pop(mm.group(1), None); r.release()
print(f"int8 convs: {n8}, fp16 layers set: {len(fp16)}")
assert r.export_rknn(dst) == 0; r.release(); print("wrote", dst)
