"""Convert any fixed-shape ONNX to RKNN for rk3566 (fp16 or int8), input shapes taken from the graph.
Usage: rknn-venv/bin/python convert_generic.py model.onnx out.rknn [--int8 dataset.txt] [--opt 3] [extra config k=v ...]
RKNN_VERBOSE=1 prints the op table (targets)."""
import os, sys, time, ast
import onnx
from rknn.api import RKNN
src, dst = sys.argv[1], sys.argv[2]; rest = sys.argv[3:]
ds = None; opt = 3; extra = {}
while rest:
    a = rest.pop(0)
    if a == "--int8": ds = rest.pop(0)
    elif a == "--opt": opt = int(rest.pop(0))
    elif "=" in a: k, v = a.split("=", 1); extra[k] = ast.literal_eval(v)
m = onnx.load(src, load_external_data=False)
init = {i.name for i in m.graph.initializer}
ins = [i for i in m.graph.input if i.name not in init]
names = [i.name for i in ins]; sizes = [[d.dim_value for d in i.type.tensor_type.shape.dim] for i in ins]
print("inputs:", list(zip(names, sizes)))
r = RKNN(verbose=bool(os.environ.get("RKNN_VERBOSE")))
r.config(target_platform="rk3566", optimization_level=opt, **extra)
assert r.load_onnx(model=src, inputs=names, input_size_list=sizes) == 0
t = time.time(); assert r.build(do_quantization=ds is not None, dataset=ds) == 0
print(f"build {time.time()-t:.0f}s"); assert r.export_rknn(dst) == 0; print("wrote", dst); r.release()
