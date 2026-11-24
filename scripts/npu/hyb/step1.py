import sys, onnx
from rknn.api import RKNN
src, ds = sys.argv[1], sys.argv[2]
m = onnx.load(src, load_external_data=False); init = {i.name for i in m.graph.initializer}
ins = [i for i in m.graph.input if i.name not in init]
names = [i.name for i in ins]; sizes = [[d.dim_value for d in i.type.tensor_type.shape.dim] for i in ins]
r = RKNN(); r.config(target_platform="rk3566", quantized_algorithm=sys.argv[3] if len(sys.argv) > 3 else "normal")
assert r.load_onnx(model=src, inputs=names, input_size_list=sizes) == 0
assert r.hybrid_quantization_step1(dataset=ds, proposal=False) == 0
r.release()
