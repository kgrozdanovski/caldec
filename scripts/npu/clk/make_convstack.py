"""NPU-bound probe: 16 x Conv1x1 1024->1024 on [1,1024,16,16] fp16 (8.6 GFLOP). Times scale ~1/f_npu."""
import numpy as np, onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh
rng = np.random.default_rng(0); nodes, inits, x = [], [], "x"
for i in range(16):
    w = (rng.standard_normal((1024, 1024, 1, 1)) / 32).astype(np.float32)
    inits.append(nh.from_array(w, f"w{i}")); nodes.append(h.make_node("Conv", [x, f"w{i}"], [f"y{i}"])); x = f"y{i}"
g = h.make_graph(nodes, "convstack", [h.make_tensor_value_info("x", T.FLOAT, [1, 1024, 16, 16])],
                 [h.make_tensor_value_info(x, T.FLOAT, [1, 1024, 16, 16])], inits)
onnx.save(h.make_model(g, opset_imports=[h.make_opsetid("", 13)]), "convstack.onnx")
from rknn.api import RKNN
r = RKNN(); r.config(target_platform="rk3566"); r.load_onnx("convstack.onnx"); r.build(do_quantization=False); r.export_rknn("convstack.rknn")
np.random.default_rng(1).standard_normal((1, 1024, 16, 16)).astype(np.float32).tofile("0_x.bin")
