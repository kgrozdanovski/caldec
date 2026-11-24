"""Toolkit simulator accuracy of a hybrid int8 build (step 2 rebuilt in-process), to tell quantisation noise from a
device-side int8 fault. Usage: sim_down.py <prefix> <mode> <n>"""
from pathlib import Path
import sys, json, numpy as np
sys.argv_saved = list(sys.argv)
pre, mode, NQ = sys.argv[1], sys.argv[2], int(sys.argv[3])
from rknn.api import RKNN
cfgp = f"{pre}.quantization.cfg"
sys.argv = ["step2s.py", pre, "unused.rknn", mode]
exec(open("step2s.py").read().split("r = RKNN()")[0])
r = RKNN()
assert r.hybrid_quantization_step2(model_input=f"{pre}.model", data_input=f"{pre}.data", model_quantization_cfg=cfgp) == 0
assert r.init_runtime() == 0
B = str(Path(__file__).resolve().parents[1]); c = np.load(f"{B}/out/cases_eval400_s256.npz"); idx = json.load(open(f"{B}/out/buckets_eval400b7.json"))["128"]
spec = [("input_ids", np.int64, (1, 128)), ("local_mask", np.float32, (128, 128)), ("global_mask", np.float32, (128, 128)),
        ("head_mask", np.float32, (128, 128)), ("marker_sel", np.float32, (1, 6, 128)), ("marker_mask", np.float32, (1, 6)), ("qtype", np.int64, (1,))]
def sm(z): z = z - z.max(); e = np.exp(z); return e / e.sum()
agree, dps = 0, []
for j in range(NQ):
    xs = [np.fromfile(f"{B}/out/cases_raw_npu_eval400b7_b128/{j}_{nm}.bin", dt).reshape(sh) for nm, dt, sh in spec]
    y = r.inference(inputs=xs, data_format=["nchw"] * len(xs))[0].reshape(-1)
    i = idx[j]; k = int(c["marker_mask"][i].sum()); p = sm(y[:k].astype(float)); q = sm(c["ref"][i][0][:k].astype(float))
    agree += int(p.argmax() == q.argmax()); dps.append(float(abs(p - q).max()))
print(json.dumps({"mode": mode, "simulator_agreement": f"{agree}/{NQ}", "max_dp": round(max(dps), 4), "median_dp": round(float(np.median(dps)), 4)}))
