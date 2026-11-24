"""Convert the CalDec ONNX graph to RKNN for the RK3566 and check it in the toolkit simulator.
Usage: rknn-venv/bin/python convert_rknn.py --seq 256 --mode fp16|int8 [--eval 40]
int8 calibration uses cases 60..119; the accuracy check uses cases 0..(eval-1), so they do not overlap.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
from rknn.api import RKNN

ap = argparse.ArgumentParser()
ap.add_argument("--seq", type=int, default=256); ap.add_argument("--mode", default="fp16")
ap.add_argument("--eval", type=int, default=40); ap.add_argument("--out", default=str(Path(__file__).resolve().parents[0]) + "/out")
ap.add_argument("--opt", type=int, default=3); ap.add_argument("--masks", action="store_true"); ap.add_argument("--safe-norm", action="store_true")
ap.add_argument("--sel", action="store_true", help="graph with one-hot marker_sel input (no GatherElements)")
ap.add_argument("--embeds", action="store_true", help="graph takes inputs_embeds [1,S,768] instead of input_ids")
ap.add_argument("--onnx", default=None, help="override the ONNX path (profiling variants)"); ap.add_argument("--tag", default=None)
a = ap.parse_args()
out = Path(a.out); c = np.load(out / f"cases_s{a.seq}.npz")
if a.masks:
    sl = "s" if a.sel else ""
    onnx = out / f"caldec_ultra150m_m{'n' if a.safe_norm else ''}{sl}_s{a.seq}.onnx"
    names = ["inputs_embeds" if a.embeds else "input_ids", "local_mask", "global_mask", "head_mask", "marker_sel" if a.sel else "marker_pos", "marker_mask", "qtype"]
    S = a.seq; sizes = [[1, S, 768] if a.embeds else [1, S], [1, 1, S, S], [1, 1, S, S], [S, S], [1, 6, S] if a.sel else [1, 6], [1, 6], [1]]
    raw = out / f"cases_raw_m{sl}{'e' if a.embeds else ''}_s{a.seq}"
    dt = {"input_ids": np.int64, "marker_pos": np.int64, "qtype": np.int64}
    def case(i): return [np.fromfile(raw / f"{i}_{k}.bin", dt.get(k, np.float32)).reshape(sz) for k, sz in zip(names, sizes)]
else:
    onnx = out / f"caldec_ultra150m_s{a.seq}.onnx"
    names = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
    sizes = [[1, a.seq], [1, a.seq], [1, 6], [1, 6], [1]]
    def case(i): return [c[k][i] for k in names]
tag = a.tag or (("mn" if a.safe_norm else "m") + ("s" if a.sel else "") + "_" if a.masks else "") + f"s{a.seq}_{a.mode}" + ("" if a.opt == 3 else f"_o{a.opt}"); log = open(out / f"convert_{tag}.log", "w")

if a.onnx: onnx = Path(a.onnx)
r = RKNN(verbose=bool(os.environ.get("RKNN_VERBOSE")))
r.config(target_platform="rk3566", optimization_level=a.opt)
assert r.load_onnx(model=str(onnx), inputs=names,
                   input_size_list=sizes) == 0, "load_onnx failed"
ds = None
if a.mode == "int8":
    cal = out / f"calib_s{a.seq}"; cal.mkdir(exist_ok=True); lines = []
    for i in range(60, min(120, c["input_ids"].shape[0])):
        fs = []
        for k, v in zip(names, case(i)):
            f = cal / f"{tag}_{k}_{i}.npy"; np.save(f, v); fs.append(str(f))
        lines.append(" ".join(fs))
    ds = out / f"calib_s{a.seq}.txt"; ds.write_text("\n".join(lines) + "\n")
t = time.time()
ret = r.build(do_quantization=(a.mode == "int8"), dataset=str(ds) if ds else None)
assert ret == 0, "build failed"
print(f"build {time.time()-t:.0f}s")
rk = out / f"caldec_ultra150m_{tag}.rknn"; assert r.export_rknn(str(rk)) == 0
print("wrote", rk, rk.stat().st_size // 2**20, "MiB")

# simulator accuracy check
assert r.init_runtime() == 0
def softmax(z): z = z - z.max(); e = np.exp(z); return e / e.sum()
agree, dmax, n = 0, 0.0, min(a.eval, c["input_ids"].shape[0])
for i in range(n):
    o = r.inference(inputs=case(i), data_format=["nchw"] * len(names) if a.masks else None)[0][0]
    k = int(c["marker_mask"][i].sum()); p, q = softmax(o[:k].astype(np.float64)), softmax(c["ref"][i][0][:k].astype(np.float64))
    agree += int(p.argmax() == q.argmax()); dmax = max(dmax, float(np.abs(p - q).max()))
res = {"tag": tag, "simulator_argmax_agreement": f"{agree}/{n}", "simulator_max_abs_dp": round(dmax, 4)}
print(json.dumps(res)); json.dump(res, open(out / f"sim_{tag}.json", "w"))
r.release()
