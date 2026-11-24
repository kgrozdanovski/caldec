"""Run the exported CalDec graph with onnxruntime (CPU) on the cases file; report agreement + latency.
Usage: python bench_ort.py <model.onnx> <cases.npz> [--threads 4] [--runs 30] [--json out.json]
Agreement: argmax over the real options vs the fp32 PyTorch reference, plus max |delta p| of the softmax.
"""
import argparse, json, platform, statistics, time, math
import numpy as np
import onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("cases")
ap.add_argument("--threads", type=int, default=4); ap.add_argument("--runs", type=int, default=30)
ap.add_argument("--json")
a = ap.parse_args()
so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
t0 = time.perf_counter()
s = ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])
load_s = time.perf_counter() - t0
c = np.load(a.cases)
names = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
n = c["input_ids"].shape[0]

def feed(i): return {k: c[k][i] for k in names}

def softmax(z): z = z - z.max(); e = np.exp(z); return e / e.sum()

agree, dmax = 0, 0.0
for i in range(n):
    out = s.run(None, feed(i))[0][0]
    k = int(c["marker_mask"][i].sum()); ref = c["ref"][i][0]
    p, q = softmax(out[:k]), softmax(ref[:k])
    agree += int(p.argmax() == q.argmax()); dmax = max(dmax, float(np.abs(p - q).max()))
for _ in range(3): s.run(None, feed(0))
lat = []
for r in range(a.runs):
    i = r % n; t = time.perf_counter(); s.run(None, feed(i)); lat.append((time.perf_counter() - t) * 1000)
lat.sort()
res = {"backend": f"onnxruntime {ort.__version__} CPU, {a.threads} threads", "host": platform.node(),
       "model": a.model.split("/")[-1], "seq": int(c["input_ids"].shape[-1]), "cases": n,
       "argmax_agreement": f"{agree}/{n}", "max_abs_dp": round(dmax, 5), "load_s": round(load_s, 1),
       "p50_ms": round(statistics.median(lat), 1), "p95_ms": round(lat[math.ceil(0.95 * len(lat)) - 1], 1),
       "runs": a.runs}
print(json.dumps(res))
if a.json: json.dump(res, open(a.json, "w"), indent=1)
