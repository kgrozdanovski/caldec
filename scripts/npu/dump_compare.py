"""Compare on-device RKNN per-layer dumps (RKNN_LOG_LEVEL=5 -> dumps/*.npy) with onnxruntime fp32 intermediates.
Usage: python dump_compare.py model.onnx cases_dir case_idx dumps_dir 'dumpprefix=onnx_tensor' ..."""
import sys, os, glob, numpy as np, onnx, onnxruntime as ort
onnx_path, cdir, ci, ddir = sys.argv[1:5]
pairs = [a.split("=", 1) for a in sys.argv[5:]]
m = onnx.load(onnx_path)
have = {o.name for o in m.graph.output}
for _, t in pairs:
    if t not in have: m.graph.output.append(onnx.ValueInfoProto(name=t))
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
feeds = {}
for i in s.get_inputs():
    dt = np.int64 if "int64" in i.type else np.float32
    feeds[i.name] = np.fromfile(f"{cdir}/{ci}_{i.name}.bin", dtype=dt).reshape(i.shape)
names = [t for _, t in pairs]
res = dict(zip(names, s.run(names, feeds)))
def align(d, r):
    d, r = np.squeeze(d), np.squeeze(r)
    if d.shape == r.shape: return d
    if d.ndim == 2 and d.T.shape == r.shape: return d.T
    if d.size == r.size:
        if d.ndim == 3 and r.ndim == 3 and d.shape[0] == r.shape[0] and d.shape[1:] == r.shape[1:][::-1]: return d.transpose(0, 2, 1)
        return d.reshape(r.shape)
    raise ValueError(f"shape {d.shape} vs {r.shape}")
for pre, t in pairs:
    f = sorted(glob.glob(f"{ddir}/{pre}*"))
    if not f: print(f"{pre:>4} MISSING dump"); continue
    d = np.load(f[0]).astype(np.float64); r = res[t].astype(np.float64)
    try: d = align(d, r)
    except ValueError as e: print(pre, t, e); continue
    cos = float((d * r).sum() / (np.linalg.norm(d) * np.linalg.norm(r) + 1e-30))
    print(f"{pre:>4} {t:<55} cos={cos:.5f} maxabs={np.abs(d-r).max():.4g} refmax={np.abs(r).max():.4g}  {os.path.basename(f[0])[:60]}")
