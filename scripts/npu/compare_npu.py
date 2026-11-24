"""Compare NPU logits (from rknn_caldec_bench) with the fp32 PyTorch reference in cases_s<seq>.npz.
Usage: python compare_npu.py <logits.bin> <cases.npz> <n>
"""
import json, sys
import numpy as np

lg, cases, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = np.load(cases); y = np.fromfile(lg, np.float32).reshape(-1, 6)[:n]

def softmax(z): z = z - z.max(); e = np.exp(z); return e / e.sum()

agree, dps, flips = 0, [], []
for i in range(n):
    k = int(c["marker_mask"][i].sum()); p = softmax(y[i, :k].astype(np.float64)); q = softmax(c["ref"][i][0][:k].astype(np.float64))
    a = p.argmax() == q.argmax(); agree += int(a); dps.append(float(np.abs(p - q).max()))
    if not a: flips.append({"case": i, "ref_p": round(float(q.max()), 3), "ref_margin": round(float(np.sort(q)[-1] - np.sort(q)[-2]), 3)})
print(json.dumps({"argmax_agreement": f"{agree}/{n}", "max_abs_dp": round(max(dps), 4),
                  "median_abs_dp": round(float(np.median(dps)), 4), "flips": flips[:10]}))
