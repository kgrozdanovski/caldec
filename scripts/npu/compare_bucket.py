"""Compare one bucket's device logits with the fp32 references.
Usage: python compare_bucket.py <logits.bin> <cases.npz> <buckets.json> <bucket>"""
import json, sys, numpy as np
lg, npz, bj, b = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
c = np.load(npz); idx = json.load(open(bj))[b]; y = np.fromfile(lg, np.float32).reshape(-1, 6)
def sm(z): z = z - z.max(); e = np.exp(z); return e / e.sum()
agree, dps, flips = 0, [], []
for j, i in enumerate(idx[: y.shape[0]]):
    k = int(c["marker_mask"][i].sum()); p = sm(y[j, :k].astype(np.float64)); q = sm(c["ref"][i][0][:k].astype(np.float64))
    a = p.argmax() == q.argmax(); agree += int(a); dps.append(float(np.abs(p - q).max()))
    if not a: flips.append(round(float(np.sort(q)[-1] - np.sort(q)[-2]), 3))
print(json.dumps({"bucket": b, "agreement": f"{agree}/{len(dps)}", "max_dp": round(max(dps), 4),
                  "median_dp": round(float(np.median(dps)), 4), "flip_margins": flips[:12]}))
