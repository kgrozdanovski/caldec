"""Agreement of packed NPU logits with packed fp32, and accuracy against the test labels (untied gold only).
Usage: python compare_packed.py <order.txt> <logits.bin>  (logits: one [1,12] float32 per order item, in order)"""
import json, pickle, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
D = pickle.load(open(HERE / "cases.pkl", "rb"))   # our own file, written by export_packed.py prep
assign = {int(k): v for k, v in json.load(open(HERE / "buckets.json")).items()}
lines = open(sys.argv[1]).read().split("\n"); nb = int(lines[0]); bks = [int(l.split()[0]) for l in lines[1:1 + nb]]
items = [tuple(map(int, l.split())) for l in lines[1 + nb:] if l.strip()]
y = np.fromfile(sys.argv[2], np.float32).reshape(len(items), 12)
agree = nq = acc_npu = acc_ref = acc_1024 = ntied = 0; dps = []; flips = []
for (bi, j), row in zip(items, y):
    c = D["cases"][assign[bks[bi]][j]]; k = 0
    for qi, (r, r1, t) in enumerate(zip(c["ref"], c["ref1024"], c["target"])):
        v = row[k:k + len(r)]; k += len(r); nq += 1
        pn = np.exp(v - v.max()); pn /= pn.sum(); pr = np.exp(r - r.max()); pr /= pr.sum()
        dps.append(np.abs(pn - pr).max())
        if v.argmax() == r.argmax(): agree += 1
        else: flips.append((c["case_id"], qi, float(np.sort(pr)[-1] - np.sort(pr)[-2])))
        t = np.array(t)
        if (t == t.max()).sum() > 1: ntied += 1; continue
        g = t.argmax(); acc_npu += v.argmax() == g; acc_ref += r.argmax() == g; acc_1024 += r1.argmax() == g
nu = nq - ntied
print(f"questions {nq}: NPU vs packed fp32 argmax {agree}/{nq}; max |dp| {max(dps):.4f}, median {np.median(dps):.4f}")
print(f"accuracy on {nu} untied: NPU {acc_npu / nu:.4f}  fp32@{D['max_len']} {acc_ref / nu:.4f}  fp32@1024 {acc_1024 / nu:.4f}")
for f in flips: print("  flip", f[0], "q", f[1], "fp32 margin %.3f" % f[2])
