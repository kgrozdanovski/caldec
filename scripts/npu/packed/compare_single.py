"""Baseline accuracy: deployed per-question NPU logits (multi_bench order = question order) vs the per-question model's fp32 and gold."""
import sys, numpy as np
c = np.load("out/cases_pk400single_s256.npz"); y = np.fromfile(sys.argv[1], np.float32).reshape(-1, 6)
n = len(y); agree = an = ar = nu = 0
for i in range(n):
    k = int(c["marker_mask"][i].sum()); v, r, g = y[i, :k], c["ref"][i][0][:k], c["gold"][i][:k]
    agree += v.argmax() == r.argmax()
    if (g == g.max()).sum() > 1: continue
    nu += 1; an += v.argmax() == g.argmax(); ar += r.argmax() == g.argmax()
print(f"questions {n}: NPU vs fp32 {agree}/{n}; accuracy on {nu} untied: NPU {an/nu:.4f}  fp32@256 {ar/nu:.4f}")
