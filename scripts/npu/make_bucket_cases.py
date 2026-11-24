"""Split an eval npz into per-bucket NPU-form case dirs: each case goes to the smallest bucket >= its length.
Usage: python make_bucket_cases.py out/cases_eval400_s256.npz eval400 128,160,192,256
Writes out/cases_raw_npu_<tag>_b<S>/<j>_<input>.bin and out/buckets_<tag>.json (bucket -> list of case indices)."""
import json, sys, numpy as np, os
npz, tag, bks = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
c = np.load(npz); K = 6; out = os.path.dirname(npz); W = 128
def masks(att, S):
    a = att[0, :S].astype(bool); i = np.arange(S)
    key = np.where(a[None, :], 0.0, -1e4).astype(np.float32)
    glo = np.broadcast_to(key, (S, S)).copy()
    loc = np.where(np.abs(i[:, None] - i[None, :]) <= W, glo, -1e4).astype(np.float32)
    return loc, glo, glo.copy()
assign = {b: [] for b in bks}
for i in range(c["input_ids"].shape[0]):
    n = int(c["attention_mask"][i].sum()); b = min(x for x in bks if x >= n); assign[b].append(i)
for b, idx in assign.items():
    d = f"{out}/cases_raw_npu_{tag}_b{b}"; os.makedirs(d, exist_ok=True)
    for j, i in enumerate(idx):
        loc, glo, head = masks(c["attention_mask"][i], b)
        sel = np.zeros((1, K, b), np.float32); sel[0, np.arange(K), c["marker_pos"][i][0]] = 1.0
        for nm, v in (("input_ids", c["input_ids"][i][:, :b]), ("local_mask", loc), ("global_mask", glo), ("head_mask", head),
                      ("marker_sel", sel), ("marker_mask", c["marker_mask"][i].astype(np.float32)), ("qtype", c["qtype"][i])):
            np.ascontiguousarray(v).tofile(f"{d}/{j}_{nm}.bin")
json.dump({str(b): v for b, v in assign.items()}, open(f"{out}/buckets_{tag}.json", "w"))
print({b: len(v) for b, v in assign.items()})
