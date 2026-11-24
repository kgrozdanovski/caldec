"""Write the multi_bench order file for a bucketed eval set.
Usage: python make_order.py out/buckets_eval400b7.json cases_raw_npu_eval400b7 gtmm out/tablesmm out/order_b7_mm.txt
The full-weight model (largest bucket) is listed first; every smaller bucket is a remove_weight model that shares its
weights (share=1). Then one line per question in original order: "<bucket line index> <index inside that bucket>"."""
import json, sys
bj, cprefix, mprefix, tprefix, dst = sys.argv[1:6]
assign = {int(k): v for k, v in json.load(open(bj)).items()}
bks = sorted(assign); full = bks[-1]; order = [full] + bks[:-1]
where = {q: (order.index(b), j) for b in bks for j, q in enumerate(assign[b])}
with open(dst, "w") as f:
    f.write(f"{len(order)}\n")
    for b in order:
        model = f"out/{mprefix}_s{b}.rknn" if b == full else f"out/{mprefix}_s{b}_now.rknn"
        f.write(f"{b} {model} {cprefix}_b{b} {tprefix}_s{b} {0 if b == full else 1}\n")
    for q in range(len(where)):
        f.write("%d %d\n" % where[q])
print("wrote", dst, len(where), "questions")
