"""Order file for multi_bench2 over the packed buckets: the 384 model (weights) first, the others share; then one line
per case in original order. Usage: python3 packed/make_order_pk.py  -> packed/order_pk.txt"""
import json
a = {int(k): v for k, v in json.load(open("packed/buckets.json")).items()}
bks = sorted(a); full = bks[-1]; order = [full] + [b for b in bks if b != full]
where = {ci: (order.index(b), j) for b in bks for j, ci in enumerate(a[b])}
with open("packed/order_pk.txt", "w") as f:
    f.write(f"{len(order)}\n")
    for b in order:
        m = f"packed/pk_s{b}.rknn" if b == full else f"packed/pk_s{b}_now.rknn"
        f.write(f"{b} {m} packed/cases_b{b} packed/tables_s{b} {0 if b == full else 1}\n")
    for ci in range(len(where)): f.write("%d %d\n" % where[ci])
print("items", len(where))
