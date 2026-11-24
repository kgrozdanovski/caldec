"""Runner act_probability (NPU head rows + numpy act head) vs PackedAgent.system_one fp32 (runner/ref_answers.jsonl).
Usage: compare_act.py answers.jsonl [ref.jsonl]"""
import json, sys
import numpy as np
npu = [json.loads(l) for l in open(sys.argv[1])]
ref = [json.loads(l) for l in open(sys.argv[2] if len(sys.argv) > 2 else "runner/ref_answers.jsonl")][: len(npu)]
d, side, n = [], 0, 0
for r, a in zip(ref, npu):
    for qid, ra in r["answers"].items():
        x, y = ra["action"]["act_probability"], a["answers"][qid]["action"]["act_probability"]
        d.append(abs(x - y)); side += (x >= 0.5) == (y >= 0.5); n += 1
d = np.array(d)
print(f"cases {len(npu)}, questions {n}: |d act_probability| max {d.max():.4f} median {np.median(d):.4f} p99 "
      f"{np.quantile(d, 0.99):.4f}; same side of 0.5: {side}/{n}")
