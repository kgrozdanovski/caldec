"""Accuracy of runner answers on untied gold (runner/gold_public200.jsonl), and latency. Usage: acc_public.py answers.jsonl"""
import json, sys
import numpy as np
gold = [json.loads(l) for l in open("runner/gold_public200.jsonl")]; ans = [json.loads(l) for l in open(sys.argv[1])]
hit = n = 0; tot, passes = [], []
for g, a in zip(gold, ans):
    tot.append(a["usage"]["total_ms"]); passes.append(a["usage"]["npu_passes"])
    for qid, x in a["answers"].items():
        keys = list(x["probabilities"]) if "probabilities" in x else ["false", "true"]
        p = np.array(list(x["probabilities"].values()) if "probabilities" in x else [1 - x["noul"], x["noul"]])
        gv = np.array([float(g[qid].get(k, 0.0)) for k in keys])
        if gv.max() <= 0 or (gv == gv.max()).sum() > 1: continue
        n += 1; hit += p.argmax() == gv.argmax()
print(f"cases {len(ans)}, untied {n}: accuracy {hit / n:.4f}; per case {np.mean(tot):.0f} ms, passes {np.mean(passes):.2f}; "
      f"per question {np.sum(tot) / sum(len(a['answers']) for a in ans):.0f} ms")
