"""Runner answers (NPU) vs caldec PackedAgent.system_one (CPU fp32): argmax agreement, max |dp| on the published
(temperature-scaled) probabilities, token counts, accuracy on untied gold, and latency split.
Usage: compare_answers.py answers.jsonl [ref.jsonl]"""
import json, sys
import numpy as np
npu = [json.loads(l) for l in open(sys.argv[1])]
ref = [json.loads(l) for l in open(sys.argv[2] if len(sys.argv) > 2 else "runner/ref_answers.jsonl")]
agree = n = tok_eq = acc_n = acc_r = nu = 0; dps = []; tot, npu_ms, passes = [], [], []
for r, a in zip(ref, npu):
    tok_eq += r["input_tokens"] == a["usage"]["input_tokens"]
    tot.append(a["usage"]["total_ms"]); npu_ms.append(a["usage"]["npu_ms"]); passes.append(a["usage"]["npu_passes"])
    for qid, ra in r["answers"].items():
        na = a["answers"][qid]; pr = np.array(list(ra["probabilities"].values()) if "probabilities" in ra else [1 - ra["noul"], ra["noul"]])
        pn = np.array(list(na["probabilities"].values()) if "probabilities" in na else [1 - na["noul"], na["noul"]])
        n += 1; agree += pr.argmax() == pn.argmax(); dps.append(np.abs(pr - pn).max())
        g = r["gold"][qid]
        keys = list(ra["probabilities"].keys()) if "probabilities" in ra else ["false", "true"]
        gv = np.array([float(g.get(k, 0.0)) for k in keys])
        if (gv == gv.max()).sum() > 1: continue
        nu += 1; acc_n += pn.argmax() == gv.argmax(); acc_r += pr.argmax() == gv.argmax()
tot, npu_ms = np.array(tot), np.array(npu_ms)
print(f"cases {len(ref)}, questions {n}: argmax agreement {agree}/{n}; |dp| max {max(dps):.4f} median {np.median(dps):.4f}; "
      f"input_tokens equal {tok_eq}/{len(ref)}; passes per case {np.mean(passes):.2f}")
print(f"accuracy on {nu} untied: runner (NPU) {acc_n / nu:.4f}  PackedAgent fp32 {acc_r / nu:.4f}")
print(f"latency per case: total mean {tot.mean():.1f} ms (median {np.median(tot):.1f}), NPU part {npu_ms.mean():.1f} ms, "
      f"host part (tokenize + inputs + pipe) {(tot - npu_ms).mean():.1f} ms; per question total {tot.sum() / n:.1f} ms")
