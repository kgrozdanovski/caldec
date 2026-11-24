"""PC: 40 public-test cases (5 questions, long) as runner requests, with PackedAgent answers at the model's own
max_len 1024 in one pass (runner/ref_public.jsonl). The runner must split these into several passes."""
from pathlib import Path
import json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import export_caldec as ex
from benchmark import load_public
agent = ex.load_agent(ex.CKPT, device="cpu"); agent.model.float()
rows = load_public("test")[:40]
with open("runner/requests_public.jsonl", "w") as fq, open("runner/ref_public.jsonl", "w") as fa:
    for r in rows:
        qs = r["questions"]
        fq.write(json.dumps({"state": r["state"], "questions": qs}) + "\n")
        out = agent.system_one(r["state"], qs)
        gold = {qid: (r.get("gold") or {}).get(qid, {}).get("probabilities", {}) for qid in qs}
        fa.write(json.dumps({"answers": out["answers"], "input_tokens": out["usage"]["input_tokens"], "gold": gold}) + "\n")
print("written", len(rows))
