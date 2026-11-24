"""PC: the 400 packed eval cases as runner requests (runner/requests.jsonl) and caldec PackedAgent.system_one answers
on CPU fp32 (runner/ref_answers.jsonl), same questions as packed/cases.pkl (those with gold labels)."""
from pathlib import Path
import json, pickle, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import export_caldec as ex
from data import load_split
D = pickle.load(open(str(Path(__file__).resolve().parents[1]) + "/packed/cases.pkl", "rb"))     # our own file
agent = ex.load_agent(ex.CKPT, device="cpu")
agent.model.float(); agent.cfg["max_len"] = 640                              # the NPU set's longest bucket
rows = {r["case_id"]: r for r in load_split(ex.CALDEC / "data", "test")}
with open("runner/requests.jsonl", "w") as fq, open("runner/ref_answers.jsonl", "w") as fa:
    for c in D["cases"]:
        r = rows[c["case_id"]]
        qs = {qid: q for qid, q in r["questions"].items() if ((r.get("gold") or {}).get(qid) or {}).get("probabilities")}
        fq.write(json.dumps({"case_id": c["case_id"], "state": r["state"], "questions": qs}) + "\n")
        out = agent.system_one(r["state"], qs)
        fa.write(json.dumps({"case_id": c["case_id"], "answers": out["answers"], "input_tokens": out["usage"]["input_tokens"],
                             "gold": {qid: r["gold"][qid]["probabilities"] for qid in qs}}) + "\n")
print("written", len(D["cases"]))
