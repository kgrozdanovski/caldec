"""PC: the first 200 public-test cases as runner requests (runner/requests_public200.jsonl) and their gold
probabilities (runner/gold_public200.jsonl), the same cases as exp_long_cases.py."""
from pathlib import Path
import json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark import load_public
rows = load_public("test")[:200]
with open("runner/requests_public200.jsonl", "w") as fq, open("runner/gold_public200.jsonl", "w") as fg:
    for r in rows:
        fq.write(json.dumps({"state": r["state"], "questions": r["questions"]}) + "\n")
        fg.write(json.dumps({qid: (r.get("gold") or {}).get(qid, {}).get("probabilities", {}) for qid in r["questions"]}) + "\n")
print("written", len(rows))
