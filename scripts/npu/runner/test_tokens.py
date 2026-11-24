"""PC check: the runner's packed inputs (tokenizers + vendored laya code) equal caldec's packed.build_case output in
packed/cases.pkl for all 400 cases, and its inputs_for equals export_packed.inputs_for byte for byte."""
from pathlib import Path
import pickle, sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]) + "/runner"); sys.path.insert(0, str(Path(__file__).resolve().parents[1]) + "/packed")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import caldec_npu as cn
import export_packed as ep
from data import load_split
D = pickle.load(open(str(Path(__file__).resolve().parents[1]) + "/packed/cases.pkl", "rb"))      # our own file
tok = cn.Tok(ep.CKPT / "tokenizer")
rows = {r["case_id"]: r for r in load_split(ep.ex.CALDEC / "data", "test")}
bad = bad_in = 0
for c in D["cases"]:
    r = rows[c["case_id"]]
    qs = [cn.to_internal(q) for qid, q in r["questions"].items() if ((r.get("gold") or {}).get(qid) or {}).get("probabilities")]
    segs = [cn.question_segment(tok, q, 192) for q in qs]
    st = tok.ids(cn.serialize_state(r["state"]).replace(tok.mask_token, " "))
    ids, views, marks = cn.build_case(tok, st, segs, 384)
    if (ids, views, marks) != (c["ids"], c["views"], c["markers"]):
        bad += 1; continue
    S = next(s for s in (96, 128, 160, 192, 224, 256, 320, 384) if ep.inputs_for(c, s, D["pad"], D["ntypes"]) is not None)
    a = cn.inputs_for(ids, views, marks, c["qtype"], S, tok.pad_token_id); b = ep.inputs_for(c, S, D["pad"], D["ntypes"])
    if any(a[k].tobytes() != b[k].tobytes() for k in ep.NAMES):
        bad_in += 1
print(f"cases {len(D['cases'])}: token/view/marker mismatches {bad}; device-input mismatches {bad_in}; pad {tok.pad_token_id} vs {D['pad']}")
