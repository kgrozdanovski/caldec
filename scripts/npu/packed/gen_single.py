"""Baseline for the packed comparison: the same 400 cases' questions as single-question sequences for the deployed
per-question model (`CALDEC_CKPT` must name a per-question checkpoint; cut at 256 as deployed), with its fp32 references and the gold targets.
Writes out/cases_pk400single_s256.npz (gen_cases.py format + 'gold' [n,6], -1 padded) for make_bucket_cases.py."""
import pickle, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import export_caldec as ex  # noqa: E402
from data import load_split  # noqa: E402
from laya.common import QTYPES, build_sequence, render_options  # noqa: E402
S = 256
D = pickle.load(open(Path(__file__).resolve().parent / "cases.pkl", "rb"))   # our own file (export_packed.py prep)
want = [(c["case_id"], len(c["markers"])) for c in D["cases"]]
torch.backends.mha.set_fastpath_enabled(False)
agent = ex.load_agent(ex.CKPT, device="cpu")
agent.model.encoder.config._attn_implementation = "eager"
core = ex.Core(agent.model).eval().float(); tok = agent.tok
rows = {r["case_id"]: r for r in load_split(ex.CALDEC / "data", "test")}
pk = {c["case_id"]: c for c in D["cases"]}
arr = {k: [] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype", "ref", "ntok", "gold")}
skipped = 0
with torch.inference_mode():
    for cid, _ in want:
        r = rows[cid]
        qids = [q for q in r["questions"] if (r.get("gold") or {}).get(q, {}).get("probabilities")]
        for qid in qids:
            q = agent._to_internal(r["questions"][qid])
            seq, markers = build_sequence(tok, r["state"], q, S, 192)
            if not (2 <= len(markers) <= ex.K) or len(markers) != len(render_options(q)):
                skipped += 1; continue
            keys = (["false", "true"] if q["t"] == "noul" else list(q["crit"] or {}) if q["t"] == "choice"
                    else [str(i) for i in range(len(q["crit"] or []))])
            g = np.full(ex.K, -1.0, np.float32); g[:len(keys)] = [float(r["gold"][qid]["probabilities"].get(k, 0.0)) for k in keys]
            ids = np.full((1, S), tok.pad_token_id, np.int64); att = np.zeros((1, S), np.int64)
            ids[0, :len(seq)] = seq; att[0, :len(seq)] = 1
            mp = np.zeros((1, ex.K), np.int64); mm = np.zeros((1, ex.K), np.float32)
            mp[0, :len(markers)] = markers; mm[0, :len(markers)] = 1
            qt = np.array([QTYPES[q["t"]]], np.int64)
            ref = core(*[torch.from_numpy(x) for x in (ids, att, mp, mm, qt)]).numpy()
            for k, v in zip(("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"), (ids, att, mp, mm, qt)):
                arr[k].append(v)
            arr["ref"].append(ref); arr["ntok"].append(len(seq)); arr["gold"].append(g)
np.savez(Path(str(Path(__file__).resolve().parents[1]) + "/out") / "cases_pk400single_s256.npz", **{k: np.stack(v) for k, v in arr.items()})
print("questions", len(arr["ntok"]), "skipped", skipped, "packed questions", sum(n for _, n in want))
