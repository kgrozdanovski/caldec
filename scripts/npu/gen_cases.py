"""Build CalDec test cases from a data split, with fp32 references from the original model, and write the
NPU-form inputs (as export_npu.py expects) for the device harness.

Usage (caldec venv): PYTHONPATH=pylib .../python gen_cases.py --split test --n 400 --seq 256 --tag eval400
Writes out/cases_<tag>_s<S>.npz (input_ids, attention_mask, marker_pos, marker_mask, qtype, ref, ntok)
and out/cases_raw_npu_<tag>_s<S>/<i>_<input>.bin.
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_caldec as ex  # noqa: E402
from data import load_split  # noqa: E402  (caldec scripts/data.py, path set by export_caldec)
from laya.common import QTYPES, build_sequence, render_options  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test"); ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--skip", type=int, default=0, help="skip the first N valid cases (e.g. the 120 already used)")
    ap.add_argument("--seq", type=int, default=256); ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", default=ex.CKPT)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[0]) + "/out")
    a = ap.parse_args(); S = a.seq; out = Path(a.out)
    torch.backends.mha.set_fastpath_enabled(False)
    agent = ex.load_agent(a.ckpt, device="cpu")
    agent.model.encoder.config._attn_implementation = "eager"
    core = ex.Core(agent.model).eval().float(); tok = agent.tok
    max_len, head_max_len = agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192)
    cases = []
    for r in load_split(ex.CALDEC / "data", a.split):
        for qid, qdef in r["questions"].items():
            q = agent._to_internal(qdef)
            seq, markers = build_sequence(tok, r["state"], q, min(max_len, S), head_max_len)
            if not (2 <= len(markers) <= ex.K) or len(markers) != len(render_options(q)):
                continue
            cases.append((seq, markers, QTYPES[q["t"]]))
        if len(cases) >= a.skip + a.n:
            break
    cases = cases[a.skip: a.skip + a.n]
    arr = {k: [] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype", "ref", "ntok")}
    d = out / f"cases_raw_npu_{a.tag}_s{S}"; d.mkdir(parents=True, exist_ok=True)
    names = ["input_ids", "local_mask", "global_mask", "head_mask", "marker_sel", "marker_mask", "qtype"]
    with torch.inference_mode():
        for i, (seq, markers, qt) in enumerate(cases):
            ids = np.full((1, S), tok.pad_token_id, np.int64); att = np.zeros((1, S), np.int64)
            ids[0, :len(seq)] = seq; att[0, :len(seq)] = 1
            mp = np.zeros((1, ex.K), np.int64); mm = np.zeros((1, ex.K), np.float32)
            mp[0, :len(markers)] = markers; mm[0, :len(markers)] = 1
            q = np.array([qt], np.int64)
            ref = core(*[torch.from_numpy(x) for x in (ids, att, mp, mm, q)]).numpy()
            for k, v in zip(("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"), (ids, att, mp, mm, q)):
                arr[k].append(v)
            arr["ref"].append(ref); arr["ntok"].append(len(seq))
            loc, glo, head = ex.make_masks(att)
            sel = np.zeros((1, ex.K, S), np.float32); sel[0, np.arange(ex.K), mp[0]] = 1.0
            for nm, v in zip(names, (ids, loc[0, 0].astype(np.float32), glo[0, 0].astype(np.float32),
                                     head.astype(np.float32), sel, mm, q)):
                v.tofile(d / f"{i}_{nm}.bin")
    np.savez(out / f"cases_{a.tag}_s{S}.npz", **{k: np.stack(v) for k, v in arr.items()})
    n = np.array(arr["ntok"])
    print(f"{len(cases)} cases from {a.split} (skip {a.skip}); tokens min {n.min()} median {int(np.median(n))} max {n.max()}; "
          f"-> {d}")


if __name__ == "__main__":
    main()
