"""CalDec on the RK3566 NPU: packed multi-question decisions through caldec_npu_server.

`CalDecNPU(model_dir).system_one(state, questions)` returns the same answer format as caldec's
`PackedAgent.system_one` (scripts/packed.py). `action.act_probability` is included when the NPU set has the second
output `hcls` (the head's first row per question: the pk2_* set) and `act_head.npz` is in model_dir; the act head
itself (Linear 772->256, GELU, Linear 256->2) runs here on the CPU.

Needs only numpy and `tokenizers` (no torch, no transformers). The pieces of `laya` it needs are copied below,
unchanged in behaviour, from laya.common / laya.agent (the version installed in the caldec repo's venv):
serialize_state, render_criterion, render_options, build_sequence (head part), _to_internal,
confidence_from_probs, temp_bucket, clamp_temperature.

model_dir layout (made by runner/stage.sh):
  tokenizer.json, tokenizer_config.json, rl_agent_config.json   (from the packed checkpoint)
  models.txt, pk_s*.rknn, tables_s*/                          (the NPU set, as for multi_bench2)
  caldec_npu_server                                           (built on the device)
  act_head.npz                                                (optional, runner/export_act_head.py)
A case whose questions exceed one pass (head rows > Sh, markers > 12) is split into several passes; each pass packs
a group of questions with the full state.
"""
import json, math, os, struct, subprocess, time
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
KT, SH_MAX, WINDOW, NTYPES = 12, 192, 128, 3
TEMP_MIN, TEMP_MAX = 0.5, 5.0


# ---- copied from laya.common / laya.agent -------------------------------------------------------------------------
def serialize_state(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def render_criterion(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q):
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if v is None or v == "" else "%s: %s" % (k, render_criterion(v)) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    f, tr = crit.get("false"), crit.get("true")
    return ["false: " + (render_criterion(f) if f not in (None, "") else "no, the statement does not hold"),
            "true: " + (render_criterion(tr) if tr not in (None, "") else "yes, the statement holds")]


def to_internal(qdef):
    t = qdef["type"]; crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}


def confidence_from_probs(p, k):
    if k < 2:
        return 1.0
    p = p[:k]; ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def temp_bucket(qtype, k):
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def clamp_temperature(t, lo=TEMP_MIN, hi=TEMP_MAX):
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if t != t or t in (float("inf"), float("-inf")):
        return 1.0
    return min(hi, max(lo, t))
# -------------------------------------------------------------------------------------------------------------------


class Tok:
    """The subset of the HF tokenizer API that build_sequence uses, over `tokenizers`."""
    def __init__(self, d):
        self.t = Tokenizer.from_file(str(Path(d) / "tokenizer.json"))
        c = json.load(open(Path(d) / "tokenizer_config.json"))
        self.mask_token = c["mask_token"]
        self.cls_token_id, self.sep_token_id = self.t.token_to_id(c["cls_token"]), self.t.token_to_id(c["sep_token"])
        self.mask_token_id, self.pad_token_id = self.t.token_to_id(c["mask_token"]), self.t.token_to_id(c["pad_token"])

    def ids(self, text):
        return self.t.encode(text, add_special_tokens=False).ids


def question_segment(tok, q, head_max_len):
    """laya.common.build_sequence(tok, "", q, ...) without [CLS] and the empty-state tail: head [SEP] options [SEP]."""
    opts = render_options(q)
    ins = str(q["ins"]).replace(tok.mask_token, " ")
    head_ids = tok.ids("%s question: %s" % (q["t"], ins))
    opt_ids = [[tok.mask_token_id] + tok.ids(" " + o.replace(tok.mask_token, " "))[:48] for o in opts]
    budget = head_max_len - sum(len(o) for o in opt_ids)
    if budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]; budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, budget)]
    body = head_ids + [tok.sep_token_id]; markers = []
    for o in opt_ids:
        markers.append(len(body)); body.extend(o)
    body.append(tok.sep_token_id)
    return body, markers


def build_case(tok, state_ids, segs, max_len):
    """packed.build_case over pre-built question segments (body, markers)."""
    ids = [tok.cls_token_id]; views, marks = [], []
    for body, offsets in segs:
        start = len(ids); ids.extend(body)
        views.append([0] + list(range(start, start + len(body)))); marks.append([1 + o for o in offsets])
    room = max(0, max_len - len(ids) - 1)
    return ids + state_ids[:room] + [tok.sep_token_id], views, marks


def inputs_for(ids, views, markers, qtypes, S, pad, cls=False):
    Sh = min(S, SH_MAX); n = len(ids)
    x = dict(input_ids=np.full((1, S), pad, np.int64))
    x["input_ids"][0, :n] = ids
    valid = np.zeros(S, bool); valid[:n] = True; i = np.arange(S)
    glo = np.broadcast_to(np.where(valid[None, :], 0.0, -1e4), (S, S)).astype(np.float32)
    x["global_mask"] = np.ascontiguousarray(glo)
    x["local_mask"] = np.where(np.abs(i[:, None] - i[None, :]) <= WINDOW, glo, -1e4).astype(np.float32)
    vs = np.zeros((S, Sh), np.float32); rt = np.zeros((NTYPES, Sh), np.float32)
    hm = np.full((Sh, Sh), -1e4, np.float32); ms = np.zeros((KT, Sh), np.float32); mm = np.zeros((1, KT), np.float32)
    cs = np.zeros((KT, Sh), np.float32)
    r0 = k = 0
    for j, v in enumerate(views):
        L = len(v); cs[j, r0] = 1.0; vs[v, np.arange(r0, r0 + L)] = 1.0; rt[qtypes[j], r0:r0 + L] = 1.0; hm[r0:r0 + L, r0:r0 + L] = 0.0
        for off in markers[j]:
            ms[k, r0 + off] = 1.0; mm[0, k] = 1.0; k += 1
        r0 += L
    for r in range(r0, Sh):
        vs[0, r] = 1.0; rt[0, r] = 1.0; hm[r, r] = 0.0
    x.update(view_sel=vs, row_type=rt, head_mask=hm, marker_sel=ms, marker_mask=mm)
    if cls:
        x["cls_sel"] = cs
    return x


_erf = np.vectorize(math.erf, otypes=[np.float64])


def act_probability(head, hrow, z):
    """packed.forward's act head for one question: hrow = head output row 0 [d], z = raw option logits [k]."""
    k = len(z); p = np.exp(z - z.max()); p = p / p.sum()
    kk = max(k, 2); ent = -(p * np.log(np.clip(p, 1e-9, None))).sum() / math.log(kk)
    top = np.sort(p)[::-1]; t2 = top[1] if k >= 2 else 0.0
    x = np.concatenate([hrow.astype(np.float64), [top[0], top[0] - t2, ent, kk / 255.0]])
    h = head["w1"] @ x + head["b1"]; h = 0.5 * h * (1.0 + _erf(h / math.sqrt(2.0)))
    a = head["w2"] @ h + head["b2"]; a = np.exp(a - a.max()); return float(a[0] / a.sum())


class CalDecNPU:
    def __init__(self, model_dir, server=None):
        d = Path(model_dir).resolve(); self.dir = d
        self.tok = Tok(d)
        cfg = json.load(open(d / "rl_agent_config.json"))
        self.head_max_len = cfg.get("head_max_len", 192)
        self.temperature = [clamp_temperature(t) for t in cfg.get("temperature", [1.0, 1.0, 1.0])]
        self.temperature_by_options = {k: clamp_temperature(v) for k, v in cfg.get("temperature_by_options", {}).items()}
        self.p = subprocess.Popen([str(server or d / "caldec_npu_server"), "models.txt"], cwd=str(d),
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.buckets = {}
        while True:
            line = self.p.stdout.readline().decode()
            if not line:
                raise RuntimeError("caldec_npu_server exited during start-up")
            if line.strip() == "READY":
                break
            f = line.split()
            if f[0] == "B":
                S = int(f[1]); self.buckets[S] = [x.split(":")[0] for x in f[3:]
                                                  if not x.endswith("|T") and not (x[0] == "O" and x[1:].isdigit())]
        self.sizes = sorted(self.buckets); self.max_len = self.sizes[-1]
        self.last_npu_ms = []
        ah = d / "act_head.npz"
        self.act_head = {k: v.astype(np.float64) for k, v in np.load(ah).items()} if ah.exists() else None

    def close(self):
        if self.p.poll() is None:
            self.p.stdin.write(struct.pack("<i", 0)); self.p.stdin.flush(); self.p.wait(10)

    def _run(self, S, x):
        w = self.p.stdin; w.write(struct.pack("<i", S))
        for nm in self.buckets[S]:
            w.write(np.ascontiguousarray(x[nm]).tobytes())
        w.flush()
        r = self.p.stdout
        st, no = struct.unpack("<ii", r.read(8)); ys = []
        for _ in range(no):
            (n,) = struct.unpack("<i", r.read(4)); ys.append(np.frombuffer(r.read(4 * n), np.float32))
        (ms,) = struct.unpack("<f", r.read(4)); self.last_npu_ms.append(ms)
        if st:
            raise RuntimeError("rknn error %d" % st)
        return ys

    def _groups(self, segs):
        """Greedy split of the questions so each pass fits the largest bucket's head (rows <= Sh, markers <= 12)."""
        sh = min(self.max_len, SH_MAX); groups, cur, rows, marks = [], [], 0, 0
        for j, (body, off) in enumerate(segs):
            r, m = 1 + len(body), len(off)
            if r > sh or m > KT:   # one question alone does not fit a pass (head rows <= 192, markers <= 12)
                raise ValueError("question %d needs %d head rows and %d option markers; one NPU pass takes at most "
                                 "%d rows and %d markers" % (j, r, m, sh, KT))
            if cur and (rows + r > sh or marks + m > KT):
                groups.append(cur); cur, rows, marks = [], 0, 0
            cur.append(j); rows += r; marks += m
        return groups + [cur]

    def system_one(self, state, questions):
        self.last_npu_ms = []
        qids = list(questions); qs = [to_internal(questions[k]) for k in qids]
        segs = [question_segment(self.tok, q, self.head_max_len) for q in qs]
        for qid, q, (_, off) in zip(qids, qs, segs):
            if len(off) != len(render_options(q)):
                raise ValueError("question %r options exceed head_max_len=%d" % (qid, self.head_max_len))
        st_ids = self.tok.ids(serialize_state(state).replace(self.tok.mask_token, " "))
        logits = [None] * len(qs); hrows = [None] * len(qs); ntok = 0
        for g in self._groups(segs):
            ids, views, marks = build_case(self.tok, st_ids, [segs[j] for j in g], self.max_len)
            rows = sum(len(v) for v in views); ntok += len(ids)
            S = next(s for s in self.sizes if s >= len(ids) and min(s, SH_MAX) >= rows)
            cls = "cls_sel" in self.buckets[S]
            ys = self._run(S, inputs_for(ids, views, marks, [QTYPES[qs[j]["t"]] for j in g], S, self.tok.pad_token_id, cls))
            y, k = ys[0], 0
            for n, (j, m) in enumerate(zip(g, marks)):
                logits[j] = y[k:k + len(m)].astype(np.float64); k += len(m)
                if len(ys) > 1:
                    hrows[j] = ys[1].reshape(KT, -1)[n]
        answers = {}
        for qid, q, z, hr in zip(qids, qs, logits, hrows):
            k = len(z); qt = QTYPES[q["t"]]
            ext = {"act_probability": round(act_probability(self.act_head, hr, z), 4)} if hr is not None and self.act_head else None
            t_scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            z = z / t_scale; p = np.exp(z - z.max()); p = p / p.sum(); conf = round(confidence_from_probs(p, k), 4)
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {"type": "choice", "choice": keys[int(p.argmax())],
                                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)}, "confidence": conf}
            elif q["t"] == "score":
                answers[qid] = {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)}, "confidence": conf}
            else:
                answers[qid] = {"type": "noul", "noul": round(float(p[1]), 4),
                                "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4)}
            if ext:
                answers[qid]["action"] = ext
        return {"model": "caldec-npu-packed", "answers": answers,
                "usage": {"input_tokens": ntok, "output_tokens": 0, "npu_passes": len(self.last_npu_ms),
                          "npu_ms": round(sum(self.last_npu_ms), 1)}}


if __name__ == "__main__":
    # CLI: one JSON object per line on stdin: {"state": ..., "questions": {...}}; one JSON answer per line on stdout.
    import sys
    m = CalDecNPU(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
    try:
        for line in sys.stdin:
            if line.strip():
                t0 = time.perf_counter(); req = json.loads(line)
                out = m.system_one(req["state"], req["questions"])
                out["usage"]["total_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
                print(json.dumps(out), flush=True)
    finally:
        m.close()
