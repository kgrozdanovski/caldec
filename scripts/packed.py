"""Packed decisions: every question of a case in one encoder pass over one copy of the state.

Laya's sequence holds one question: [CLS] question [SEP] [MASK] opt ... [SEP] state [SEP].
A case with three questions therefore encodes its state three times, which is free at
130 tokens and dominant at 8K (an 8K state with three questions costs three 8K passes).
The packed sequence holds them all:

    [CLS] q1 [SEP] [MASK] o [MASK] o [SEP] q2 [SEP] [MASK] o ... [SEP] ... state [SEP]

The encoder runs once. The decision head then runs per question, as it does in Laya,
but over [CLS] and that question's own segment instead of the whole sequence: the
state reaches the head through the encoder, and each question's options never mix
with another's. Every parameter is the one `laya.common.DecisionModel` already has,
so a Laya-format checkpoint initializes it unchanged; fine-tuning teaches the head the
narrower view. A checkpoint trained this way records `"packed": true` in
`rl_agent_config.json`, and `data.load_agent` then returns a `PackedAgent`, whose
`system_one` matches `laya.Agent.system_one`'s output format.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch


def question_segment(tok, q: Dict, head_max_len: int):
    """The token ids of one question's segment and its marker offsets, built exactly as
    `laya.common.build_sequence` builds the part before the state."""
    from laya.common import build_sequence

    seq, markers = build_sequence(tok, "", q, 10 ** 9, head_max_len)
    # build_sequence returns [CLS] head [SEP] options [SEP] <empty state> [SEP]; keep the
    # middle: head [SEP] options [SEP].
    body = seq[1:-1]
    return body, [m - 1 for m in markers]


def build_case(tok, state, questions: List[Dict], max_len: int, head_max_len: int):
    """One packed sequence for a case: ids, and per question the absolute positions of
    its head view ([CLS] + its segment) and its markers' offsets in that view."""
    from laya.common import serialize_state

    ids = [tok.cls_token_id]
    views, markers = [], []
    for q in questions:
        body, offsets = question_segment(tok, q, head_max_len)
        start = len(ids)
        ids.extend(body)
        views.append([0] + list(range(start, start + len(body))))
        markers.append([1 + o for o in offsets])
    room = max(0, max_len - len(ids) - 1)
    st = tok(serialize_state(state).replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"][:room]
    ids = ids + st + [tok.sep_token_id]
    return ids, views, markers


def collate_cases(cases: List[Dict[str, Any]], pad_id: int, device) -> Dict[str, torch.Tensor]:
    """Pad packed cases. Each case has `ids`, `views`, `markers`, `qtype` and optionally
    `target` (lists, one entry per question)."""
    bs, L = len(cases), max(len(c["ids"]) for c in cases)
    qs = [(b, j) for b, c in enumerate(cases) for j in range(len(c["views"]))]
    S = max(len(cases[b]["views"][j]) for b, j in qs)
    K = max(len(cases[b]["markers"][j]) for b, j in qs)
    ids = torch.full((bs, L), pad_id, dtype=torch.long)
    att = torch.zeros((bs, L), dtype=torch.long)
    view = torch.zeros((len(qs), S), dtype=torch.long)
    view_mask = torch.zeros((len(qs), S), dtype=torch.bool)
    mpos = torch.zeros((len(qs), K), dtype=torch.long)
    mmask = torch.zeros((len(qs), K), dtype=torch.bool)
    qtype = torch.zeros(len(qs), dtype=torch.long)
    has_target = "target" in cases[0]
    target = torch.zeros((len(qs), K), dtype=torch.float32)
    for b, c in enumerate(cases):
        ids[b, : len(c["ids"])] = torch.tensor(c["ids"])
        att[b, : len(c["ids"])] = 1
    for n, (b, j) in enumerate(qs):
        v, m = cases[b]["views"][j], cases[b]["markers"][j]
        view[n, : len(v)] = torch.tensor(v) + b * L          # flat index into (bs * L)
        view_mask[n, : len(v)] = True
        mpos[n, : len(m)] = torch.tensor(m)
        mmask[n, : len(m)] = True
        qtype[n] = cases[b]["qtype"][j]
        if has_target:
            t = cases[b]["target"][j]
            target[n, : len(t)] = torch.tensor(t, dtype=torch.float32)
    out = dict(input_ids=ids, attention_mask=att, view=view, view_mask=view_mask,
               marker_pos=mpos, marker_mask=mmask, qtype=qtype)
    if has_target:
        out["target"] = target
    return {k: v.to(device) for k, v in out.items()}


def forward(model, batch) -> tuple:
    """Option logits [questions, K] and act logits for a packed batch, using the
    weights of a `laya.common.DecisionModel`."""
    h = model.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    hq = h.reshape(-1, h.size(-1))[batch["view"]]                     # questions, S, d
    hq = hq + model.type_emb(batch["qtype"])[:, None, :]
    if model.head is not None:
        pad = ~batch["view_mask"]
        for layer in model.head.layers:
            hq = layer(hq, src_key_padding_mask=pad)
    idx = batch["marker_pos"][:, :, None].expand(-1, -1, hq.size(-1))
    m = torch.gather(hq, 1, idx)
    logits = model.scorer(m).squeeze(-1).float().masked_fill(~batch["marker_mask"], -1e4)
    p = torch.softmax(logits.detach(), -1)
    k = batch["marker_mask"].sum(-1).clamp(min=2).float()
    ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
    top2 = p.topk(2, -1).values if p.size(-1) >= 2 else torch.cat([p, torch.zeros_like(p)], -1)
    feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
    act = model.act_head(torch.cat([hq[:, 0].float(), feats], -1))
    return logits, act


def case_items(rows, tok, cfg) -> List[Dict[str, Any]]:
    """One packed training item per case, with the per-question targets that
    `data.build_items` would give; a question whose markers do not match its options
    is dropped from its case."""
    from laya.agent import Agent
    from laya.common import QTYPES, render_options

    items, dropped = [], 0
    for row in rows:
        qs, targets, qtypes, qids = [], [], [], []
        for qid, question in row["questions"].items():
            gold = (row.get("gold") or {}).get(qid)
            if not gold or "probabilities" not in gold:
                continue
            q = Agent._to_internal(question)
            keys = (["false", "true"] if q["t"] == "noul" else list(q["crit"] or {}) if q["t"] == "choice"
                    else [str(i) for i in range(len(q["crit"] or []))])
            target = [float(gold["probabilities"].get(k, 0.0)) for k in keys]
            if sum(target) <= 0 or len(render_options(q)) != len(target):
                dropped += 1
                continue
            qs.append(q)
            targets.append([v / sum(target) for v in target])
            qtypes.append(QTYPES[q["t"]])
            qids.append(qid)
        if not qs:
            continue
        ids, views, markers = build_case(tok, row["state"], qs, cfg["max_len"], cfg["head_max_len"])
        keep = [j for j in range(len(qs)) if len(markers[j]) == len(targets[j])]
        dropped += len(qs) - len(keep)
        if not keep:
            continue
        items.append(dict(ids=ids, views=[views[j] for j in keep], markers=[markers[j] for j in keep],
                          qtype=[qtypes[j] for j in keep], target=[targets[j] for j in keep],
                          questions=[qids[j] for j in keep], site=row["site"], case_id=row["case_id"]))
    if dropped:
        print(f"  note: dropped {dropped} question(s) whose markers did not match their options.")
    return items


class PackedAgent:
    """`system_one` over a packed checkpoint; wraps the `laya.Agent` that loaded it."""

    def __init__(self, agent):
        self.agent = agent
        self.model, self.tok, self.cfg, self.device = agent.model, agent.tok, agent.cfg, agent.device
        self.dtype = getattr(agent, "dtype", torch.bfloat16)

    def __getattr__(self, name):
        return getattr(self.agent, name)

    @torch.no_grad()
    def system_one(self, state, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        from laya.common import QTYPES, confidence_from_probs, render_options, temp_bucket

        qids = list(questions)
        qs = [self.agent._to_internal(questions[k]) for k in qids]
        ids, views, markers = build_case(self.tok, state, qs, self.cfg.get("max_len", 512),
                                         self.cfg.get("head_max_len", 192))
        for qid, q, m in zip(qids, qs, markers):
            if len(m) != len(render_options(q)):
                raise ValueError("question %r options exceed head_max_len=%d" % (qid, self.cfg.get("head_max_len", 192)))
        case = dict(ids=ids, views=views, markers=markers, qtype=[QTYPES[q["t"]] for q in qs])
        batch = collate_cases([case], self.tok.pad_token_id, self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            logits, act = forward(self.model, batch)
        logits = logits.float().cpu().numpy()
        act = torch.softmax(act.float(), -1).cpu().numpy()
        answers = {}
        for r, (qid, q) in enumerate(zip(qids, qs)):
            k = len(markers[r])
            qt = QTYPES[q["t"]]
            t_scale = self.agent.temperature_by_options.get(temp_bucket(qt, k), self.agent.temperature[qt])
            z = logits[r, :k] / t_scale
            p = np.exp(z - z.max())
            p = p / p.sum()
            conf = round(confidence_from_probs(p, k), 4)
            ext = {"act_probability": round(float(act[r, 0]), 4)}
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {"type": "choice", "choice": keys[int(p.argmax())],
                                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                                "confidence": conf, "action": ext}
            elif q["t"] == "score":
                answers[qid] = {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                                "confidence": conf, "action": ext}
            else:
                answers[qid] = {"type": "noul", "noul": round(float(p[1]), 4),
                                "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4), "action": ext}
        return {"model": "laya-rl-agent", "answers": answers,
                "usage": {"input_tokens": len(ids), "output_tokens": 0}}
