"""A Decision Index engine for CalDec Laya-format checkpoints.

The Decision Index kit (github.com/apolinario/decision-index) runs any engine with
`__call__(state, questions) -> (response, raw)`; `laya.Agent.system_one` already
speaks that wire format. This adapter loads a checkpoint through `data.load_agent`
and, following the kit's no-truncation rule, raises `Unsupported` for a question
whose sequence would not fit `max_len` instead of cutting the state.

    PYTHONPATH=<kit>:scripts python -m decision_index run --engine jdi_engine:CalDecEngine \\
        --option checkpoint=checkpoints/caldec-ultra-150m --option max_len=8192 \\
        --rows <rows.jsonl.gz> --out runs/jdi/caldec-ultra-150m
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decision_index.engines.base import Engine, Unsupported  # noqa: E402


class CalDecEngine(Engine):
    name = "caldec"

    def __init__(self, checkpoint: str, max_len: int = 512, device: str | None = None,
                 batch_tokens: int = 24576, **options):
        super().__init__(checkpoint=checkpoint, max_len=max_len, batch_tokens=batch_tokens, **options)
        self.batch_tokens = int(batch_tokens)
        from data import load_agent
        from laya.common import build_sequence, serialize_state

        self.agent = load_agent(checkpoint, device=device)
        self.on_gpu = self.agent.device.type == "cuda"
        self.agent.cfg["max_len"] = int(max_len)
        self.max_len = int(max_len)
        self._build, self._serialize = build_sequence, serialize_state
        self.provenance = {"kind": "caldec", "checkpoint": checkpoint, "max_len": self.max_len,
                           "head_max_len": self.agent.cfg.get("head_max_len", 192)}

    def __call__(self, state, questions):
        tok = self.agent.tok
        state_len = len(tok(self._serialize(state), add_special_tokens=False)["input_ids"])
        for qid, q in questions.items():
            internal = self.agent._to_internal(q)
            seq, markers = self._build(tok, "", internal, 10 ** 9, self.agent.cfg.get("head_max_len", 192))
            if len(seq) + state_len > self.max_len:
                raise Unsupported(f"{qid}: {len(seq) + state_len} tokens exceed max_len {self.max_len}")
        # A per-question checkpoint encodes the state once per question, so a case with
        # many questions over a long state is answered in groups that fit `batch_tokens`;
        # each question is still its own sequence, so the answers do not change. A packed
        # checkpoint takes the whole case in one sequence.
        packed = hasattr(self.agent, "agent")
        per_call = len(questions) if packed else max(1, self.batch_tokens // max(1, state_len + 256))
        keys = list(questions)
        out = {"answers": {}, "usage": {"input_tokens": 0}}
        for start in range(0, len(keys), per_call):
            part = {k: questions[k] for k in keys[start:start + per_call]}
            try:
                res = self.agent.system_one(state, part)
            except ValueError as exc:            # options beyond head_max_len
                raise Unsupported(str(exc)) from None
            if self.on_gpu and self.agent.device.type == "cpu":
                raise RuntimeError("laya fell back to CPU after a GPU out-of-memory error")
            out["answers"].update(res["answers"])
            out["usage"]["input_tokens"] += res.get("usage", {}).get("input_tokens", 0)
        answers = {}
        for qid, a in out["answers"].items():
            if a["type"] == "choice":
                answers[qid] = {"type": "choice", "choice": a["choice"], "probabilities": a["probabilities"]}
            else:
                answers[qid] = {"type": "noul", "noul": a["noul"]}
        return {"model": self.provenance["checkpoint"], "answers": answers,
                "usage": out.get("usage", {})}, None

    def synchronize(self):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()


class UltraZeroShotEngine(Engine):
    """Raw UltraBERT through `ultra.ZeroShotAgent` (the paper's one-mask MCQA readout),
    the base-model reference for the CalDec Ultra models. A state longer than
    `max_len` raises `Unsupported` instead of being cut."""
    name = "ultrabert-zeroshot"

    def __init__(self, model: str = "dmis-lab/UltraBERT-380M", max_len: int = 8192, device: str = "cuda", **options):
        super().__init__(model=model, max_len=max_len, **options)
        import ultra

        ultra.register()
        self.agent = ultra.ZeroShotAgent(model, device, max_len=int(max_len))
        self.max_len = int(max_len)
        self.provenance = {"kind": "zero-shot masked LM", "model": model, "max_len": self.max_len}

    def __call__(self, state, questions):
        from laya.common import serialize_state

        if len(self.agent.tok(serialize_state(state), add_special_tokens=False)["input_ids"]) + 256 > self.max_len:
            raise Unsupported(f"state exceeds max_len {self.max_len}")
        out = self.agent.system_one(state, questions)["answers"]
        answers = {}
        for qid, q in questions.items():
            a = out[qid]
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": float(a["noul"])}
            else:
                p = {k: float(v) for k, v in a["probabilities"].items()}
                answers[qid] = {"type": "choice", "choice": max(p, key=p.get), "probabilities": p}
        return {"model": self.provenance["model"], "answers": answers}, None
