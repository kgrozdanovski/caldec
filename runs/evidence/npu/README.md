# Evidence: CalDec Ultra 150M on the RK3566 NPU

The answers behind the numbers in [NPU.md](../../../NPU.md). Measured on 2026-11-24 on one Panther X2 (RK3566,
4 GB), kernel 6.18, librknnrt 2.5.0, through the installed `caldec-npu` service (which sets the CPU governor to
`performance` while it runs), with the model set listed in `model-set.SHA256SUMS`.

| File | What |
|---|---|
| `assistant400.npu.jsonl` | NPU answers for the first 400 assistant-test cases (858 questions with gold labels), one line per case, with `usage` (`total_ms`, `npu_ms`, `npu_passes`, `input_tokens`) |
| `assistant400.fp32.jsonl` | `PackedAgent.system_one` answers of the same checkpoint in fp32 on the CPU, `max_len` 640 (every case fits), with the gold probabilities |
| `public200.npu.jsonl` | NPU answers for the first 200 cases of `LocalLLaMA/typed-decisions` test, at the pinned revision |
| `public200.fp32-split.txt` | fp32 accuracy on the same 200 cases: one pass at a given `max_len`, and the runner's split rule (`runner/exp_long_cases.py`) |
| `model-set.SHA256SUMS` | sha256 of every file of the published model set |

Recompute from `scripts/npu/`:

```
python3 runner/compare_answers.py ../../runs/evidence/npu/assistant400.npu.jsonl ../../runs/evidence/npu/assistant400.fp32.jsonl
python3 runner/compare_act.py     ../../runs/evidence/npu/assistant400.npu.jsonl ../../runs/evidence/npu/assistant400.fp32.jsonl
$CALDEC_PY runner/make_public200.py      # writes runner/gold_public200.jsonl (and the requests)
python3 runner/acc_public.py ../../runs/evidence/npu/public200.npu.jsonl
```

Expected: 856/858 equal answers, max |Δp| 0.0121, accuracy 0.8489 for both; `act_probability` max |Δ| 0.0006,
858/858 on the same side of 0.5; public test 0.7462 on 981 untied labels.
