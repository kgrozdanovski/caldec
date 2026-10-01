# Hosted typed-decision reference: Jev and Kev

TypeSafe Jev 1.13 and the open-weight Kev 4B were measured through OpenRouter's Decisions endpoint as zero-shot references. Their outputs were used for evaluation only, never as training labels. The reusable client and scorer are in `scripts/jev_decisions.py`.

## Full assistant test

Jev answered all **1,647 cases and 3,452 decisions** in the released test split, including **3,372 untied targets**. The local checkpoints below were scored on those exact cases and targets. Accuracy and ECE exclude tied targets; distribution metrics include all decisions.

| Model | Accuracy | Soft accuracy | Brier | Soft NLL | ECE |
|---|---:|---:|---:|---:|---:|
| CalDec GLiNER | 0.843 | 0.859 | 0.039 | 0.552 | 0.039 |
| Jev 1.13, zero-shot | 0.836 | 0.831 | 0.044 | 0.811 | 0.042 |
| CalDec Laya | 0.823 | 0.839 | 0.041 | 0.559 | 0.072 |
| GLiNER2.5-Decide | 0.650 | 0.664 | 0.105 | 0.758 | 0.097 |
| Laya specialist | 0.578 | 0.655 | 0.123 | 0.805 | 0.041 |
| Laya base | 0.557 | 0.643 | 0.161 | 0.989 | 0.158 |

`Laya base` is [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya). `Laya specialist` is [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions), which used the Typed Decisions train split.

Jev's 0.65-point gap behind CalDec GLiNER is not significant in the paired exact McNemar test (p = 0.4072); its 1.33-point lead over CalDec Laya has p = 0.0898. Jev's two-decimal output precision raises its standard soft NLL when it assigns zero probability to a target option. Flooring each probability at 0.005 and renormalizing gives a second reading of **0.599** soft NLL on this test. The recorded OpenRouter charge for the full assistant-test run was **$0.0307**; that is a run record, not a future price quote.

## `LocalLLaMA/typed-decisions` test

The [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) test split has 400 cases, 2,000 decisions and 1,965 untied targets. Accuracy and ECE below exclude tied targets; distribution metrics use all decisions.

| Model | Accuracy | Soft accuracy | Brier | Soft NLL | ECE |
|---|---:|---:|---:|---:|---:|
| CalDec Laya, trained on the public train split | 0.778 | 0.860 | 0.015 | 0.858 | 0.149 |
| Laya specialist | 0.773 | 0.826 | 0.018 | 0.884 | 0.218 |
| Jev 1.13, zero-shot | 0.738 | 0.750 | 0.041 | 2.272 | 0.045 |
| Kev 4B, zero-shot | 0.668 | — | — | — | — |
| CalDec GLiNER | 0.582 | 0.709 | 0.061 | 1.089 | 0.091 |
| GLiNER2.5-Decide | 0.540 | 0.662 | 0.060 | 1.123 | 0.147 |
| Laya base | 0.361 | 0.590 | 0.101 | 1.344 | 0.174 |

Jev's probabilities are returned to two decimal places and can be exactly zero or one. A wrong certainty is heavily penalized under the evaluator's 1e-12 probability clip. Flooring each probability at 0.005 and renormalizing gives Jev a soft NLL of **1.151** on this split. That is an additional reading, not a replacement for the standard score. Jev's measured accuracy on all 2,000 decisions, including option-order tie breaks, is **0.732**; the provider's published figure is **0.727** under its own tie convention.

Kev's saved assistant-case measurement does not cover the full release test, so it is not in the full-test table.

## Measurement path

The script sends one request per case, with all its questions, to `https://openrouter.ai/api/alpha/decisions`. A `noul` answer is P(yes); `choice` and `score` answers contain probabilities keyed by option. The responses go through the same target ordering and metric code as the local checkpoints. Calls are journalled and cached under ignored `runs/` files, with a spend cap. Set `LLM_OPENROUTER_API_KEYS` in the process environment; no credential is written to the journal.

```bash
LLM_OPENROUTER_API_KEYS=... python3 scripts/jev_decisions.py run \
  --model typesafe/jev-1.13 --name jev --out runs/my-jev --sets internal,public
```

Network latency measured from the run location is not directly comparable with a resident local model's forward pass. Provider aliases, pricing and terms can change. Check them when reproducing the run. Jev and Kev outputs must remain evaluation-only under their provider terms. Only aggregate Jev summaries are tracked; raw responses and per-decision predictions remain local.
