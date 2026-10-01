# Evaluation results

All assistant-test figures use the same released test split. The CalDec rows use the release weight files in `checkpoints/v1-laya/final` and `checkpoints/v1-gliner/final`; the base rows use pinned upstream checkpoints. Jev was queried through OpenRouter. The reports are summarized in `runs/numbers.json`. A fresh clone can run `python3 scripts/check_evidence.py` to recalculate local-model scores from committed prediction vectors. Jev's aggregate report is committed without its per-decision outputs.

## Scoring contract

A case may contain several typed questions. The test split contains **1,647 cases and 3,452 decisions**; **3,372** decisions have a unique highest-probability target. Hard accuracy and expected calibration error (ECE) use those untied decisions. Soft accuracy, Brier score, soft negative log-likelihood (NLL), and score mean absolute error use every decision. `noul` options are ordered `false, true`. Accuracy means agreement with the synthetic target's argmax, not independent correctness. The test split was used during development.

## Full assistant test

| Model | Accuracy | Soft accuracy | Brier | Soft NLL | ECE | Score MAE |
|---|---:|---:|---:|---:|---:|---:|
| **CalDec GLiNER** | **0.843** | **0.859** | **0.039** | **0.552** | **0.039** | **0.298** |
| Jev 1.13, zero-shot via OpenRouter | 0.836 | 0.831 | 0.044 | 0.811 | 0.042 | 0.354 |
| **CalDec Laya** | 0.823 | 0.839 | 0.041 | 0.559 | 0.072 | 0.362 |
| GLiNER2.5-Decide | 0.650 | 0.664 | 0.105 | 0.758 | 0.097 | 0.734 |
| Laya specialist | 0.578 | 0.655 | 0.123 | 0.805 | 0.041 | 0.737 |
| Laya base | 0.557 | 0.643 | 0.161 | 0.989 | 0.158 | 0.753 |

CalDec GLiNER exceeds GLiNER2.5-Decide by **19.25 accuracy points**; CalDec Laya exceeds Laya base by **26.57 points**. Jev falls between the two CalDec models. CalDec GLiNER alone is correct on 332 untied decisions where Jev is wrong, and Jev alone on 310 where CalDec GLiNER is wrong (exact paired McNemar p = **0.4072**). CalDec Laya alone is correct on 314 where Jev is wrong, and Jev alone on 359 where CalDec Laya is wrong (p = **0.0898**). These two small accuracy gaps are not statistically resolved by this test.

For both GLiNER rows, this repository normalizes each question's independent sigmoid scores to sum to one. Softmax-based evaluations can report different ECE, Brier and soft NLL; argmax accuracy is unchanged. Compare probability metrics only within this repository's scoring contract.

CalDec GLiNER and CalDec Laya predict different labels on 469 untied decisions. Exactly one is correct on 439: CalDec GLiNER alone on 253 and CalDec Laya alone on 186; the two-sided exact McNemar p-value is **0.0016**. They have different backbones, losses, training schedules and public-data use, so this is a comparison of released recipes, not an isolated backbone effect. Both are one-seed measurements.

The majority-class baseline, fitted on the training split and weighted by decision, scores **0.574**. The saved benchmark report gives the individual Laya and baseline metrics; `scripts/evaluate.py --compare` gives the paired test.

## `LocalLLaMA/typed-decisions` test

[**Typed Decisions** (`LocalLLaMA/typed-decisions`)](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) test contains 400 cases, 2,000 decisions and 1,965 untied targets. All rows below were scored against the same pinned test revision and target vectors. CalDec Laya includes this benchmark's *train* split in its training recipe; CalDec GLiNER does not.

| Model | Accuracy | Soft accuracy | Brier | Soft NLL | ECE |
|---|---:|---:|---:|---:|---:|
| **CalDec Laya** | **0.778** | **0.860** | **0.015** | **0.858** | 0.149 |
| Laya specialist | 0.773 | 0.826 | 0.018 | 0.884 | 0.218 |
| TypeSafe Jev 1.13, zero-shot via OpenRouter | 0.738 | 0.750 | 0.041 | 2.272 | **0.045** |
| **CalDec GLiNER** | 0.582 | 0.709 | 0.061 | 1.089 | 0.091 |
| GLiNER2.5-Decide | 0.540 | 0.662 | 0.060 | 1.123 | 0.147 |
| Laya base | 0.361 | 0.590 | 0.101 | 1.344 | 0.174 |

CalDec GLiNER improves on GLiNER2.5-Decide by **4.17 accuracy points** here, but trails CalDec Laya by **19.59 points**. This public comparison includes different training data for the two CalDec checkpoints. Jev was also measured on the full assistant test above. Its two-decimal probabilities can be exactly zero or one, which strongly affects soft NLL; [JEV.md](JEV.md) explains the scoring and the 0.005-floored alternative. The Jev outputs were evaluation evidence only, never training data.

## By assistant decision site

Accuracy on untied targets. The per-site sample size is the number of untied test decisions. This table describes the saved full-test predictions; small differences by site need their own uncertainty analysis before deployment decisions.

| Site | Untied | CalDec Laya | Laya base | CalDec GLiNER |
|---|---:|---:|---:|---:|
| `ambient.react` | 70 | 0.643 | 0.429 | 0.557 |
| `ambient.salience` | 398 | 0.741 | 0.523 | 0.791 |
| `compute.irreversible` | 217 | 0.705 | 0.415 | 0.765 |
| `gateway.refusal` | 77 | 0.883 | 0.364 | 0.831 |
| `memory.preference` | 190 | 0.805 | 0.642 | 0.768 |
| `memory.session_split` | 82 | 0.890 | 0.622 | 0.878 |
| `memory.triage` | 288 | 0.795 | 0.608 | 0.861 |
| `scheduled.speak` | 135 | 0.852 | 0.637 | 0.874 |
| `skill.review` | 335 | 0.794 | 0.382 | 0.848 |
| `task.route` | 75 | 0.907 | 0.707 | 0.920 |
| `tool.risk` | 224 | 0.848 | 0.496 | 0.839 |
| `tools.injection` | 102 | 0.931 | 0.608 | 0.902 |
| `turn.tier` | 202 | 0.837 | 0.530 | 0.842 |
| `turn.toolsets` | 398 | 0.912 | 0.729 | 0.930 |
| `voice.addressee` | 138 | 0.855 | 0.464 | 0.891 |
| `voice.backchannel` | 76 | 0.934 | 0.684 | 0.921 |
| `voice.endpoint` | 222 | 0.860 | 0.635 | 0.869 |
| `workflow.outcome` | 143 | 0.790 | 0.566 | 0.804 |

## Reproduce and inspect

```bash
USE_TF=0 python3 scripts/benchmark.py \
  --model caldec-laya=checkpoints/v1-laya/final \
  --model laya=convaiinnovations/laya \
  --model laya-td=convaiinnovations/laya-typed-decisions \
  --data data --out runs/my-benchmark
python3 scripts/gliner2_transfer.py eval --model checkpoints/v1-gliner/final \
  --out runs/my-gliner.json --predictions runs/my-gliner.predictions.json
python3 scripts/gliner2_transfer.py zeroshot --model fastino/GLiNER2.5-Decide \
  --out runs/my-gliner2.5-decide.json --predictions runs/my-gliner2.5-decide.predictions.json
python3 scripts/gliner2_transfer.py eval --model checkpoints/v1-gliner/final \
  --split public --out runs/my-gliner-public.json \
  --predictions runs/my-gliner-public.predictions.json
python3 scripts/gliner2_transfer.py zeroshot --model fastino/GLiNER2.5-Decide \
  --split public --out runs/my-decide-public.json \
  --predictions runs/my-decide-public.predictions.json
```

`runs/evidence/` contains the tracked local-model reports, compact prediction files and Jev aggregate summaries. Hosted response caches and per-decision Jev outputs stay under ignored `runs/`. New benchmark runs write their reports and predictions under ignored `runs/` paths. Use `python3 scripts/evaluate.py --compare a.json b.json` on compatible prediction files. The exact state overlap reported by `scripts/check_data.py` is a data limitation; cases and targets were not changed to make a result look cleaner.
