# CalDec GLiNER transfer

The same typed-decision data also trains CalDec GLiNER from [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) checkpoint. The local release weights are at `checkpoints/v1-gliner/final`, and the Hub destination is [`kgrozdanovski/caldec-v1-gliner2.5-decide`](https://huggingface.co/kgrozdanovski/caldec-v1-gliner2.5-decide).

## Full assistant test

The test contains **1,647 cases, 3,452 decisions and 3,372 untied targets**. All rows use the scoring contract in [RESULTS.md](RESULTS.md).

| Model | Accuracy | Soft accuracy | Brier | Soft NLL | ECE | Score MAE |
|---|---:|---:|---:|---:|---:|---:|
| **CalDec GLiNER** | **0.843** | **0.859** | **0.039** | **0.552** | **0.039** | **0.298** |
| CalDec Laya | 0.823 | 0.839 | 0.041 | 0.559 | 0.072 | 0.362 |
| GLiNER2.5-Decide | 0.650 | 0.664 | 0.105 | 0.758 | 0.097 | 0.734 |

CalDec GLiNER alone is correct on 253 untied decisions where CalDec Laya is wrong; CalDec Laya alone is correct on 186 where CalDec GLiNER is wrong (exact McNemar p = **0.0016**). Both are one-seed results. The two CalDec models differ in backbone, loss, training schedule, site weighting, state rendering and use of the `LocalLLaMA/typed-decisions` train split. This comparison shows the performance of the two released recipes, not a controlled estimate of the backbone's effect.

## `LocalLLaMA/typed-decisions` test

The pinned [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) test has 400 cases, 2,000 decisions and 1,965 untied targets. CalDec Laya includes the benchmark train split in its recipe; CalDec GLiNER uses only the assistant dataset train split.

| Model | Accuracy | Soft accuracy | Brier | Soft NLL | ECE |
|---|---:|---:|---:|---:|---:|
| CalDec Laya | 0.778 | 0.860 | 0.015 | 0.858 | 0.149 |
| CalDec GLiNER | 0.582 | 0.709 | 0.061 | 1.089 | 0.091 |
| GLiNER2.5-Decide | 0.540 | 0.662 | 0.060 | 1.123 | 0.147 |

CalDec GLiNER exceeds GLiNER2.5-Decide by 4.17 accuracy points on this test. The two GLiNER runs use identical cases, targets and scoring code.

This evaluator normalizes GLiNER's independent sigmoid scores per question. Softmax-based evaluations can yield different ECE and other probability metrics; argmax accuracy is unchanged.

## Training and inference

GLiNER2's public classification trainer uses hard labels. `scripts/gliner2_transfer.py` patches its example conversion and target construction to carry the full teacher distribution into binary cross-entropy. It disables label augmentation so labels cannot be dropped or renamed. The script validates the soft-target round trip before starting an update.

```bash
python3 scripts/gliner2_transfer.py train --soft --out checkpoints/my-gliner-run
python3 scripts/gliner2_transfer.py eval --model checkpoints/v1-gliner/final \
  --out runs/my-gliner.json --predictions runs/my-gliner.predictions.json
python3 scripts/gliner2_transfer.py eval --model checkpoints/v1-gliner/final \
  --split public --out runs/my-gliner-public.json
python3 scripts/gliner2_transfer.py zeroshot --model fastino/GLiNER2.5-Decide \
  --split public --out runs/my-decide-public.json
```

The release recipe uses the assistant dataset train split, three epochs, batch 2 with accumulation 8, encoder LR 1e-5, task LR 5e-4 and bf16. The served model returns independent sigmoid scores; normalize each question's label scores to sum to one, as the evaluator does. Render a state with `render` in the script for the input representation used in the reported test.

CalDec GLiNER is larger than CalDec Laya (about 486M versus 421M parameters). The recipes target one 16 GB CUDA GPU. The release training logs do not retain the actual training GPU or elapsed time, so there is no measured training-time claim.

Resident-model latency was measured on an **NVIDIA GeForce RTX 4080 SUPER (16 GB)** using the same Assistant Decisions case, 10 warm-up calls and 40 timed calls per question count, with CUDA synchronized around each call. The compact reports and the runnable `scripts/latency.py` are in `runs/evidence/` and `scripts/`.

| Model | One question p50 | One question p95 | Three questions p50 | Three questions p95 |
|---|---:|---:|---:|---:|
| CalDec Laya | 16.9 ms | 17.7 ms | 18.3 ms | 19.3 ms |
| CalDec GLiNER | 23.7 ms | 24.6 ms | 24.1 ms | 25.8 ms |

These are single-case latency measurements, not throughput or a deployment guarantee.

## Limits

Accuracy is agreement with a synthetic teacher, not a claim that either model is correct or safe in production. The test split informed development. The security examples were generated, not collected from adversaries. The split files include a few exact state overlaps, reported by `scripts/check_data.py`.
