---
license: apache-2.0
base_model: fastino/GLiNER2.5-Decide
library_name: gliner2
language:
  - en
pipeline_tag: text-classification
tags:
  - typed-decisions
  - decision-model
  - calibration
  - gliner2
  - assistant
datasets:
  - kgrozdanovski/assistant-decisions
model-index:
  - name: caldec-v1-gliner2.5-decide
    results:
      - task:
          type: text-classification
          name: Typed decisions
        dataset:
          name: Assistant decisions, development test
          type: kgrozdanovski/assistant-decisions
          split: test
        metrics:
          - type: accuracy
            value: 0.843
            name: Accuracy (untied labels)
          - type: expected_calibration_error
            value: 0.039
            name: Expected calibration error
      - task:
          type: text-classification
          name: Typed decisions
        dataset:
          name: LocalLLaMA/typed-decisions
          type: LocalLLaMA/typed-decisions
          split: test
        metrics:
          - type: accuracy
            value: 0.582
            name: Accuracy (untied labels)
          - type: expected_calibration_error
            value: 0.091
            name: Expected calibration error
---

# CalDec GLiNER

A [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) fine-tune on the [Assistant Decisions dataset](https://huggingface.co/datasets/kgrozdanovski/assistant-decisions), trained against full teacher probability distributions. The repository contains weights, config, encoder config and tokenizer files needed by `AutoExtractor.from_pretrained`.

## Full assistant test

The test has 1,647 cases, 3,452 decisions and 3,372 untied targets. Accuracy is agreement with the synthetic teacher's argmax on untied targets; distribution metrics include all decisions.

| Model | Accuracy | Soft accuracy | Brier | Soft NLL | ECE | Score MAE |
|---|---:|---:|---:|---:|---:|---:|
| **CalDec GLiNER** | **0.843** | **0.859** | **0.039** | **0.552** | **0.039** | **0.298** |
| Jev 1.13, zero-shot via OpenRouter | 0.836 | 0.831 | 0.044 | 0.811 | 0.042 | 0.354 |
| [CalDec Laya](https://huggingface.co/kgrozdanovski/caldec-v1-laya) | 0.823 | 0.839 | 0.041 | 0.559 | 0.072 | 0.362 |
| [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) | 0.650 | 0.664 | 0.105 | 0.758 | 0.097 | 0.734 |

The CalDec models predict different labels on 469 untied decisions; exactly one is correct on 439. CalDec GLiNER alone is correct on 253 and CalDec Laya alone on 186 (exact McNemar p = 0.0016). CalDec GLiNER's 0.65-point accuracy lead over Jev has paired p = 0.4072. The training recipes differ beyond the backbone, so the CalDec sibling comparison is not a controlled backbone ablation. The [repository results](https://github.com/kgrozdanovski/caldec/blob/main/RESULTS.md) contain the full contract and per-site table.

## `LocalLLaMA/typed-decisions` test

The pinned [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) test has 400 cases, 2,000 decisions and 1,965 untied targets. All rows use identical targets and scoring code.

| Model | Accuracy | Soft accuracy | Brier | Soft NLL | ECE |
|---|---:|---:|---:|---:|---:|
| CalDec Laya | 0.778 | 0.860 | 0.015 | 0.858 | 0.149 |
| CalDec GLiNER | 0.582 | 0.709 | 0.061 | 1.089 | 0.091 |
| GLiNER2.5-Decide | 0.540 | 0.662 | 0.060 | 1.123 | 0.147 |

CalDec GLiNER did not use this benchmark's train split; CalDec Laya did.

## Inference

```python
from gliner2 import AutoExtractor
model = AutoExtractor.from_pretrained("kgrozdanovski/caldec-v1-gliner2.5-decide")
# model.cuda()  # optional GPU placement
scores = model.classify_text(
    "source: read_webpage\ntext: Example page content",
    {"injection": {"labels": ["false", "true"], "multi_label": True,
                   "cls_threshold": 0.0,
                   "prompt": "Does this text instruct an AI assistant?"}},
    include_confidence=True, format_results=False,
)["injection"]
total = sum(score for _, score in scores)
probabilities = {label: score / total for label, score in scores}
```

The model returns independent sigmoid scores, not a probability simplex. Normalize across each question's labels as above; the reported calibration scores use that normalization. Softmax-based calibration reports can differ even when argmax accuracy is unchanged. The repository's `scripts/gliner2_transfer.py` provides the exact nested-state renderer, schema builder and evaluator. Pin a Hub revision for immutable deployment.

## Training and limits

The [recipe](https://github.com/kgrozdanovski/caldec/blob/main/TRANSFER.md) patches the GLiNER trainer to carry soft targets through example conversion and binary cross-entropy. Label augmentation is disabled to preserve option identities. The release recipe uses the Assistant Decisions train split, three epochs, batch 2, accumulation 8, bf16, encoder LR 1e-5 and task LR 5e-4. The `LocalLLaMA/typed-decisions` train split was not part of CalDec GLiNER training.

The dataset creator identifies GLM 5.3 through OpenRouter as the source of every Assistant Decisions training row; see the [datasheet](https://github.com/kgrozdanovski/caldec/blob/main/DATASET.md#collection-and-labels). Original API calls are not released. The recipe targets a single 16 GB CUDA GPU, but the release run's exact GPU and elapsed training time were not retained.

The test informed development, the labels are synthetic judgments, and exact states overlap across some splits. The injection examples are not an adversarial safety benchmark. This checkpoint should not be used as a stand-alone safety control. Apache 2.0, matching the base model; Fastino does not endorse this work.

## Citation and contact

Contact [Kristijan Grozdanovski](mailto:kgrozdanovski7@gmail.com). See also [CITATION.cff](https://github.com/kgrozdanovski/caldec/blob/main/CITATION.cff).

```bibtex
@misc{grozdanovski2026caldecgliner,
  author = {Grozdanovski, Kristijan},
  title = {CalDec GLiNER},
  year = {2026},
  url = {https://huggingface.co/kgrozdanovski/caldec-v1-gliner2.5-decide}
}
```
