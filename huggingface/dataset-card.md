---
license: apache-2.0
language:
  - en
task_categories:
  - text-classification
tags:
  - typed-decisions
  - decision-model
  - calibration
  - synthetic
  - assistant
pretty_name: Assistant Decisions
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
      - split: validation
        path: validation.jsonl
      - split: test
        path: test.jsonl
---

# Assistant Decisions

**7,436 synthetic cases and 15,471 labelled decisions** across 18 assistant decision sites. Targets are probability distributions for `noul` (yes/no), `choice` and `score` questions. This dataset supports the [CalDec training and evaluation recipes](https://github.com/kgrozdanovski/caldec).

| Split | Cases | Decisions |
|---|---:|---:|
| Train | 4,984 | 10,375 |
| Validation | 805 | 1,644 |
| Test | 1,647 | 3,452 |

Each row has six string fields: `case_id`, `site`, `workflow`, `state`, `questions`, and `gold`. Decode the last three from JSON text:

```python
import json
from datasets import load_dataset
row = load_dataset("kgrozdanovski/assistant-decisions")["train"][0]
state = json.loads(row["state"])
questions = json.loads(row["questions"])
gold = json.loads(row["gold"])
```

`sites.json` gives the 18 site definitions; `prompts.json` contains site generation and labelling prompts. The dataset creator identifies **GLM 5.3 through OpenRouter as the source of every released row**, including generated states and target distributions. The `exp-`, `gen-` and `fill1-` case-ID prefixes identify generation builds, not providers. Original API calls and response journals are not released, so individual calls cannot be verified from the public files. The public recipe defaults to GLM 5.3 and shows how to configure other models and OpenAI-compatible providers for new builds. The [datasheet](https://github.com/kgrozdanovski/caldec/blob/main/DATASET.md#collection-and-labels) describes collection and limitations.

## Model comparisons

Assistant accuracy is agreement with the synthetic target's argmax on 3,372 untied test decisions; it is not independent correctness. ECE also excludes ties; distribution metrics include all 3,452 assistant decisions.

| Model | Assistant accuracy | ECE | Soft NLL | `LocalLLaMA/typed-decisions` test accuracy |
|---|---:|---:|---:|---:|
| [CalDec GLiNER](https://huggingface.co/kgrozdanovski/caldec-v1-gliner2.5-decide) | **0.843** | **0.039** | **0.552** | 0.582 |
| Jev 1.13, zero-shot via OpenRouter | 0.836 | 0.042 | 0.811 | 0.738 |
| [CalDec Laya](https://huggingface.co/kgrozdanovski/caldec-v1-laya) | 0.823 | 0.072 | 0.559 | **0.778** |
| [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) | 0.650 | 0.097 | 0.758 | 0.540 |
| Laya specialist | 0.578 | 0.041 | 0.805 | 0.773 |
| Majority class, fitted on train | 0.574 | — | — | — |
| Laya base | 0.557 | 0.158 | 0.989 | 0.361 |

`Laya base` means the upstream [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) checkpoint. `Laya specialist` means the upstream [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) checkpoint, fine-tuned on the Typed Decisions train split.

The final column uses the pinned test split of [**Typed Decisions** (`LocalLLaMA/typed-decisions`)](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), with 2,000 decisions and 1,965 untied targets. CalDec Laya used that benchmark's train split; CalDec GLiNER did not. The full comparison is in [RESULTS.md](https://github.com/kgrozdanovski/caldec/blob/main/RESULTS.md).

The test split was used during development. Three exact state values occur in more than one split; the repository's `scripts/check_data.py` prints the case ids. This limits test independence slightly. The full evaluation contract and per-site table are in [RESULTS.md](https://github.com/kgrozdanovski/caldec/blob/main/RESULTS.md).

## Use and limits

The states are synthetic, English, and generally short. Some contain **prompt-injection text and invented credentials by design**. Treat all state text as untrusted. These are generated cases, not an adversarial safety benchmark. Targets reflect model judgments, including their errors. Evaluate a trained model on real, independently labelled cases before consequential use.

The repository's [datasheet](https://github.com/kgrozdanovski/caldec/blob/main/DATASET.md) documents collection, format, processing, split overlap and intended use. The license is Apache 2.0. Review the hosted model providers' terms when using the recipe to create new data.

## Citation and contact

Contact [Kristijan Grozdanovski](mailto:kgrozdanovski7@gmail.com). The repository's [CITATION.cff](https://github.com/kgrozdanovski/caldec/blob/main/CITATION.cff) is the machine-readable citation.

```bibtex
@misc{grozdanovski2026assistantdecisions,
  author = {Grozdanovski, Kristijan},
  title = {Assistant Decisions: A Dataset and Recipe for Small Typed-Decision Models},
  year = {2026},
  url = {https://huggingface.co/datasets/kgrozdanovski/assistant-decisions}
}
```
