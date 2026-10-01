# Assistant Decisions datasheet

The dataset supports training and evaluating small models that answer an assistant's closed, typed questions. It follows the questions in *Datasheets for Datasets*: why the data exists, how it was made, what it contains, and where its limits matter.

## Motivation and ownership

A real-time assistant makes repeated yes/no, choice and ordered-score decisions while handling turns, tools, memory, security and background work. The dataset provides states and probability targets for those decisions. It is self-funded and released under Apache 2.0. Contact: [kgrozdanovski7@gmail.com](mailto:kgrozdanovski7@gmail.com). Citation metadata is in [CITATION.cff](CITATION.cff).

## Contents

| Split | Cases | Decisions | Tied targets |
|---|---:|---:|---:|
| `data/train.jsonl` | 4,984 | 10,375 | 250 |
| `data/validation.jsonl` | 805 | 1,644 | 34 |
| `data/test.jsonl` | 1,647 | 3,452 | 80 |
| **Total** | **7,436** | **15,471** | **364** |

There are **18 sites** in six workflows: voice, turn routing, security, memory, proactivity and quality. One case is a state plus one or more questions and a teacher distribution for each question. `noul` has `false` and `true` options, `choice` has named options, and `score` has ordered levels. The test set is a development test; it was used during model development.

Each JSONL row has six **string** columns: `case_id`, `site`, `workflow`, `state`, `questions`, `gold`. The last three columns contain JSON text because states have different shapes and Arrow requires one column type. For example:

```python
import json
from datasets import load_dataset
rows = load_dataset("json", data_files="data/train.jsonl")["train"]
row = rows[0]
state = json.loads(row["state"])
questions = json.loads(row["questions"])
gold = json.loads(row["gold"])
```

`data/sites.json` defines the site questions. `data/prompts.json` contains the site generation and labelling prompts used by the recipe. `scripts/data.py` loads and decodes rows for training and evaluation.

## Collection and labels

The dataset creator identifies **GLM 5.3 through OpenRouter as the source of every released row**, including its generated state and target distributions. The `exp-`, `gen-` and `fill1-` case-ID prefixes identify generation builds, not model providers. Original API calls and response journals are not released, so the public files do not independently verify individual calls. The recipe uses separate generation and labelling calls, retains probability distributions, and supports other models and OpenAI-compatible providers for new builds; see [METHOD.md](METHOD.md#create-cases).

The released dataset contains no scraped conversations, inboxes or user accounts. Some states mention public sites as context. Some contain **prompt-injection attempts** and **invented credential strings** because those are the phenomena the questions ask about. They must be handled as untrusted text. `scripts/check_data.py` scans every split for live email domains, unapproved hosts, public IPs, key-shaped strings and card-shaped strings. The checker also validates row and probability-vector schemas.

## Processing and split limitations

Rows with malformed distributions are rejected by the generation recipe. Generation checks exact and near-duplicate states against the current splits at the time it runs. A later data audit found **three exact state values shared across splits**; one shared value occurs in several rows. Two overlaps are train to test and one is validation to test. The checker prints the relevant case ids and rejects any new cross-split state overlap. The labels and split rows are retained so the published dataset and reported full-test scores refer to the same files. This overlap slightly limits the independence of the test estimate.

A target is tied when two options share its highest probability. The scoring code excludes tied targets from hard accuracy and ECE, while retaining them for distribution metrics. Exact case ids are unique across splits. There are also some repeated states within validation and test; `check_data.py` reports cross-split overlaps, and the data should not be treated as a duplicate-free benchmark.

## Intended uses

Use the set to train a typed-decision model, compare predictions against synthetic target distributions, test calibration, or study transfer between model backbones. It can also serve as a schema and prompt recipe for a new collection. The model checkpoints on Hugging Face are separate artifacts. A model trained on this dataset should be evaluated on real, independently labelled states before it controls consequential actions.

## Limitations

- Targets are synthetic GLM 5.3 judgments. Agreement with them is not independent correctness.
- The short, synthetic states may be easier than real assistant traffic.
- Injection cases were generated, not created by adaptive adversaries; this is not a safety benchmark.
- English is the only language intentionally covered.
- The test split informed development, and exact states overlap across some splits.
- Hosted-model output terms may matter for downstream redistribution and training. Review them for the models and providers you use in a new build.

The model evaluation contract and full-test measurements are in [RESULTS.md](RESULTS.md). The dataset card in [huggingface/dataset-card.md](huggingface/dataset-card.md) describes the Hub layout.
