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

The dataset creator identifies **GLM 5.3 through OpenRouter as the source of every row in `data/`**, including its generated state and target distributions; the [long states](#long-states) and the [relabelled test](#relabelled-test) below name their own model families. The `exp-`, `gen-` and `fill1-` case-ID prefixes identify generation builds, not model providers. Original API calls and response journals are not released, so the public files do not independently verify individual calls. The recipe uses separate generation and labelling calls, retains probability distributions, and supports other models and OpenAI-compatible providers for new builds; see [METHOD.md](METHOD.md#create-cases).

The released dataset contains no scraped conversations, inboxes or user accounts. Some states mention public sites as context. Some contain **prompt-injection attempts** and **invented credential strings** because those are the phenomena the questions ask about. They must be handled as untrusted text. `scripts/check_data.py` scans every split for live email domains, unapproved hosts, public IPs, key-shaped strings and card-shaped strings. The checker also validates row and probability-vector schemas.

## Processing and split limitations

Rows with malformed distributions are rejected by the generation recipe. Generation checks exact and near-duplicate states against the current splits at the time it runs. A later data audit found **three exact state values shared across splits**; one shared value occurs in several rows. Two overlaps are train to test and one is validation to test. The checker prints the relevant case ids and rejects any new cross-split state overlap. The labels and split rows are retained so the published dataset and reported full-test scores refer to the same files. This overlap slightly limits the independence of the test estimate.

A target is tied when two options share its highest probability. The scoring code excludes tied targets from hard accuracy and ECE, while retaining them for distribution metrics. Exact case ids are unique across splits. There are also some repeated states within validation and test; `check_data.py` reports cross-split overlaps, and the data should not be treated as a duplicate-free benchmark.

## Long states

`data-long/` adds long states in the same row format, for training and testing [CalDec Ultra](ULTRA.md). The files in `data/` are unchanged.

| Split | Cases | Decisions | Tied targets | Median state |
|---|---:|---:|---:|---:|
| `data-long/train.jsonl` | 117 | 244 | 5 | 15.8K characters |
| `data-long/validation.jsonl` | 24 | 49 | 1 | 25.7K characters |
| `data-long/test.jsonl` | 77 | 162 | 4 | 18.3K characters |

They cover nine sites whose real states run long: conversation history, fetched pages, accessibility trees, memory dumps and workflow transcripts. Each split comes from builds aimed at about 8K tokens (`long8k_*` case ids: 39 train, 24 validation and 32 test cases) and about 4K tokens (`long4k_*`: 78 train and 45 test cases); `builds/long*.json` are the configurations and `scripts/briefs/long*.json` the length briefs.

Unlike the original splits, each case was generated by one of three model families at maximum reasoning effort, **GLM 5.3, MiMo-V2.6-Pro and DeepSeek V4.1 Flash**, and labelled independently, also at maximum reasoning, by the **two families that did not write it**; the target is the mean of the two distributions. Annotator agreement on the argmax was 0.90 (8K train), 0.88 (4K train), 0.82 (validation), 0.94 (8K test) and 0.86 (4K test). Three GLM generation batches of the 8K test build were abandoned after GLM spent its 64K-token output budget reasoning without finishing a state, and three cases (one validation batch, one test batch) were dropped when DeepSeek returned an invalid label vector on every attempt. The builds cost $22.81. `check_data.py --data data-long` passes. The long test is small: 158 untied decisions, so differences under about 7 points are noise.

## Relabelled test

`data-relabelled/test.jsonl` holds the 1,647 test states of `data/test.jsonl`, in the same order, with new targets: each case was labelled by **two of Claude Opus 5.5, GPT-5.6 Sol and Gemini 3.8 Flash**, rotated across cases, at medium reasoning effort through each model's first-party provider, and the target is the mean of the two. It exists to measure how much of a model's test accuracy is agreement with GLM 5.3 in particular. The two annotators agree on 87.8% of decisions; the new targets' argmax agrees with GLM's on 90.4% of decisions untied under both. One six-case `gateway.refusal` batch was blocked by Claude Opus 5.5's provider content filter on every attempt and was labelled by Gemini 3.8 Flash in its place. 41 of its 3,452 targets are tied. The relabel cost $9.49.

These targets are **evaluation data only**: no released model was trained on Claude, GPT or Gemini output. Use them to score a model, not to train one.

## Intended uses

Use the set to train a typed-decision model, compare predictions against synthetic target distributions, test calibration, or study transfer between model backbones. It can also serve as a schema and prompt recipe for a new collection. The model checkpoints on Hugging Face are separate artifacts. A model trained on this dataset should be evaluated on real, independently labelled states before it controls consequential actions.

## Limitations

- Targets are synthetic model judgments: GLM 5.3 in `data/`, two of three model families per case in `data-long/` and `data-relabelled/`. Agreement with them is not independent correctness.
- The short, synthetic states may be easier than real assistant traffic.
- Injection cases were generated, not created by adaptive adversaries; this is not a safety benchmark.
- English is the only language intentionally covered.
- The test split informed development, and exact states overlap across some splits.
- Hosted-model output terms may matter for downstream redistribution and training. Review them for the models and providers you use in a new build.

The model evaluation contract and full-test measurements are in [RESULTS.md](RESULTS.md). The dataset card in [huggingface/dataset-card.md](huggingface/dataset-card.md) describes the Hub layout.
