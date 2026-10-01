# Method and reproducible recipes

## Data contract

Each case contains a state, its decision site and workflow, typed questions, and a teacher probability distribution for every question. `data/sites.json` defines the 18 sites; `data/prompts.json` contains the generation and labelling prompts. The files in `data/` are the public dataset, with `state`, `questions` and `gold` stored as JSON strings for Arrow compatibility. `scripts/data.py` decodes them before training or scoring.

The current splits have **4,984 train, 805 validation and 1,647 test cases**. A case can hold more than one decision. The local weights are the release checkpoints used for the reported evaluations. The scripts below train new checkpoints from the available data; the release weights are preserved under their own directories.

## Create cases

`scripts/generate.py` builds cases in two separate calls: one writes a state from a site brief, another sees that state and its questions without the generation brief and returns probability distributions. The default [example config](scripts/generate.example.json) uses `z-ai/glm-5.3` through OpenRouter. A single-model run uses independent blind calls; for more independent labels, configure multiple families and set `annotators_per_case` accordingly. The target for multiple annotators is the mean distribution.

The generator validates the state shape, rejects exact and near duplicates against the current splits, rejects states with a finding of the publishing scan, journals API calls and reservations, caches responses by request hash, and enforces a spend cap. `freeze` records the accepted cases and hashes; `fold` scans them again and appends only frozen cases to the chosen split of the build's working dataset directory (`data_dir`), never to the v1 files in `data/`. Working records remain in ignored `data/provenance/`. Review the retained shares, labels and model terms before folding. Run `python3 scripts/check_data.py --data <data_dir>` before publishing any new split.

OpenRouter is built in. For another OpenAI-compatible chat-completions service, add a `providers` entry with an HTTPS `endpoint` and environment-variable name for its key; point a model at that provider. Set `family` when the model id prefix does not identify the underlying model family. The model's price estimate is used for the budget reservation and must be reviewed against its provider's current pricing. [README.md](README.md#generate-your-own-data) includes a config fragment and runnable command sequence.

## Train CalDec Laya

The CalDec Laya script tokenizes each case-question pair using `laya.common.build_sequence`, which is also used by the served model. It optimizes soft cross-entropy against the teacher distribution. `--balance-sites` weights items by the inverse square root of their site's item count. `--with-public` adds the [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) train split; its test split stays outside training. The optional policy-gradient term is disabled by default (`--rl-weight 0`); it is not part of the released checkpoint recipe.

```bash
USE_TF=0 python3 scripts/train.py --data data --out checkpoints/my-laya-run \
  --epochs 6 --balance-sites --with-public --legacy-drop-tail
```

The command uses AdamW, encoder LR 2.5e-5, head LR 1e-4, weight decay 0.01, micro-batch 4 and accumulation 8 by default. The release run discarded the last incomplete accumulation group; `--legacy-drop-tail` preserves that behavior when replaying it. Omit the flag for new runs to train on every decision. The script saves a final `model.safetensors`, `rl_agent_config.json`, tokenizer and encoder config. `--dry-run` builds and checks the training items without updating weights. `--out` must be a new directory, so a training run cannot overwrite the release checkpoint by accident. The training script expects a local dataset directory, so use the checked-in `data/` as shown above.

The fitted temperatures live in `rl_agent_config.json`. Evaluation uses `laya.Agent.system_one`, applying the same temperature and dtype behavior that inference uses. Proposed decision thresholds are fitted on validation, never on test. Keep states within the roughly 320-token state budget implied by the 512-token context and 192-token option head.

## Train CalDec GLiNER from GLiNER2.5-Decide

The CalDec GLiNER path renders a state as lines of `key: value` text and converts each typed question to a classification with ordered labels. Its training patches carry the teacher distribution through GLiNER2's example conversion and target builder, so binary cross-entropy receives soft probabilities rather than one-hot labels. Label augmentation is disabled because dropping or renaming options would corrupt a distribution. Before training, the script checks that a soft target survives the example round trip.

```bash
python3 scripts/gliner2_transfer.py train --soft --out checkpoints/my-gliner-run
python3 scripts/gliner2_transfer.py eval --model checkpoints/my-gliner-run/final
```

The CalDec GLiNER recipe uses three epochs, batch 2, accumulation 8, encoder LR 1e-5, task LR 5e-4 and bf16 on GPU. The training source is the assistant dataset; it does not add the `LocalLLaMA/typed-decisions` train split. The independent sigmoid outputs are normalized across each question's labels for evaluation and use. [TRANSFER.md](TRANSFER.md) gives its measured comparison with CalDec Laya.

Both release recipes target one 16 GB CUDA GPU. The exact GPU and elapsed time of the release training runs were not retained; [TRANSFER.md](TRANSFER.md#training-and-inference) reports a separate, reproducible latency measurement on an RTX 4080 SUPER.

## Evaluate

`benchmark.py` evaluates CalDec Laya and upstream Laya checkpoints through the served path on the Assistant Decisions test and the `LocalLLaMA/typed-decisions` test. `gliner2_transfer.py eval` evaluates CalDec GLiNER, and `gliner2_transfer.py zeroshot` evaluates upstream GLiNER2.5-Decide. Both GLiNER commands accept `--split public` for the pinned `LocalLLaMA/typed-decisions` test split. All paths use the same option order, tie policy and metric definitions. Accuracy and ECE exclude tied targets; distribution metrics include all targets. `evaluate.py --compare` runs an exact paired McNemar test and requires matching decision coverage, split, dataset fingerprint or revision when present, and targets. Upstream model and `LocalLLaMA/typed-decisions` revisions are pinned in `scripts/data.py`. `scripts/check_evidence.py` recalculates the released local-model scores and paired comparison from the compact, committed prediction files.

This is synthetic-label agreement, not correctness against a human or observed outcome. The test split was used during development. Exact duplicate states across splits are printed by `scripts/check_data.py` and should be considered when interpreting the scores.
