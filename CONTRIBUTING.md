# Contributing

Issues, corrections and new decision sites are welcome. Include the case or metric
that motivates a change, and keep generated data and model output separate from
manually reviewed examples.

## Local checks

Python 3.10 or newer can run the offline checks with the NumPy range below; they do
not download models or call a provider. The pinned full environment in
`requirements.txt` is tested with Python 3.12:

```bash
python3 -m pip install 'numpy>=2.2,<3'
python3 scripts/check_data.py
python3 scripts/check_numbers.py
python3 scripts/check_evidence.py
python3 scripts/test_generate.py
```

Use a virtual environment if the system Python is externally managed. CI runs
these checks for pull requests. Model training and inference need the larger
`requirements.txt` environment and should be reported separately with the
dataset revision, model revision, command, seed and hardware.

## Extend the dataset

The v1 files in `data/` are the released dataset: the reported scores,
`runs/numbers.json` and `runs/evidence/` are computed from them, and CI checks that
they agree. New data therefore goes into a working copy, never into `data/`:

```bash
mkdir data-next && cp data/*.json data/*.jsonl data-next/
```

Set `"data_dir": "data-next"` in the build config (the example already does).
`generate.py` reads sites, prompts and existing cases from that directory and
`fold` appends to its split files; both refuse `data/` itself. Check the copy with
`python3 scripts/check_data.py --data data-next`, and train or evaluate on it with
`--data data-next`. Git ignores `data-next/`.

Follow [README.md](README.md#add-a-decision-site) for a new site, editing the
working copy's `sites.json`, `prompts.json` and split files. Keep the question
specification of an existing site stable: every existing row carries that spec and
`check_data.py` checks it against `sites.json`. To change its questions or option
order, add a new site identifier such as `memory.preference.v2` and leave the old
site and rows intact. Add a seed row, a prompt entry and a generation brief for the
new identifier.

Generate into an ignored build under `data/provenance/`, inspect its rejection and
annotation summaries, and fold only after generation and labelling complete.
Keep raw provider responses, credentials, private conversations and call journals
out of commits. `generate` and `fold` apply the data scrub; still review the new rows
manually and disclose any split overlap or intended test reuse. A new benchmark
claim needs a fresh test set that was not used to select a model or tune thresholds.

To propose new data, open an issue or pull request with the `sites.json` and
`prompts.json` changes, the build config and the `freeze.json` summary, and attach
or link the build's `cases.jsonl` rather than committing the working copy. A new
dataset version replaces the files in `data/` in one reviewed change that also
re-scores the models on it, refreshes `runs/numbers.json`, `runs/evidence/` and its
manifest, the figures and the documented totals.
`check_numbers.py` and `check_evidence.py` fail until all of those agree.

## Report model results

Use the scoring contract in [RESULTS.md](RESULTS.md). Include per-decision
predictions, the ordered dataset fingerprint or immutable revision, the exact
checkpoint, and the number of decisions scored. Add compact, non-sensitive
evidence under `runs/evidence/`; update its hash manifest and
`scripts/check_evidence.py` for a new release. Do not edit the v1 split files,
weights or saved v1 prediction vectors to change a reported score.
