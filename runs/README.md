# Evaluation run outputs

`runs/numbers.json` is the tracked summary used by the README charts. It contains Assistant Decisions and `LocalLLaMA/typed-decisions` test results for CalDec Laya, CalDec GLiNER, upstream GLiNER2.5-Decide, the Laya baselines and Jev, plus per-site rows, split totals and source paths. The compact local-model reports and per-decision predictions in `runs/evidence/` are also tracked. `scripts/check_evidence.py` verifies their SHA-256 manifest, recalculates local-model metrics and checks the CalDec Laya/CalDec GLiNER paired result. These files contain prediction vectors and identifiers, not raw provider responses or state text. `scripts/charts.py` redraws the six SVG assets from `numbers.json`:

```bash
python3 scripts/charts.py --numbers runs/numbers.json --out assets
```

Other files under `runs/` are local and ignored by Git. The benchmark and GLiNER evaluators write new reports and predictions there. Keep those files to run paired comparisons with `python3 scripts/evaluate.py --compare a.json b.json`. The public Hub model cards report the same full-test metrics as `numbers.json`.

Hosted-model call ledgers, cached responses and per-decision outputs stay local. The Jev metrics for both named test sets in `numbers.json` come from committed aggregate summaries, so `check_evidence.py` checks those summaries but cannot independently recalculate Jev scores from its per-decision outputs in a fresh clone.

The two `*.latency.json` files record resident-checkpoint timings on an RTX 4080 SUPER. Reproduce them with `scripts/latency.py` using the release checkpoints. They use one case, 10 warm-up calls and 40 synchronized measurements for one and three questions. Training wall times were not retained.
