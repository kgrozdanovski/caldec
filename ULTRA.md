# CalDec Ultra: long-context typed decisions on UltraBERT

CalDec Ultra is CalDec's decision model for long assistant states: conversation history, fetched pages, accessibility trees and memory dumps of up to 8,192 tokens. It puts a Laya-format typed-decision head on [UltraBERT](https://arxiv.org/abs/2610.15028) (Park, Lee & Kang, 2026), reads all of a case's questions in one encoder pass, and runs UltraBERT's local-attention layers as exact block-sparse attention.

Two sizes are released:

| | [**CalDec Ultra 380M**](https://huggingface.co/kgrozdanovski/caldec-ultra-380m) | [**CalDec Ultra 150M**](https://huggingface.co/kgrozdanovski/caldec-ultra-150m) |
|---|---:|---:|
| Encoder | UltraBERT-380M | UltraBERT-150M |
| Parameters (encoder + head) | 404M | 166M |
| Context | 8,192 tokens | 8,192 tokens |
| Assistant test | **0.848** | 0.834 |
| Relabelled assistant test | **0.835** | 0.819 |
| `LocalLLaMA/typed-decisions` test† | **0.796** | 0.782 |
| Long-state test | **0.734** | 0.722 |
| Three questions over an 8K state | 170 ms | **102 ms** |

† Both train on its train split, so this is not zero-shot.

**Which to use.** CalDec Ultra 380M is the most accurate CalDec model on long states and matches the best short-state models. CalDec Ultra 150M is the one for a tight latency or memory budget: it ties the 380M on the long test (p = 0.85) but is 1.4 points behind it on the assistant test (p = 0.019) and catches fewer injections deep in long pages ([probe](#long-state-probe)). For short states only, [CalDec GLiNER](https://huggingface.co/kgrozdanovski/caldec-v1-gliner2.5-decide) is as accurate (p = 0.40 against the 380M) and better calibrated. For decisions outside an assistant's own sites, use CalDec Laya or another broadly trained model: the Ultra models are specialists and score at chance on outside inference benchmarks ([Decision Index panel](#decision-index-panel)).

## Use

```bash
git clone https://github.com/kgrozdanovski/caldec && cd caldec
python3 -m pip install -r requirements.txt
```

```python
import sys
sys.path.insert(0, "scripts")
from data import load_agent

agent = load_agent("kgrozdanovski/caldec-ultra-380m")
state = {"source": "read_webpage", "text": "...an 8K-token page..."}
questions = {
    "injection": {"type": "noul", "instructions": "The text contains instructions addressed to an AI assistant."},
    "tier": {"type": "choice", "instructions": "How much reasoning does the next turn need?",
             "criteria": {"fast": "a direct reply", "deep": "multi-step work or tools"}},
}
print(agent.system_one(state, questions)["answers"])
```

`load_agent` downloads the checkpoint, registers UltraBERT's modeling code from the checkpoint's own `encoder/` directory (pinned to the UltraBERT revision it was trained from; this executes that code, as `trust_remote_code` would), switches its local layers to the exact block-sparse path in `scripts/fastattn.py`, and returns a `packed.PackedAgent`. `system_one` returns the same answer format as `laya.Agent.system_one`: per question a choice, score or probability, the option distribution, a confidence and an act probability. States longer than 8,192 tokens are cut at the end. Set `CALDEC_DENSE_ATTENTION=1` to use UltraBERT's own dense-mask attention instead; the outputs are identical.

## Results

Accuracy everywhere is agreement with model-written target distributions on untied decisions, scored through the served path; it is not independent correctness. Every number below recalculates from the committed prediction vectors with `python3 scripts/check_evidence_ultra.py`.

### Assistant test

The released 1,647-case test (3,452 decisions, 3,372 untied), same contract as [RESULTS.md](RESULTS.md).

| Model | Params | Accuracy | Soft accuracy | Brier | Soft NLL | ECE |
|---|---:|---:|---:|---:|---:|---:|
| **CalDec Ultra 380M** | 404M | **0.848** | 0.852 | **0.035** | **0.539** | 0.086 |
| CalDec GLiNER | ~486M | 0.843 | **0.859** | 0.038 | 0.552 | **0.039** |
| Jev 1.13, zero-shot | — | 0.836 | — | — | 0.811 | 0.042 |
| **CalDec Ultra 150M** | 166M | 0.834 | 0.836 | 0.041 | 0.559 | 0.083 |
| CalDec Laya | 421M | 0.823 | 0.839 | 0.041 | 0.559 | 0.072 |
| GLiNER2.5-Decide | — | 0.650 | 0.664 | 0.105 | 0.758 | 0.097 |
| Laya specialist | 421M | 0.578 | 0.655 | 0.123 | 0.805 | 0.041 |
| Majority class, fitted on train | — | 0.574 | — | — | — | — |
| Laya base | 421M | 0.557 | 0.643 | 0.161 | 0.989 | 0.158 |
| UltraBERT-380M, zero-shot | 380M | 0.460 | 0.612 | 0.151 | 0.893 | 0.152 |
| UltraBERT-150M, zero-shot | 152M | 0.361 | 0.551 | 0.209 | 1.065 | 0.300 |

Exact paired McNemar tests on untied decisions (first model only correct, second only correct):

| Comparison | First | Second | p |
|---|---:|---:|---:|
| Ultra 380M vs CalDec Laya | 250 | 166 | < 0.0001 |
| Ultra 380M vs CalDec GLiNER | 187 | 170 | 0.40 |
| Ultra 380M vs Ultra 150M | 215 | 168 | 0.019 |
| Ultra 150M vs CalDec Laya | 234 | 197 | 0.083 |
| Ultra 150M vs CalDec GLiNER | 206 | 236 | 0.17 |

The raw UltraBERT encoders, read zero-shot at one mask token as in the paper's multiple-choice protocol, score below the majority class: the decision skill comes from fine-tuning. CalDec GLiNER remains the best-calibrated model. The Ultra models are under-confident rather than over-confident: CalDec Ultra 380M's decisions served at about 0.70 confidence are right 82% of the time, and at about 0.85 right 95% of the time. A deployment that gates on raw confidence should use the bands `evaluate.py` fits on validation.

### Relabelled assistant test

Every short target above is one GLM 5.3 judgment, and the test informed CalDec's development, so a model could score well by learning GLM's habits. The same 1,647 states were relabelled by two of Claude Opus 5.5, GPT-5.6 Sol and Gemini 3.8 Flash per case (`data-relabelled/test.jsonl`, [DATASET.md](DATASET.md#relabelled-test)). The new targets agree with GLM's argmax on 90.4% of decisions.

| Model | Original targets | Relabelled targets |
|---|---:|---:|
| **CalDec Ultra 380M** | 0.848 | **0.835** |
| CalDec GLiNER | 0.843 | 0.828 |
| CalDec Ultra 150M | 0.834 | 0.819 |
| CalDec Laya | 0.823 | 0.811 |
| GLiNER2.5-Decide | 0.650 | 0.639 |
| Laya specialist | 0.578 | 0.573 |
| Laya base | 0.557 | 0.551 |

Every model loses 1–2 points; the CalDec fine-tunes lose about half a point more than base models that never saw GLM labels, so agreement specific to GLM is worth at most about a point. The ranking does not change, and the gaps hold: Ultra 380M beats CalDec Laya 254 to 172 (p = 0.0001) and Ultra 150M 221 to 167 (p = 0.0071), and stays level with CalDec GLiNER (193 to 169, p = 0.23). The two target sets disagree on about one decision in ten, about as often as the best model misses.

### `LocalLLaMA/typed-decisions` test

400 cases, 2,000 decisions, 1,965 untied targets, at the pinned revision. CalDec Ultra and CalDec Laya train on its train split; CalDec GLiNER does not.

| Model | Accuracy | Soft accuracy | Soft NLL | ECE |
|---|---:|---:|---:|---:|
| **CalDec Ultra 380M**† | **0.796** | **0.862** | **0.851** | 0.195 |
| CalDec Ultra 150M† | 0.782 | 0.853 | 0.861 | 0.181 |
| CalDec Laya† | 0.778 | 0.860 | 0.858 | 0.149 |
| Laya specialist† | 0.772 | 0.826 | 0.884 | 0.218 |
| Jev 1.13, zero-shot | 0.738 | — | — | — |
| CalDec GLiNER | 0.582 | 0.709 | 1.089 | **0.090** |
| GLiNER2.5-Decide | 0.540 | 0.662 | 1.123 | 0.147 |

† Trained on the benchmark's train split. Ultra 380M beats CalDec Laya here 161 to 125 (p = 0.038); Ultra 150M and CalDec Laya are level (134 to 126, p = 0.66).

### Long-state test

77 held-out cases on nine sites (`data-long/test.jsonl`): 32 at about 4.5–8K tokens and 45 at about 3–5K, 158 untied decisions. Each was generated by one of GLM 5.3, MiMo-V2.6-Pro and DeepSeek V4.1 Flash at maximum reasoning effort and labelled by the two families that did not write it. Every model reads up to 8,192 tokens (18 cases are longer and are cut) except CalDec GLiNER (‡); differences under about 7 points are noise at this size.

| Model | All (158) | 8K build (68) | 4K build (90) |
|---|---:|---:|---:|
| **CalDec Ultra 380M** | **0.734** | 0.706 | **0.756** |
| **CalDec Ultra 150M** | 0.722 | **0.765** | 0.689 |
| Packed CalDec Laya, long-trained (development) | 0.665 | 0.676 | 0.656 |
| Step 1: short-context Ultra 380M (development) | 0.677 | 0.647 | 0.700 |
| CalDec GLiNER‡ | 0.614 | 0.647 | 0.589 |
| Step 1: short-context Ultra 150M (development) | 0.570 | 0.500 | 0.622 |
| CalDec Laya | 0.563 | 0.588 | 0.544 |

‡ CalDec GLiNER's encoder has no input limit of its own, but whole 8K states run out of memory on a 16 GB GPU, so each input is cut to its first 2,048 tokens. That scored best of the cuts tried: 512 tokens scored 0.551, 3,072 scored 0.608, and 4,096 ran out of memory on the 8K build.

Paired tests on the 158 decisions: Ultra 380M beats CalDec Laya 41 to 14 (p = 0.0004) and CalDec GLiNER 33 to 14 (p = 0.0079), and Ultra 150M beats CalDec Laya 37 to 12 (p = 0.0005). The two sizes are level (15 to 13, p = 0.85).

### Long-state probe

`scripts/needle.py` embeds each `tools.injection` test case's text at the start, middle or end of a 4K- or 8K-token page built from train chunks the teacher rated clean, keeping the case's own target (612 decisions). It measures reading position, not real long pages.

Each cell holds 102 decisions, 35 of them injected; answering "clean" everywhere scores 0.657.

| Model | 4K start | 4K middle | 4K end | 8K start | 8K middle | 8K end |
|---|---:|---:|---:|---:|---:|---:|
| **CalDec Ultra 380M** | **0.931** | **0.931** | **0.922** | **0.931** | **0.804** | **0.745** |
| **CalDec Ultra 150M** | 0.873 | 0.824 | 0.824 | 0.902 | 0.725 | 0.676 |
| Step 1: short-context Ultra 380M (development) | 0.922 | 0.814 | 0.745 | 0.882 | 0.696 | 0.676 |
| CalDec Laya | 0.912 | 0.755 | 0.667 | 0.912 | 0.667 | 0.676 |
| CalDec GLiNER (first 2,048 tokens) | 0.951 | 0.843 | 0.657 | 0.961 | 0.657 | 0.657 |
| Packed CalDec Laya, long-trained (development) | 0.853 | 0.775 | 0.725 | 0.569 | 0.461 | 0.412 |

Share of injected texts caught, and of clean pages flagged:

| Model | 4K start | 4K middle | 4K end | 8K start | 8K middle | 8K end | Clean flagged (4K / 8K) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **CalDec Ultra 380M** | 0.89 | 0.86 | 0.83 | 0.89 | 0.46 | 0.29 | 0.03 / 0.02 |
| **CalDec Ultra 150M** | 0.77 | 0.57 | 0.51 | 0.80 | 0.23 | 0.06 | 0.04 / 0.02 |
| CalDec Laya | 0.94 | 0.29 | 0.03 | 0.97 | 0.06 | 0.06 | 0.03 / 0.04 |

On 4K pages CalDec Ultra 380M finds an injection about as well wherever it sits. On 8K pages **neither Ultra model is reliable past the first few thousand tokens**: the 380M catches 29% of injections at the end and the 150M 6%, against 6% for CalDec Laya. The second seed of the 380M catches fewer at every position, 34% at the end of a 4K page ([seed variance](#seed-variance)), so the 380M's 4K numbers are partly the luck of one run. The long-trained packed CalDec Laya flags 75% of clean 8K pages as injected.

### Latency

One real assistant case with three questions, its state padded with other test states to each length, through the served `system_one` path; RTX 4080 SUPER, bfloat16, median of 20 timed calls after warm-up, no other compute on the GPU (`runs/evidence/ultra/latency/long-release.json`).

| Model | 512 | 1,024 | 2,048 | 4,096 | 8,192 tokens |
|---|---:|---:|---:|---:|---:|
| CalDec Laya (one pass per question) | 33 ms | 56 ms | 123 ms | 330 ms | 920 ms |
| Step 1: short-context Ultra 380M (one pass per question) | 33 ms | 58 ms | 114 ms | 275 ms | 631 ms |
| Step 1: short-context Ultra 150M (one pass per question) | 24 ms | 32 ms | 67 ms | 159 ms | 404 ms |
| Packed CalDec Laya (development) | 21 ms | 24 ms | 43 ms | 95 ms | 262 ms |
| **CalDec Ultra 380M** | 25 ms | 26 ms | 39 ms | 75 ms | 170 ms |
| **CalDec Ultra 150M** | 23 ms | 26 ms | 28 ms | 43 ms | **102 ms** |

At short lengths every model sits near a 20–35 ms floor of per-layer launch overhead. Two changes make the long end fast: packing (one encoder pass for all questions instead of one per question) and exact block-sparse attention (encoder-only, one sequence: at 8K tokens UltraBERT-150M runs in 92 ms instead of 174 ms and UltraBERT-380M in 160 ms instead of 263 ms; `latency/encoder-*.json`).

### Decision Index panel

To check what fine-tuning does to skills outside the assistant's sites, nine benchmarks from the [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) were rebuilt with its reproduction kit (`suite rebuild --only ...`) and run through `scripts/jdi_engine.py` at up to 8,192 tokens. Each is a deterministic sample of at most 1,000 requests (`runs/evidence/ultra/index/sample.json`), scored by the kit's own scorers with edition 0.2.1; iSarcasmEval uses the board's metric, sarcasm F1 on track A, English. These are subset scores, not leaderboard entries. As a harness check, Laya base scores 0.489 ANLI macro-F1 here against 0.487 on the published board, and 0.147 iSarcasmEval F1 against 0.149. Jev 1.13, GLiNER2.5-Decide and Laya rows marked "board" are the published full-benchmark scores (`index/board.json`).

| Benchmark (metric) | Jev 1.13 (board) | GLiNER2.5-Decide (board) | Laya base | UltraBERT-380M zero-shot | CalDec Laya | **CalDec Ultra 380M** | **CalDec Ultra 150M** | Chance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ContractNLI (macro-F1) | 0.717 | 0.218 | 0.291 | 0.210 | 0.252 | 0.210 | 0.283 | 0.31 |
| ANLI (macro-F1) | 0.748 | 0.309 | 0.489 | 0.270 | 0.437 | 0.167 | 0.187 | 0.33 |
| MuSR (accuracy) | 0.661 | 0.452 | 0.461 | 0.424 | 0.464 | 0.388 | 0.379 | 0.37 |
| iSarcasmEval (sarcasm F1) | 0.505 | 0.324 | 0.147 | 0.245 | 0.091 | 0.235 | 0.045 | 0.22 |
| VAST (macro-F1) | 0.646 | 0.353 | 0.377 | 0.210 | 0.388 | 0.319 | 0.262 | 0.33 |
| NLI4CT (macro-F1) | 0.841 | 0.480 | 0.483 | 0.235 | 0.461 | 0.294 | 0.467 | 0.49 |
| CLadder (accuracy) | 0.726 | 0.501 | 0.535 | 0.486 | 0.552 | 0.530 | 0.511 | 0.5 |
| PhishNChips (verdict accuracy) | 0.626 | 0.500 | 0.502 | 0.420 | 0.502 | 0.501 | 0.561 | 0.5 |
| When2Call (accuracy) | 0.810 | 0.334 | 0.109 | 0.007 | 0.185 | 0.274 | 0.173 | 0.25 |

**This is the cost of the specialisation, stated plainly.** Laya base was trained broadly (its card reports XNLI English 0.860) and CalDec Laya, fine-tuned from it, keeps most of that. The Ultra models start from a masked-LM encoder and a fresh head and learn decisions mostly from assistant data, so they sit at or below chance on ANLI, ContractNLI and VAST. Mixing 2,700 MNLI and BoolQ rows into training lifted five of the seven benchmarks that separate models on a development run but did not give them Laya's breadth. On When2Call, the benchmark closest to an assistant's own tool decisions, CalDec Ultra 380M is the only model run here above chance. Most models are at chance on CLadder and PhishNChips, so those two separate little. Jev 1.13, a hosted generative model, leads every benchmark.

### Seed variance

Each released recipe was trained a second time with a different seed (`--seed 20261019`) and scored the same way (`*-seed2` files). The released weights are the first seed, chosen before the second was run; picking the better seed afterwards would fit the test.

| | Ultra 380M | 380M, seed 2 | Ultra 150M | 150M, seed 2 |
|---|---:|---:|---:|---:|
| Assistant test | 0.848 | 0.853 | 0.834 | 0.820 |
| Relabelled test | 0.835 | 0.842 | 0.819 | 0.805 |
| `typed-decisions` test† | 0.796 | 0.808 | 0.782 | 0.792 |
| Long-state test | 0.734 | 0.709 | 0.722 | 0.696 |
| Probe, 8K end | 0.745 | 0.696 | 0.676 | 0.676 |
| ANLI macro-F1 (Decision Index panel) | 0.167 | 0.295 | 0.187 | 0.246 |
| When2Call accuracy | 0.274 | 0.361 | 0.173 | 0.189 |

On the assistant test the 380M seeds are level (121 to 138, p = 0.32); the 150M seeds are not (207 to 161, p = 0.019), so the released 150M is the better of its two runs by about 1.4 points and a retrained 150M should be expected to land anywhere in that range. On the long test both pairs are level (11 to 7, p = 0.48; 10 to 6, p = 0.45). Outside benchmarks move by up to 0.18 between seeds (iSarcasmEval F1: 150M 0.045 and 0.224) while staying below chance on ANLI, so the panel's per-benchmark differences between the Ultra models are not meaningful.

## How we got here

CalDec Ultra was built in four steps, each checked on the assistant test, the public test and, from step three, long states and outside benchmarks. None of the intermediate checkpoints is released; their predictions are in `runs/evidence/ultra/` under `dev-*` names.

**1. A short-context model on UltraBERT.** The first CalDec Ultra models kept CalDec Laya's data, objective and 512-token budget, and replaced its ModernBERT-large backbone with UltraBERT under a fresh decision head (Laya's pretrained head cannot move to a new encoder) and UltraBERT's fine-tuning recipe: encoder LR 1e-5 (150M) or 8e-6 (380M), AdamW β2 0.98, ε 1e-6, weight decay 0.1, linear warm-up and decay, and epoch selection on validation soft NLL. The 380M model reached 0.851 on the assistant test. A ModernBERT-large control with the same fresh head and recipe reached 0.837, so the gain over CalDec Laya came in two significant steps: the recipe with a fresh head (0.823 → 0.837, p = 0.020) and the backbone (0.837 → 0.851, p = 0.012). The recipes differ in several settings at once, so the first step measures the recipe as a whole.

**2. A meta-analysis pointed at long states.** `scripts/meta_analysis.py` bucketed every model's test decisions by how sure the teacher was (`runs/evidence/ultra/meta.json`). A fifth of the targets have a teacher top probability below 0.7, and every model sits near 0.55–0.65 on them, close to the teacher's own uncertainty; the top 30% are solved by every model; models differ almost only in between. 94.1% of decisions are answered correctly by at least one model. Short-state accuracy was close to what single-annotator labels allow. Meanwhile an assistant's real states are long, while the dataset's median state is 132 tokens and every model stopped at 512. Long states are where UltraBERT's design differs, and where latency, not accuracy, becomes the constraint.

**3. Making long states fast.** UltraBERT gives two of every three layers a 128-token window, but its `sdpa` path builds a dense L × L mask and runs full attention under it, so the windows saved nothing. `scripts/fastattn.py` runs each windowed layer as 128-token query blocks against their own and neighbouring key blocks through SDPA's fused kernels, and global layers with a key-padding mask; outputs equal the original path exactly in float32 on 300- to 8,192-token inputs, and re-scoring the assistant test gives identical predictions. Laya's sequence also holds one question, so a case with three questions over an 8K state costs three 8K passes. `scripts/packed.py` puts every question ahead of one copy of the state, runs the encoder once, and runs the decision head per question over [CLS] and that question's own segment, with exactly the parameters a Laya-format head already has. Packing cost no measurable accuracy: at a 1,024-token budget the packed 150M scored 0.832 against 0.830 (p = 0.79), and two seeds of the packed 380M scored 0.844 and 0.851 against the per-question 0.851 (p = 0.16 and 1.0). At a 512-token budget the packed model lost 7.6 points on the public test because its five-question cases no longer fit, which is why the budget grows with the question count.

**4. Training on long states.** With no long training data, every model lost 17–30 points between the assistant test and the long test, and on the probe at 8K tokens models trained only on short states caught 0–23% of injections placed past the start of the page. Two small long train builds (39 cases at about 8K tokens and 78 at about 4K, generated and labelled like the long test) and a 24-case long validation build were added, with the MNLI and BoolQ mix, to the assistant and public train splits, at `max_len` 8,192 with length-bucketed micro-batches and gradient checkpointing. That is the released recipe ([below](#recipe)). Against step 1, it kept short-state accuracy (380M: 0.848 against 0.851, p = 0.53; 150M: 0.834 against 0.830, p = 0.56) and lifted the long test: 150M from 0.570 to 0.722 (37 to 13, p = 0.0009), 380M from 0.677 to 0.734 (26 to 17, p = 0.22, within the test's noise).

**Also tried.** A packed CalDec Laya trained the same way keeps Laya's breadth on the Decision Index panel (ANLI 0.463, MuSR 0.491) and CalDec Laya's assistant accuracy (0.827, p = 0.52), but CalDec Ultra 380M beats it on the assistant test (250 to 179, p = 0.0007) and it flags 75% of clean 8K pages as injected on the probe. Weighting each small long-data site as its own site in the loss cost that Laya model 1.9 assistant-test points (p = 0.0015); weighting each extra file as one site fixed it, and the Ultra models, on a lower decaying learning rate, were unaffected. Neither Laya variant is released.

## Recipe

```bash
python3 scripts/general_data.py --out data-general        # the MNLI/BoolQ mix, pinned revisions
USE_TF=0 python3 scripts/ultra.py init --encoder dmis-lab/UltraBERT-380M --out checkpoints/ultra-380m-init
USE_TF=0 python3 scripts/train.py --data data --base checkpoints/ultra-380m-init --out checkpoints/my-ultra-380m \
  --epochs 6 --balance-sites --with-public --packed --max-len 8192 --max-batch-tokens 16384 --grad-checkpoint \
  --extra-train data-general/train.jsonl --extra-train data-long/train.jsonl \
  --extra-val data-general/validation.jsonl --extra-val data-long/validation.jsonl --select-epoch \
  --micro-batch 2 --grad-accum 8 --lr-encoder 8e-6 --lr-head 1e-4 \
  --weight-decay 0.1 --adam-beta2 0.98 --adam-eps 1e-6 --schedule linear --warmup-ratio 0.05
USE_TF=0 python3 scripts/benchmark.py --model "ultra=checkpoints/my-ultra-380m/final" --data data --out runs/my-ultra-380m
USE_TF=0 python3 scripts/evaluate.py --data data-long --checkpoint checkpoints/my-ultra-380m/final --max-len 8192 \
  --predictions runs/my-ultra-380m.long.predictions.json
```

For 150M use `--encoder dmis-lab/UltraBERT-150M`, `--micro-batch 4 --grad-accum 4` and `--lr-encoder 1e-5`. On an RTX 4080 SUPER the 380M run took 61 minutes (peak 8.1 GB) and the 150M run 34 minutes; epoch selection kept epoch 5 for 380M and epoch 6 for 150M. `--balance-sites` weights each assistant site by 1/√n. The probe data comes from `python3 scripts/needle.py`, the Decision Index runs from the kit and `scripts/jdi_engine.py`, and latency from `scripts/latency_long.py`.

UltraBERT ships as remote code before transformers 5.20. The pinned 5.17 environment registers it through `ultra.register`, which `data.resolve_laya` calls for any checkpoint whose encoder is UltraBERT, so `train.py`, `evaluate.py` and `benchmark.py` need no other change.

## Evidence

`runs/evidence/ultra/` holds the per-decision prediction vectors behind every number on this page (assistant and public tests at the top level, `long/`, `probe/`), the Decision Index subset summaries with iSarcasmEval's confusion counts (`index/`), latency measurements (`latency/`), the meta-analysis, a hash manifest and `numbers.json`. `python3 scripts/check_evidence_ultra.py` checks every hash, checks that each file was scored against the released test split, `data-long/test.jsonl` or the pinned public revision with matching targets, and recalculates every metric, relabelled score and paired test.

## Limits

- **Labels are model judgments.** Every target is a model's probability distribution: GLM 5.3 for the assistant splits, two of three model families for the long builds and the relabelled test. Accuracy is agreement with them.
- **The long test is small and synthetic.** 158 decisions, so differences under about 7 points are noise, generated by the same three-family recipe as the long training data. It measures that distribution, not real assistant traffic.
- **The long training data is small.** 117 cases, because GLM 5.3 at maximum reasoning costs about $0.20 per 8K state.
- **Specialists.** The Ultra models are at chance on outside inference benchmarks ([panel](#decision-index-panel)); they are for an assistant's own decision sites.
- **Calibration.** CalDec Ultra 380M's ECE is 0.086 against CalDec GLiNER's 0.039; it is under-confident.
- **Seeds.** Each released model is one training run. A second 150M run scored 1.4 assistant-test points lower (p = 0.019); probe and outside-benchmark scores move more than that ([seed variance](#seed-variance)).
- **Injections deep in a long page.** On 8K probe pages the 380M catches 29% of injections placed at the end and the 150M 6%. Do not rely on either to find an injection in the second half of an 8K state.
- **The assistant test informed development** of every CalDec model; the relabelled test checks the labels on those states, not new states.
