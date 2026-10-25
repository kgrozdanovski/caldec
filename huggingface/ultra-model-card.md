---
license: apache-2.0
base_model: {{encoder}}
language:
  - en
pipeline_tag: text-classification
tags:
  - typed-decisions
  - decision-model
  - long-context
  - ultrabert
  - assistant
datasets:
  - kgrozdanovski/assistant-decisions
  - LocalLLaMA/typed-decisions
  - nyu-mll/glue
  - google/boolq
model-index:
  - name: {{model}}
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
            value: {{metrics|@M|test|accuracy:.3f}}
            name: Accuracy (untied labels)
          - type: expected_calibration_error
            value: {{metrics|@M|test|ece:.3f}}
            name: Expected calibration error
      - task:
          type: text-classification
          name: Typed decisions
        dataset:
          name: Assistant decisions, long-state test
          type: kgrozdanovski/assistant-decisions
          config: long
          split: test
        metrics:
          - type: accuracy
            value: {{long|@M|accuracy:.3f}}
            name: Accuracy (untied labels)
      - task:
          type: text-classification
          name: Typed decisions
        dataset:
          name: LocalLLaMA/typed-decisions
          type: LocalLLaMA/typed-decisions
          split: test
        metrics:
          - type: accuracy
            value: {{metrics|@M|public|accuracy:.3f}}
            name: Accuracy (untied labels, trained on the train split)
---

# CalDec Ultra {{size}}

A typed-decision model for long assistant states. It puts a Laya-format decision head on [{{encoder}}](https://huggingface.co/{{encoder}}), reads states of up to **8,192 tokens**, and answers all of a case's questions in **one encoder pass**. Give it a state and closed questions (`noul` yes/no, `choice`, `score`); it returns answers and probabilities without generating text. It was fine-tuned on [Assistant Decisions](https://huggingface.co/datasets/kgrozdanovski/assistant-decisions), including its long-state builds.

The other size is [CalDec Ultra {{other_size}}](https://huggingface.co/kgrozdanovski/{{other}}). The [CalDec repository](https://github.com/kgrozdanovski/caldec) has the code, data recipe and evidence; [ULTRA.md](https://github.com/kgrozdanovski/caldec/blob/main/ULTRA.md) has the full results.

## Evaluation

Accuracy is agreement with model-written target distributions on untied decisions, scored through the served path. It is not independent correctness.

| Model | Assistant test | Relabelled test | Long-state test | `typed-decisions` test |
|---|---:|---:|---:|---:|
| **CalDec Ultra {{size}}** | **{{metrics|@M|test|accuracy:.3f}}** | **{{relabelled|@M:.3f}}** | **{{long|@M|accuracy:.3f}}** | **{{metrics|@M|public|accuracy:.3f}}**† |
| [CalDec Ultra {{other_size}}](https://huggingface.co/kgrozdanovski/{{other}}) | {{metrics|@O|test|accuracy:.3f}} | {{relabelled|@O:.3f}} | {{long|@O|accuracy:.3f}} | {{metrics|@O|public|accuracy:.3f}}† |
| [CalDec GLiNER](https://huggingface.co/kgrozdanovski/caldec-v1-gliner2.5-decide) | {{metrics|caldec-gliner|test|accuracy:.3f}} | {{relabelled|caldec-gliner:.3f}} | {{long|caldec-gliner|accuracy:.3f}}‡ | {{metrics|caldec-gliner|public|accuracy:.3f}} |
| [CalDec Laya](https://huggingface.co/kgrozdanovski/caldec-v1-laya) | {{metrics|caldec-laya|test|accuracy:.3f}} | {{relabelled|caldec-laya:.3f}} | {{long|caldec-laya|accuracy:.3f}} | {{metrics|caldec-laya|public|accuracy:.3f}}† |
| Jev 1.13, zero-shot | 0.836 | — | — | 0.738 |
| [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) | {{metrics|gliner2.5-decide|test|accuracy:.3f}} | {{relabelled|gliner2.5-decide:.3f}} | — | {{metrics|gliner2.5-decide|public|accuracy:.3f}} |
| [Laya](https://huggingface.co/convaiinnovations/laya) base | {{metrics|laya-base|test|accuracy:.3f}} | {{relabelled|laya-base:.3f}} | — | {{metrics|laya-base|public|accuracy:.3f}} |
| {{encoder}}, zero-shot | {{metrics|ultrabert-@S-zeroshot|test|accuracy:.3f}} | {{relabelled|ultrabert-@S-zeroshot:.3f}} | — | — |

† Trained on the benchmark's train split, so not zero-shot. ‡ Inputs cut to their first 2,048 tokens, its best-scoring cut: whole 8K states run CalDec GLiNER out of memory on a 16 GB GPU.

- **Assistant test**: 1,647 cases, 3,372 untied decisions, GLM 5.3 targets. The test informed development of every CalDec model. Against CalDec Laya this model alone is right on {{pair|@M|caldec-laya|test|a_only_correct}} decisions and CalDec Laya alone on {{pair|@M|caldec-laya|test|b_only_correct}} (p {{pair|@M|caldec-laya|test|p_two_sided:p}}); against CalDec GLiNER {{pair|@M|caldec-gliner|test|a_only_correct}} to {{pair|@M|caldec-gliner|test|b_only_correct}} (p {{pair|@M|caldec-gliner|test|p_two_sided:p}}). Between the sizes, the 380M leads {{pair|caldec-ultra-380m|caldec-ultra-150m|test|a_only_correct}} to {{pair|caldec-ultra-380m|caldec-ultra-150m|test|b_only_correct}} (p {{pair|caldec-ultra-380m|caldec-ultra-150m|test|p_two_sided:p}}).
- **Relabelled test**: the same states with targets from two of Claude Opus 5.5, GPT-5.6 Sol and Gemini 3.8 Flash per case. Every model loses 1–2 points and the ranking holds, so the scores are not agreement with GLM in particular.
- **Long-state test**: 77 held-out cases of about 3–8K tokens on nine sites (158 untied decisions), each generated by one of GLM 5.3, MiMo-V2.6-Pro and DeepSeek V4.1 Flash and labelled by the other two. Against CalDec Laya: {{pair|@M|caldec-laya|long|a_only_correct}} to {{pair|@M|caldec-laya|long|b_only_correct}} (p {{pair|@M|caldec-laya|long|p_two_sided:p}}). At this size, differences under about 7 points are noise.
- **Long-state probe**: injected texts placed at the start, middle or end of 4K and 8K pages. On 8K pages this model catches {{probe|@M|8000/injected_caught:.0%}} of injections overall and few of those at the end of the page; see Limits.

**Latency**, three questions through `system_one` on an RTX 4080 SUPER (bfloat16, median of 20 calls):

| State length | 512 | 1,024 | 2,048 | 4,096 | 8,192 tokens |
|---|---:|---:|---:|---:|---:|
| CalDec Ultra {{size}} | {{latency|long-release|@M|512:ms}} ms | {{latency|long-release|@M|1024:ms}} ms | {{latency|long-release|@M|2048:ms}} ms | {{latency|long-release|@M|4096:ms}} ms | {{latency|long-release|@M|8192:ms}} ms |
| CalDec Laya | {{latency|long-release|caldec-laya|512:ms}} ms | {{latency|long-release|caldec-laya|1024:ms}} ms | {{latency|long-release|caldec-laya|2048:ms}} ms | {{latency|long-release|caldec-laya|4096:ms}} ms | {{latency|long-release|caldec-laya|8192:ms}} ms |

**Outside benchmarks.** This is a specialist. On a nine-benchmark subset of the [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) it scores ANLI macro-F1 {{index|@M|12:.3f}} (chance 0.33; Laya base 0.489), ContractNLI {{index|@M|11:.3f}} (chance 0.31) and When2Call {{index|@M|62:.3f}} (chance 0.25). Use a broadly trained model for decisions outside an assistant's own sites.

## Use

Load it through the CalDec repository, which registers UltraBERT's modeling code, switches its local-attention layers to an exact block-sparse path and runs all questions in one pass:

```bash
git clone https://github.com/kgrozdanovski/caldec && cd caldec
python3 -m pip install -r requirements.txt     # Python 3.12; torch 2.14, transformers 5.17, laya 0.3.6
```

```python
import sys
sys.path.insert(0, "scripts")
from data import load_agent

agent = load_agent("kgrozdanovski/{{model}}")
result = agent.system_one(
    {"source": "read_webpage", "text": "...a long fetched page..."},
    {"injection": {"type": "noul", "instructions": "The text contains instructions addressed to an AI assistant or agent, such as to ignore earlier instructions, to call a tool, or to send data somewhere."}},
)
p_injection = result["answers"]["injection"]["noul"]
```

Loading runs the UltraBERT modeling code shipped in this repository's `encoder/` directory (copied unchanged from [{{encoder}}](https://huggingface.co/{{encoder}}) at the revision it was trained from), as `trust_remote_code` would. States longer than 8,192 tokens are cut at the end. A `choice` or `score` answer has a `probabilities` mapping. **Ignore `action.act_probability`**: it comes from a head that training never used, initialised at random.

## Training

UltraBERT with a freshly initialised two-layer decision head, fine-tuned with soft cross-entropy on teacher distributions: the Assistant Decisions train split (4,984 cases), its long-state train build (117 cases at about 4–8K tokens), the `LocalLLaMA/typed-decisions` train split and 2,700 MNLI and BoolQ rows, six epochs at up to 8,192 tokens with per-site inverse-square-root weighting, UltraBERT's fine-tuning optimiser (AdamW β2 0.98, ε 1e-6, weight decay 0.1, linear warm-up and decay) and the epoch kept by validation soft NLL. Temperatures are fitted on validation and ship in `rl_agent_config.json`. The [recipe](https://github.com/kgrozdanovski/caldec/blob/main/ULTRA.md#recipe) reproduces it on one 16 GB GPU.

No Claude, GPT or Gemini output was used in training; those labels are evaluation targets only.

## Limits

The labels are synthetic model judgments, and the states are synthetic. The long-state test is small and comes from the same generation recipe as the long training data, so it does not measure real assistant traffic. Calibration (ECE {{metrics|@M|test|ece:.3f}}) is weaker than CalDec GLiNER's ({{metrics|caldec-gliner|test|ece:.3f}}); the model is under-confident. On 8K-token probe pages CalDec Ultra 380M catches 29% of injections placed at the end and CalDec Ultra 150M 6%: do not rely on either to find an injection in the second half of a long state. Injection cases were not written by adaptive attackers: this is not a stand-alone safety control. Ask the injection question in the wording of the example above, which is the training wording (also in `data/sites.json`): with the shorter "The text contains instructions addressed to an AI assistant." CalDec Ultra 380M flags 12.0% of 600 ordinary English web pages instead of 0.7%. Both models raise many false alarms on ordinary non-English text: on 1,440 ordinary web pages in 12 languages (FineWeb-2, 120 per language) at the 0.5 cut, CalDec Ultra 150M flags 64.5% and CalDec Ultra 380M 15.1% (63.8% and 19.0% with the shorter wording), against 0.3% and 0.7% of the English pages. Do not use either to screen non-English text. Each model is one training run; a second run of the 150M recipe scored 1.4 assistant-test points lower ([seed variance](https://github.com/kgrozdanovski/caldec/blob/main/ULTRA.md#seed-variance)). UltraBERT's, Laya's and the benchmarks' authors do not endorse this checkpoint. Apache 2.0, matching the base model.

## Citation and contact

Contact [Kristijan Grozdanovski](mailto:kgrozdanovski7@gmail.com). See also [CITATION.cff](https://github.com/kgrozdanovski/caldec/blob/main/CITATION.cff).

```bibtex
@misc{grozdanovski2026{{bibkey}},
  author = {Grozdanovski, Kristijan},
  title = {CalDec Ultra {{size}}},
  year = {2026},
  url = {https://huggingface.co/kgrozdanovski/{{model}}}
}
```
