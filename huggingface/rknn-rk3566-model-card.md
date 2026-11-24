---
license: apache-2.0
base_model: kgrozdanovski/caldec-ultra-150m
base_model_relation: quantized
language:
  - en
pipeline_tag: text-classification
tags:
  - typed-decisions
  - decision-model
  - rknn
  - rockchip
  - rk3566
  - rk3568
  - npu
  - edge
---

# CalDec Ultra 150M for the Rockchip RK3566 NPU (RKNN, fp16)

[CalDec Ultra 150M](https://huggingface.co/kgrozdanovski/caldec-ultra-150m) converted for the NPU of a Rockchip
RK3566 with rknn-toolkit2 2.5.0, in fp16, without retraining. On a Panther X2 board it answers at **311 ms per
question** end to end and agrees with the fp32 model on **856 of 858** assistant-test questions.

The write-up (results, how it works, the driver fix, limits) is
[NPU.md](https://github.com/kgrozdanovski/caldec/blob/main/NPU.md) in the CalDec repository, and the scripts that
built and validated these files are in [`scripts/npu/`](https://github.com/kgrozdanovski/caldec/tree/main/scripts/npu).

## Source model

| | |
|---|---|
| Model | [`kgrozdanovski/caldec-ultra-150m`](https://huggingface.co/kgrozdanovski/caldec-ultra-150m/tree/fcad869cd222f235f9366804c721f319fcfd842b) at revision `fcad869cd222f235f9366804c721f319fcfd842b` (the revision the CalDec repo pins) |
| `model.safetensors` sha256 | `ddf6769ae0666a92dd38c88bb05b4864b778489f219d123973d26d43b0f78b3b` |
| `tokenizer/` and `rl_agent_config.json` | copied unchanged from that revision |

## Files

| File | What |
|---|---|
| `pk2_s384.rknn` | the 384-token bucket model, with the weights (381 MB) |
| `pk2_s{96,128,160,192,224,256,320,512,640}_now.rknn` | weightless bucket models; they share `pk2_s384.rknn`'s weights at run time (`RKNN_FLAG_SHARE_WEIGHT_MEM`) |
| `tables_s<S>/` | per-bucket constant inputs (`rope_tab.bin`, `s_scale.bin`, `h_scale.bin`), written once at load |
| `act_head.npz` | the act head (Linear 772→256, GELU, Linear 256→2) for the CPU, as numpy arrays |
| `tokenizer/`, `rl_agent_config.json` | from the source model: the tokenizer and the calibration temperatures |
| `SHA256SUMS` | sha256 of every file above |

## Requirements

- **NPU:** RK3566 (target platform `rk3566`). The driver also targets RK3568; these files were not tested there.
- **Runtime:** `librknnrt.so` **2.5.0** and `rknn_api.h` from
  [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) tag `v2.5.0`. They are not included here.
  `.rknn` files are tied to the runtime version they were built for.
- **Kernel driver:** on a mainline kernel, the out-of-tree `rknpu` driver with the fixes in
  [this fork](https://github.com/kgrozdanovski/rk3568-npu-ODROID-M1/tree/panther-x2) (sent upstream as
  [PR #4](https://github.com/vitalijborissow/rk3568-npu-ODROID-M1/pull/4)). Without them the model runs at about half
  speed. A vendor (BSP) kernel usually ships the driver built in; that was not tested.
- **On the board:** Python with numpy and `tokenizers` for the runner; no torch.

## Use

The runner (`caldec_npu_server.c` + `caldec_npu.py`) and a systemd socket service are in the CalDec repository. With
the repo checked out on a PC:

```
hf download kgrozdanovski/caldec-ultra-150m-rknn-rk3566 --local-dir caldec-npu-set
NPU_MODELS=$PWD/caldec-npu-set RKNN_RT=<dir with librknnrt.so and rknn_api.h> \
  scripts/npu/runner/install-service.sh user@board
ssh -t user@board 'sudo bash /tmp/caldec-npu-stage.XXXXXXXX/caldec-npu-install.sh pk2'   # the path it prints
ssh user@board caldec-npu < requests.jsonl > answers.jsonl
```

A request line is `{"state": ..., "questions": {...}}`, as for `system_one` in the source model's card; the answer
has the same format (choice, probabilities, confidence, act probability). Cases longer than one pass (640 tokens,
192 head rows or 12 option markers) are split into several passes.

## Validation

Measured on one Panther X2 (RK3566, 4 GB), kernel 6.18, librknnrt 2.5.0, through the installed service, against the
same model in fp32 PyTorch:

| | Assistant test | Public test |
|---|---:|---:|
| Cases, questions | 400, 858 (2.15 per case) | 200, 1,000 (5 per case) |
| NPU passes per case | 1.00 | 2.00 (split: more than 192 head rows) |
| **Time per question**, end to end | **311 ms** | 625 ms |
| Time per case, median (p95) | 626 ms (1,083) | 2,163 ms (5,300) |
| Same answer as fp32 | **856/858** (both flips at fp32 margins ≤ 0.007) | |
| \|Δp\| vs fp32, max / median | 0.012 / 0.0011 | |
| `act_probability` vs fp32 | max \|Δ\| 0.0006; same side of 0.5 on 858/858 | |
| **Accuracy, NPU** (untied labels) | **0.8489** (834) | **0.7462** (981) |
| Accuracy, fp32, same split rule | 0.8489 | 0.7452 |
| Accuracy, fp32, one pass (assistant: full; public: 1,024 tokens) | 0.8489 | 0.7686 |
| NPU memory, all 10 buckets loaded | about 785 MB | |

Accuracy is agreement with the benchmarks' synthetic target labels on untied decisions. The assistant
cases are the first 400 of the test split, and every one fits a single pass. The public cases are the first 200 of
`LocalLLaMA/typed-decisions` test: each needs two passes, which costs 2.2 points against one fp32 pass over the whole
case. The service ran with the CPU governor at `performance`. The answers behind these numbers are in
[`runs/evidence/npu/`](https://github.com/kgrozdanovski/caldec/tree/main/runs/evidence/npu).

## Limits

- One board and one runtime version were tested.
- fp16 only. int8 was tried on earlier development checkpoints and failed the accuracy bar.
- Per pass, the state is cut to fit 640 tokens; the source model reads up to 8,192.
- The probabilities use the source model's calibration temperatures; calibration on the NPU was not re-measured
  beyond the agreement and |Δp| figures above.

## Licence

The converted files are derived from CalDec Ultra 150M and carry its licence. The Rockchip runtime is not included
and has its own terms.
