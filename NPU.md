# CalDec Ultra on a Rockchip NPU (RK3566)

[CalDec Ultra 150M](ULTRA.md) runs on the NPU of a Rockchip RK3566, an edge chip found in cheap single-board computers, at **311 ms per question** end to end, in fp16, without retraining. On 858 assistant-test questions it
picks the same answer as the original fp32 model on **856**, and scores **0.8489** on the untied labels, the same
as fp32. The converted model set is on Hugging Face as
[`kgrozdanovski/caldec-ultra-150m-rknn-rk3566`](https://huggingface.co/kgrozdanovski/caldec-ultra-150m-rknn-rk3566).

Getting there took a kernel driver fix, an attention rewrite that keeps every op on the NPU, and a packed graph that
answers all of a case's questions in one encoder pass. This page has the result, how to run it, how it works and
what did not work. The appendices in [`npu/`](npu/) have the detail, and [`scripts/npu/`](scripts/npu/) has every
script.

| | |
|---|---|
| Model | [CalDec Ultra 150M](https://huggingface.co/kgrozdanovski/caldec-ultra-150m) at its pinned revision, `model.safetensors` sha256 `ddf6769ae0666a92…` (full hash in the model card) |
| Hardware | Panther X2 (RK3566: 4x Cortex-A55, one NPU core, 4 GB), passively cooled |
| Software | Armbian trixie, mainline kernel 6.18, out-of-tree `rknpu` driver with [our fixes](#the-driver-fix), rknn-toolkit2 / librknnrt **2.5.0** |
| Precision | fp16 throughout; no int8, no retraining |
| Context | packed buckets of 96-640 tokens; longer cases split into several passes |

## Results

Measured on the board through the installed service (`caldec-npu`), text in and answers out, so the times include
tokenisation, input building and the CPU post-processing. References are the same checkpoint in fp32 PyTorch
(`PackedAgent.system_one`) on the PC.

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

**Reading the numbers.**
- **Agreement** is the share of questions where the NPU picks the same option as fp32. **Accuracy** is agreement
  with the benchmark's synthetic target labels on untied decisions, as everywhere in this repo. The two are
  different measures.
- **Assistant test:** the first 400 cases of the test split. All of them fit one pass of at most 640 tokens, so
  the fp32 reference is the full model on the same input.
- **Public test:** the first 200 cases of `LocalLLaMA/typed-decisions` test. These have 5 questions over ~600-token
  states, so none fits one pass; the runner splits them into two. Its accuracy is compared with fp32 under the same
  split rule (`runner/exp_long_cases.py` re-implements it) and with fp32 in one 1,024-token pass, which reads all of
  these states. [ULTRA.md](ULTRA.md)'s 0.782 is over all 400 public cases, so it is not comparable with this
  200-case subset.
- **Timing conditions:** the service sets the CPU governor to `performance` while it runs; the NPU runs at its
  default governor (up to 900 MHz). The answers behind every number are in
  [`runs/evidence/npu/`](runs/evidence/npu/).
- **Not measured on the released model:** a CPU baseline. On an earlier development checkpoint the same encoder took
  4782 ms per question on the board's four CPU cores (onnxruntime fp32, 256 tokens) against 863-911 ms on the NPU at
  the same length ([development.md](npu/development.md)).

## Run it

The full guide, with versions and the board-specific steps, is [npu/run-on-rockchip.md](npu/run-on-rockchip.md).
In short:

1. **NPU driver.** On a mainline kernel, the out-of-tree `rknpu` driver from
   [our fork](https://github.com/kgrozdanovski/rk3568-npu-ODROID-M1/tree/panther-x2) (DKMS), plus a device-tree
   overlay on boards whose tree has no NPU node (`scripts/npu/board/` for the Panther X2). A vendor (BSP) kernel
   usually has the driver built in; that path is untested here.
2. **Runtime.** `librknnrt.so` 2.5.0 and `rknn_api.h` from
   [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) `v2.5.0`. They are not redistributed
   here, and the `.rknn` files need exactly this version.
3. **Model set and service.**
   ```
   hf download kgrozdanovski/caldec-ultra-150m-rknn-rk3566 --local-dir caldec-npu-set
   NPU_MODELS=$PWD/caldec-npu-set RKNN_RT=<runtime dir> scripts/npu/runner/install-service.sh user@board
   ssh -t user@board 'sudo bash /tmp/caldec-npu-stage.XXXXXXXX/caldec-npu-install.sh pk2'   # the path it prints
   ssh user@board caldec-npu < requests.jsonl > answers.jsonl
   ```
   A request line is `{"state": ..., "questions": {...}}`, the same as `system_one` in [ULTRA.md](ULTRA.md); the
   answer has `system_one`'s format. The board needs only numpy and `tokenizers`, no torch.
4. **Rebuild from the checkpoint** (optional): `scripts/npu/packed/export_packed.py`, `build_act.sh` and
   `build_long.sh` export and convert the 10 bucket models in about 30 minutes on 4 PC cores
   ([run guide §4](npu/run-on-rockchip.md#4-rebuild-the-set-from-the-checkpoint)).

## How it works

The RK3566 NPU has no kernels for Softmax, Exp, ReduceMax, elementwise Max or a working GatherElements. Every such op
falls back to the CPU, and on this chip each CPU op costs more than its arithmetic because the runtime converts
layouts around it. The graph is therefore rewritten so that nothing runs on the CPU inside the model, and the
rewrite is checked in fp32 against the original model (max |Δlogit| about 1e-5) before conversion.

1. **Attention without CPU ops:** the row max is a `MaxPool`, `exp` is `(ELU(z/2) + 1)^2` (ELU has an NPU kernel and
   the halving avoids an fp16 cancellation), and the row sum is an `AvgPool`. Heads are the batch of a 3-D matmul.
2. **Token grid:** the hidden state stays a `[1, 768, S/16, 16]` feature map, so every linear layer is a 1x1
   convolution, the NPU's fastest path (8 chained 768-wide linears: 13.8 ms as convs on the grid, 21.5 ms as
   matmuls).
3. **Selection by one-hot matmuls:** the option markers and each question's view are picked with one-hot matrices
   instead of `GatherElements`, which returns wrong values on the device.
4. **Packed questions:** one encoder pass reads all of a case's questions ahead of one copy of the state; the
   decision head runs over every question's view at once under a block-diagonal mask, which is exactly
   per-question attention. On an earlier development checkpoint, with 2.1 questions per assistant case, this cut
   time per question by about a third against the per-question graph.
5. **Length buckets and weight sharing:** 10 static-shape models (96 to 640 tokens) share the weights of the 384
   model at run time, so all of them fit in about 785 MB of NPU memory. Bucketing is exact because padding is masked.
6. **RoPE as an input table**, with the half-swap written as a permutation matmul: the forms that look simpler crash
   the toolkit or put reshapes on the CPU.

[npu/how-it-works.md](npu/how-it-works.md) explains each step with code and measurements.

## The driver fix

The out-of-tree `rknpu` driver that brings the RK3566/RK3568 NPU to mainline kernels
([vitalijborissow/rk3568-npu-ODROID-M1](https://github.com/vitalijborissow/rk3568-npu-ODROID-M1)) maps NPU buffers
**uncached** on mainline arm64, so every CPU step of the runtime ran about 10x slower than it should; CalDec ran at
half speed. Our fork, branch
[`panther-x2`](https://github.com/kgrozdanovski/rk3568-npu-ODROID-M1/tree/panther-x2), fixes that and three more
problems, and is sent upstream as
[pull request #4](https://github.com/vitalijborissow/rk3568-npu-ODROID-M1/pull/4):

- the cacheable shmem-pages allocation path, as every other mainline port uses (CPU reads of NPU memory 165 →
  1970-2840 MB/s);
- the `iommu_dma_cookie` layout of Linux ≥ 6.15, without which that path hung the board;
- a GEM `.close` callback, so a crashed process no longer leaks its NPU memory until reboot;
- a DRM-only build (the misc-device path oopsed).

Effect on the first CalDec graph (an earlier development checkpoint, 256 tokens): 3674 → 1778-1809 ms per question,
with bit-identical outputs. The mainline `rocket`
driver is gaining RK3568 support, which would eventually make the out-of-tree driver unnecessary.

## Limits

- **One board, one runtime.** Everything was measured on one Panther X2 with rknn-toolkit2 / librknnrt 2.5.0. Other
  RK3566 and RK3568 boards should behave the same but were not tested, and the `.rknn` files are tied to the 2.5.0
  runtime and the `rk3566` target.
- **Length.** One pass holds 640 tokens, 192 head rows and 12 option markers. Larger cases are split into passes,
  each with the state cut to fit. On the public test that split costs 2.2 points against one full fp32 pass (0.7462
  vs 0.7686), and on states much longer than 640 tokens the NPU sees less than the 8,192-token model would.
- **Latency.** The fastest case took 264 ms and one-question cases a median of 383 ms. Per question it
  took 272 ms for cases up to 128 tokens, 264 ms at 129-256 (more questions share each pass), 453 ms at 257-384,
  and 625 ms on the public test's 5-question, ~570-token cases, which need two passes. This is a model for background decisions, not for per-keystroke use.
- **One request at a time.** There is one NPU core; the service serialises requests.
- **Calibration.** The probabilities use the released model's temperatures; their calibration on the NPU was not
  re-measured beyond the |Δp| figures above.

## What did not work

All measured on earlier development checkpoints with the same architecture.

- **int8** (tried on an earlier development checkpoint). Three int8 variants failed a fixed accuracy bar (240/400,
  382/400 and 48/90 agreement with fp32), and where int8 did run it saved only 10-14%. RKNN quantises activations per tensor, and the encoder's `down_proj`
  inputs have outlier channels thousands of times larger than the median; SmoothQuant-style migration did not fix
  it. int8 would need quantisation-aware training or rotations.
- **Splitting linears along the token axis:** faster in isolation, slower in the full model (Concat/Split traffic).
- **Fused projections and pruning the last head layer:** the compiler splits large convs back, and the small
  matmuls cost as much as they save.
- **Finer buckets:** at most 2-6% for many more models.

Every failure along the way, with its cause and fix, is in [npu/failures.md](npu/failures.md); the history and
every intermediate measurement are in [npu/development.md](npu/development.md).

## What could still help

- **A shallower encoder** (for example 16 of 24 layers, distilled): the encoder was 82% of the time at 128 tokens on
  the development checkpoint, so about 25-30% if accuracy holds. Needs a retrain.
- **int8 with outlier handling** (rotations or QAT matched to RKNN's per-tensor scheme): 15-25% at best with the
  current toolkit, by estimate.
- **Newer toolkit releases:** NPU Softmax/Exp kernels, or a fixed per-op profiler, would simplify the graph and the
  tuning.
- **The mainline `rocket` driver:** a maintenance path rather than a speed one; whether its Mesa userspace runs a
  transformer like this is unverified.

## Files

| Path | What |
|---|---|
| [npu/run-on-rockchip.md](npu/run-on-rockchip.md) | run guide: versions, overlay, driver, install, rebuild, the model's input contract, the C API, gotchas |
| [npu/how-it-works.md](npu/how-it-works.md) | the technical explanations, with code |
| [npu/development.md](npu/development.md) | history and every intermediate measurement (development checkpoints) |
| [npu/failures.md](npu/failures.md) | every failure, cause and fix |
| [scripts/npu/](scripts/npu/) | export, conversion, runner, service, validation and experiment sources |
| [Hugging Face: caldec-ultra-150m-rknn-rk3566](https://huggingface.co/kgrozdanovski/caldec-ultra-150m-rknn-rk3566) | the converted set, `SHA256SUMS` and the model card |

## Sources

### Driver

- Upstream out-of-tree driver for RK3568/RK3566 on mainline: <https://github.com/vitalijborissow/rk3568-npu-ODROID-M1>
  (base commit `66c96ec`)
- Fork with the fixes, branch `panther-x2` (dd74fb6): <https://github.com/kgrozdanovski/rk3568-npu-ODROID-M1/tree/panther-x2>
- Upstream pull request with the fixes: <https://github.com/vitalijborissow/rk3568-npu-ODROID-M1/pull/4>
- Other mainline ports read for comparison:
  <https://github.com/simonswine/rknpu-mainline>,
  <https://github.com/QNAP-android-internal/rknpu-mainline>,
  <https://github.com/heliosrun/linux-rknpu-rk3588>,
  <https://github.com/DrDitto/rknpu-module>,
  <https://github.com/lurenJBD/rknpu-mainline-dkms>
- Linux 6.18 `drivers/iommu/dma-iommu.c` (the `iommu_dma_cookie` layout without the leading `type` field):
  <https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/drivers/iommu/dma-iommu.c?h=v6.18>

### Mainline `rocket` driver and RK3568

- Kernel documentation: <https://docs.kernel.org/accel/rocket/index.html>
- RK3568 support, RFC v3, June 2026: <https://lists.infradead.org/pipermail/linux-rockchip/2026-June/071790.html>
- RFC v4, June 2026: <https://lists.infradead.org/pipermail/linux-rockchip/2026-June/072157.html>
- Port plan on the ROCK 3B: <https://github.com/MidG971/rk3568-npu-rocket-port/blob/main/PLAN.md>
- Userspace experiments: <https://github.com/mcthesw/rocket-userspace>

Status as found on 2026-11-22: patch series under review. No merge was found, and that was not checked against the
mainline tree.

### Rockchip toolkit and runtime

- rknn-toolkit2 and the runtime (`librknnrt.so`, `rknn_api.h`), tag `v2.5.0`: <https://github.com/airockchip/rknn-toolkit2>
- Model zoo (its transformer example, lite_transformer, ships fp16 only): <https://github.com/airockchip/rknn_model_zoo>

### Quantisation

- SmoothQuant (scale migration from activations to weights): Xiao et al., 2022, <https://arxiv.org/abs/2211.10438>
- LLM.int8() (outlier features): Dettmers et al., 2022, <https://arxiv.org/abs/2208.07339>
- Quantizable Transformers: removing outliers by helping attention heads do nothing (clipped softmax, gated
  attention; BERT, OPT and ViT): Bondarenko et al., 2023, <https://arxiv.org/abs/2306.12929>
- QuaRot (Hadamard rotations for outlier-free 4-bit inference, decoder LLMs): Ashkboos et al., 2024,
  <https://arxiv.org/abs/2404.00456>
- Rotation-based quantisation and learned vs fixed rotations (LLMs), 2025: <https://arxiv.org/abs/2506.15689>
- W8A8 PTQ on BERT-base collapses under uniform scaling; mixed-precision PTQ (2026): <https://arxiv.org/abs/2603.04308>
- AMD Quark QuaRot tutorial (per-tensor W8A8 on Llama-3-8B):
  <https://quark.docs.amd.com/release-0.9/pytorch/tutorial_quarot.html>
- Unsloth QAT (TorchAO fake-quant inside LoRA): <https://unsloth.ai/docs/blog/quantization-aware-training-qat>
