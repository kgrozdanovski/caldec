# NPU scripts: CalDec Ultra on the Rockchip RK3566 NPU

The export, conversion, runner and experiment sources behind [NPU.md](../../NPU.md). The released pipeline is
listed first; the rest are the experiments the write-up cites, kept as they were run (on an earlier development
checkpoint; not maintained).

**Paths and settings.** Scripts find this repo from their own location, read and write their outputs next to
themselves (`out/`, `packed/onnx/`, `packed/tables_s*/`, `runner/`; all git-ignored), and take:
- `CALDEC_CKPT`: the checkpoint (default `kgrozdanovski/caldec-ultra-150m` at the revision `scripts/data.py` pins;
  a local directory works too);
- `CALDEC_PY`: this repo's Python (the shell scripts default to `../../.venv/bin/python`);
- `BOARD`, `SSH_OPTS`: the board's `user@host` and extra ssh options, for the scripts that drive it;
- `NPU_MODELS`, `CALDEC_CKPT_DIR`, `RKNN_RT`: for `runner/install-service.sh` ([npu/run-on-rockchip.md](../../npu/run-on-rockchip.md) §3).

**Environments:**
- **Python export scripts:** this repo's environment with `PYTHONPATH=pylib` (onnx 1.17.0, onnxscript 0.7.2 and
  ml_dtypes 0.7.0, installed with `pip --target pylib`) and `CALDEC_DENSE_ATTENTION=1`.
- **Conversion scripts:** rknn-toolkit2 2.5.0 in its own venv, `rknn-venv/bin/python`.
- **C programs:** built on the board with `gcc -O2 -o X X.c -I. -L. -lrknnrt -Wl,-rpath,'$ORIGIN'`, next to
  `librknnrt.so` and `rknn_api.h` (rknn-toolkit2 2.5.0 runtime; not included).
- **`board/`:** the Panther X2 device-tree overlays (with and without the NPU IOMMU).

## Released pipeline (packed model)

| File | Runs on | Purpose |
|---|---|---|
| `packed/export_packed.py` | PC | `prep`: 400 packed assistant-test cases with fp32 references; `export --seq S` (`ACT=1`): the packed grid model with the `hcls` output, with an fp32 check against `packed.forward` |
| `packed/build_act.sh`, `packed/build_long.sh` | PC | the `pk2` set: S=384 with weights, weightless 96-320, then 512 and 640 |
| `convert_generic.py` | PC, rknn venv | ONNX → RKNN for rk3566: fp16 (or `--int8 dataset.txt`), `--opt N`, extra `config()` keys such as `remove_weight=True`; `RKNN_VERBOSE=1` prints op targets |
| `export_grid.py` | PC | the encoder graph the packed model builds on: token grid, NPU-only attention, `--tables-input` for weight sharing, `ROPE_SWAP` (default `mm`) |
| `export_caldec.py` | PC | shared helpers: `CALDEC`, `CKPT`, `load_agent`, masks |
| `runner/export_act_head.py` | PC | the act head as `runner/act_head.npz` |
| `runner/caldec_npu_server.c`, `runner/caldec_npu.py` | board | the runner: all buckets loaded with shared weights and zero-copy inputs; numpy + `tokenizers` front-end with `system_one` |
| `runner/service/`, `runner/install-service.sh` | board / PC | the socket-activated service and its installer |
| `runner/test_tokens.py`, `make_ref.py`, `compare_answers.py`, `compare_act.py`, `make_public200.py`, `acc_public.py`, `exp_long_cases.py` | PC | the validation behind the numbers in NPU.md |

## Per-question pipeline (development)

| File | Runs on | Purpose |
|---|---|---|
| `gen_cases.py` | PC | test or validation cases with fp32 reference logits (`out/cases_<tag>_s256.npz`) plus raw device inputs |
| `make_bucket_cases.py`, `make_order.py` | PC | per-bucket device inputs and the `multi_bench` order file |
| `multi_bench.c`, `multi_bench2.c` | board | deployment benchmark: all bucket models in one process, weight sharing, `ZC=1` zero-copy; `multi_bench2` finds inputs by name |
| `compare_npu.py`, `compare_bucket.py` | PC | agreement, max and median \|Δp\|, flips with fp32 margins |

## Earlier graphs and harnesses

| File | Purpose |
|---|---|
| `export_npu.py` | NPU-only attention in the row layout (`[S, E]` matmuls): `--exp-pow`, `--fuse`, `--opt2`, `--prune-head`, `--softmax npu\|npu-sum\|std` |
| `convert_rknn.py` | Conversion of the original export (`--masks --sel [--embeds] [--onnx --tag]`), with a simulator check |
| `rknn_caldec_bench.c` | Single-model device benchmark: p50/p95, CPU user/sys time inside `rknn_run`, logits out |
| `bench_buckets.py` | Bucketed run driven from the PC over SSH: copies a model per bucket, runs it, reassembles and scores the logits |
| `bench_ort.py` | The CPU baseline with onnxruntime |
| `share_test.c` | Weight-sharing test: a `remove_weight` child model initialised from the full model gives bit-identical outputs |
| `native_attr.c` | Prints the native input attributes (type, layout, stride) of a model |
| `memspeed.c` | CPU read/write speed of NPU buffers vs `malloc` (found the uncached mapping) |
| `dump_compare.py` | Per-layer device dumps (`RKNN_LOG_LEVEL=5`) against onnxruntime intermediates (found the GatherElements bug) |
| `fp16_check.py`, `hybrid_check.py` | fp16 range check in PyTorch; scoring of hybrid int8 builds |

## Micro-experiments

| Directory | Purpose |
|---|---|
| `ops/` | Which softmax and activation formulations land on the NPU (`softmax_variants.py`, `sm2.py`, `act.py`, `targets.sh` reads the op tables) |
| `attn/` | Attention block formulations (`attn_variants.py`, `attn3d.py`): the NPU-only attention came from here |
| `acc/` | Activation LUT accuracy on the device (`act_acc.py`): found the ELU cancellation |
| `parts/` | Timing of single components (q·k, P·V, RMSNorm, MLP, elementwise chains) |
| `lay/` | Layout test (`lay.py`: matmul vs conv vs grid) and the grid-shape and length sweep (`sweep.py`) |
| `clk/` | NPU-bound conv probe for clock scaling (`make_convstack.py`) |
| `hyb/`, `hyb_full/`, `hyb_grid/` | Hybrid int8 (`hybrid_quantization_step1/2`): 3-layer, full row-layout and grid MLP variants. The generated `.quantization.cfg` files were not copied (step 1 recreates them) |

## Round 2: chunking and int8

| File | Purpose |
|---|---|
| `chunk/chunk.py`, `chunk/chunk2.py` | Token-axis split of 1x1 conv chains (768x768; wide 768↔2048/2304/3072), plain vs halves |
| `export_grid.py` env `CHUNK_HALVES=1`, `CHUNK_MLP=1` | Same split in the full model (measured slower or no change; off by default) |
| `smooth/smooth_stats.py`, `smooth/export_smooth.py` | Per-channel MLP activation maxima; exact SmoothQuant-style scale migration before export |
| `hyb_grid/step2s.py` (modes `gate_up`, `down` added), `hyb_grid/step2f.py` | Hybrid int8 selection; step2f is the variant the toolkit rejects (inputs/constants as keys) |
| `compare_bucket.py` | Agreement for one bucket's logits against the fp32 references |
| `packed_fit.py` | Tokenizer-only count of packed cases that fit 256/384/512 tokens |

## Round 3: packed multi-question encoding (`packed/`)

| Script | Where | What |
|---|---|---|
| `packed/export_packed.py` | PC | `prep`: 400 packed test cases with fp32 references (at 384 and 1024); `export --seq S`: the packed grid model (encoder grid + one-hot view gather + block-diagonal head), with an fp32 exactness check; `cases`: per-bucket device inputs |
| `packed/build384.sh`, `build_rest.sh` | PC | the 384 model with weights, and the `remove_weight` buckets 96-320 |
| `packed/make_order_pk.py` | PC | `multi_bench2` order file (384 first, others share) |
| `packed/compare_packed.py` | PC | agreement with packed fp32, accuracy on untied labels |
| `packed/gen_single.py`, `compare_single.py` | PC | the like-for-like baseline: the same cases' questions for the deployed per-question set |
| `packed/stats.py`, `bucket_model.py` | PC | packed shape statistics; expected time for other bucket sets |
| `packed/build_sh128.sh` | PC | head-row cap experiment (S=256, 128 rows) |
| `multi_bench2.c` | device | `multi_bench` with inputs found by name (tables from `<tablesdir>/<name>.bin`), for any model |
| `rknn_caldec_bench.c` | device | now 16 input slots (was 8) |

## Runner (`runner/`)

| Script | Where | What |
|---|---|---|
| `runner/caldec_npu_server.c` | device | all bucket models loaded (shared weights, zero-copy); binary requests on stdin/stdout |
| `runner/caldec_npu.py` | device | numpy + tokenizers front-end: `CalDecNPU(dir).system_one(state, questions)` and a JSON-lines CLI |
| `runner/stage.sh` | PC | the round-3 staging of the runner (`pk_*` set, a development board layout under `~/npu-bench`); superseded by `install-service.sh` |
| `runner/test_tokens.py` | PC | runner inputs vs caldec `packed.build_case` / `export_packed.inputs_for` (400 cases) |
| `runner/make_ref.py`, `make_ref_public.py` | PC | requests and `PackedAgent.system_one` references |
| `runner/compare_answers.py`, `compare_public.py` | PC | agreement, accuracy, latency split |
| `packed/stats_public.py` | PC | packed shapes on the public test (none fits one pass) |

## Round 4: act probability, long states, the service (`packed/`, `runner/`)

| File | Where | What |
|---|---|---|
| `packed/export_packed.py` with `ACT=1` | PC | adds input `cls_sel` and output `hcls` (each question's head row 0) for the act head; files `pk2_*` |
| `packed/build_act.sh`, `packed/build_long.sh` | PC | the pk2 set: S=384 with weights + weightless 96…320, then 512 and 640 |
| `runner/export_act_head.py` | PC | the act head as `act_head.npz` |
| `runner/caldec_npu.py`, `runner/caldec_npu_server.c` | device | `act_probability` (numpy act head); server returns every model output |
| `runner/test_act_head.py`, `runner/compare_act.py` | PC | numpy vs torch act head; NPU vs fp32 `act_probability` |
| `runner/exp_long_cases.py` | PC | fp32: one pass vs the runner's split at max_len 384/512/640/1024 on public cases |
| `runner/make_public200.py`, `runner/acc_public.py` | PC | 200 public requests + gold; accuracy and latency of runner answers |
| `runner/service/` | device | `caldec_npu_service.py` (socket-activated JSON-lines service), `caldec-npu` client, systemd units |
| `runner/install-service.sh` | PC | installs the service in `/opt/caldec-npu` on a board, from the build outputs or a downloaded set (`NPU_MODELS`) |


## Generic harness (`research/`)

| File | Where | What |
|---|---|---|
| `research/npu_run_generic.c` | device | runs any RKNN model (fp16 or int8, up to 64 inputs/outputs) through `rknn_inputs_set`; `NPU_U8_NHWC=1` feeds image inputs as uint8 NHWC |
| `research/npu_generic.py`, `research/npu_client.py` | device | batch driver (npz in/out, timing) and an interactive client, same protocol as `caldec_npu_server` |

Used to run other models (image and audio) on the same runtime.
