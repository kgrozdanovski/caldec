# CalDec on the RK3566 NPU: run guide

Part of [NPU.md](../NPU.md). This is the tested path: a Panther X2 (RK3566, 4 GB), Armbian trixie, mainline kernel
6.18 and the out-of-tree `rknpu` driver from a patched fork. Steps marked **(board-specific)** are for this board on
a mainline kernel. A board on a Rockchip vendor (BSP) kernel normally has the NPU driver built in and skips §1-§2;
that is **unverified** here. The driver also targets RK3568; running these files there is unverified.

## Versions used

| Component | Version |
|---|---|
| Kernel | 6.18.63 (Armbian trixie, aarch64) |
| Boot arguments **(board-specific)** | `cma=1024M movablecore=256M pd_ignore_unused clk_ignore_unused`. The driver's 32-bit DMA path cannot use the RAM bank above 4 GiB: either keep that bank movable (`movablecore=256M`, the current setup) or drop it (`mem=3800M`, used during development) |
| Overlay **(board-specific)** | `scripts/npu/board/panther-npu.dts`: NPU power domain 6, `vdd_npu` always-on, the NPU node `npu@fde40000` with clocks and OPPs, and the NPU IOMMU `iommu@fde4b000` |
| NPU driver | out-of-tree `rknpu` 0.9.8 (DKMS package `rknpu/1.0`), [fork branch `panther-x2`](https://github.com/kgrozdanovski/rk3568-npu-ODROID-M1/tree/panther-x2) @ `dd74fb6` = upstream `vitalijborissow/rk3568-npu-ODROID-M1` @ `66c96ec` + 5 commits, sent upstream as [PR #4](https://github.com/vitalijborissow/rk3568-npu-ODROID-M1/pull/4) |
| Device runtime | `librknnrt.so` 2.5.0 and `rknn_api.h` from [`airockchip/rknn-toolkit2`](https://github.com/airockchip/rknn-toolkit2) tag `v2.5.0` (`rknpu2/runtime/Linux/librknn_api/aarch64/`). Not redistributed here; the `.rknn` files need this version |
| Device compiler, Python | gcc 14.2, Python 3.13 with numpy and `tokenizers` (no torch) |
| Host converter | rknn-toolkit2 2.5.0 in its own venv (x86-64) |
| Host exporter | this repo's environment (torch), plus onnx 1.17.0, onnxscript 0.7.2 and ml_dtypes 0.7.0 installed with `pip --target scripts/npu/pylib` |

## 1. Device tree overlay (board-specific)

- **Why:** the mainline Panther X2 device tree has no NPU node.
- **What:** `scripts/npu/board/panther-npu.dts` adds the power domain, the regulator flags, the NPU with its clocks
  and OPP table (up to 900 MHz) and the NPU IOMMU, with values from the Panther's stock BSP tree.
  `panther-npu-noiommu.dts` is the variant without the IOMMU: small models run, large ones fail with a job timeout
  or an oops in `rknpu_dma_buf_sync`.
- **Build and install:** `dtc -@ -I dts -O dtb -o panther-npu.dtbo panther-npu.dts`, copy the `.dtbo` to
  `/boot/overlay-user/` and add `panther-npu` to `user_overlays=` in `/boot/armbianEnv.txt`.

## 2. NPU driver (mainline kernel)

1. **Sources:** the fork branch `panther-x2` (link above).
2. **DKMS:** copy `dkms.conf` and `drivers/` to `/usr/src/rknpu-1.0/`, then
   `dkms build -m rknpu -v 1.0 --force && dkms install -m rknpu -v 1.0 --force`. Do not run the upstream
   `install.sh`; it targets the ODROID-M1.
3. **Load:** `modprobe dma32_heap; modprobe rknpu` (or from a boot unit).
4. **Check:** `dmesg | grep -i rknpu` shows no warnings; `ls /dev/dri/by-path | grep npu` lists a render node (the
   build is DRM-only, with no `/dev/rknpu`); `cat /sys/class/devfreq/fde40000.npu/governor` prints
   `simple_ondemand`.

**Why the patched driver.** Unpatched, the driver maps NPU buffers uncached on mainline arm64, which makes every CPU
step of the runtime about 10x slower (CalDec at half speed); it hangs on kernels ≥ 6.15 once the cacheable path is
enabled (an `iommu_dma_cookie` layout change); and it leaks the NPU memory of a crashed process until reboot.
[how-it-works.md](how-it-works.md) §2 has the details.

**The clock.** `clk_summary` shows `clk_npu` at 200 MHz. That is a misreading: the real clock comes from a PVTPLL
that the firmware programs. The default governor reaches 900 MHz; no pinning is needed.

## 3. Install the released model set

The converted set is on Hugging Face:
[`kgrozdanovski/caldec-ultra-150m-rknn-rk3566`](https://huggingface.co/kgrozdanovski/caldec-ultra-150m-rknn-rk3566).
It holds the 10 bucket models (`pk2_s384.rknn` with the weights; `pk2_s{96..640}_now.rknn` without), their
`tables_s*/`, `act_head.npz`, the tokenizer and `rl_agent_config.json` of
[CalDec Ultra 150M](https://huggingface.co/kgrozdanovski/caldec-ultra-150m), and `SHA256SUMS`.

On the PC, with this repo checked out:

```
hf download kgrozdanovski/caldec-ultra-150m-rknn-rk3566 --local-dir caldec-npu-set
NPU_MODELS=$PWD/caldec-npu-set RKNN_RT=<dir with librknnrt.so and rknn_api.h> \
  scripts/npu/runner/install-service.sh user@board            # SSH_OPTS="-i key ..." if needed
ssh -t user@board 'sudo bash /tmp/caldec-npu-stage.XXXXXXXX/caldec-npu-install.sh pk2'   # the path it prints
```

This installs `/opt/caldec-npu` (read-only for a system user `caldec`), builds `caldec_npu_server` on the board,
makes a venv with numpy and `tokenizers`, and enables `caldec-npu.socket` (socket activation; the service loads the
10 models in about 6 s on the first connection and exits after 10 idle minutes). While it runs, the service sets
the CPU governor to `performance` and sets it back to `schedutil` when it stops; the published timings were measured
this way. Members of the group
`caldec-clients` may use it:

```
echo '{"state": {"source": "chat", "text": "..."}, "questions": {"tier": {"type": "choice", "instructions": "...", "criteria": {"fast": "...", "deep": "..."}}}}' \
  | caldec-npu                                    # on the board
ssh user@board caldec-npu < requests.jsonl        # from elsewhere: SSH is the authentication
```

Each request line is `{"state": ..., "questions": {...}}`, as for `system_one` in [ULTRA.md](../ULTRA.md). Each
answer line is `PackedAgent.system_one`'s answer format, with `usage` (`total_ms`, `npu_ms`, `npu_passes`).

Without the service, in Python on the board: `from caldec_npu import CalDecNPU;
CalDecNPU("/opt/caldec-npu").system_one(state, questions)`.

## 4. Rebuild the set from the checkpoint

On an x86 PC, from `scripts/npu/`, with this repo's environment as `$CALDEC_PY`, rknn-toolkit2 2.5.0 in
`rknn-venv/` and the extra onnx packages in `pylib/` (`PYTHONPATH=pylib` is set only for `$CALDEC_PY`; the toolkit's
own onnx and protobuf must not be shadowed). `CALDEC_CKPT` names the checkpoint: by default the Hub id
`kgrozdanovski/caldec-ultra-150m` at the revision `scripts/data.py` pins; a local directory also works.

```
cd scripts/npu
export CALDEC_PY=../../.venv/bin/python OMP_NUM_THREADS=4
PYTHONPATH=pylib CALDEC_DENSE_ATTENTION=1 $CALDEC_PY packed/export_packed.py prep --n 400 --max-len 384
bash packed/build_act.sh        # pk2_s384.rknn (weights) + weightless 96..320, with an fp32 check per bucket
bash packed/build_long.sh       # weightless 512 and 640
PYTHONPATH=pylib CALDEC_DENSE_ATTENTION=1 $CALDEC_PY runner/export_act_head.py      # runner/act_head.npz
```

- `prep` writes `packed/cases.pkl`: the first 400 assistant-test cases (858 questions) with fp32 reference logits.
  `export` checks every bucket graph, run in fp32 PyTorch, against `packed.forward` on those cases and prints the max
  |Δlogit| (about 1e-5) and the argmax agreement (all equal).
- **Conversion:** `convert_generic.py` builds fp16 for `target_platform='rk3566'` at `optimization_level=3`, taking
  the shapes from the ONNX graph; `remove_weight=True` makes the weightless bucket models. `RKNN_VERBOSE=1` prints
  the op table with each op's target. The 384 bucket has 17 CPU ops (13 reshapes, mostly of the inputs; the embedding Gather;
  3 transposes around the selection matmuls) and none in the attention.
- **Time:** about 30 minutes for all 10 buckets on 4 cores.
- **Install:** `CALDEC_CKPT_DIR=<local checkpoint dir> RKNN_RT=... runner/install-service.sh user@board` uses these
  outputs instead of a download.
- **Validation** (`runner/`): `test_tokens.py` (runner inputs vs `packed.build_case`, byte for byte), `make_ref.py`
  (requests and `PackedAgent.system_one` references), `compare_answers.py` and `compare_act.py` (NPU answers vs fp32),
  `make_public200.py` and `acc_public.py` (public-test accuracy), `exp_long_cases.py` (fp32 accuracy under the
  runner's split).

## 5. The model's inputs (contract)

One model per bucket length S (96, 128, 160, 192, 224, 256, 320, 384, 512, 640), batch 1. `Sh = min(S, 192)` head
rows, `Kt = 12` marker slots over all questions of a pass.

| Name | dtype | Shape | Content |
|---|---|---|---|
| `input_ids` | int64 | `[1, S]` | the packed sequence (`packed.build_case`: every question's segment, then the state), padded with `pad_token_id` |
| `local_mask` | float32 | `[S, S]` | 0 where key j is a real token and \|i − j\| ≤ 128, else −1e4 |
| `global_mask` | float32 | `[S, S]` | 0 where key j is a real token, else −1e4 |
| `view_sel` | float32 | `[S, Sh]` | column r one-hot at the source token of head row r (each view is `[CLS]` + one question's segment) |
| `row_type` | float32 | `[3, Sh]` | one-hot question type of head row r (choice, score, noul) |
| `head_mask` | float32 | `[Sh, Sh]` | block-diagonal: 0 within a view, −1e4 elsewhere; padding rows see only themselves |
| `marker_sel` | float32 | `[12, Sh]` | row k one-hot at the head row of the k-th option marker, questions in order |
| `marker_mask` | float32 | `[1, 12]` | 1 for real marker slots |
| `cls_sel` | float32 | `[12, Sh]` | row j one-hot at question j's first head row (for the act head) |
| `rope_tab` | float32 | `[4, 128, S]` | per RoPE theta (10,000, then 2,560,000): cos, then sin' = cat(sin[64:], −sin[:64]); from `tables_s<S>/` |
| `s_scale`, `h_scale` | float32 | `[1]` | S and Sh as floats; from `tables_s<S>/` |

- **Outputs:** `logits [1, 12]` (option logits of all questions in order; unused slots about −1e4) and
  `hcls [12, 768]` (each question's head row 0, the act head's input).
- **Post-processing** (in `caldec_npu.py`): the calibration temperatures of `rl_agent_config.json` and a softmax
  over each question's options, as in `scripts/packed.py`; the act head (Linear 772→256, GELU, Linear 256→2) on the
  CPU.
- **Bucket rule:** the smallest S that fits the tokens, with at most 192 head rows and 12 markers. A case that does
  not fit one pass is split into several passes, each with the full state (cut to fit) and a group of questions.
- **Tables:** `rope_tab`, `s_scale` and `h_scale` are written once per bucket at load. Models and tables must come
  from the same export.

## 6. Using the runtime from your own program (C API)

This is the sequence `caldec_npu_server.c` uses, abbreviated:

```c
rknn_context full, b128;                       // load the full-weight model first
rknn_init(&full, buf384, len384, 0, NULL);
rknn_init_extend ext = {0}; ext.ctx = full;    // weightless bucket models share its weights
rknn_init(&b128, buf128, len128, RKNN_FLAG_SHARE_WEIGHT_MEM, &ext);   // keep buf128 alive

// zero-copy inputs, once per model and input k:
rknn_tensor_attr na = {.index = k};
rknn_query(b128, RKNN_QUERY_NATIVE_INPUT_ATTR, &na, sizeof na);
rknn_tensor_mem *m = rknn_create_mem(b128, na.size_with_stride);
rknn_set_io_mem(b128, m, &na);
// write rope_tab / s_scale / h_scale once (fp16 if na.type == RKNN_TENSOR_FLOAT16), then rknn_mem_sync

// per request: write the other inputs, sync, run, read the outputs
rknn_mem_sync(b128, m, RKNN_MEMORY_SYNC_TO_DEVICE);
rknn_run(b128, NULL);
rknn_output o[2] = {{.want_float = 1}, {.want_float = 1}};
rknn_outputs_get(b128, 2, o, NULL); /* logits, hcls */ rknn_outputs_release(b128, 2, o);
```

- **One request at a time:** `rknn_run` is synchronous and there is one NPU core. Two processes using it at once
  slow each other down; serialise requests in one process, as the service does.
- **Without zero-copy:** `rknn_inputs_set` with float32/int64 buffers also works, about 4 ms slower per run.
- **Python on the board:** `rknn-toolkit-lite2` wraps the same runtime. It was not tested here, and its support for
  weight sharing and zero-copy is unverified.

## 7. Gotchas (all hit during this work)

| Symptom | Cause | Fix |
|---|---|---|
| wrong answers (68/120) | `GatherElements` is wrong on the device; the simulator is right | one-hot selection matmuls |
| NPU "stuck at 200 MHz" | misread clock tree (PVTPLL behind SCMI) | none needed |
| 2-3 s per question, CPU busy inside `rknn_run` | CPU ops (Softmax, Transposes) on uncached NPU buffers | the patched driver; NPU-only attention |
| board hang on the first `rknn_run` with the cacheable driver | `iommu_dma_cookie` layout changed in Linux 6.15 | the driver's cookie fix |
| oops in the `/dev/rknpu` misc-device path | misc-device code path | DRM-only build |
| NPU memory gone after a crashed process | GEM tracking reference never dropped | the driver's `.close` fix |
| accuracy drop with the NPU softmax | fp16 cancellation in ELU+1 | exp = (ELU(z/2)+1)^2 |
| toolkit segfault in OpFusing (rc 139) | slice/concat half-swap of a product with an input | permutation matmul (`ROPE_SWAP=mm`) |
| hand-written softmax folded back into the CPU `exSoftmax13` | toolkit pattern matching | MaxPool / ELU / AvgPool / Div pieces |
| per-op profiler crashes | NULL dereference in librknnrt on these graphs | ablation models instead |
| wrong logits from a bucket model | models and tables from different exports | build and ship them together |
| service crash loop `failed to open rknpu module` | the runtime opens `/dev/dri/card1` (group `video`) | `SupplementaryGroups=render video` |
