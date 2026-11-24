# CalDec on the RK3566 NPU: failures and fixes

Every failure of the RK3566 NPU work, in order. Part of [NPU.md](../NPU.md); the context of each is in
[development.md](development.md).

| # | Step | Symptom | Cause | Fix |
|---:|---|---|---|---|
| 1 | First boot with the NPU, no IOMMU | small models ran; CalDec-sized jobs timed out or oopsed in `rknpu_dma_buf_sync` | buffers above the 32-bit DMA range without an IOMMU | enable the NPU IOMMU; boot with `mem=3800M` |
| 2 | Opening `/dev/rknpu` | kernel oops | the driver's misc-device path | hide the device, later a DRM-only build |
| 3 | First fp16 conversion (`--opt 3`, and a parallel `--opt 1`) | both stalled in `OpFusing 1 : 0%` for more than 10 min | the in-graph mask construction (cause inside the toolkit not established) | masks as precomputed float inputs: converted in 151 s, exact (max \|diff\| 0.0) |
| 4 | First NPU benchmark | 68/120 wrong | `GatherElements` wrong on the device; the simulator is right | one-hot `marker_sel` MatMul |
| 5 | Early diagnosis "the head breaks" | wrong statement | assumed, not measured | per-layer dumps found GatherElements |
| 6 | "NPU stuck at 200 MHz" | wrong statement | clock tree misread (PVTPLL via SCMI) | register read and a clock-scaling probe |
| 7 | 3.6 s per question | CPU busy inside `rknn_run` | CPU ops on uncached NPU buffers | driver fix; NPU-only attention |
| 8 | NPU softmax rewrite in the original 4-D layout | slower (1303 vs 1174 ms, 3 layers) | layout change added CPU Transposes | the 3-D form (overnight) |
| 9 | Enabling the driver's cacheable pages path | board hang (network dead, kernel alive) | `iommu_dma_cookie` layout changed in Linux 6.15 | the cookie guard |
| 10 | A crashed or killed rknn process | its NPU memory lost until reboot | GEM tracking reference never dropped | the `.close` callback |
| 11 | Per-op profiler (`RKNN_FLAG_COLLECT_PERF_MASK`, `RKNN_LOG_LEVEL>=4`) | segfault in `rknn_run` | NULL dereference in librknnrt on non-conv graphs | ablation models |
| 12 | Driver README's patch instructions | `patch -p2` did not apply | wrong strip level | `-p1` from `/usr/src/rknpu-1.0` |
| 13 | NPU-form model, plain ELU exp | 118/120, max \|Δp\| 0.115 | fp16 cancellation in ELU+1 | exp = (ELU(z/2)+1)^2 |
| 14 | `op_target={'/Softmax':'npu'}` | `Invalid key` | wants internal names; no RK3566 NPU Softmax anyway | rewrite |
| 15 | Hand-written softmax | folded back into the CPU `exSoftmax13` | toolkit pattern matching | MaxPool / ELU / AvgPool / Div pieces |
| 16 | int8 dataset list | `FileNotFoundError .../out/out/...` | paths resolved relative to the list file | absolute paths |
| 17 | Full w8a8 | 1452 ms, 240/400 | int8 ops on the CPU; per-tensor activation scales vs outliers | not used |
| 18 | Hybrid step 2 with op names | `Invalid operands name` (322 times) | the cfg wants tensor names | tensor names |
| 19 | Hybrid int8 q/k/v/gate/up | 382/400, median \|Δp\| 0.016 | int8 accuracy | not used |
| 20 | Grid int8 MLP incl. `down_proj` | 48/90 | `down_proj` input outliers | not used |
| 21 | Two benchmarks on the NPU at once | 3-layer times of 750 ms | one NPU core shared | serialise |
| 22 | RoPE `x·cos + cat(slice, slice)(x·sin')` | toolkit segfault in OpFusing, rc 139, reproducible | toolkit bug | permutation matmul (`ROPE_SWAP=mm`) |
| 23 | RoPE swap as `flip` | 19 Reshapes on the CPU | layout | not used |
| 24 | Tables as inputs | ~4% slower | lost `ConvAdd` fusion (17 → 11 in 3 layers) | the matmul RoPE form |
| 25 | Exact head pruning (row and grid layouts) | no gain, slower in the grid | small-K matmuls cost as much as they save | not used |
| 26 | Fused QKV / gate+up | no gain | compiler splits them back | kept fused in the export, harmless |
| 27 | Old `gt_*` models with the new `tables_s*` | wrong logits (sin vs sin') | models and tables from different exports | keep models and tables from one export together |
| 28 | Token-axis chunking in the full model | slower (256: 1010 vs 875 ms) | Concat/Split traffic outweighs the micro tiling gain | not used |
| 29 | Chunked 256 + unchunked weightless buckets | `size = 0! while weight tensor need share!`, wrong ≤128 logits | constant names differ between graph variants | share only within one graph family |
| 30 | Hybrid cfg with graph inputs/constants as keys | `Invalid operands name 'local_mask'` | not allowed by the toolkit | — |
| 31 | int8 `down_proj`, smoothed or not | 49-50/90 (simulator 12/20) | quantisation error (the simulator fails too), likely the output into the residual stream | not used |
| 32 | `multi_bench` with the packed model | not usable | 9 input names hard-coded | `multi_bench2.c` (inputs by name) |
| 33 | `rknn_caldec_bench` with 11 inputs | `Bad address`, segfault | fixed arrays of 8 | raised to 16 |
| 34 | Per-op profiling of the packed graph | segfault in `librknnrt.so` (`rknn_run`) | runtime bug with `RKNN_FLAG_COLLECT_PERF_MASK` | A/B instead |
| 35 | Runner CLI, first run | `FileNotFoundError` in `Popen` | relative model dir: server path without `./` | `Path(model_dir).resolve()` |
| 36 | One question larger than a pass (synthetic) | bare `StopIteration` from the bucket choice | 195 head rows > 192, or > 12 options | clear `ValueError` with the sizes |
| 37 | Service first start (`caldec-npu.service`) | crash loop `failed to open rknpu module` | the runtime opens `/dev/dri/card1` (group `video`); the service user had `render` only | `SupplementaryGroups=render video` |
| 38 | Other models, same runtime (2.5.0): a 521-wide fully connected layer (YAMNet) | outputs from class ~438 up wrong | RKNN fault, one case observed | pad the layer to 528, slice on the CPU |
| 39 | Other models: int8 YOLOv8n with float32 input via `rknn_inputs_set` | 69/260 detections | float input path of a quantized image model | uint8 NHWC input: 248/260 |
