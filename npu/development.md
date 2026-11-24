# CalDec on the RK3566 NPU: development history and measurements

Part of [NPU.md](../NPU.md). This is how the result was reached, with every intermediate measurement and wrong turn.

**Which model.** Everything here was measured on earlier, unreleased development checkpoints of CalDec Ultra 150M
(the same UltraBERT-150M encoder and head architecture): the per-question form through phase 4, and a packed form
in round 3. The released model was converted with the same pipeline afterwards; its numbers are in
[NPU.md](../NPU.md). Hardware: one Panther X2 (RK3566, 4 GB), Armbian trixie, kernel 6.18, rknn-toolkit2 /
librknnrt 2.5.0. Times are UTC.

## Phase 1: bring-up (2026-11-21)

1. **Board prep.** The mainline Panther X2 device tree has no NPU. An overlay was written from the stock BSP
   tree, with power domain 6, `vdd_npu`, the NPU node, OPPs and later the IOMMU.
   - **Driver:** the out-of-tree `rknpu` 0.9.8 DKMS port for the ODROID-M1 (RK3568).
   - **Runtime and toolkit:** `librknnrt` 2.5.0, with rknn-toolkit2 2.5.0 on the PC.
2. **First boot with the NPU.** Without the IOMMU, small models ran, but CalDec-sized jobs timed out or
   oopsed. With the IOMMU and `mem=3800M` (the driver's 32-bit DMA cannot reach the bank above 4 GiB), the NPU
   worked.
   - **Oops:** opening the misc device `/dev/rknpu` oopsed. The device was hidden, and later the driver was rebuilt
     DRM-only.
3. **Export and conversion.**
   - **Export:** `export_caldec.py` exports the per-question model (masks as inputs, fixed 256 tokens).
   - **Conversion:** the first fp16 conversion stalled in the toolkit's OpFusing for more than 10 minutes, at
     optimisation level 3 and at 1. Moving the attention masks out of the graph into float inputs fixed it
     (151 s, exact). The graph has taken masks as inputs ever since.
   - **CPU baseline:** onnxruntime fp32 on 4 cores, **4782 ms**.
   - **First NPU run:** **3547 ms and 68/120 correct**.

## Phase 2: root causes (2026-11-21 evening)

4. **Wrong answers.** Per-layer dumps from the device (`RKNN_LOG_LEVEL=5`, `dump_compare.py`) against
   onnxruntime matched through all 24 encoder layers and both head layers (cosine ≥ 0.9998).
   - **First wrong tensor:** `GatherElements`, which picks the hidden state at each option marker. Its output held
     only 0 and −10000. The toolkit's simulator computed it correctly (20/20).
   - **An earlier note was wrong:** it had said "the head breaks".
   - **Fix:** a one-hot `marker_sel` input and a MatMul. **120/120**, at 3674 ms.
5. **"NPU clock stuck at 200 MHz".** The clock tree said 200 MHz, but the CRU register showed `clk_npu`
   running from the PVTPLL, which the firmware programs through SCMI.
   - **Probe:** an NPU-bound conv probe ran at 209.9 / 68.3 / 53.3 ms at 200 / 600 / 900 MHz.
   - **Conclusion:** not a problem.
6. **Why so slow.**
   - **Compute is small:** CalDec at 200 vs 900 MHz took 5578 vs 3678 ms, so only about 650 ms was NPU compute.
   - **CPU time:** 2836 of 3650 ms was CPU time inside `rknn_run`, spent in the CPU ops (Softmax per layer,
     Transposes, the head's attention).
   - **Cause:** `memspeed.c` showed CPU reads of NPU buffers at **165 MB/s**, against 1937 MB/s for `malloc`
     memory. Every NPU buffer was mapped uncached.
   - **Driver source:** `dma_mmap_attrs` on mainline arm64 forces Normal Non-Cacheable. The vendor kernel had
     `DMA_ATTR_NON_CONSISTENT`, which mainline removed.
   - **First fix attempt:** enabling the driver's disabled cacheable path **hung the board**.
7. **More bugs.**
   - **Leak:** every crashed rknn process leaked its NPU memory until reboot.
   - **Profiler:** the runtime's per-op profiler crashes on CalDec.
   - **Hang:** the cacheable-path hang left the kernel alive but the network dead. A power cut recovered it.

## Phase 3: the driver fix (2026-11-21/22)

8. **Research.** Five other mainline ports of the same driver were read: simonswine, QNAP, heliosrun, DrDitto
   and lurenJBD.
   - **What they share:** all of them use the cacheable pages path, and all carry a guard for the
     `iommu_dma_cookie` layout change in Linux 6.15.
   - **What the fork lacked:** that guard, which caused the hang.
9. **Fix and verification.** The changes: the cookie layout, the pages path on, `force_contig` off, the sync
   fix, the leak fix and a DRM-only build.
   - **Memory speed:** CPU reads of NPU buffers at 1970-2840 MB/s.
   - **Correctness:** CalDec **120/120 at 1778-1809 ms**, bit-identical over three passes.
   - **Leaks:** none after SIGKILL or a crash.
   - **Kernel:** no warnings.
10. **Publication.**
    - **Fork:** [`kgrozdanovski/rk3568-npu-ODROID-M1`, branch `panther-x2`](https://github.com/kgrozdanovski/rk3568-npu-ODROID-M1/tree/panther-x2).
    - **Upstream:** [PR #4](https://github.com/vitalijborissow/rk3568-npu-ODROID-M1/pull/4) to `vitalijborissow/rk3568-npu-ODROID-M1`.
    - **Driver and kernel choice:** keep the out-of-tree driver on 6.18 until the mainline `rocket` driver
      gains RK3568 support.

## Phase 4: overnight optimisation (2026-11-22, 02:09-06:56 UTC)

The goal was a per-question latency of 300-500 ms. Times are phase ranges.

| Time | Step | Result |
|---|---|---|
| 02:20-02:26 | `perf` on the 1.78 s graph | hottest loop = the runtime's generic N-D copy (CPU Transposes) |
| 02:27-02:43 | Attention micro-blocks: 12 formulations, op placement | NPU-only attention: 49.3 → 24.7 ms |
| 02:43-02:57 | Whole model in the NPU form (`export_npu.py`); ELU LUT accuracy probe → cancellation | 929 ms but 118/120; `--exp-pow 2`: 119/120, 947 ms |
| 02:57-03:40 | Fused projections, folded scales (no gain); int8 full and hybrid (240/400, 382/400: fail); ablation; length buckets 128-256 | buckets: mean 584 ms, median 465 ms |
| 03:40-04:10 | Layout test → token grid (`export_grid.py`); grid + buckets; grid-shape sweep → 64/96 buckets; head pruning (no gain); int8 MLP (48/90: fail) | linears 21.5 → 13.8 ms; mean 489 ms, median 390 ms |
| 04:10-04:40 | All buckets in one process (`multi_bench.c`); weight sharing (tables as inputs); zero-copy | 478 ms at 2447 MB → 497 ms at 535 MB |
| 04:40-05:25 | Lost ConvAdd fusion traced to RoPE; toolkit segfault; `flip` puts Reshapes on the CPU; matmul half-swap | fusion restored; rebuild started |
| 05:25-06:56 | `gtmm_*` result; S=128 time breakdown; confirmation (3 rounds pinned + 1 round default governor); reboot | **493.0 / 489.4 ms**, 398/400, bit-identical |

The NPU clock stayed at most 900 MHz, the stock top OPP, throughout.

## What the history teaches

- **Find the first wrong tensor.** Device dumps compared with onnxruntime layer by layer found the
  `GatherElements` bug in one pass. Up to then the guess had been "the head is broken".
- **Measure CPU time inside `rknn_run`.** Where the NPU looks slow, the cause has so far been CPU work in the runtime
  (placement, layout copies, uncached memory), not NPU compute.
- **Read the verbose op table after every graph change.** A single CPU op (a Transpose, a Reshape, a Softmax) costs
  more than whole layers of NPU work.
- **Exactness first.** Every rewrite was checked in fp32 against the original model (max |logit diff| ~1e-5)
  before converting, so device differences could only come from fp16 and the NPU's LUTs.

## Measurements


All runs on one Panther X2 (RK3566, 4 x Cortex-A55, 4 GB, NPU at up to 900 MHz). Model: the per-question form of an earlier, unreleased development checkpoint of CalDec Ultra 150M: UltraBERT-150M encoder with 24 layers, hidden 768, 6 heads x 128, MLP 1024
(SwiGLU), and alternating local (window 128) and global layers with two RoPE thetas (10,000 and 2,560,000). On top
sit a 2-layer head (`nn.TransformerEncoderLayer`, 12 heads x 64, FFN 3072, norm-first) and the option scorer. One
question is one forward pass over one sequence, batch 1, with up to 6 options. The output is 6 option logits.

**Reference:** fp32 PyTorch on the PC, from the same inputs. **Agreement** is the argmax over valid options.
**|Δp|** is the largest absolute difference of the softmax probabilities over valid options, at temperature 1.

### R1. Baselines and the path to 1.8 s (fixed 256 tokens)

120 real questions from the assistant test split (`cases_s256.npz`).

| Backend on the board | Graph | Agreement | p50 per question |
|---|---|---:|---:|
| CPU, onnxruntime 1.32, 4 threads, fp32 | `caldec_ultra150m_s256` | 120/120 | **4782 ms** |
| NPU, RKNN 2.5.0 fp16, model as exported | `m_s256` | **68/120, wrong** (GatherElements bug) | 3547 ms |
| NPU, one-hot option select, original driver | `ms_s256` | 120/120 (max \|Δp\| 0.0067) | 3674 ms |
| NPU, one-hot select, **patched driver** | `ms_s256` | 120/120 (max \|Δp\| 0.0067) | **1778-1809 ms** |

- **NPU clock:** 200 MHz gives 5578 ms and 900 MHz gives 3678 ms, so only about 650 ms was NPU compute at 900 MHz.
- **CPU time inside `rknn_run`:** 2836 of 3650 ms before the driver fix, 934-966 ms after it.
- **NPU-bound probe** (16 chained 1x1 convs, 1024 channels): 209.9 ms at 200 MHz, 68.3 ms at 600 MHz and 53.3 ms at
  900 MHz, about 160 GFLOPS fp16, or 69% MAC use.
- **Memory bandwidth for CPU reads of NPU buffers** (`memspeed.c`): 165 MB/s with the original driver, 1970-2840 MB/s
  patched. `malloc` memory reads at 1937 MB/s. Strided loads ran at 8.7 M/s before and 84 M/s after.

### R2. Where the CPU time went after the driver fix (1.78 s graph)

- **CPU inside `rknn_run`:** 979 ms, of which user 897 and sys 83.
- **`perf record`:** librknnrt 76.7%, kernel 19.9%. About 30% of all samples fall in one loop, a generic
  element-by-element N-D copy with one `ldrh`/`strh` per fp16 element. That is the runtime's CPU Transpose and layout
  conversion around the CPU ops, not the softmax arithmetic.
- **CPU ops** (from the verbose build log): the Softmax of every encoder layer (`exSoftmax13`), one Transpose of K per
  layer (two more in global layers), Reshape + Transpose + Softmax in each head layer, and the embedding Gather.

### R3. Attention micro-block (S=256, 6 x 128, hidden 768)

| Formulation | CPU ops | p50 | CPU in run |
|---|---|---:|---:|
| As exported: 4-D, `[1,1,S,S]` mask | Transpose, Softmax, Transpose | 49.3 ms | 33.8 ms |
| 3-D `bmm` on `[H,S,D]`, `[S,S]` mask, `torch.softmax` | Softmax | 34.7 ms | 18.2 ms |
| **3-D + MaxPool(1,S) max + ELU+1 exp + sum + Div** | **none** | **24.7 ms** | 4.1 ms |
| 3-D + max as a ReLU tree | none | 26.6 ms | 4.3 ms |

Accuracy of every variant: cosine 0.999999 against onnxruntime fp32.

**Op placement on RK3566** (toolkit 2.5.0, verbose op table):
- **CPU:** Softmax on any axis, Exp, ReduceMax on any axis, elementwise Max, GatherElements (also wrong on the
  device).
- **NPU:** MaxPool, AvgPool, ELU, Sigmoid, Div, Conv, MatMul, and the fused `exNorm`.

### R4. Whole model in the NPU form (`export_npu.py`, S=256, 120 questions)

| Build | p50 | CPU in run | Agreement | max \|Δp\| |
|---|---:|---:|---:|---:|
| Patched driver, original graph | 1807 ms | ~960 ms | 120/120 | 0.0067 |
| NPU-only attention, exp = ELU+1 | **929 ms** | 29 ms | **118/120** | **0.115** |
| exp = (ELU(z/2)+1)^2 (`--exp-pow 2`) | 947 ms | | 119/120 (flip at margin 0.003) | 0.0071 |
| exp = (ELU(z/8)+1)^8 (`--exp-pow 8`) | 968 ms | | 120/120 | |

- **ELU LUT accuracy** (`acc/act_acc.py`, z in [-20, 0]): plain ELU+1 reaches a max relative error of 78 at z in
  [-12, -8], because the NPU computes e^z − 1 in fp16 and adding 1 cancels it. Sigmoid and tanh forms are limited
  by about 7e-4 absolute LUT error.
- **fp32 exactness of the rewrite:** max |logit diff| 8e-6, 120/120.

### R5. Measured, no gain

| Change | Result |
|---|---|
| Fused QKV and gate+up projections | the compiler splits them back; 935 vs 947 ms |
| Folding q scaling and 1/EXP_POW into q, normalising after P·V (`--opt2`) | within ±5 ms |
| Last head layer only at the 6 marker rows, exact (`--prune-head`) | row layout: −42 ms in an encoder-free model, no change in the full model. Grid: 112-118 vs 110-113 ms, slower |
| 16x16 pooling of the scores | 25.9 vs 24.3 ms |
| `enable_flash_attention=True` in the toolkit | placement unchanged |

### R6. int8 (bar fixed before converting)

- **Calibration:** 50 validation-split questions. **Evaluation:** the 400 test-split questions.
- **Bar:** ≥ 98% argmax agreement, flips only at fp32 margin < 0.05, median |Δp| ≤ 0.01.

| Variant | Speed | Agreement | Verdict |
|---|---:|---:|---|
| Full w8a8 (row layout, S=256) | 1452 ms (several int8 ops fall back to the CPU) | 240/400 | fails |
| Hybrid: int8 only for q/k/v/gate/up convs, rest fp16 (incl. `down_proj`) | 805 ms vs ~930 (−13%) | 382/400, median \|Δp\| 0.016 | fails |
| Grid, int8 only for the MLP convs, incl. `down_proj` (S=128) | 354 vs 395 ms (−10%) | 48/90 | fails |

### R7. Layout and buckets

**8 chained 768x768 linears** (`lay/lay.py`):

| Layout | Time |
|---|---:|
| `[S,E]` matmuls | 21.5 ms |
| 1x1 conv on (1,768,1,256) | 17.5 ms |
| 1x1 conv on (1,768,16,16) | **13.8 ms** |

**Grid-shape sweep** (8-conv chain, `lay/sweep.py`): the aspect ratio does not matter. Cost by token count:

| Tokens | 64 | 96 | 128 | 160 | 192 |
|---|---:|---:|---:|---:|---:|
| Time | 3.3 ms | 4.8 ms | 6.3 ms | **10.3 ms** | 11.1 ms |

Cost is linear up to 128 tokens and jumps at 160.

**Bucket runs** (400 test questions, one model per bucket, fp16, exact rewrite):

| Graph | Buckets | mean | median | p95 | Agreement |
|---|---|---:|---:|---:|---:|
| Row layout | 128/160/192/256 | 584 ms | 465 ms | 933 ms | 398/400 |
| Grid | 128/160/192/256 | 524.8 ms | 400.2 ms | 840.8 ms | 398/400 |
| Grid | 64/96/128/192/224/256 | 488.7 ms | 389.6 ms | 863.0 ms | 398/400 |

3-layer grid model: 228 ms, against 270 ms for the row layout.

### R8. Deployment: all buckets in one process (`multi_bench.c`)

The 400 test questions run in their original order, so buckets interleave. Each question runs on the smallest bucket
that fits it. Timing is end to end from prepared inputs. Buckets: 64/96/128/160/192/224/256.

| Deployment | mean | median | p95 | max | Agreement | NPU memory (Shmem) |
|---|---:|---:|---:|---:|---:|---:|
| 7 standalone models (constant tables), 900 MHz pinned, 1 round | 478.4 ms | 394.6 ms | 875.7 ms | 902.9 ms | 398/400 | 2447 MB |
| 1 full + 6 weight-sharing, plain-sin tables (`gt_*`), zero-copy, 1 round | 497.4 ms | 404.7 ms | 901.0 ms | 916.3 ms | 398/400 | 535 MB |
| **1 full + 6 weight-sharing, matmul-swap RoPE (`gtmm_*`), zero-copy, 900 MHz, 3 rounds** | **493.0 ms** | 411.9 ms | 869.1 ms | 884.3 ms | 398/400 | **535 MB** |
| same, default `simple_ondemand` governor after a fresh `modprobe`, 1 round | 489.4 ms | 400.7 ms | 867.1 ms | 882.3 ms | 398/400 | 535 MB |

- **Pinning:** the 900 MHz pin is not needed, because the default governor gives the same result.
- **Repeatability:** logits are bit-identical across runs and governors.
- **Disagreements:** both are near-ties, at fp32 margins of 0.004 and 0.012. Max |Δp| is 0.0119; median |Δp| 0.0011.
- **Load time:** 1.06 s for the full model, 0.12-0.26 s for each weight-sharing one.
- **Memory with all 7 loaded:** MemAvailable drops from 3415 to 2614 MB, including about 300 MB of preloaded inputs.
- **Weight sharing check** (`share_test.c`): bit-identical to the same model with its own weights, 10/10.
- **Cost of table inputs:** the table-input graph with plain sin tables was about 4% slower than baked constants,
  because it lost the fused ConvAdd ops. The matmul-swap RoPE form restored the fusion.
- **Zero-copy:** bit-identical, and about 4 ms faster.

**Per-bucket latency** (`gtmm_*`, 3 rounds, e2e p50):

| Tokens | Questions (of 400) | Per question |
|---|---:|---:|
| ≤ 64 | 28 | 199.5 ms |
| 65-96 | 87 | 295.1 ms |
| 97-128 | 90 | 410.1 ms |
| 129-160 | 68 | 557.7 ms |
| 161-192 | 69 | 650.6 ms |
| 193-224 | 30 | 743.4 ms |
| 225-256 | 28 | 878.2 ms |

**Fixed 256 tokens** (the like-for-like comparison with the 4782 ms CPU baseline): 863-911 ms, about 5.3x faster.

### R9. Where the time goes now (grid model, S=128)

From 3-layer ablation models (environment variable `ABL`), 3 rounds each, p50. The runtime's per-op profiler
(`RKNN_FLAG_COLLECT_PERF_MASK`) crashes on every model except plain conv chains, so ablation was the only way to
measure this.

| Part | Time | Share |
|---|---:|---:|
| Per encoder layer, total | 13.4 ms | |
| Per encoder layer, MLP | 5.0 ms | |
| Per encoder layer, attention block | 8.4 ms | |
| Within attention: softmax chain | 2.2 ms | |
| Within attention: RoPE | 1.3 ms | |
| Within attention: q/k/v/o projections and the two matmuls | ~4.9 ms | |
| 24 encoder layers | 322 ms | 82% |
| 2-layer head | 62 ms | 16% |
| Embedding, scorer, I/O | 9.8 ms | 2% |
| **Sum** | **394 ms** | measured: 395 ms |

- **Conv throughput:** the convs run at about 65 GMAC/s, against an fp16 peak of about 115.
- **Head cost:** each head layer costs about 31 ms, against 13.4 ms per encoder layer. The head is 768 wide with
  12 heads and FFN 3072 (4·d), while the encoder MLP is 1024.
- **At S=256** (row layout, earlier): attention block 20.0 ms (softmax chain 5.6, RoPE 1.6) and MLP 10.5 ms per
  layer, with a fixed part of about 176 ms.

The build is reproducible. Re-running the documented recipe for S=64 on 2026-11-22 gave a byte-identical ONNX and
`.rknn` and identical RoPE tables, with max |logit diff| 4.2e-6 against fp32 (28/28).

### R10. Round 2: chunking and int8 (2026-11-22 afternoon)

| Experiment | Result |
|---|---|
| 8 x 768→768 convs, token axis split in halves (micro, run-only) | 160: 10.4 → 8.3 ms, 192: 11.2 → 9.8, 224: 12.5 → 11.5, 256: 13.8 → 12.8 |
| Same split in the full model, 160-256 buckets (deployment A/B, 3 rounds each) | mean 513.8 vs **489.8 ms**; 256: 1010 vs 875 ms (slower) |
| Split per MLP block, S=256 | 860 vs 859 ms (no change), 28/28 |
| int8 `down_proj` only, unsmoothed / smoothed (bucket 128, 90 questions) | 50/90 / 49/90; toolkit simulator 12/20 (quantisation error, not a device fault) |
| int8 gate/up only, unsmoothed / smoothed | 84/90 / 86/90 (median \|Δp\| 0.008); 387 / 375 ms vs ~395 fp16 |
| int8 encoder MLP / all linears, smoothed | 50/90 / 44/90 |
| Packed sequence length, assistant test (1,647 cases, 2.10 questions each) | fits 256: 80.0%, 384: 97.7%, 512: 99.8%; p50 187 tokens (single 135) |
| Thermals, 20 min continuous NPU load | CPU zone max 50.6 °C, GPU zone 48.9 °C (trip 70 °C) |

### R11. Round 3: packed multi-question encoding (2026-11-22 evening)

Same 400 assistant-test cases (858 questions) for both sets; fp16, default governor, `multi_bench`/`multi_bench2`
with `ZC=1`.

| | Per-question set (development checkpoint, cut 256) | Packed set (packed development checkpoint, cut 384) |
|---|---:|---:|
| NPU time per question, mean | 478.2 ms | **321.9 ms** |
| per case, mean | 1025.8 ms | 690.5 ms (median 659.1, p95 1131.3) |
| agreement with its fp32 model | 855/858 | 853/858 (flip margins ≤ 0.003) |
| accuracy, 834 untied labels, NPU / fp32 | 0.8453 / 0.8417 | **0.8465** / 0.8489 |
| weight memory | 521 MB (7 buckets) | 600 MB (8 buckets) |

Per-bucket e2e p50 (ms):

| S | 96 | 128 | 160 | 192 | 224 | 256 | 320 | 384 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| packed | 299.3 | 406.2 | 567.3 | 659.4 | 752.3 | 862.0 | 1123.6 | 1356.0 |
| per question | 290.6 | 407.8 | 555.5 | 644.0 | 751.8 | 893.0 | - | - |

| Experiment | Result |
|---|---|
| fp32 torch run of the packed NPU graph vs `packed.forward` | max \|Δlogit\| 5-9e-6, all argmax equal, every bucket |
| Head capped at 128 rows vs 192 (S=256, 39 cases, A/B/A/B) | 846.1 / 830.3 vs 857.0 / 856.3 ms: 1-3% |
| Bucket granularity (model from measured times, all 1,647 cases) | 8 buckets 346.3 ms/question; every 32: 339.3; every 16 (21 models): 326.4; no padding 314.7 |
| CPU time inside `rknn_run`, S=384 | 48 ms of 1343 ms (17 CPU ops: input reshapes, embedding gather, transposes around the selection matmuls) |
| Per-op profiling (`RKNN_FLAG_COLLECT_PERF_MASK`) | segfault inside `librknnrt.so` 2.5.0 in `rknn_run` |

