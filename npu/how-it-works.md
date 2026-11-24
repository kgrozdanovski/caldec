# CalDec on the RK3566 NPU: how it works

Each section explains a problem, why it happens on this hardware and software, and what the fix does. Part of
[NPU.md](../NPU.md). Code is from `scripts/npu/export_grid.py` (the encoder graph) and
`scripts/npu/packed/export_packed.py` (the packed model), unless noted.

**Which model the numbers come from.** The mechanisms were worked out on the per-question form of an earlier,
unreleased development checkpoint of CalDec Ultra 150M (same UltraBERT-150M encoder and head architecture). Timings
and agreement counts in §1-§10 are from that model, at a fixed 256 tokens unless noted
([development.md](development.md)). The released model's numbers are in [NPU.md](../NPU.md).

## 1. The RK3566 NPU and the RKNN stack

- **Hardware:** one NPU core of the RKNPU2 family, rated about 0.8 TOPS int8. int8 has 4x the fp16 MAC rate
  (commonly quoted; not measured here, and not checked against a Rockchip datasheet). fp16 measured about 160 GFLOPS at 900 MHz on a conv probe
  (69% MAC use). There is no w8a16 mode.
- **Software stack:**
  - On the PC, **rknn-toolkit2** converts ONNX to a `.rknn` file. It fuses and lowers the graph, assigns each op to
    the NPU or the CPU, and lays out the tensors.
  - On the device, **librknnrt** (the C runtime) loads the `.rknn` and runs the NPU ops through the kernel driver.
    It runs the CPU ops itself.
  - The driver is **rknpu**: a DRM GEM allocator, job submission and the IOMMU.
- **Placement is the first thing to check.** The toolkit decides per op. On RK3566, Softmax, Exp, ReduceMax,
  elementwise Max and GatherElements have no NPU kernel ([development.md](development.md)).
  - **Each CPU op costs more than its arithmetic.** The runtime converts layouts on both sides (NPU tensors are
    stored in a native layout with channel alignment) and runs a generic element-by-element copy loop.
  - **How to see the placement:** `RKNN_VERBOSE=1 scripts/npu/convert_generic.py ...` prints the op table with each op's
    target.
- **The per-op profiler is unusable here.** `RKNN_FLAG_COLLECT_PERF_MASK` and `RKNN_LOG_LEVEL>=4` crash inside
  `rknn_run` (a NULL dereference) on everything except plain conv chains. Every time breakdown in these notes
  therefore comes from **ablation**: the same graph with one part removed, timed on the device.

## 2. The driver bug: NPU buffers mapped uncached (fixed in the driver)

- **Symptom:** the 1.8 s → 3.6 s gap. The CPU spent 2836 of 3650 ms inside `rknn_run`. CPU reads of an NPU buffer
  ran at 165 MB/s, against 1937 MB/s for `malloc` memory. A buffer requested as CACHEABLE was just as slow.
- **Cause (source reading):** in the out-of-tree driver, `rknpu_gem_alloc_buf` allocates with `dma_alloc_attrs`,
  and `rknpu_gem_mmap_buffer` maps it with `dma_mmap_attrs`. On mainline arm64 that applies `dma_pgprot()`, which is
  Normal Non-Cacheable and overrides the cacheable `vm_page_prot`. The vendor kernel kept the mapping cacheable
  through `DMA_ATTR_NON_CONSISTENT`, which mainline removed. Every CPU op of the runtime therefore read and wrote
  uncached memory.
- **Fix** (fork branch [`panther-x2`](https://github.com/kgrozdanovski/rk3568-npu-ODROID-M1/tree/panther-x2) @ `dd74fb6`):
  1. **Pages path:** enable the driver's `RKNPU_GEM_ALLOC_FROM_PAGES` path, the vendor design used by every other
     mainline port. It allocates cacheable shmem pages, maps them through the IOMMU, and makes `MEM_SYNC` call
     `dma_sync_*`.
  2. **Cookie layout:** enabling the pages path first **hung the board**. The driver's private mirror of the
     kernel's `struct iommu_dma_cookie` still had the leading `enum type` field that Linux 6.15 removed, so every
     IOVA allocation used a pointer 8 bytes into the real `iova_domain`. The fix guards the field with
     `#if KERNEL_VERSION(6, 15, 0) > LINUX_VERSION_CODE`, as the other mainline ports do.
  3. **Sync:** the fork's extra whole-buffer sync now runs only without `fake_dev`.
  4. **Leak:** a GEM `.close` callback drops the fork's tracking reference. Without it, every crashed or killed
     process lost all its NPU buffers (341 MiB for CalDec) until reboot.
  5. **Build:** DRM-only (no misc `/dev/rknpu` device, whose path oopsed).
- **Effect:**
  - CPU reads of NPU buffers: 1970-2840 MB/s.
  - CalDec: 3674 → 1778-1809 ms, with bit-identical outputs.
  - CPU inside `rknn_run`: 2836 → 934-966 ms.
- **Status:** sent upstream as [PR #4](https://github.com/vitalijborissow/rk3568-npu-ODROID-M1/pull/4) (not yet
  reviewed). The mainline `rocket` driver is getting RK3568 support (patch series under review in mid-2026), which
  would make the out-of-tree driver unnecessary.

## 3. Attention without any CPU op

Standard attention on this NPU puts the Softmax on the CPU, plus Transposes around it. The rewrite keeps every op on
the NPU:

```python
def attn_core(q, k, v, mask, S):                       # q, k, v [H, D, S]; mask [S, S] (0 or -1e4)
    H = q.shape[0]
    z = torch.bmm(q.transpose(1, 2), k) + mask                          # scores [H, Sq, Sk]
    m = F.max_pool2d(z.view(H, S, 1, S), (1, S)).view(H, S, 1)          # row max: MaxPool, not ReduceMax
    e = F.elu(z - m) + 1.0                                              # exp(z/2 - m/2) via ELU (q pre-scaled)
    p = EXP_POW
    while p > 1:
        e = e * e; p //= 2                                              # square: exp(z - m)
    avg = F.avg_pool2d(e.view(H, S, 1, S), (1, S)).view(H, S, 1)        # row sum = mean * S: AvgPool
    ssum = avg * SSCALE if SSCALE is not None else avg * S
    o = torch.bmm(e, v.transpose(1, 2)) / ssum                          # normalise after P·V (Div is on the NPU)
    return o.transpose(1, 2)
```

- **3-D layout.** Heads are the batch of a 3-D `bmm` (`[H, D, S]`), not the 4-D `[1, H, S, D]` form. The 4-D form
  made the toolkit insert CPU Transposes. The mask is `[S, S]` and broadcasts over heads.
- **Max by MaxPool.** `MaxPool(1, S)` over a `view(H, S, 1, S)` computes the row max on the NPU. ReduceMax and the
  elementwise Max are CPU ops on RK3566. Subtracting the max is what keeps exp in range.
- **Exp by ELU.** For z ≤ 0, `ELU(z) = e^z − 1`, and ELU has an NPU LUT kernel; Exp does not. Since z − m ≤ 0,
  `ELU(z − m) + 1 = e^(z − m)`.
- **Sum by AvgPool.** The row sum is `AvgPool(1, S) · S`. A plain `sum(-1)` was lowered to a conv plus CPU
  Transposes.
- **Fixed-pattern softmax is avoided.** Any graph that looks like a softmax (max-subtract, exp, sum, div) is folded
  back into the toolkit's `exSoftmax13`, which runs on the CPU. The pieces above do not match that pattern.
- **Result:** attention block 49.3 → 24.7 ms at S=256 ([development.md](development.md)). The whole model went from 1807 to
  929 ms, with CPU time inside the run at 29 ms.

## 4. The ELU exp loses accuracy, and range reduction fixes it

- **Problem:** the first full-model build gave 118/120 with max |Δp| 0.115.
- **Cause:** `scripts/npu/acc/act_acc.py` evaluated the NPU's ELU on z in [-20, 0]. The LUT computes e^z − 1 in **fp16**. For a
  very negative z the result sits near −1, where the fp16 spacing is 4.9e-4. Adding 1 then leaves only that noise.
  So every small attention weight got an absolute error of about 5e-4 (relative error up to 78 at z in [-12, -8]).
  Over 256 keys this adds up to a visible probability shift.
- **Fix:** range reduction. `e^z = (e^(z/2))^2`, and `e^(z/2) = ELU(z/2) + 1`. Halving z keeps the LUT in a range
  where e^z is not tiny next to 1, and squaring on the NPU restores e^z. The 1/2 is folded into the q projection
  weights, which costs nothing:

  ```python
  w = torch.cat([a.q_proj.weight * (a.scaling / EXP_POW), a.k_proj.weight, a.v_proj.weight], 0)
  ```

  With `EXP_POW = 2`: 119/120 (the only flip at an fp32 margin of 0.003) and max |Δp| 0.0071, the same as the old
  CPU-softmax build's 0.0067, at 947 ms. `EXP_POW = 8` gives 120/120 for 21 ms more. The final graph uses 2.

## 5. The token grid: linears as 1x1 convs on `[1, E, S/16, 16]`

- **Why it matters:** the NPU's convolution path is its fastest path. A Linear on `[S, E]` lowers to a MatMul. The
  same weights as a 1x1 conv on a `[1, E, h, w]` feature map, with S = h·w, use the conv engine. A square-ish
  spatial shape tiles better than `[1, E, 1, S]`.
- **Measured:** 8 chained 768x768 linears take 21.5 ms as matmuls, 17.5 ms as convs on `(1, 768, 1, 256)` and
  13.8 ms on `(1, 768, 16, 16)`.
- **Implementation:** the hidden state stays a grid for the whole network. Norms reduce over the channel axis, and
  the toolkit still fuses them to `exNorm`. Attention reshapes the conv output to `[H, D, S]` and back, which is
  free in memory order:

  ```python
  def conv(x, lin_w, lin_b=None):                  # 1x1 conv with a Linear's [out, in] weight
      return F.conv2d(x, lin_w[:, :, None, None], lin_b)

  def rms_c(x, weight, eps):                       # RMSNorm over channels of [1, E, h, w]
      v = x.pow(2).mean(1, keepdim=True)
      return x * torch.rsqrt(v + eps) * weight[None, :, None, None]

  x = enc.embed_tokens(input_ids)[0]               # [S, E]
  x = x.t().reshape(1, -1, S // 16, 16)            # grid [1, E, h, w]
  ```

- **Result:** the 3-layer model went from 270 to 228 ms. Buckets 128-256 went from a mean of 584 to 525 ms.
- **Option select.** The option logits are read at each option's `[MASK]` marker. The original model gathers those
  rows with `GatherElements`, which gives **wrong results on the device** (68/120; the toolkit simulator is
  correct). The graph instead takes a one-hot `marker_sel [1, 6, S]` input and computes
  `marker_sel @ hidden^T`, which is an exact MatMul.

## 6. Length buckets (exact)

- **Why it is exact:** padding positions are masked with −1e4 in every attention (key mask), and the output is
  read only at the marker rows. A question of n tokens therefore gives the same logits in any graph of length
  S ≥ n. Measured: max |Δp| is unchanged between bucket and 256-token runs.
- **Bucket rule:** the smallest of 64/96/128/160/192/224/256 that is ≥ n. This is `make_bucket_cases.py`. RKNN
  models have static shapes, so each bucket is a separate model.
- **Choice of buckets:** conv cost is linear in tokens up to 128 and jumps at 160 (8-conv chain: 6.3 ms at 128,
  10.3 ms at 160; [development.md](development.md)). So 160 is a poor bucket, but it still beats running those questions at 192.
  Attention cost is quadratic, so small buckets pay off more than linearly.
- **The local window at short lengths.** With S ≤ 128 the local mask (|i − j| ≤ 128) equals the global mask. Local
  and global layers then compute the same attention pattern, with their own weights and RoPE theta.

## 7. Weight sharing across the 7 bucket models

- **Problem:** 7 standalone models need 2447 MB of NPU memory, out of about 3.4 GB available on a 4 GB board.
- **Mechanism:** librknnrt can initialise a model **without its own weights** (converted with
  `remove_weight=True`) and point it at the weights of an already loaded context. The calls are
  `rknn_init(..., RKNN_FLAG_SHARE_WEIGHT_MEM, &ext)` with `ext.ctx` set to the full model's context. This only works
  when both models' constant blobs are byte-identical.
- **What differed between buckets:** the RoPE cos/sin tables (`[D, S]`, length-dependent) and the softmax sum scale
  S. `--tables-input` turns both into inputs (`rope_tab [2·n_theta, D, S]`, `s_scale [1]`). Every other constant
  then matches across buckets, and the tables are written to the input buffers once at load.
- **Result:** 535 MB for all 7 buckets, against 2447 MB. The weightless models are 13-33 MB each, and load in
  0.12-0.26 s once the 256 model (1.06 s) is resident. Outputs are bit-identical to the same model with its own
  weights (`scripts/npu/share_test.c`, 10/10).

## 8. RoPE with input tables, and why its form matters

- **Problem:** with constant tables, the compiler folded `rotate_half(x) · sin` into a permutation conv fused with
  the following Add (`ConvAdd`). With tables as inputs it could not: a 3-layer model fell from 17 to 11 `ConvAdd`
  ops, and the full model was about 4% slower.
- **Fix:** rewrite RoPE so that the input table multiplies first and the fixed permutation comes after:

  ```
  rotate_half(x) · sin  =  swap(x · sin'),   sin' = cat(sin[h:], −sin[:h]),   h = D/2
  rope(x)               =  x · cos + swap(x · sin')
  ```

  `swap` exchanges the two halves of the head dimension. Written as `cat(slice, slice)`, it **segfaulted
  rknn-toolkit2 2.5.0 in OpFusing**, reproducibly. Written as `flip` on a `[H, 2, h, S]` view, it put 19 Reshapes on
  the CPU. Written as an explicit permutation matmul, the fused `ConvAdd` ops returned:

  ```python
  P = torch.roll(torch.eye(D), half, 1)          # half-swap permutation, a constant
  sw = torch.matmul(P, x * sin_prime)            # x [H, D, S]
  return x * cos + sw
  ```

  This is the default `ROPE_SWAP=mm`. The table file carries `sin'`, so models and tables must come from the same
  export. The superseded plain-sin `gt_*` models give **wrong logits** with the current tables.

## 9. Zero-copy inputs

- **Without zero-copy:** `rknn_inputs_set` converts each input (float32 → fp16, NCHW → native layout) into the
  runtime's own buffer.
- **With zero-copy** (`ZC=1` in `scripts/npu/multi_bench.c`), each model's input buffers are created once, in the native format
  the model wants:
  1. `rknn_query(RKNN_QUERY_NATIVE_INPUT_ATTR)` gives the native format.
  2. `rknn_create_mem(size_with_stride)` and `rknn_set_io_mem` create and bind the buffer.
  3. Per question, the application writes the values (fp16 conversion on the CPU) and calls
     `rknn_mem_sync(TO_DEVICE)`.
  4. The table inputs are written once at load.
- **Result:** bit-identical outputs, about 4 ms faster per question. It needs the driver fix: these buffers are
  NPU memory, and with uncached mappings the CPU writes were slow.

## 10. What did not help, and why (short)

- **Fused QKV / gate+up:** the compiler splits large convs back into tiles, so the fusion gains nothing (935 vs
  947 ms).
- **Pruning the last head layer to the 6 marker rows:** exact, and faster in isolation. In the full model the K = 6
  row matmuls cost as much as the full-sequence convs they replace.
- **int8:** RKNN quantises activations per tensor. The CalDec encoder's `down_proj` inputs have max/median ratios of
  10^4-10^5 (measured on the CPU with fake-quant). A per-tensor int8 scale then wipes out the small values. `down_proj` was int8 in the 48/90 build and fp16 in the 382/400 one.

## 11. Packed multi-question encoding (the released form)

The design below is what the released model set uses. The figures in this section (exactness, the third saved
per question) were measured on a packed development checkpoint; the released model's numbers are in
[NPU.md](../NPU.md).

CalDec Ultra is a packed model: one encoder pass reads every question of a case ahead of one copy of the state, and
the decision head then runs per question over `[CLS]` and that question's own segment (`scripts/packed.py`). The NPU
graph (`scripts/npu/packed/export_packed.py`) keeps that exactly:

- **Encoder:** the token-grid rewrite above (1x1 convs on `[1, 768, S/16, 16]`, NPU-only attention, RoPE tables and
  the softmax scale as inputs for weight sharing).
- **Gathering the views:** each question's view is `[CLS]` + its segment. All views are gathered into one head grid
  of `Sh = min(S, 192)` rows by a one-hot matmul, `x[E,S] @ view_sel[S,Sh]`; the per-row question type is added as
  `type_emb.T @ row_type[T,Sh]`.
- **Head:** the 2 head layers run once over the `Sh` rows under a block-diagonal `head_mask`, so a row sees only
  rows of its own view. That is exactly per-question attention; `[CLS]` is copied into every view because its head
  output differs per question.
- **Padding rows** copy token 0 with type 0 and attend only to themselves, so no all-zero row reaches LayerNorm in
  fp16 and no row is fully masked.
- **Output:** `marker_sel[12,Sh]` picks the option markers of all questions in order, then the scorer runs:
  `logits[1,12]`, masked slots at −1e4. A second output, `hcls[12,E]`, carries each question's head row 0 for the
  act head, which runs on the CPU (a 2-layer MLP, in numpy in the runner).
- **Exactness:** the NPU graph run in fp32 PyTorch matches `packed.forward` to a max |Δlogit| of about 1e-5, with
  every argmax equal, at every bucket.
- **Buckets:** 96, 128, 160, 192, 224, 256, 320, 384, 512 and 640 tokens. The 384 model carries the weights; the
  others are weightless and share them. A case that does not fit one pass (more than 640 tokens, more than 192 head
  rows or more than 12 option markers) is split by the runner into several passes, each with the full state cut to
  fit and a subset of the questions.
- **Why it pays:** an assistant case carries 2.1 questions on average, and the packed pass costs about the same as
  one per-question pass of the same length, so time per question drops by about a third.
