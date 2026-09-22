# E11 — measured inputs for the hybrid GPU + PIM model (π0)

**Task T010** · π0 (`lerobot/pi0_libero_finetuned_v044` @ `45dcc8f`) · MAXN, 16 SMs, GPU pinned
1300.5 MHz, EMC 3199 MHz, 12 CPU cores at 2.2 GHz · bf16 · batch 1 · P = 816, S = 51

Deliverable: `results/20260922_E11_pi0_inputs/model_inputs.csv`, consolidating seven runs.
`python docs/reference_scripts/check_model_inputs.py results/*_E11*/model_inputs.csv` exits 0.

---

## The answer first

`docs/00_context.md` prices every GPU-side term of the hybrid model analytically:

```
host_op_time  = max(FLOPs / (42 TFLOP/s x 0.4), bytes / (204.8 GB/s x 0.7)) + kernels x 20 us
transfer_time = bytes / (204.8 GB/s x 0.7)
```

with synchronisation free, one kernel per host segment, `T_host_layer` ≈ 0.16 ms, and the
prefix unaffected by memory-side activity. Six of those terms carry an explicit
**[ASSUMPTION]** tag. Here is what each is worth on this device.

| term | assumed | measured | verdict |
|---|---|---|---|
| kernels per host segment | 1 (2 for attention) | **9–10 for the norms, 7.1 for attention** | 6× too few |
| `T_host_layer`, eager | 0.16 ms | **1.502 ms** | 9.4× optimistic |
| `T_host_layer`, CUDA-graph layer | — | **0.177 ms** | the assumption is about right *here* |
| `T_host_layer`, compiled sequence | — | **0.085 ms** | 1.9× better than assumed |
| transfer fixed cost | **0** | **40 µs** (pinned, either direction) | not zero |
| transfer bandwidth | 143 GB/s | **36 GB/s** (pinned) | 4× optimistic |
| synchronisation | **free** | 10.6 µs idle sync, 59 µs kernel round trip, **108.8 µs stream handoff** | not free |
| `kv_copied_bytes_per_step` | **0** | **15,040,512 B — the whole cache, every step** | not zero |
| `t_kv_materialise` | **0** | 0.48–0.58 ms | not zero |

The single most consequential correction is not in the table. **`T_host_layer` ≈ 0.16 ms is not
wrong — it is a statement about an implementation that does not exist in the eager stack.** A
CUDA-graph layer hits 0.177 ms and a compiled sequence 0.085 ms, so the mapping's arithmetic is
achievable; the 9.4× gap is the cost of driving it from Python, one launch at a time.

---

## Method notes that shaped every number

**The segments are not where the plan's table implies.** `denoise_step`
(`modeling_pi0.py:894`) calls `paligemma_with_expert.forward` with
`inputs_embeds=[None, suffix_embs]`, which takes the `elif inputs_embeds[0] is None` branch
(`:478`) and runs `gemma_expert.model.forward` — the ordinary decoder stack. The layer body is
therefore `pi_gemma.py:157-188`. `compute_layer_complete` (`modeling_pi0.py:230`) serves the
joint prefix+suffix path and is *not* what a denoising step executes. Timing the wrong one
would have looked entirely plausible.

**Where the loaded model differs from the plan's segment table, the model wins** (recorded in
`segments.csv.notes`):

* `embed_suffix` returns **`adarms_cond = None`** for π0 (`modeling_pi0.py:734`), so
  `PiGemmaRMSNorm` takes its unconditioned branch and both residuals are plain `x + y`, not
  `x + y*gate`. π0.5 uses AdaRMS and is expected to differ — which is why the branch is
  recorded per model rather than assumed.
* `eager_attention_forward` calls **`repeat_kv`** (`modeling_gemma.py:205-206`), materialising
  the single KV head into 8 before both matmuls. That expansion is traffic the analytic model
  does not price. Unique KV bytes are counted once in `bytes_host_segment`.
* The model's own score cast is fp32 → **bf16**, inside softmax (`modeling_gemma.py:213`), not
  fp32 → fp16. The mapping's fp16 variant is kept as `cast_scores_fp32_fp16`, marked
  `mapping_only`.

**Two measurement errors found and fixed before any number was reported.**

1. The first attention replay rebuilt keys from the suffix only, giving **T = 51 instead of the
   real T = P + S = 867** — a 17× understatement of every attention quantity. It now reads the
   prefix K/V out of the cache *before* the layer updates it, and `scores` is `[1, 8, 51, 867]`.
2. CUPTI needs root on this device, and attaching `torch.profiler` inflated every subsequent
   call by ~30 %. Kernel counting and timing therefore run as **two separate passes** — count
   under `sudo`, time as the ordinary user — which is the same separation §13.6 mandates for
   CPU utilisation. Timing runs never attach a profiler.

---

## E11a — the host segments

`results/20260922_E11a_pi0_hostseg/` · `bench/host_segments.py` · ≥ 20 warm-ups and 1,000 timed
calls per (segment, style, timing), layers 0 and 9, on the model's own tensors captured from one
real denoising step.

At layer 0, `wall_isolated`, `gap_ms = 0` (analytic bytes and FLOPs from the real shapes, with
the formulas in `segments.csv.flops_formula`):

| segment | eager ms | compiled ms | cudagraph ms | kernels, eager | bytes | FLOPs |
|---|---|---|---|---|---|---|
| `rmsnorm1` | 0.3927 | 0.2330 | **0.0461** | 9.0 | 210,944 | 417,792 |
| `rope` | 0.4526 | 0.1947 | **0.0620** | 10.1 | 705,024 | 705,024 |
| `softmax` | 0.1051 | 0.1425 | **0.0528** | 2.1 | 1,414,944 | 1,768,680 |
| `residual_rmsnorm2` | 0.4308 | 0.2445 | **0.0496** | 10.1 | 522,240 | 470,016 |
| `gelu_gate_mul_up` | 0.1057 | 0.1497 | **0.0221** | 2.1 | 1,253,376 | 1,880,064 |
| `partials_residual` | 0.1115 | 0.1456 | **0.0222** | 2.1 | 626,688 | 208,896 |
| `residual` | 0.0629 | 0.1480 | **0.0223** | 1.1 | 313,344 | 52,224 |
| `attention_gpu` | 0.5273 | 0.3894 | **0.3226** | 7.1 | 1,305,600 | 362,225,664 |
| `cast_scores_fp32_fp16` | 0.0690 | 0.1450 | **0.0227** | 1.0 | 2,122,416 | 0 |

Three things in that table matter more than the individual numbers.

**Kernel counts are 2–10×, not 1.** The model charges one launch per segment at 20 µs.
`rmsnorm1` is **9** kernels, `rope` **10.1**, `residual_rmsnorm2` **10.1**, `attention_gpu`
**7.1**. The six mapping segments are ~35 launches per layer, not 6. `PiGemmaRMSNorm` is a
Python composite — cast to fp32, square, mean, rsqrt, multiply, weighted multiply, cast back —
and nothing fuses it in eager mode.

**Compiling a segment on its own is counter-productive; compiling the layer is not.** For five
of the nine segments `compiled` is *slower* than `eager` (`softmax` 0.142 vs 0.105, `residual`
0.148 vs 0.063): a compiled callable still pays a fixed per-call cost, and these segments are
too small to amortise it. Yet the six-segment **sequence** compiles to 0.085 ms against
1.502 ms eager — an 18× win, because fusion crosses the segment boundaries that the mapping
treats as separable. A mapping that hands each segment to the GPU independently forfeits that.

**One graph per layer beats six graphs per layer.**

| `t_host_layer`, layer 0 | ms | vs the analytic 0.16 ms |
|---|---|---|
| `sum_of_segments`, eager (derived) | 1.598 | 10.0× |
| `sequence`, eager | **1.502** | **9.4×** |
| `sum_of_segments`, compiled (derived) | 1.110 | 6.9× |
| `sum_of_segments`, cudagraph (derived) | 0.255 | 1.6× |
| `whole_layer_graph` | **0.177** | **1.1×** |
| `sequence`, compiled | **0.085** | **0.53×** |

`docs/00_context.md` already anticipates the `whole_layer_graph` case — "one controller macro
replaces per-segment launches (launches per layer = 1)" — and prices it at one 20 µs launch.
Measured, that layer costs 0.177 ms, so the macro is worth having but is not free.

Per-step extras (`order_in_layer = -1`): `suffix_embed` 1.428 ms eager / 0.845 compiled
(27.2 kernels), `final_norm_out_proj` 0.587 ms (11 kernels), `euler_update` 0.100 ms. The
analytic `T_extra` is 0.08 ms per step; measured they total **2.115 ms** eager.

Two items are **not measured**, with their exact errors, per the task's escape hatch:
`suffix_embed` under `cudagraph` (`cudaErrorStreamCaptureInvalidated`) and
`final_norm_out_proj` under `compile` (inductor: *"Offset increment outside graph capture
encountered unexpectedly"*). Both are recorded in `meta.json.not_measured`.

---

## E11b — transfers and synchronisation

`results/20260922_E11b_xfer/` · `bench/xfer_latency.py` · `model = none`, because this part is
model-independent and measured once (plan part 2).

Least squares on the per-size medians, `t = a + b·bytes`, MB = 10⁶ B:

| method | direction | **fit_a** | fit_b | 1/b | r² |
|---|---|---|---|---|---|
| pinned | h2d | **40.3 µs** | 0.02765 ms/MB | **36.17 GB/s** | 0.9996 |
| pinned | d2h | **39.7 µs** | 0.02778 ms/MB | 35.99 GB/s | 0.9987 |
| pageable | h2d | 74.1 µs | 0.06945 ms/MB | 14.40 GB/s | 0.9986 |
| pageable | d2h | 59.0 µs | 0.12973 ms/MB | 7.71 GB/s | 0.99995 |

**Both terms of the transfer model are wrong.** The fixed cost is ~40 µs, not zero, and the
achievable rate is 36 GB/s, not 143. For the score matrix the mapping actually moves
(8 × 51 × 867 × 4 B = 1.41 MB) the model predicts 9.8 µs; the measured cost is
40.3 + 39.1 = **79.4 µs, 8× more**. At the 0.1 MB activation tile the fixed term is the whole
cost and the ratio is worse still.

Synchronisation, ≥ 1,000 samples each, against an assumed **zero**:

| | median |
|---|---|
| `event_query` | 0.8 µs |
| `event_synchronize_done` | 1.1 µs |
| `synchronize_idle` | 10.6 µs |
| `tiny_kernel_roundtrip` | 59.0 µs |
| `stream_handoff` | **108.8 µs** (net of the same two kernels on one stream, as §13.3 requires) |

A hand-over that synchronises once per host segment would pay 109 µs per boundary — comparable
to the whole eager segment it separates.

**`zero_copy` works on this device.** A `CUDAPluggableAllocator` over `cudaMallocManaged`
(`bench/managed_alloc.cu`, built by the `nvcc` line in its header) gives one allocation both
sides address; the CPU writes it through a numpy view of `data_ptr()` and the GPU reads the same
pages. It runs in a dedicated subprocess because `change_current_allocator` is global and
irreversible, which is also why its control — the same reduction on an ordinary device tensor —
is measured in a different process and labelled as such.

**The CPU blocks rather than spins while waiting** (process CPU / wall = 0.0036 during a long
synchronise). §13.3 asks for this explicitly, because a spinning wait would show up as 100 % CPU
in E11e and must not be read as work. It does not.

---

## E11c — the hand-over object

`results/20260922_E11c_pi0_handover/` · `bench/handover.py`

| | value |
|---|---|
| KV bytes per layer | 835,584 |
| **KV bytes total** | **15,040,512 B (15.04 MB)** |
| cross-check, 18 × 2 × 816 × 1 × 256 × 2 | 15,040,512 — reported, not forced |
| valid prefix tokens | **533 of 816**; 283 padded, stored and copied like any other |
| `t_kv_materialise`, `stacked` | 0.579 ms wall, 0.506 ms GPU |
| `t_kv_materialise`, `v_transposed` | 0.480 ms wall, 0.400 ms GPU |
| **`copy.deepcopy` per step** | **5.53 ms** |
| **`kv_copied_bytes_per_step`** | **15,040,512 B — the whole cache** |

The model assumes the last two are zero. They are not: `modeling_pi0.py:918` deep-copies the
entire cache on every denoising step, so at N = 10 that is **55 ms per inference of pure
copying** — more than the analytic budget for all 18 layers of host segments combined.

**Getting the evidence right took two attempts.** Sampling `data_ptr()` only on *entry* to
`denoise_step` shows one stable address across all steps, which reads as "the cache is
referenced, not copied" — and is wrong. The same object is passed in every step; the copy is
made inside. Sampling again at the expert forward shows a *different* buffer, which is the proof
of the copy. That second address is itself stable across steps, and that is the caching
allocator handing the same freed 15 MB block back — not reuse of the data. `kv_reuse.md` now
carries both columns and says so.

---

## E11d — how the prefix scales with its input

`results/20260922_E11d_pi0_{eager,reduce-overhead}_prefixsweep/` · `bench/prefix_sweep.py` ·
images {1, 2, 3} × prompt {short, typical, long} = 9 points per mode, N = 10.

**The prefix cost does not depend on the input at all.** Across all nine points:

| region (eager) | min ms | max ms | spread |
|---|---|---|---|
| `vision` | 241.58 | 241.71 | **1.0005×** |
| `vlm_prefix` | 234.44 | 234.79 | **1.0015×** |
| `ae_total` | 574.64 | 583.15 | 1.0148× |
| `t_e2e` | 1057.29 | 1065.79 | 1.0080× |
| `t_e2e`, `reduce-overhead` | 466.47 | 467.36 | 1.0019× |

Meanwhile `P_valid` moves from **262 to 807**, a factor of 3.1. Supplying one camera and a
six-token prompt costs the same as three cameras and a thirty-nine-token prompt, to within
0.15 % on the two prefix regions.

The reason is in the input format, and it is worth stating plainly because the hybrid model
parameterises the prefix by P:

* `config.input_features` declares **three** image slots (`image`, `image2`,
  `empty_camera_0`) and `config.empty_cameras = 1`. A camera that is not supplied is **not
  dropped** — `_preprocess_images` synthesises a −1-filled tensor with a zero mask and it is
  still encoded into 256 prefix tokens.
* `config.tokenizer_max_length = 48`, so every prompt is padded to 48 tokens.

So **P_processed = 3 × 256 + 48 = 816 for every input this model will ever see**, and the
attention mask — not the work — is what changes. Protocol §13 anticipates exactly this
("models that pad the prompt to a fixed length will show a constant processed P — report that,
it is a result"), and it is why `prefix_latency_fit` has **no slope to report** against P:
`prefix_fit.csv` records `x is constant at 816 across the sweep; no slope can be fitted, which
is itself the result`.

The one fit that does exist, `vision_vs_images`, gives a = 241.57 ms and
b = **0.047 ms per image** (r² = 0.948 over a 3-point range) — i.e. flat to within a twentieth
of a millisecond per additional camera, for the same reason.

**What this means for the hybrid model.** `t_vision` and `t_vlm_prefix` can be treated as
constants for this checkpoint — 241.6 ms and 234.4 ms eager, and the pair costs 467 ms
end-to-end under `reduce-overhead` at N = 10 (2.27× faster than eager). But the constancy is a
property of the *input format*, not of the hardware: a deployment that trimmed the empty camera
slot and the prompt padding would be encoding 533 of 816 tokens, and none of these numbers would
carry over. That is a design lever the model does not currently represent.

Peak memory is likewise flat: 9,037–9,040 MB eager, 8,912–8,924 MB compiled.

Regions are instrumented in **eager only**: `bench/regions.py`'s context managers force a graph
break under `torch.compile` (`bench/run_e4.py` disables them for the same reason), so the
compiled sweep reports `t_e2e` alone. The third image slot of the 3-image points carries a copy
of the second, because the dataset supplies two cameras; what varies is how many slots arrive
with a valid mask, which is what `P_valid` measures.

---

## E11e — what memory pressure actually costs the prefix

`results/20260922_E11e_pi0_{eager,reduce-overhead}_mempressure/` · `bench/mempressure.py` +
`bench/bw_hog.py` · prefix-only loop, 60 s per condition, foreground pinned to cores 0–5 and
the CPU hog to 6–11.

The model assumes **`prefix_slowdown = 1.0`**. Eager:

| load | prefix median ms | **slowdown** | bg GB/s during |
|---|---|---|---|
| none | 481.7 | 1.000 | — |
| `gpu_hog_25` | 1075.5 | **2.233** | 78.4 |
| `gpu_hog_50` | 1074.2 | 2.230 | 55.8 |
| `gpu_hog_75` | 1068.8 | 2.219 | 78.5 |
| `gpu_hog_100` | 1067.7 | 2.217 | 78.5 |
| **`gpu_spin_control`** | 843.6 | **1.752** | 5.5 |
| `cpu_hog_25…100` | 510.9–511.1 | 1.061 | ~49.8 |
| **`cpu_spin_control`** | 490.5 | **1.018** | (in-cache) |

**The controls change the answer, which is exactly why §13.5 demands them.** Taken alone, the
`gpu_hog` rows say "2.2× slowdown from memory pressure". But `gpu_spin_control` runs the same
duty cycle with cache-resident GEMMs — **5.5 GB/s of background traffic, 27× less** — and still
costs **1.752×**. So most of the 2.22× is the hog taking GPU time slices, not taking bandwidth.
The bandwidth-attributable part is 2.217 / 1.752 = **1.27×**.

Reporting the 2.2× as an interference result would have been wrong, and nothing in the
foreground measurement would have revealed it.

Two more things the table says:

* **Duty cycle barely matters.** 25 % and 100 % give 2.233 and 2.217. Whatever the hog takes,
  it takes at 25 % duty and more of it changes nothing — consistent with a scheduler that
  hands the GPU to the contending context for a slice regardless of how much work it has.
* **CPU-side pressure is nearly free for the prefix** (1.061×), and its own control accounts
  for a third of that. The prefix is not CPU-starved by six competing workers.

**And the CPU is not free either.** Per-phase utilisation of the launching thread, measured in
a separate run (§13.6):

| phase | thread CPU |
|---|---|
| prefix-only | 69.8 % |
| **action-expert-only** | **99.8 %** |
| full inference | 56.5 % |

The action expert saturates a core. The hybrid model assumes the CPU is free to drive the
memory-side command stream; on this device the denoising loop already consumes essentially all
of one core issuing launches — which is the same fact T005 found from the other direction
(969 kernels per step, ~62 µs of dispatch each).

The §13.3 spin-wait check matters here: the CPU **blocks** rather than spins while waiting
(CPU/wall = 0.0036), so 99.8 % is real work, not a busy-wait being miscounted.

---

## E11f — throughput, GPU idle, and sharing the GPU

`results/20260922_E11f_pi0_pipelining/` · `bench/pipelining.py`

### Throughput (60 s back-to-back loops)

| phase | eager inf/s | `reduce-overhead` inf/s |
|---|---|---|
| prefix-only | 2.101 | 2.937 |
| action-expert-only | 4.293 | 4.353 |
| full inference | 1.402 | 2.759 |

### The GPU is idle for most of the action-expert region

From the eager nsys captures, region wall time minus the **union** of kernel intervals:

| N | `ae_total` ms | kernel busy ms | **GPU idle** | **idle share** | kernels |
|---|---|---|---|---|---|
| 4 | 1174.9 | 580.9 | 593.9 ms | **50.6 %** | 10,377 |
| 10 | 2198.2 | 743.6 | 1454.6 ms | **66.2 %** | 22,005 |

`docs/00_context.md` lists `gpu_idle_ae_total` as **unknown**. It is half to two thirds of the
region. More denoising steps make it *worse*, not better: at N = 10 the kernel work grows by
28 % while the region grows by 87 %, because each additional step adds launches faster than it
adds GPU work.

For the pipelined design this is the most encouraging number in the task — there is already
more GPU time free during denoising than the next frame's prefix needs (476 ms of prefix
against 594–1455 ms of idle). It is also a warning: that idle time exists because the GPU is
waiting on the CPU, and a hybrid design that hands the CPU *more* to do will consume it.

### Sharing the GPU

| sharing | host sequence alone | with the prefix loop | slowdown |
|---|---|---|---|
| two processes, eager | 250.4 ms | 251.1 ms | **1.003** |
| two streams, eager | 250.4 ms | 479.4 ms | 1.914 |
| two processes, compiled | 250.3 ms | 250.5 ms | **1.001** |
| two streams, compiled | 250.3 ms | 448.1 ms | 1.790 |

**The two-stream numbers are confounded and should not be read as GPU contention.** The prefix
loop was driven from a second Python thread, so it contends for the GIL with the host sequence's
own Python work, and the host sequence spends 216 ms of its 250 ms *sleeping* (the mapping's
2 ms pause × 6 segments × 18 layers). A clean implementation launches both from one thread onto
two streams; this one does not, and the 1.79–1.91× is at least partly interpreter contention.
Recorded as a deviation, and T011 should fix the harness before repeating it.

The **two-process** figure is sound and is the useful one: **1.001–1.003×**, i.e. a second
process running the prefix loop costs the host-segment sequence essentially nothing — which is
what `host_layer_slowdown = 1.0` assumed, and on this evidence the assumption holds for that
sharing model. The high-priority stream argument was accepted by the driver
(`meta.json.high_priority_stream_accepted`).

---

## What this changes for the hybrid model

1. **The 0.16 ms host layer is achievable, but not the way the stack runs today.** Eager it is
   1.502 ms (9.4×). As one CUDA graph per layer it is 0.177 ms — within 11 % of the assumption
   — and a compiled six-segment sequence is 0.085 ms. The mapping should therefore state which
   implementation it assumes, because the answer moves by 18×.
2. **Price the launches, not the FLOPs.** Every host segment is far below the ridge point and
   none is compute-bound; what they cost is 2–10 kernel launches each at ~29–62 µs of CPU
   dispatch (T005). `kernels x 20 us` is the right *shape* of model and the wrong constants in
   both factors.
3. **Transfers and synchronisation are not free and not bandwidth-limited at these sizes.**
   40 µs fixed + 36 GB/s, and 109 µs for a stream handoff. At the 0.1–1.4 MB objects the
   mapping moves, the fixed terms dominate: the score matrix is 8× the modelled cost.
4. **The KV hand-over is copied every step, not loaded once.** 15.04 MB and 5.53 ms per step —
   ~55 ms per inference at N = 10. Removing the `copy.deepcopy` is worth more than any change
   to the host segments, and T003 already showed a byte-identical `crop` alternative
   (10.37 ms/step against 16.30).
5. **The prefix is a constant for this checkpoint, but only because of padding.** 816 tokens
   whatever the input; 533 of them valid. Trimming the empty camera slot and the prompt padding
   is a design lever worth more than it looks and the model does not represent it.

## Measured, derived, not measured

**Measured:** every segment, operator, transfer, sync and loop time; kernel counts per call;
KV geometry and the deepcopy cost; region latencies under the prefix sweep; background
bandwidth; CPU utilisation; rail power.
**Derived:** analytic bytes and FLOPs from the real shapes (`timing=analytic`, formulas in
`segments.csv.flops_formula`); `achieved_bw_GBps` / `achieved_tflops` as analytic quantity ÷
`gpu_event` time; `t_launch_per_kernel` as `wall_launch` ÷ kernels per call;
`t_host_layer` key `sum_of_segments`; all `fit_a`/`fit_b`/`fit_r2` by least squares on per-size
medians; `gpu_idle_ae_total` as region time − union of kernel intervals.
**Not measured**, each with its exact error in the owning run's `meta.json.not_measured`:
`suffix_embed` under `cudagraph`; `final_norm_out_proj` under `compile`. Quantities absent from
the consolidated file are listed in its `meta.json.quantities_absent` — §13.8 forbids writing a
non-finite value, so an unmeasured quantity is left out rather than written blank.

## Reproduce

```bash
sudo jetson_clocks
# E11a: count kernels under root (CUPTI needs it), then time as the ordinary user —
# attaching a profiler inflates every subsequent call by ~30 %
sudo -n sh -c "HOME=$HOME HF_HOME=$HF_HOME PATH=$PATH \
    python bench/host_segments.py --outdir scratch/T010/counts --mode count"
python bench/host_segments.py --outdir results/<date>_E11a_pi0_hostseg \
    --timed 1000 --kernels-csv scratch/T010/counts/kernels_per_call.csv
nvcc -O3 -arch=sm_87 --shared -Xcompiler -fPIC \
     -o bench/managed_alloc.so bench/managed_alloc.cu     # the zero_copy route
python bench/xfer_latency.py  --outdir results/<date>_E11b_xfer
python bench/handover.py      --outdir results/<date>_E11c_pi0_handover
python bench/prefix_sweep.py  --outdir-prefix results/<date>_E11d_pi0
python bench/mempressure.py   --outdir-prefix results/<date>_E11e_pi0 --seconds 60
python bench/pipelining.py    --outdir results/<date>_E11f_pi0_pipelining --seconds 60
python bench/consolidate_e11.py --outdir results/<date>_E11_pi0_inputs
python docs/reference_scripts/check_model_inputs.py results/*_E11*/model_inputs.csv
python bench/check_report_numbers.py --task=T010 <date>
```
