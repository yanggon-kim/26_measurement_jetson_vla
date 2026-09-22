# E11 — measured inputs for the hybrid GPU + PIM model (π0)

**Task T010** · π0 (`lerobot/pi0_libero_finetuned_v044` @ `45dcc8f`) · MAXN, 16 SMs, GPU pinned
1300.5 MHz, EMC 3199 MHz, 12 CPU cores at 2.2 GHz · bf16 · batch 1 · P = 816, S = 51

Deliverable: `results/20260922b_E11_pi0_inputs/model_inputs.csv` — 1,564 rows, 33 of the 35
quantities §13.8 names, consolidated from eight runs.
`python docs/reference_scripts/check_model_inputs.py results/20260922*_E11*/model_inputs.csv`
exits 0 for every one of them, run in place and never adapted.

> **This report replaces an earlier version whose headline number measured nothing.** The
> directories it was written from are kept with a `SUPERSEDED.md` naming each defect. What
> changed, and how it was caught, is in *Method notes* below — it is not an aside, it is the
> main methodological result of this task.

---

## The answer first

`docs/00_context.md` prices every GPU-side term of the hybrid model analytically:

```
host_op_time  = max(FLOPs / (42 TFLOP/s x 0.4), bytes / (204.8 GB/s x 0.7)) + kernels x 20 us
transfer_time = bytes / (204.8 GB/s x 0.7)
```

with synchronisation free, one kernel per host segment, `T_host_layer` ≈ 0.16 ms, and the
prefix unaffected by memory-side activity. Six of those terms carry an explicit
**[ASSUMPTION]** tag. Here is what each is worth on this device. Every row is regenerated from
the CSVs by `scratch/T010/headline_table.py`; none is typed.

| term | assumed | measured | verdict |
|---|---|---|---|
| kernels per host segment | 1 (2 for attention) | **9.0-10.1 for the norms, 7.1 for attention** | 10x too few |
| `T_host_layer`, sum_of_segments / eager | same 0.16 ms | **1.6162 ms** | 10.10x the assumption |
| `T_host_layer`, sequence / eager | 0.16 ms | **1.5290 ms** | 9.56x the assumption |
| `T_host_layer`, sum_of_segments / compiled | same 0.16 ms | **1.1287 ms** | 7.05x the assumption |
| `T_host_layer`, sequence / compiled | same 0.16 ms | **0.4455 ms** | 2.78x the assumption |
| `T_host_layer`, sum_of_segments / cudagraph | same 0.16 ms | **0.2548 ms** | 1.59x the assumption |
| `T_host_layer`, whole_layer_graph / cudagraph | same 0.16 ms | **0.1827 ms** | 1.14x the assumption |
| transfer fixed cost, pinned d2h | **0** | **38.2 us** | not zero |
| transfer bandwidth, pinned d2h | 143 GB/s | **35.69 GB/s** | 4.0x optimistic |
| transfer fixed cost, pinned h2d | **0** | **39.7 us** | not zero |
| transfer bandwidth, pinned h2d | 143 GB/s | **36.00 GB/s** | 4.0x optimistic |
| synchronisation: synchronize_idle | **free** | **10.4 us** | not free |
| synchronisation: event_synchronize_done | **free** | **1.2 us** | not free |
| synchronisation: event_query | **free** | **0.8 us** | not free |
| synchronisation: tiny_kernel_roundtrip | **free** | **60.7 us** | not free |
| synchronisation: stream_handoff | **free** | **68.7 us** | not free |
| `kv_copied_bytes_per_step` | **0** | **15,040,512 B — the whole cache, every step** | not zero |
| `t_kv_materialise`, stacked | **0** | **0.5726 ms** (n=1000, `wall_isolated`) | not zero |
| `t_kv_materialise`, v_transposed | **0** | **0.4789 ms** (n=1000, `wall_isolated`) | not zero |
| `t_kv_materialise`, deepcopy_per_step | **0** | **5.5824 ms** (n=1000, `wall_isolated`) | not zero |

**The single most consequential correction.** `T_host_layer` ≈ 0.16 ms is not wrong — it is a
statement about an implementation that does not exist in the eager stack, and only *one* of the
six implementations measured here comes near it. **One CUDA graph over the whole layer costs
0.1827 ms, 1.14× the assumption.** Everything else is worse: six separate graphs cost 1.6×, a
compiled sequence 2.8×, and the eager sequence the mapping is implicitly compared against
9.6×. Fusion across segment boundaries is real and worth 2.5× (a compiled sequence against six
separately compiled segments), but it does not get you to 0.16 ms. A graph does.

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

**Four measurement errors found and fixed before any number in this version was reported.**
The first two were caught during the measurement; the last two by the third adversarial
review, *after* a first version of this report had been written and its numbers quoted. That
version's directories are kept with a `SUPERSEDED.md` naming each defect.

1. The first attention replay rebuilt keys from the suffix only, giving **T = 51 instead of the
   real T = P + S = 867** — a 17× understatement of every attention quantity. It now reads the
   prefix K/V out of the cache *before* the layer updates it, and `scores` is `[1, 8, 51, 867]`.
2. CUPTI needs root on this device, and attaching `torch.profiler` inflated every subsequent
   call by ~30 %. Kernel counting and timing therefore run as **two separate passes** — count
   under `sudo`, time as the ordinary user — which is the same separation §13.6 mandates for
   CPU utilisation. Timing runs never attach a profiler.
3. **The timed sequence returned `None`, so `torch.compile` deleted its body.** A compiled
   callable whose outputs are discarded is dead-code-eliminated. Measured here, on this
   device, deliberately, to size the error: a compiled 2048³ bf16 matmul costs **0.0662 ms**
   when its result is discarded and **0.6605 ms** when it is returned — a factor of 10. The
   superseded report's "compiled sequence = 0.085 ms", and the "18× win", the "0.53×" and the
   "1.9× better than assumed" built on it, measured nothing. Every timed callable now returns
   its results.
4. **The background-load generator's duty cycle never applied.** One 256 MiB device-to-device
   copy is ~3.4 ms of GPU time, so queueing copies for `d × 20 ms` and synchronising
   afterwards drained a queue far longer than the 20 ms period: the remaining sleep was always
   negative, never ran, and every duty cycle collapsed to 100 %. The superseded E11e table is
   one condition repeated four times. The hog now synchronises inside its duty window.

**Two conventions that decide whether a number means anything.**

* `gpu_idle` is reported **corrected for the profiler's own overhead**. nsys instruments every
  CUDA API call, and this workload is launch-bound, so the region wall time it reports is
  inflated; kernel durations are not, because nsys reads them from the GPU. Dividing
  `(profiled wall − busy)` by the profiled wall therefore charges nsys's overhead to the model.
  Each capture is paired with the identical loop run unprofiled in the same process seconds
  earlier, and the corrected fraction is `(unprofiled wall − measured kernel busy) /
  unprofiled wall`. Both figures and their ratio are in `gpu_idle.csv`.
* Idle is computed **per inference and averaged**, never as one envelope over the whole
  capture. An envelope across all NVTX instances counts the gap between inferences — the next
  inference's vision and VLM prefix, which saturate the GPU — as action-expert idle.

---

## E11a — the host segments

`results/20260922b_E11a_pi0_hostseg/` · `bench/host_segments.py` · ≥ 20 warm-ups and 1,000
timed calls per (segment, style, timing), layers 0 and 9, on the model's own tensors captured
from one real denoising step. Full tables in that directory's `summary.md`; distributions,
`gap_ms = 2`, `gpu_event` and `wall_launch` in `segment_summary.csv`; 216,000 per-iteration
samples in `raw/segment_samples.csv.zst`; kernel counts, their `key_averages` table and one
Chrome trace per (segment, layer, style) in `raw/profiler/` (61 traces, 496 KB compressed).

At layer 0, `wall_isolated`, `gap_ms = 0`:

| segment | eager ms | compiled ms | cudagraph ms | kernels, eager | bytes r + w | FLOPs |
|---|---:|---:|---:|---:|---|---:|
| `rmsnorm1` | 0.3961 | 0.2663 | **0.0463** | 9.0 | 106,496 + 104,448 | 417,792 |
| `rope` | 0.4618 | 0.1930 | **0.0617** | 10.1 | 287,232 + 235,008 | 705,024 |
| `softmax` | 0.1038 | 0.1403 | **0.0518** | 2.1 | 2,122,416 + 2,122,416 | 1,768,680 |
| `residual_rmsnorm2` | 0.4328 | 0.2410 | **0.0499** | 10.1 | 313,344 + 208,896 | 470,016 |
| `gelu_gate_mul_up` | 0.1072 | 0.1462 | **0.0226** | 2.1 | 835,584 + 417,792 | 1,880,064 |
| `partials_residual` | 0.1144 | 0.1420 | **0.0225** | 2.1 | 522,240 + 104,448 | 208,896 |
| `residual` | 0.0630 | 0.1446 | **0.0227** | 1.1 | 208,896 + 104,448 | 52,224 |
| `attention_gpu` | 0.5295 | 0.3870 | **0.3154** | 7.1 | 9,433,776 + 3,038,784 | 362,225,664 |
| `cast_scores_fp32_fp16` | 0.0685 | 0.1407 | **0.0227** | 1.0 | 1,414,944 + 707,472 | 0 |

Layer 9 agrees with layer 0 to within 3 % on every row, which is the intended check that these
are properties of the layer and not of its position.

**Kernel counts are 1–10, not 1.** The model charges one launch per segment at 20 µs.
`rmsnorm1` is **9**, `rope` and `residual_rmsnorm2` **10.1**, `attention_gpu` **7.1**. The six
mapping segments are ~35 launches per layer, not 6. `PiGemmaRMSNorm` is a Python composite —
cast to fp32, square, mean, rsqrt, multiply, weighted multiply, cast back — and nothing fuses
it in eager mode. The counts reproduced exactly across three independent passes.

**Compiling a segment on its own is counter-productive; compiling the sequence is not; a graph
is better than either.** For five of the nine segments `compiled` is *slower* than `eager`
(`residual` 0.145 vs 0.063, `softmax` 0.140 vs 0.104): a compiled callable still pays a fixed
per-call cost and these segments are too small to amortise it. The layer-level table is where
the effect shows:

| `t_host_layer`, layer 0 | ms | vs the analytic 0.16 ms |
|---|---:|---:|
| `sum_of_segments`, eager (derived) | 1.6162 | 10.10× |
| `sequence`, eager | 1.5290 | 9.56× |
| `sum_of_segments`, compiled (derived) | 1.1287 | 7.05× |
| `sequence`, compiled | 0.4455 | 2.78× |
| `sum_of_segments`, cudagraph (derived) | 0.2548 | 1.59× |
| **`whole_layer_graph`** | **0.1827** | **1.14×** |

`docs/00_context.md` anticipates the last case — "one controller macro replaces per-segment
launches (launches per layer = 1)" — and prices it at one 20 µs launch. Measured, that layer
costs 0.1827 ms: the macro is worth having, and it is not free.

**Per-step extras** (`order_in_layer = -1`, once per denoising step rather than once per layer):
`suffix_embed` 1.4466 ms eager / 0.8492 compiled (27.2 kernels), `final_norm_out_proj` 0.5920 ms
eager / 0.0795 as a graph (11 kernels), `euler_update` 0.1036 / 0.1483 / 0.0228 (2.1 kernels).
The analytic `T_extra` is 0.08 ms per step; measured they total **2.14 ms** eager.

Two items are **not measured**, with their exact errors, per the task's escape hatch:
`suffix_embed` under `cudagraph` (`cudaErrorStreamCaptureInvalidated`) and
`final_norm_out_proj` under `compile` (inductor: *"Offset increment outside graph capture
encountered unexpectedly"*). Both are in `meta.json.not_measured` and named in `summary.md`.
Compile time is logged per callable: 5.36 s in total, of which the sequence is 0.5 s.

---

## E11b — transfers and synchronisation

`results/20260922b_E11b_xfer/` · `bench/xfer_latency.py` · `model = none`, because this part is
model-independent and measured once.

**The fixed term is the point.** `t = a + b·bytes`, fitted over 13 sizes:

| method | direction | a (µs) | b (ms/MB) | GB/s from b | r² |
|---|---|---:|---:|---:|---:|
| pinned | h2d | **39.7** | 0.0278 | 36.00 | 0.9995 |
| pinned | d2h | **38.2** | 0.0280 | 35.69 | 0.9990 |
| pageable | h2d | 73.9 | 0.0689 | 14.52 | 0.9984 |
| pageable | d2h | 56.7 | 0.1300 | 7.69 | 0.9999 |

The model assumes `a = 0` and prices every transfer at 143 GB/s. Both halves are wrong in the
same direction: there is a ~40 µs floor no transfer can go below, and the slope corresponds to
36 GB/s, **4× lower** than assumed. For π0's own boundary sizes the floor dominates — an
`activation_tile` of 104,448 B takes 0.0437 ms pinned, of which the fixed term is 91 %.

**Synchronisation is not free**: `event_query` 0.8 µs, `event_synchronize` on a completed event
1.2 µs, `synchronize` on an idle device 10.4 µs, a tiny-kernel round trip 60.7 µs, and a
**stream hand-off 68.7 µs** net of the same two kernels on one stream. A mapping that hands work
between the GPU and a memory-side engine once per layer pays one of these per hand-over: at 18
layers × 4 steps that is 4.9 ms per inference for the hand-off alone, against a measured
eager `ae_total` of 221 ms.

Zero-copy through a `CUDAPluggableAllocator` over `cudaMallocManaged` works and is reported
per part (`cpu_write`, `gpu_read`, `roundtrip`) with its own control in the same process,
because `change_current_allocator` is global and irreversible.

---

## E11c — the hand-over object

`results/20260922b_E11c_pi0_handover/` · `bench/handover.py`

The object the VLM prefix hands to the action expert is the prefix KV cache: 18 layers × (K, V)
at `[1, 1, 816, 256]` bf16 — **835,584 B per layer, 15,040,512 B in total**, which matches the
task's independent cross-check `18 × 2 × P × 1 × 256 × 2 B` exactly. Of the 816 positions only
**533 carry a valid mask**; the padded ones are stored, copied and attended over identically,
so 5,216,256 B of the total is padding.

**The model assumes `kv_copied_bytes_per_step = 0` and `t_kv_materialise = 0`. Both are
measurably not.** `copy.deepcopy(past_key_values)` at `modeling_pi0.py:918` runs **once per
denoising step** and costs **5.5824 ms** (n = 1000, p5 5.5445, p95 5.6274) moving the whole
15.04 MB cache. At N = 4 that is 22.3 ms per inference, 10 % of the measured eager `ae_total`,
spent copying a tensor that is never written to. Building the cache once costs 0.5726 ms
stacked or 0.4789 ms with V transposed.

`kv_reuse.md` carries the evidence from both sides: the code path, and `data_ptr()` sampled on
entry *and* after the deepcopy across steps — the copy's address differs from the original,
which is the proof, and is stable across steps only because the caching allocator reuses the
same block.

---

## E11d — how the prefix scales with its input

`results/20260922b_E11d_pi0_{eager,reduce-overhead}_prefixsweep/` · `bench/prefix_sweep.py` ·
images {1, 2, 3} × prompt {short, typical, long} = 9 points per mode, N = 10, 100 timed
inferences per point.

**The prefix cost does not depend on the input at all.** Across all nine points, eager:

| region | min ms | max ms | spread |
|---|---:|---:|---:|
| `vision` | 239.81 | 241.23 | **1.0059×** |
| `vlm_prefix` | 233.14 | 234.78 | **1.0070×** |
| `ae_total` | 572.63 | 575.53 | 1.0051× |
| `e2e` | 1053.43 | 1057.01 | 1.0034× |
| `e2e`, `reduce-overhead` | 462.78 | 465.01 | 1.0048× |

Meanwhile `P_valid` moves from **262 to 807**, a factor of 3.1. One camera and a six-token
prompt costs the same as three cameras and a thirty-nine-token prompt, to within 0.7 %.

The reason is the input format, and it matters because the hybrid model parameterises the
prefix by P:

* `config.input_features` declares **three** image slots and `config.empty_cameras = 1`. A
  camera that is not supplied is **not dropped** — `_preprocess_images` synthesises a −1-filled
  tensor with a zero mask and it is still encoded into 256 prefix tokens. `image_tokens` is 768
  at every point in the sweep.
* `config.tokenizer_max_length = 48`, so every prompt is padded to 48.

So **P_processed = 3 × 256 + 48 = 816 for every input this checkpoint will ever see**, and what
changes is the attention mask, not the work. `docs/01_measurement_plan.md` anticipates exactly
this ("models that pad the prompt to a fixed length will show a constant processed P — report
that, it is a result"), and it is why `prefix_latency_fit` has **no slope against P**:
`prefix_fit.csv` records `x is constant at 816 across the sweep; no slope can be fitted, which
is itself the result`.

The one fit that exists, `vision_vs_images`, gives a = 241.74 ms and b = **−0.66 ms per image**
(r² = 0.92 over three points) — a *negative* slope of the same size as the run-to-run noise,
which is the honest way of saying flat.

**What this means for the hybrid model.** `t_vision` and `t_vlm_prefix` can be treated as
constants for this checkpoint. But the constancy is a property of the *input format*, not of the
hardware: a deployment that trimmed the empty camera slot and the prompt padding would encode
533 of 816 tokens and none of these numbers would carry over. That is a design lever the model
does not currently represent. **π0.5 has `tokenizer_max_length = 200` and `empty_cameras = 0`,
so this result is not expected to transfer to it** — T011 measures it rather than assuming.

---

## E11e — what memory pressure actually costs the prefix

`results/20260922d_E11e_pi0_{eager,reduce-overhead}_mempressure/` · `bench/mempressure.py` +
`bench/bw_hog.py` · prefix-only loop, 60 s per condition, foreground pinned to cores 0–5 and
the competitor to 6–11, 20 ms period, tegrastats attached throughout. Eager shown; the
`reduce-overhead` table is in that directory's `summary.md` and agrees to within 0.01 on every
slowdown.

| load | asked | **achieved** | bg GB/s alone | bg GB/s during | % of 187.47 | prefix median ms | slowdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| `none` | 0 | — | — | — | — | 479.12 | **1.0** |
| `gpu_hog_25` | 25 | 26.56 | 36.799 | 8.629 | 4.6 | 528.44 | **1.1029** |
| `gpu_hog_50` | 50 | 51.51 | 72.796 | 16.322 | 8.7 | 578.87 | **1.2082** |
| `gpu_hog_75` | 75 | 75.2 | 107.05 | 21.609 | 11.5 | 619.27 | **1.2925** |
| `gpu_hog_100` | 100 | 99.99 | 145.725 | 24.656 | 13.2 | 647.65 | **1.3517** |
| `cpu_hog_25` | 25 | 29.67 | 26.741 | 27.709 | 14.8 | 495.28 | **1.0337** |
| `cpu_hog_50` | 50 | 63.06 | 32.882 | 39.547 | 21.1 | 507.19 | **1.0586** |
| `cpu_hog_75` | 75 | 89.11 | 32.711 | 46.029 | 24.6 | 512.37 | **1.0694** |
| `cpu_hog_100` | 100 | 99.99 | 32.911 | 48.933 | 26.1 | 516.52 | **1.0781** |
| `gpu_spin_control` | 100 | 99.99 | 21.906 | 0.705 | — | 558.80 | **1.1663** |
| `cpu_spin_control` | 100 | 99.99 | 186.006 | 183.042 | — | 491.44 | **1.0257** |

**Read the achieved duty, not the requested one.** The unit of work cannot be made arbitrarily
small — a 32 MiB GPU chunk is ~0.4 ms against a 5 ms window at duty 25 — so a hog asked for 25 %
delivers 26.6 %. The achieved figure is measured from the time the hog actually spent working in
each 20 ms period, and it is what the analysis uses.

**What the controls are for, and what they say.** Both controls run at the same duty and take
the same time slices as their hog; only the hog takes bandwidth. That separates the two effects:

| | total slowdown | control (time slices only) | attributable to lost bandwidth |
|---|---:|---:|---:|
| `gpu_hog_100` vs `gpu_spin_control` | 1.352× | 1.166× | **1.159×** |
| `cpu_hog_100` vs `cpu_spin_control` | 1.078× | 1.026× | **1.051×** |

A GPU competitor that takes **24.656 GB/s during the run — 13.2 % of the
measured 187.47 GB/s peak — costs the prefix 35 %, but 17 points
of that is simply having to share the GPU**, because a compute-bound load with a small memory
footprint at the same duty costs 1.166× on its own. The bandwidth itself is worth
1.159×. A CPU competitor moving 48.933 GB/s of real DRAM traffic
from the other six cores costs only 7.8 %, of which arithmetic that stays in
cache accounts for 2.6 %.

That is consistent with everything else measured here: the prefix is not bandwidth-starved, so
taking bandwidth away from it hurts far less than taking its GPU time slices away.

**Limits — protocol §13.5, repeated as it requires.** This is a **proxy** for memory-side
activity on the same DRAM. A real PIM occupies DRAM channels through the command bus and blocks
ordinary accesses to them; a software hog instead competes at the memory controller and in the
caches, and additionally takes GPU time slices (`gpu_hog`) or CPU cores (`cpu_hog`). The controls
bound those side effects; **the result is the sensitivity of the GPU work to lost memory
bandwidth, not a measurement of PIM interference.**

One consequence of the granularity fix deserves stating, because it changes what the numbers
mean. The superseded pass issued indivisible 256 MiB copies and blocked the GPU for ~3.4 ms at a
stretch, giving `gpu_hog_100` a slowdown of 1.95×; the chunked hog yields every ~0.4 ms and gives
1.35×. Those are two different competitors — a coarse time-slice competitor and a bandwidth
competitor — and the chunked one is the right proxy for §13.5's purpose. The number to compare a
PIM against is the bandwidth column, 1.159×, not the total.

### CPU utilisation per phase

Three scopes, because they answer different questions and disagree by 10×: `thread` is what the
Python loop itself burns, `process` adds the CUDA driver's worker threads, and
`system_all_cores` / `busiest_core` come from the kernel's own `/proc/stat` accounting, the only
one that sees the whole machine. Per-core samples are in `raw/cpu_util_samples.csv`.

| phase | thread % | process % | system, all 12 cores % | cores busy | busiest core % |
|---|---:|---:|---:|---:|---:|
| `vlm_prefix` | 69.1 | 69.2 | 21.0 | 2.52 | 74.5 |
| `ae_total` | **99.1** | **99.2** | 11.0 | 1.32 | **100.0** |
| `e2e` | 56.5 | 56.6 | 6.3 | 0.76 | 66.0 |

The action-expert loop pins **one core at 100 %** while the GPU sits 75.6 % idle (E11f). That is
the launch-bound result stated in CPU terms: the bottleneck is a single Python thread issuing
kernels, and eleven of twelve cores are doing nothing.

### Power per phase (four INA3221 rails)

| phase | VDD_GPU_SOC | VDD_CPU_CV | VIN_SYS_5V0 | VDDQ_VDD2_1V8AO | module total |
|---|---:|---:|---:|---:|---:|
| `vlm_prefix` | 39,410 mW | 2,388 | 8,797 | 3,326 | **53,921 mW** |
| `ae_total` | 10,757 | 3,187 | 5,755 | 1,310 | **21,009 mW** |
| `e2e` | 23,914 | 2,391 | 7,269 | 2,218 | **35,793 mW** |

Medians over 28–85 samples; means, p5 and p95 in `power.csv`. `VDDQ_VDD2_1V8AO` is the memory
**regulator input**, not DRAM power, and is never rescaled to 1.8 V. The four rails do not
double-count (established in T004/E7's rail check).

The GPU rail draws **3.7× more during the prefix than during the denoising loop** (39.4 W vs
10.8 W) — the prefix saturates the GPU and the expert does not. The expert phase draws *more* CPU
rail power than the prefix (3.19 W vs 2.39 W) despite drawing a third of the GPU power, which is
the same story again from the power side.

---

## E11f — throughput, GPU idle, and sharing the GPU

`results/20260922b_E11f_pi0_pipelining/` · `bench/pipelining.py`

### Throughput (60 s back-to-back loops)

The right-hand column is headed **"compile + TF32"**, not "compiled", and the distinction is
not pedantic — see the correction below.

| loop | eager ms | eager inf/s | compile + TF32 ms | inf/s | what actually ran compiled |
|---|---:|---:|---:|---:|---|
| prefix only | 478.65 | 2.090 | 340.39 | 2.938 | `eager_subpath` — **nothing** |
| action expert only | 221.13 | 4.521 | 221.81 | 4.508 | `eager_subpath` — **nothing** |
| full inference | 704.13 | 1.420 | 363.20 | 2.753 | `compiled` |

Under `reduce-overhead` only the *full* loop runs a compiled graph. The prefix-only and AE-only
loops drive sub-paths that `torch.compile` never wrapped, which is why the AE loop is identical
in both columns (221.13 vs 221.81 ms).

**Correction, 2026-09-22 — the prefix loop's 1.41× is not compilation.** An earlier version of
this report read that column as a compile result. It is not. `modeling_pi0.py:591-595` enables
TF32 for every fp32 matmul in the process —

```python
if config.compile_model:
    torch.set_float32_matmul_precision("high")      # ← global, one line before the compile
    self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
```

— and `bench/pipelining.py:274` loads the policy with `compile_mode=`, which sets that flag. So
every "compiled" row here also has TF32 on, including the two rows that were never compiled at
all. Measured with an uncompiled policy and **only** the precision toggled
(`results/20260922_E4b_pi0_tf32_confound/`):

| loop | `matmul_precision=highest` | `=high` | TF32 alone |
|---|---:|---:|---:|
| prefix | 478.01 ms | 341.44 ms | **1.400×** |
| action expert | 232.76 ms | 230.11 ms | 1.012× |
| full inference | 716.25 ms | 575.72 ms | **1.244×** |

1.400× against the 1.406× in the table above: **none** of the prefix loop's gain is
compilation. The full loop's 1.94× decomposes as **1.244× TF32 × 1.56× compilation**. The
action expert is untouched by TF32 (1.012×), as expected for bf16 work that is launch-bound.

**What compilation does do, and where.** It is not visible in this table, because the AE loop
here was never compiled. T003/E4's `step_only` variant compiles `denoise_step` itself, with
TF32 off throughout (`bench/run_e4.py:210` loads with `compile_mode=None`), and gives the clean
answer:

| region | eager N=4 | compiled N=4 | | eager N=10 | compiled N=10 | |
|---|---:|---:|---:|---:|---:|---:|
| `vision` | 239.35 | 241.58 | 1.00× | 239.60 | 240.42 | 1.00× |
| `vlm_prefix` | 233.08 | 234.64 | 1.00× | 232.93 | 234.59 | 0.99× |
| **`ae_total`** | 215.53 | **68.88** | **3.13×** | 533.03 | **170.31** | **3.13×** |

**Compilation speeds up the action expert and nothing else.** Vision and the VLM prefix are flat
because they are already GPU-saturated — 39.4 W on the GPU rail against 10.8 W for the expert
(E11e) — so there are no launch gaps to close. The expert is the only launch-bound region, and
removing its launches is exactly what `reduce-overhead` does. `results/20260922_E4c_pi0_opt_ladder/`
separates TF32, compilation, CUDA graphs and autotuning in a single ladder, one factor per rung.

### The GPU is idle for 75.6 % of the action-expert region

| mode | N | ae_total under nsys | ae_total unprofiled | inflation | kernel busy | **idle (corrected)** |
|---|---:|---:|---:|---:|---:|---:|
| eager | 4 | 339.11 ms | 221.81 ms | ×1.53 | 54.15 ms | **75.6 %** |
| eager | 10 | 857.76 ms | 554.05 ms | ×1.55 | 135.28 ms | **75.6 %** |

Two corrections are folded into that column and both matter. Idle is computed **per inference
and averaged**, not as one envelope over the capture — an envelope counts the gap between
inferences, which is the next inference's vision and VLM prefix saturating the GPU, as expert
idle. And nsys inflates the region's wall time by ×1.53 while leaving kernel durations alone, so
`(profiled wall − busy) / profiled wall` charges the profiler's overhead to the model and reads
84 %. The corrected figure divides the same measured kernel-busy time by the **unprofiled** wall
time of the identical loop, measured in the same process seconds earlier.

75.6 % idle at both N = 4 and N = 10 is the same statement T005 made from the other direction:
the expert is not waiting for memory, it is waiting to be given work.

The compiled captures produce no idle figure at all, and the reason is structural, not a
failure: `bench/regions.py`'s context managers force a graph break, so a compiled run with
regions on would not be the graph the model executes. The compiled captures therefore run with
`--no-regions` and have no `ae_step` ranges to project onto. This is the "known wrinkle" the
task anticipates; it is recorded in `gpu_idle.csv.note`.

### Sharing the GPU: the limit is Python, not the GPU

| sharing | eager slowdown | of the non-sleep part | compiled slowdown | of the non-sleep part |
|---|---:|---:|---:|---:|
| two CUDA streams, one process | 2.245× | **10.38×** | 1.691× | **6.18×** |
| two processes | 1.002× | 1.012× | 1.002× | 1.016× |

A prefix loop running beside the host-segment sequence costs **nothing measurable when the two
are separate processes** (1.002×) and costs a factor of 2.2 when they are two CUDA streams in
one process. The high-priority stream request was accepted (range (0, −3), requested 0, got 0),
so this is not a priority failure. Both halves of the sequence's 249 ms are reported because
216 ms of it is `time.sleep` standing in for memory-side work; the `non_sleep` column divides
out that constant and makes the effect visible: **10.4×** on the part that actually executes.

The interpretation follows from everything else in this report. Two streams share one Python
interpreter, and this workload is launch-bound — 969 kernels per step, ~62 µs of CPU per
`F.linear`. Contending for the GIL costs far more than contending for the GPU. An overlap
scheme that runs the memory side in the same process as the GPU side will not get the overlap
it expects; one that uses a separate process will.

---

## What this changes for the hybrid model

1. **`kernels × 20 µs` under-counts by ~10× per segment and by ~6× per layer.** Use the measured
   counts in `n_kernels_host_segment` and ~29–62 µs per launch (T005/E2), not 20 µs.
2. **`T_host_layer` = 0.16 ms is reachable only as one CUDA graph per layer** (0.1827 ms). If
   the mapping assumes per-segment dispatch, the honest figure is 1.53 ms eager or 0.45 ms
   compiled.
3. **Transfers have a ~40 µs floor and run at 36 GB/s, not 143.** For every π0 boundary tensor
   except the KV cache the fixed term dominates.
4. **Synchronisation costs 0.8–69 µs depending on what is synchronised.** A per-layer hand-off
   costs 4.9 ms per inference at N = 4.
5. **`kv_copied_bytes_per_step` is the whole 15.04 MB cache, every step, at 5.58 ms.** This is
   pure software cost — the cache is never written — and it is 10 % of `ae_total`.
6. **The prefix is a constant for this checkpoint**, because the input format fixes P at 816.
   Do not parameterise it by P without checking the model's padding.
7. **Overlap belongs in a separate process.** Two streams in one process cost 10.4× on the
   executing part; two processes cost 1.2 %.

---

## Measured, derived, not measured

**Measured**: every latency, byte count, kernel count, bandwidth, power and utilisation figure
above, each with median, p5, p95 and n in the CSV it cites.
**Derived**: `sum_of_segments` (the six mapping segments added), `t_launch_per_kernel`
(`wall_launch` ÷ kernels), the corrected idle fraction (measured busy ÷ measured unprofiled
wall), achieved GB/s and TFLOP/s (analytic bytes and FLOPs ÷ measured `gpu_event` time, and
published only for the `cudagraph` style where the timed region is the work).
**Not measured**, each with its exact error in `meta.json.not_measured` and its summary:
`t_preprocess` and `t_postprocess` (no region instrumented for them); `suffix_embed` under
`cudagraph`; `final_norm_out_proj` under `compile`; the GPU-idle fraction under
`reduce-overhead`.
