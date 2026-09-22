# E1 / E2 — what this Orin can do, and what the action expert actually does with it

**Task T005** · π0 (`lerobot/pi0_libero_finetuned_v044` @ `45dcc8f`) · MAXN, 16 SMs, GPU pinned
1300.5 MHz, EMC 3199 MHz · bf16 · `bench/` commit recorded in each `meta.json`

Results: `results/20260921_E1_peak/`, `results/20260921_E2_ae_kernels/`,
`results/20260921_E2_kernel_counts{,_N2,_N10}/`, `results/20260921_E7_rails_check/`

---

## The answer first

**1. This device reaches 91 % of its spec bandwidth and 80 % of its spec bf16 throughput.**

| quantity | spec | measured | ratio |
|---|---|---|---|
| memory bandwidth | 204.8 GB/s | **185.59 GB/s** | 0.906 |
| bf16 dense GEMM | 42 TFLOP/s | **33.55 TFLOP/s** | 0.799 |
| ridge point | task's ≈ 208 FLOP/B (spec-derived 205.1) | **180.8 FLOP/B** | 0.87 |

**2. E11's analytic host model is pessimistic by 2× on compute.** `docs/00_context.md` carries
two fractions marked **[ASSUMPTION]**; both are now measured:

| fraction | assumed | measured | assumption is |
|---|---|---|---|
| `eff_flops_frac` | 0.40 | **0.799** | 2.00× too pessimistic |
| `eff_bw_frac` | 0.70 | **0.906** | 1.29× too pessimistic |

**3. The action expert's GEMMs are not the problem.** With the Python interpreter out of the
timed region, the expert's 126 projection GEMMs — 622.85 MB, i.e. 97.6 % of the 638 MB that
T002's `ae_step` floor was built on — run at **1.31× their memory floor**. There is very little
left there for a faster memory system to win.

**4. A denoising step launches 969 GPU kernels, and only 17 % of them are GEMMs.** Measured by
slope over N ∈ {2, 10}, not assumed: 48 % of the launches are elementwise kernels, 12 % copies,
8 % KV-cache concatenation, 8 % normalisation.

**5. The step is CPU-dispatch-bound.** The same 126 projections take **8.82 ms** issued from
Python and **4.42 ms** replayed as one CUDA graph — a 2.0× gap with identical GPU work.

**6. The L2 sysmem-miss substitute for DRAM traffic works — for reads only, and the protocol's
definition of the write half is wrong.** Reads validate to within 0.9 % on four kernels with
analytically known traffic. The *write* path is **identically zero** in every `*_lookup_miss`
counter; it is recovered exactly by `lts__t_sectors_aperture_sysmem_op_write.sum` instead.

**7. The four power rails do not double-count and may be summed.**

---

## E1 — the peaks

`results/20260921_E1_peak/` · `bench/micro_peak.py`

**Method.** GEMM rows: median of **100 timed calls after 20 warm-ups** (protocol §2). Bandwidth
rows: **3 independent repeats** of (20 warm-ups + 100 timed calls), and the reported GB/s is the
median of the **best** repeat — a deliberately optimistic estimator, because this is a *peak*;
the pooled median and p5–p95 across all three repeats are in each row's `note`, and every
individual call is in `raw/latency_samples.csv`. Every GEMM is checked against an fp32 reference
**before** its time is used; `correct` and `rel_err` are per row, and only correct rows could set
a peak.

### Bandwidth, by access pattern

| kernel | bytes counted | GB/s | % of 204.8 spec |
|---|---|---|---|
| `mul_scalar_fp32` (`z = 2·x`, fp32) | r+w | **185.59** | **90.6 %** |
| `add_out_of_place` (`z = x+y`) | r+r+w | 170.32 | 83.2 % |
| `mul_scalar_out_of_place` | r+w | 168.24 | 82.1 % |
| `copy_fp32` | r+w | 166.90 | 81.5 % |
| `axpy_inplace` (`y += a·x`) | r+r+w | 165.11 | 80.6 % |
| `fill_` (pure write) | w | 163.43 | 79.8 % |
| `clone_allocating`, 256 MiB / 1 / 2 GiB | r+w | 163.25 / 159.94 / 155.81 | 76.1–79.7 % |
| `copy_out_of_place` | r+w | 155.65 | 76.0 % |
| `sum_reduction` (pure read) | r | 103.43 | 50.5 % |

Eight patterns rather than one, because a read-modify-write, a pure write and a copy that
allocates its own output are not the same kernel. The `clone_allocating` rows show what the
caching allocator costs and are **not** peaks; `sum_reduction` is limited by the reduction tree.
Which pattern comes top varies between runs — see the variability section below.

### Compute

| dtype | best shape | TFLOP/s | note |
|---|---|---|---|
| **bf16** | 2048³ | **33.55** | 79.9 % of the 42 spec |
| fp16 | 2048³ | 27.25 | |
| tf32 | 4096³ | 15.66 | fp32 tensors, `allow_tf32=True` |
| fp32 | 8192³ | 3.48 | true IEEE fp32 (CUDA cores) |
| int8 | 8192³ | 10.49 | **not a hardware peak — see below** |

bf16 peaks at n = 2048 and falls to 21.7 TFLOP/s at n = 8192. That is **not** throttling:
`raw/sustained_load.csv` holds 12 consecutive repeats of an 8192² bf16 GEMM, during which the
GPU clock stayed at 1300.5 MHz while the die went 51 → 56 °C and throughput showed no downward
trend (19.3 → 23.3 TFLOP/s, the first repeat being the slowest). The fall-off is cuBLAS kernel
selection at large n.

**int8 is a software-path number.** `torch._int_mm` reaches 10.5 TOP/s, roughly an eighth of
this GPU's dense INT8 capability. The result is bit-exact against an fp32 reference, so this is
kernel selection in PyTorch on `sm_87`, not silicon. Recorded as measured and explicitly **not**
offered as a hardware constant.

### The ridge point

Measured **180.8 FLOP/B** (33.55 TFLOP/s ÷ 185.59 GB/s), against the ≈ 208 FLOP/B the task
expected and the 205.1 that `docs/00_context.md` derives from the spec sheet. It comes out
**13 % below** both, because the bandwidth peak is reached more nearly than the compute peak
(0.906 vs 0.799 of spec). A kernel needs ~181 FLOP per byte here to be compute-bound.

### ⚠ Bandwidth varies run to run, and I cannot explain it

Repeated whole-run and single-kernel measurements do not agree, and the disagreement is large
enough to matter for every floor in this campaign.

| evidence, all committed | spread |
|---|---|
| `raw/bandwidth_modes.jsonl` — 22 fresh processes, same kernel | 165.1 → 188.6 GB/s (**1.14×**) |
| `raw/earlier_passes/` — three earlier whole-E1 passes | 161.8, 183.4, 189.1 GB/s (**1.17×**) |

Across all 22 probe samples the GPU clock was 1300.5 MHz, the EMC clock 3199 MHz (sampled
*during* the timed loop, not just around it) and the die 45–47 °C — every one of them constant
while bandwidth moved by 14 %. The shift applies uniformly to every memory-bound kernel in a
given run, so a run is "fast" or "slow" as a whole rather than one pattern being erratic.

What was checked, and what that check rests on:

| hypothesis | how it was checked | result | in a committed file? |
|---|---|---|---|
| GPU clock | sampled either side of every timed row, and per probe sample | pinned 1300.5 MHz throughout | **yes** (`latency_summary.csv`, `bandwidth_modes.jsonl`) |
| EMC clock | sampled before, **during** and after the timed loop | pinned 3199 MHz throughout | **yes** (`bandwidth_modes.jsonl`) |
| die temperature | recorded with every sample | 45–47 °C across the whole range | **yes** (`bandwidth_modes.jsonl`) |
| thermal throttling | 12 sustained 8192² GEMM repeats | clock stable, no decay | **yes** (`sustained_load.csv`) |
| buffer placement | `data_ptr()` recorded per sample | identical pointers in fast and slow samples | **yes** (`bandwidth_modes.jsonl`) |
| competing processes | `ps` during runs | none | no — checked interactively |
| accumulator value drift | refilled `y` with fresh `N(0,1)` each repeat | no effect | no — checked interactively |
| buffer reuse vs reallocation | both | no effect | no — checked interactively |
| allocator pool size and ordering | replayed a fast run's allocation sequence | no effect | no — checked interactively |
| re-applying `jetson_clocks` | measured immediately after | no effect | no — checked interactively |
| idle time | 4 minutes idle | no effect | no — checked interactively |

The first five rows are reproducible from committed files. **The last six were run interactively
and their output was not retained**; they are reported as reasoning, not as evidence, and should
be treated as weaker.

**How it is handled.** `peak_memory_bandwidth_GBps` is this run's best observed value, which is
what "peak" should mean for a roofline. **Consequence for the campaign: any bandwidth-derived
floor carries roughly ±7 %, and a measured-vs-floor ratio should use a peak measured in the same
session.** E2 below was run immediately after this E1 and normalised against its peak. Worth
re-checking after the power-mode reboots, which clear whatever this is.

---

## How many kernels a denoising step launches — measured, not assumed

`results/20260921_E2_kernel_counts{,_N2,_N10}/` · `bench/count_step_kernels.py`,
`bench/analyze_step_kernels.py`

An earlier draft of this report asserted that "a denoising step issues on the order of 400
operations". That was a guess, and it was wrong by a factor of 2.4. It is now measured with
`torch.profiler` (CUDA activity needs root on this device, like `ncu`), at N ∈ {2, 4, 10}, and
the per-step figure is taken as a **slope** so that vision and the VLM prefix — which run once
per inference, not once per step — fall out:

| N | total kernel launches |
|---|---|
| 2 | 4,567 |
| 4 | 6,505 |
| 10 | 12,320 |

**969 kernel launches per denoising step**, plus 2,629 fixed per inference. The two-point fit
predicts the N = 4 run exactly (6,505 predicted, 6,505 measured), which is the check that the
split is real.

| category | launches per step | share |
|---|---|---|
| elementwise | 466.1 | **48.1 %** |
| GEMM (tensor / CUDA core) | 168.0 | **17.3 %** |
| copy | 119.0 | 12.3 % |
| concatenate (KV cache) | 78.0 | 8.0 % |
| reduction / normalisation | 75.0 | 7.7 % |
| other | 42.0 | 4.3 % |
| softmax | 18.0 | 1.9 % |
| memset | 3.0 | 0.3 % |

**The expert's GEMMs are 17 % of the launches in a step.** The other 83 % is elementwise work,
copies, KV-cache concatenation and normalisation — all of it cheap per kernel and all of it
paying full dispatch cost.

Two counts worth singling out, because they turn assumptions into measurements:

* **37 `reduce_kernel` + 37 `rsqrt` per step.** RMSNorm is applied twice per layer
  (`input_layernorm` and `post_attention_layernorm` in `lerobot/policies/pi_gemma.py`), so 18
  layers × 2 = 36 invocations per step is what those counts say, within rounding. The earlier
  draft assumed that multiplier; it is now confirmed from the profiler.
* **168 GEMMs per step** against the 126 projections the expert contains — the extra ~42 are
  the attention products (QK<sup>T</sup> and PV, 18 layers × 2) plus a few more.

Times are deliberately **not** taken from this run: `torch.profiler` perturbs them. Only the
counts are used. Traces are committed compressed (`zstd -19`, ~23×), with each original's
sha256 and size in `raw/ORIGINALS.json`.

---

## E2 — the action expert, one kernel at a time

`results/20260921_E2_ae_kernels/` · `bench/micro_ae_kernels.py`

The shapes are read from the loaded checkpoint, never from a table:

> hidden 1024 · intermediate 4096 · 18 layers · 8 query heads · **1 KV head** · head_dim 256
>
> `q_proj` 1024→2048 · `k_proj` 1024→256 · `v_proj` 1024→256 · `o_proj` 2048→1024
> `gate_proj`/`up_proj` 1024→4096 · `down_proj` 4096→1024

`docs/01_measurement_plan.md` listed one `kv_proj (k and v) 1024 → 512`. The expert is
**multi-query**, so k and v are two *separate* 1024 → 256 GEMMs: the same weight bytes, two
kernel launches instead of one. The doc has been corrected in place, with the confirmation from
the checkpoint's module shapes (T002) and the Inductor autotune log's `36 × mm(51x1024,
1024x256)` (T003).

### Two measurements that were wrong, and how

**(a) One call per event pair measures the harness.** The obvious method — one `F.linear`
between one CUDA-event pair — gave **0.10 ms for every projection regardless of shape**, and
68 µs for an empty kernel. At these sizes the Python interpreter cannot feed the GPU. The CPU
was not at fault: all 12 cores were pinned at 2.2 GHz.

**(b) A single timed call carries a fixed ~45 µs that is not the call.** Fitting the project's
own chain measurements, `t(k) = fixed + k·marginal` gives marginal = **60.5 µs** and fixed =
**44.9 µs** (the fit predicts the 8-call chain to 0.3 %). An earlier version of this report
multiplied that 45 µs constant by 18 and 36 to build a per-step budget — the same class of
error as T002's "+3.13 ms finding that was 74 % instrumentation". Every operator is therefore
now measured three ways, and the budget built from microbenchmarks has been **replaced** by the
measured kernel counts above.

* **`eager_single_ms`** — one Python call between one event pair. Reported, never multiplied.
* **`eager_chain_ms`** — the marginal cost of one call inside a 64-call chain. *This* is what an
  operation costs when the model issues it back to back.
* **`graph_cold_ms`** — the projection across all 18 layers inside one CUDA graph, so the weights
  stream from DRAM as they do in the real model. The honest kernel time; floors use it.
* **`graph_hot_ms`** — layer 0's weights replayed 18 times, so they sit in L2.

### Every projection is CPU-bound, and the big ones are at the speed of memory

At S = 51, against the **measured** 185.59 GB/s peak:

| projection | weight | CPU/call | GPU/call | CPU-bound? | × floor | % of peak BW |
|---|---|---|---|---|---|---|
| `gate_proj` | 8.39 MB | 63.7 µs | 53.4 µs | yes | **1.11×** | 89.9 % |
| `up_proj` | 8.39 MB | 64.1 µs | 53.4 µs | yes | 1.11× | 90.0 % |
| `down_proj` | 8.39 MB | 62.5 µs | 55.1 µs | yes | 1.15× | 87.1 % |
| `o_proj` | 4.19 MB | 62.0 µs | 28.8 µs | yes | 1.18× | 84.5 % |
| `q_proj` | 4.19 MB | 62.1 µs | 30.8 µs | yes | 1.27× | 78.8 % |
| `k_proj` | 0.52 MB | 62.0 µs | 8.0 µs | yes | 2.28× | 43.8 % |
| `v_proj` | 0.52 MB | 62.1 µs | 8.1 µs | yes | 2.28× | 43.8 % |

Two things stand out. **CPU dispatch is ~62 µs per projection whatever the shape** — a 0.52 MB
k_proj and an 8.39 MB gate_proj cost the CPU the same — while GPU time varies 7× with the
weight size. Every projection is therefore CPU-bound at S = 51. And the three MLP projections,
which carry 73 % of the expert's weight bytes, run at **87–90 % of measured peak bandwidth**;
only the two tiny multi-query projections do badly (44 %), and they are 3 % of the bytes, where
a 0.52 MB weight at S = 51 is latency-bound rather than bandwidth-bound.

Note the floor definitions differ between the two tables that follow, and are stated in the CSVs:
`kernels.csv` floors count **weights + input and output activations** (`total_bytes`), while the
126-projection stack floor counts **weights only**. At S = 51 activations are ~1 % of the bytes.

### Floor multiple against sequence length

The multiple of the memory floor each projection achieves, S = 1 … 256 (`kernels.csv`):

| S | q_proj | k_proj | o_proj | gate_proj | down_proj |
|---|---|---|---|---|---|
| 1 | 1.20× | 2.14× | 1.23× | 1.17× | 1.17× |
| 2 | 1.22× | 2.77× | 1.43× | 1.18× | 1.27× |
| 4 | 1.22× | 2.75× | 1.43× | 1.18× | 1.27× |
| 8 | 1.22× | 2.71× | 1.42× | 1.18× | 1.27× |
| 16 | 1.20× | 2.66× | 1.43× | 1.17× | 1.27× |
| 32 | 1.19× | 2.44× | 1.20× | 1.12× | 1.15× |
| **51** (the model) | **1.27×** | **2.28×** | **1.18×** | **1.11×** | **1.15×** |
| 64 | 1.26× | 2.17× | 1.18× | 1.11× | 1.14× |
| 128 | 1.19× | 1.77× | 1.28× | 1.99× | 1.29× |
| 256 | 1.92× | 1.71× | 1.56× | 3.17× | 1.53× |

The multiple is almost flat from S = 1 to S = 64: these kernels stream a fixed weight matrix and
the activations are negligible beside it, so lengthening the sequence costs nothing until the
GEMM itself becomes the bottleneck. The rise at S = 128–256 for `gate_proj` (1.99×, 3.17×) is
cuBLAS selecting a worse kernel once the output grows, the same effect E1 sees at n = 8192.

### At which S does a layer stop being launch-bound?

`docs/00_context.md` carries "our *S ≈ 16 crossover*" as an analytic guess. It is too low. The
marginal wall time per call is `max(CPU dispatch, GPU time)`; the ratio of that to GPU time says
which side dominates (`cpu_dispatch_us ÷ gpu_time_us`, both committed in `kernels.csv`):

| S | 1 | 16 | 51 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| `k_proj` | 10.2× | 8.9× | **7.7×** | 7.7× | 7.8× | 5.7× |
| `q_proj` | 2.3× | 2.3× | **2.0×** | 2.0× | 2.0× | 1.0× |
| `o_proj` | 2.2× | 2.1× | **2.2×** | 2.1× | 1.9× | 1.3× |
| `gate_proj` | 1.2× | 1.2× | **1.2×** | 1.2× | 1.0× | 1.0× |
| `down_proj` | 1.2× | 1.2× | **1.1×** | 1.1× | 1.0× | 1.0× |

**At S = 16 nothing has crossed over.** The MLP projections become GPU-bound at S ≈ 128, `q_proj`
and `o_proj` at S ≈ 256, and `k_proj`/`v_proj` never do within S ≤ 256 — they spend 6–10× longer
being launched than being executed. At the model's own S = 51 every projection is still
launch-bound, though the MLP ones only mildly (1.1–1.2×).

### The whole projection stack

All 126 projections of a denoising step, floor counting **weights only** (622.85 MB):

| | ms | µs per launch | × floor |
|---|---|---|---|
| analytic floor (622.85 MB ÷ 185.59 GB/s) | 3.356 | — | 1.00× |
| **replayed as one CUDA graph (GPU time)** | **4.411** | 35.0 | **1.31×** |
| issued back to back from Python (eager) | 8.674 | 68.8 | 2.58× |

**This is the central result of T005.** Those 622.85 MB are 97.6 % of the 638 MB that T002's
`ae_step` floor was built on. With the interpreter removed they stream at **1.31× the floor**.
Meanwhile a whole eager denoising step is 52.56 ms against a 3.438 ms floor at the measured
peak — **15.3×**. The factor of 15 is not weight streaming.

### The non-GEMM operators

At S = 51, T = P + S = 867 keys (`ops.csv`):

| op | CPU/call | GPU/call | CPU ÷ GPU | what it is |
|---|---|---|---|---|
| `rmsnorm` | **282.0 µs** | 17.6 µs | **16.0×** | Python composite: mean, rsqrt, casts, weighted multiply |
| `rope` | **198.6 µs** | 21.2 µs | **9.4×** | Python composite: slice, negate, concat, two multiplies, add |
| `sdpa_attention` | 78.5 µs | 41.0 µs | 1.9× | one fused SDPA call |
| `softmax` | 67.8 µs | 22.0 µs | 3.1× | |
| `gelu` | 30.5 µs | 4.8 µs | 6.4× | single kernel |
| `elementwise_add` | 29.2 µs | 2.1 µs | 13.9× | single kernel |

Per-kernel dispatch is ~30 µs for a single-kernel operator and ~62 µs for an `F.linear`; the
282 µs of `rmsnorm` and 199 µs of `rope` are those costs multiplied by the number of kernels
each Python composite expands into. `PiGemmaRMSNorm` is not a fused kernel, and in the real
model it does *more* than the version benchmarked here — it also has a conditioning path
(`dense(cond)`, `chunk`, two further elementwise ops) that this benchmark omits, so 282 µs is a
**lower bound** on what the model pays.

> **These operators run on activations, so `bytes` in `ops.csv` is L2 traffic, not off-chip
> traffic.** The whole working set at the model's sizes is a couple of MB and never leaves the
> 4 MB L2 — `l2_resident` is `True` for every row but `softmax` at S = 256. Dividing that by a
> DRAM peak is meaningless, and the CSV therefore reports `l2_GBps` and no DRAM roofline column.
> The proof that this is the right call is in the numbers: `elementwise_add` reaches
> **447 GB/s** at S = 256 and `gelu` **226 GB/s**, 2.4× and 1.2× the DRAM peak. An earlier
> version of this report quoted `x_floor` values of 1.1–1.3× for these operators as though they
> were near a DRAM roofline. They are not; they never touch DRAM.

### Launch cost

| method | µs per operation |
|---|---|
| empty kernel (`add_` on one element), Python | 69.0 |
| smallest real GEMM (`q_proj` at S = 1), Python, single timed call | 106.5 |
| `q_proj` at S = 51, single timed call | 105.5 |
| 8 back to back | 65.9 |
| **64 back to back (the marginal cost)** | **60.7** |
| the same 64 replayed as one CUDA graph (GPU time only) | 21.5 |
| the 126-projection stack, Python | 68.8 |
| the 126-projection stack, CUDA graph | 35.0 |

The gap between a single timed call (105.5 µs) and the marginal cost in a chain (60.7 µs) is the
fixed ~45 µs of the timed region itself, not a property of the call.

**CPU dispatch for one `F.linear` is ~61 µs**, and it is shape-independent: a 0.52 MB `k_proj`
and an 8.39 MB `gate_proj` both cost the CPU ~62 µs. It is *not* 61 − 21 = 40 µs: CUDA launches
are asynchronous, so a chain runs at `max(CPU, GPU)` per call, and subtracting the GPU time
would assume the two serialise — which is exactly the opposite of what the measurement shows.
The GPU work hides underneath the dispatch.

### What the L2 is worth

`graph_hot_ms` replays one layer's weights 18 times instead of 18 different layers':

| projection | weight | 18× working set | hot speed-up at S = 1 | at S = 51 |
|---|---|---|---|---|
| `q_proj` | 4.19 MB | 75 MB | **1.51×** | 1.33× |
| `k_proj` | 0.52 MB | 9.4 MB | 1.44× | 1.02× |
| `gate_proj` | 8.39 MB | 151 MB | 1.00× | 1.01× |

Weights around the size of the 4 MB L2 gain up to 1.5× from being resident; the 8.39 MB MLP
weights gain nothing, because they cannot fit however they are scheduled. This is the closest
thing in T005 to a direct measurement of what keeping expert weights near the compute units
would buy — and it caps out at ~1.5× for the projections near the L2 size and 1× for the large ones, *given
that those already run at 87–90 % of peak bandwidth*.

Two caveats on the "cold" side of that comparison. For `k_proj`/`v_proj` the 18-layer working
set is **9.4 MB, only 2.25× the L2**, so "cold" is not fully cold — consistent with their hot
speed-up being just 1.02× at S = 51. And `timed()` replays the graph 100 times back to back with
no cache flush, so some of the previous replay can survive into the next. For the ≥ 4.19 MB
projections (75–151 MB working sets) neither caveat bites.

---

## Validating the DRAM-traffic substitute (protocol §3.3, research-side question 1)

This Tegra exposes **no `dram__*` counters at all** — T001 found 0 of 4426 metrics, and an
explicit request returns `n/a`. Every off-chip byte this campaign reports must therefore come
from the L2's sysmem counters. The question was whether that substitute is trustworthy.

Four kernels whose traffic is known by construction, one profiled launch each
(`bench/ncu_validation.py`, driven by `bench/run_ncu_validation.sh`):

| case | analytic read | read from misses | **ratio** | analytic write | write from `op_write` | **ratio** |
|---|---|---|---|---|---|---|
| `copy_256mib` | 268.44 MB | 268.44 MB | **1.0000** | 268.44 MB | 268.44 MB | **1.0000** |
| `gemm_cold` | 33.97 MB | 34.28 MB | **1.0089** | 0.42 MB | 0.42 MB | **1.0010** |
| `gemm_warm` | 33.97 MB | 34.26 MB | **1.0085** | 0.42 MB | 0.42 MB | **1.0010** |
| `gemm_l2` | 1.15 MB | 1.15 MB | **1.0002** | 0.05 MB | 0.05 MB | **1.0080** |
| `gemm_l2_twice` | 1.15 MB | 1.15 MB | **1.0002** | 0.05 MB | 0.05 MB | **1.0080** |

**Reads: the substitute is exact.** `lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum ×
32 B` recovers analytic read traffic to within 0.9 % across two orders of magnitude.

**Writes: the counter the protocol names is identically zero, and a different one works.**
`lts__t_sectors_aperture_sysmem_op_write_lookup_miss.sum` reads **0 on every case**, including
the 256 MiB copy that demonstrably writes 256 MiB. The reason is visible in the hit counter:
write sectors are recorded as lookup *hits* (the L2 is write-back / write-allocate), so a write
can never appear in a miss counter. The total write-sector counter
`lts__t_sectors_aperture_sysmem_op_write.sum × 32 B` recovers the traffic exactly.

> **Correction to the substitute definition.** Off-chip traffic on this device is
> `read = op_read_lookup_miss.sum × 32 B` **+** `write = op_write.sum × 32 B`.
> Using `*_lookup_miss` for both, as `docs/02_protocol.md` §3.3 has it, would report every
> kernel's write traffic as zero and understate DRAM traffic by the whole write stream — for a
> copy, by half.

Two deviations recorded rather than hidden:

* The task asked for `clone()` of a 256 MiB tensor. A contiguous device-to-device `clone()`
  dispatches to `cudaMemcpyAsync`, a copy-engine operation and not a kernel: ncu reports
  *"No kernels were profiled."* An elementwise `z = 2·x` with identical traffic was used
  instead.
* The task expected case (d), a working set inside the 4 MiB L2, to show ≈ 0 misses. It shows
  full misses, and a control that ran the same kernel twice inside one profiled window
  (`gemm_l2_twice`, `-c 2`) gave **byte-identical counters for both launches** (36,039 miss /
  25,190 hit each). That is ncu's kernel replay restoring device state between passes, not a
  statement about the L2. The independent answer comes from E2 with no profiler attached:
  replaying a 0.52 MB-weight projection from L2 is **1.44×** faster than streaming 18 distinct
  ones, and 4.19 MB weights give **1.51×** at S = 1. The L2 does retain; ncu cannot show it
  this way.

The per-run SM-clock check of protocol §3.2 is committed for each of the five profiled runs in
`raw/ncu_clock_check.csv` (GPU 1300.5 MHz and EMC 3199 MHz, pinned, for all five).

---

## Do the four power rails double-count? (protocol §4)

Four INA3221 rails sampled at 10 Hz for 60 s per load, after a 3 s settle
(`bench/power_rails_check.py`):

| load | VDD_GPU_SOC | VDD_CPU_CV | VIN_SYS_5V0 | VDDQ_VDD2_1V8AO | sum |
|---|---|---|---|---|---|
| idle (absolute) | 4783 mW | 1210 mW | 4156 mW | 405 mW | 10555 mW |
| `clone_256mib` | +18681 | +1522 | +7265 | +4929 | +32396 mW |
| `copy_kernel_256mib` | +22174 | +1522 | +7849 | +5231 | +36776 mW |
| `gemm_8192_bf16` | +50720 | +1132 | +8802 | +5531 | +66186 mW |

**No double counting; the four may be summed.** The argument is behavioural: the bf16 GEMM
raises `VDD_GPU_SOC` by 50.7 W while raising `VIN_SYS_5V0` by only 8.8 W. If `VIN_SYS_5V0`
contained the GPU rail it would have to rise by at least as much as the rail it contains. The
same load separates the other pairs: `VDD_CPU_CV` *falls* going from the copy loop (+1.52 W) to
the GEMM loop (+1.13 W) even as GPU power more than doubles, so it is not a superset of the GPU
either, and `VDDQ_VDD2_1V8AO` responds to DRAM traffic (+4.93 W on the pure-copy load, which
does almost no arithmetic) rather than to compute.

Two things worth carrying into E7. The memory rail is **not** where memory traffic shows up
most: the copy loop costs +4.9 W on `VDDQ_VDD2_1V8AO` but +18.7 W on `VDD_GPU_SOC`, because the
GPU rail also feeds the memory controller and the fabric. And a bandwidth-saturating load costs
about half what a tensor-core load costs above idle (32.4 W vs 66.2 W) — the energy side of the
same story E2 tells about time.

This answers the double-counting question only. The absolute scaling of `VDDQ_VDD2_1V8AO`
remains open (its INA3221 channel reports a ≈ 5 V bus voltage, not 1.8 V); that is question 2
in `STATUS.md`.

---

## What this changes for the PIM study

1. **The action expert's weight streaming is already near-optimal.** The projections hold
   622.85 MB — 97.6 % of the 638 MB that T002's `ae_step` floor is built on — and with dispatch
   removed they run at 1.31× that floor. A memory system that removed *all* off-chip weight
   traffic for them would save at most the ~4.4 ms those GEMMs take, out of a 52.56 ms step.
2. **The gap is dispatch and pointwise work.** 969 kernels per step, 83 % of them not GEMMs,
   each paying a CPU dispatch cost comparable to or larger than its GPU time. This is a
   software-stack cost, and T003 already showed `torch.compile` + CUDA graphs recovers most of
   it (52.56 → 16.30 ms/step).
3. **E11's analytic host model should use 0.799 and 0.906**, not 0.40 and 0.70. The compute
   fraction is wrong by 2×, and it propagates into every segment estimate.
4. **Off-chip traffic must be counted with two *different* counters.** Reads from
   `op_read_lookup_miss.sum`, writes from `op_write.sum`. Using `*_lookup_miss` for both, as
   protocol §3.3 has it, reports zero write traffic for every kernel.
5. **Power rails may be summed**, and a bandwidth-saturating load costs about half what a
   tensor-core load costs above idle.
6. **Every floor in this campaign now has a measured row.** `bench/fill_measured_floors.py`
   filled the 90 `peak_source = measured` rows that T002 and T003 left blank across every
   `floor.csv`. Two of them change the story: with measured peaks `ae_step`'s floor is
   **3.438 ms** (not 3.115 at spec), so the eager step is **15.3× floor** rather than 16.9×;
   and `vlm_prefix` is **compute**-bound, not memory-bound (122.0 ms compute floor against a
   27.0 ms memory floor).

## Measured, derived, not measured

**Measured:** every bandwidth and FLOP/s figure; every per-projection and per-operator time;
the chained marginal cost of a call; CUDA-graph replay times; kernel launch counts per step;
ncu sector counters; rail power; clock and temperature under sustained load.
**Derived:** floors (bytes ÷ measured peak); `× floor` and `% of peak`; the ridge point;
per-step kernel counts (slope over N); dispatch cost (2-point fit of `t(k) = fixed + k·marginal`).
**Not measured:** how the 969 kernels per step divide into *time* — T004/E5 does that with nsys,
and nothing here should be read as a time budget; the hardware's true dense INT8 peak (PyTorch
cannot reach it); the cause of the run-to-run bandwidth variation; whether the L2 retains across
launches *under ncu* (its replay restores state — answered instead by E2's `graph_hot` numbers,
taken with no profiler attached).

## Reproduce

```bash
sudo jetson_clocks
python bench/micro_peak.py         --outdir results/<date>_E1_peak
python bench/sustained_load_probe.py --out   results/<date>_E1_peak/raw/sustained_load.csv
for i in $(seq 1 22); do python bench/bandwidth_mode_probe.py; done \
    > results/<date>_E1_peak/raw/bandwidth_modes.jsonl
python bench/micro_ae_kernels.py   --outdir results/<date>_E2_ae_kernels
bash   bench/run_ncu_validation.sh         results/<date>_E2_ae_kernels
python bench/parse_ncu_validation.py       results/<date>_E2_ae_kernels
# CUDA activity in torch.profiler needs root on this device, as ncu does
for N in 2 4 10; do sudo -n sh -c "HOME=$HOME PATH=$PATH $(which python) \
    bench/count_step_kernels.py --outdir results/<date>_E2_kernel_counts_N$N --steps $N"; done
python bench/analyze_step_kernels.py --lo <...>_N2 --hi <...>_N10 --out <...>_kernel_counts
python bench/power_rails_check.py   --outdir results/<date>_E7_rails_check --seconds 60
python bench/fill_measured_floors.py
python bench/refresh_raw_files.py results/<date>_E1_peak results/<date>_E2_ae_kernels ...
```

T007's ncu target, as T005 requires — warms up, then runs **one** kernel between
`cudaProfilerStart/Stop` inside NVTX range `micro`:

```bash
sudo ncu --profile-from-start off --nvtx --nvtx-include "micro/" -c 1 --set full \
     python bench/micro_ae_kernels.py --outdir /tmp/x --ncu-window q_proj@51
```

Verified end to end: it profiles exactly one kernel,
`ampere_bf16_s16816gemm_bf16_64x128_sliced1x2_ldg8_` (a tensor-core bf16 GEMM), 37.3 µs, whose
134,919 read-miss sectors × 32 B = 4.32 MB matches q_proj's 4.19 MB of weights plus 0.10 MB of
activations to within 0.4 %. `--layer <proj>` and `--S <list>` restrict the sweep.

The ncu reports are committed compressed (`zstd -19`, 32–114×) and the chrome traces likewise
(~23×); each original's sha256 and size is in the relevant `ORIGINALS.json` and in the
`raw_files` entry itself.
