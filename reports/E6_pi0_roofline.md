# E6 — π0 on the Orin's own roofline

**Task T004 step 3. Binding: plan E6, protocol §8 (the roof is this device's), §9 (floor and
gap ratio), §3.3 (off-chip bytes), §2 (region times). Result:
`results/20260925_E6_pi0/roofline.csv`; the CSV is the deliverable and this is the short
reading of it.**

> **This experiment was missing.** T004 is titled "E5, E6, E7" and was marked `done` on
> 2026-09-25 with a status block covering E5 and E7 and silent on E6. No `results/*_E6_*`
> existed for any model. `bench/roofline.py` had been written for it and never run. T004 is
> re-opened; the correction is in its status block and in `STATUS.md`.

## The answer

π0, eager, N = 10, MAXN, `results/20260921_E3_pi0_eager_N10`. Roof = **this Orin's measured
peaks** (E1/T005: 187.47 GB/s, 33.85 TFLOP/s bf16, ridge **180.57 FLOP/B**), with spec peaks
(204.8 GB/s, 42 TFLOP/s) alongside in the CSV.

| region | measured ms | intensity | side of the ridge | % peak BW | % peak compute | floor ms | **gap × floor** |
|---|---:|---:|---|---:|---:|---:|---:|
| `vision` | 239.60 | 398.1 FLOP/B | compute | 3.69 % | 8.14 % | 8.85 | **27.07×** |
| `vlm_prefix` | 232.93 | 816.0 FLOP/B | compute | 11.49 % | 51.92 % | 120.94 | **1.93×** |
| **`ae_step`** | **52.79** | **60.0 FLOP/B** | **memory** | **6.45 %** | **2.14 %** | **3.40** | **15.51×** |

Protocol §9's reading of `gap_ratio`: < 1.5× near-optimal, 1.5–3× moderate, 3–10× substantial
software overhead, **> 10× a structural problem (launch-bound, idle GPU, too little
parallelism)**.

## What this says about the study's premise, and it cuts both ways

**The premise holds analytically.** The action expert's operational intensity is **60.0 FLOP/B,
below this device's measured ridge of 180.57**, so on the roofline it is a memory-side region.
A memory-side accelerator is aimed at the right half of the chart. (Our analytic intensities
are 816.0 for the VLM backbone and 60.0 for the expert, against arXiv:2604.24447's 840 and
64.5 — close enough to confirm both derivations describe the same model.)

**The premise does not hold operationally, and this is the finding that matters.** The expert
reaches **6.45 % of peak bandwidth** and **2.14 % of peak compute**, and runs **15.51× its
physical floor**. It is memory-*bound* in the sense that memory is the nearer of two distant
roofs — but it is not memory-*limited*, because it is nowhere near either roof. Something other
than bandwidth or arithmetic is consuming 94 % of its time.

**Four independent measurements now agree on what that something is**, which is why this is
stated as a conclusion rather than a conjecture:

| evidence | experiment | figure |
|---|---|---|
| GPU idle during the expert | E5 | **84.2 %** idle over 968 kernels/step (eager) |
| no configuration is memory-latency-bound | E5b | long-scoreboard stall share **19.5–47.6 %**, none meets §7's ≥ 60 % |
| the expert does not use the SMs | E12 §5 | 4× fewer SMs costs it **1.3 %** (eager) / 4.9 % (`bf16_both`) |
| the host issues for the whole wall clock | E12 §1 | `issue/wall` = **1.0000** |
| **distance from the physical floor** | **E6, here** | **15.51×** |

π0's action expert on Orin is **launch-bound**. Its kernels are small, numerous and short, and
the GPU spends most of the phase waiting for the next one.

**Consequence for a processing-in-memory case, stated plainly.** A PIM accelerator that raises
effective bandwidth targets a ceiling this workload currently uses 6.45 % of. On these
measurements the latency win from moving the expert off the GPU would come from **eliminating
per-kernel launch overhead**, not from relieving a bandwidth limit — a different argument,
resting on a different mechanism, and one the numbers here support where the bandwidth argument
is not supported. If the paper argues from bandwidth, this table contradicts it; if it argues
from launch and dispatch overhead, this table is evidence for it.

This does not say a PIM design is unattractive. It says the reason must change, and it says the
achievable win is bounded by what launch overhead costs — which E5 puts at 84.2 % of the phase.

## The other two regions

* **`vlm_prefix` is the efficient region** — 1.93× its floor and **51.92 % of peak compute**,
  the only region anywhere near a roof. It is compute-side (816.0 FLOP/B) and is doing roughly
  what the hardware allows. There is little to win here by any means.
* **`vision` is the worst offender** at **27.07× floor** while using 8.14 % of peak compute.
  Its floor is small (8.85 ms) and it takes 239.60 ms. This is not a target of the study, but
  it is the largest single inefficiency in the frame and is recorded because it is where the
  most wall-clock time is recoverable.

## Measured / derived / not measured

**Measured.** Region times (protocol §2 NVTX regions, from `latency_summary.csv` of the E3
run); the peaks (E1/T005); off-chip bytes per kernel for the 300 `source = ncu` rows, from
`lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum × 32 B` plus the write counterpart
(§3.3 [Orin] — there are no DRAM counters on this device, and ncu's `Memory Throughput` is L2
throughput and is never read as off-chip bandwidth).

**Derived.** FLOPs and bytes per region are **analytic**, from the loaded model's own shapes:
`floor.csv` for `vlm_prefix` and `ae_step`, and `bench/vision_flops.py` for `vision`, whose
`bytes_as_loaded_fp32` was cross-checked against `floor.csv`'s `bytes_read` and agrees exactly
at 1 659 214 784 B — the run refuses to proceed if they disagree, since taking FLOPs from one
derivation and bytes from another is only safe when the two agree. Intensity, achieved GB/s and
TFLOP/s, % of peak, and `gap_ratio = measured ÷ floor` follow arithmetically.

**Two choices worth stating because they change numbers.**

1. **`ae_step` time is the median over the 10 measured steps (52.7908 ms), not step 0.** Step 0
   alone is 55.9312 ms, **+5.9 %** — a first-step effect. `floor.csv` names the region `ae_step`
   while the measured summary names them `ae_step[0]`…`[9]`; before this was handled the
   `ae_step` row silently received no time and was **dropped from the roofline entirely**, which
   would have omitted the one region the study is about.
2. **The 300 `ncu` rows carry bytes but no rates.** Protocol §3.3(3) requires achieved GB/s to
   use an nsys duration, never ncu `Duration`, and no per-kernel nsys duration is joined here.
   The rate columns are left **empty rather than computed from the wrong clock**.

**Not measured.**

* **E6 for the other five models.** π0 is the only model with a `floor.csv`; π0.5, SmolVLA and
  the three GR00T variants have none, so their analytic bytes and FLOPs do not exist yet. T008
  and T009 both name `reports/E6_<m>_roofline.md` in their acceptance and both are therefore
  still unmet. `bench/regen_floor.py` reads `region_parameter_bytes` and `tokens_detail` from a
  run's own `meta.json`, so the floors are derivable without re-measuring **if** those fields
  were recorded for those models — not yet checked.
* **Per-kernel roofline points with rates**, which need an nsys/ncu join on kernel identity.
* **A compiled-rung roofline.** These are eager numbers. Compiling cuts the expert's kernels
  from 968 to 372 per step (E5), so the `gap_ratio` should fall sharply — the measurement that
  would quantify how much of the 15.51× is launch overhead is the same table rebuilt from a
  `bf16_both` run, and it is the obvious next step.
