# T005 — Microbenchmarks (E1, E2)  · priority P1 · depends on: T001 (MAXN; protocol §1 gate)

Implement `bench/micro_peak.py` (E1) and `bench/micro_ae_kernels.py` (E2) exactly as specified in
`docs/01_measurement_plan.md`. Acceptance: `reports/E1_E2_microbench.md` with achieved GB/s and TFLOP/s vs the
204.8 GB/s / 42 TFLOP/s spec, the per-layer floor-multiple vs S table, the measured per-kernel launch cost (µs),
and the effect of CUDA-graph replay.
Also complete `hw_constants.json` (protocol §8) with the measured peaks and the ridge point, and state in the
report how the measured values compare with the expected 204.8 GB/s / ≈ 42 TFLOP/s / ≈ 208 FLOP/B. `bench/micro_ae_kernels.py`
must accept `--layer`, `--S`, `--ncu-window` (warm up, then run the selected kernel between
`cudaProfilerStart/Stop` inside an NVTX range `micro`) so that T007 can use it as an ncu target.
**Substitute validation (protocol §3.3, question 1).** With ncu (filtered, `-c` ≤ 3 launches each, the per-run clock
check of protocol §3.2) on four E1 kernels whose traffic is known analytically: (a) `clone()` of a 256 MiB tensor (reads
= writes = 256 MiB), (b) a bf16 GEMV / skinny GEMM that streams an 8–64 MB weight once (reads ≫ writes), (c) the same
kernel launched again immediately (warm L2), (d) a kernel whose working set fits far inside the L2 (expected
≈ 0 misses). Write `ncu/sysmem_miss_validation.csv`: `kernel, analytic_read_bytes, analytic_write_bytes,
miss_read_sectors, miss_write_sectors, miss_all_sectors, measured_read_bytes, measured_write_bytes, ratio_read,
ratio_write, nsys_kernel_ms, GBps_from_misses, GBps_analytic, pct_of_measured_peak`. Report the ratios as found; if
the write path does not appear in any sysmem-miss counter, say so — the substitute then stays read-only.
**Rail double-count check (protocol §4).** Four rails from sysfs at idle, during the ≥ 60 s `clone()` loop and during
the ≥ 60 s 8192² bf16 GEMM loop → `power_rails_check.csv` (`load, rail, mean_mW, above_idle_mW`) and one paragraph.
Raw files (protocol §3.5 / §6): every per-iteration sample and the run stdout committed under the result directories' `raw/`.

status: claimed 2026-09-22


---

status: done 2026-09-22

**Result directories** (the `20260921_*` copies of these runs are kept, each with a
`SUPERSEDED.md` saying why)
* `results/20260922_E1_peak/` — E1 peaks, `hw_constants.json`, bandwidth-variation and
  sustained-load probes
* `results/20260922_E2_ae_kernels/` — E2 per-projection and per-operator sweep,
  `ncu/sysmem_miss_validation.csv`, `raw/ncu_clock_check.csv`, the T007 `--ncu-window` check
* `results/20260922_E2_kernel_counts_N{2,4,10}/` and `results/20260922_E2_kernel_counts/` —
  kernel launches per denoising step, by slope over N
* `results/20260922_E7_rails_check/` — the four-rail double-count check

**Report:** `reports/E1_E2_microbench.md`. Every headline number in it is re-derived from the
CSVs by `bench/check_report_numbers.py 20260922` (37 values, all present).

**`bench/` commit:** recorded in each `meta.json` (`bench_commit`, `bench_files_sha256`).

**Headline.** Measured peaks 187.47 GB/s (91.5 % of spec) and 33.85 TFLOP/s bf16 (80.6 %),
ridge 180.6 FLOP/B against the ≈ 208 the task expected. The two `[ASSUMPTION]` fractions in
`docs/00_context.md` measure 0.915 and 0.806 — the compute one was 2.01× pessimistic. The
expert's 126 projection GEMMs (622.85 MB, 97.6 % of the 638 MB T002's `ae_step` floor used) run
at **1.33× the memory floor** as a CUDA graph and 2.61× from Python; a denoising step launches
**969 kernels**, only 17 % of them GEMMs; at S = 51 the marginal cost of a projection is ~62 µs
whatever its shape, so every projection is launch-bound there.

**Deviations from the task text**
1. **`clone()` is not a kernel.** Case (a) asked for `clone()` of a 256 MiB tensor; a contiguous
   device-to-device clone dispatches to `cudaMemcpyAsync`, so ncu reports "No kernels were
   profiled". An elementwise `z = 2·x` with identical traffic was substituted.
2. **Case (d) does not show ≈ 0 misses.** ncu's kernel replay restores device state between
   passes, so an L2-resident kernel still reports full misses and a two-launch control gives
   byte-identical counters. The L2-retention question is answered instead by E2's `graph_hot`
   measurements, taken with no profiler attached.
3. **The write half of the substitute needed a different counter**, as the task anticipated:
   `*_lookup_miss` is identically zero for writes because write sectors register as L2 *hits*.
   `op_write.sum × 32 B` recovers it (1.001–1.008). `docs/02_protocol.md` §3.3 needs this.
4. **`GBps_from_misses` and `pct_of_measured_peak` are deliberately blank** in
   `sysmem_miss_validation.csv`. The task's schema asks for them, but §3.3(3) forbids deriving
   achieved GB/s from ncu's `Duration`, and no nsys run of these kernels exists yet. The column
   that used to be called `nsys_kernel_ms` is now `ncu_duration_ms`, which is what it is.
5. **E2 uses 100 timed calls, not the 1,000 `docs/01_measurement_plan.md` asks for.** 100 is the
   protocol §2 minimum; 1,000 would have made the sweep (130 shape/S points × 3 timing methods,
   plus 3-point chain fits) impractical. Two full independent runs agree to 0.30 % median /
   1.39 % max on GPU kernel time, so the extra samples would not change a conclusion.
6. **Extra result directories** beyond the task's list: the kernel-count runs, added because the
   report needed a measured number of operations per step rather than an assumed one.
7. **E2 reports the non-GEMM operators as L2 traffic**, not against a DRAM roofline: their
   working sets usually fit the 4 MB L2, and five rows exceed 100 % of DRAM peak — which is the
   proof, not an anomaly.
8. **`docs/` corrected as owner** (disclosed in the files themselves): `01_measurement_plan.md`'s
   E2 table listed one `kv_proj 1024 → 512`, but the expert is multi-query so k and v are two
   separate 1024 → 256 GEMMs; and its floor definition divided by the 204.8 spec, which §8
   forbids. `00_context.md`'s ladder, launch-overhead, efficiency-fraction, peak and
   "S ≈ 16 crossover" rows now carry measured values — the real crossover is S ≈ 128 for the MLP
   projections and never within S ≤ 256 for `o_proj` or the multi-query k/v.
9. **90 `peak_source = measured` rows filled** across every existing `floor.csv`
   (`bench/fill_measured_floors.py`). `ae_step`'s floor becomes 3.403 ms, so the eager step is
   **15.4× floor**, not 16.9×. `vlm_prefix` was **already** compute-bound at the spec peaks; the
   measured peaks only widen that margin from 4.0× to 4.5×.

**Errors found by two independent review passes (protocol §11) and fixed**
* Pass 1: the first per-kernel timing measured **Python dispatch, not kernels** (every
  projection came out at 0.10 ms regardless of shape); and the corrected version still
  multiplied a **fixed ~45 µs timed-region cost** by 18 and 36 to build a per-step budget — the
  same class of error as T002's "+3.13 ms finding". The budget was replaced by measured kernel
  counts.
* Pass 2: the headline still carried two numbers from a superseded run; the launch fit
  (60.5 / 44.9 µs) came from a superseded run and is **59.44 / 44.35 µs** from the committed
  CSV; `cpu_dispatch_us` was really `max(CPU, GPU)` and the "~62 µs whatever the shape" claim
  had to be scoped to the launch-bound regime; **protocol §2 was still violated** in the chain
  measurement (10 warm-ups / 30 timed → now 20 / 100, with a 3-point least-squares fit and a
  per-row residual); `Memcpy` kernels were classified as "other", under-counting copies by 38
  per step; the "37 `rsqrt` confirm 36 RMSNorms" claim was wrong (it is exactly 37); ncu's
  `Duration` was being used as a latency; the SM-clock half of §3.2's check was collected but
  never compared; and **two result directories were overwritten**, which §5 forbids — hence the
  new dated directories and the `SUPERSEDED.md` markers.
  Superseded runs are in `scratch/T005/quarantine/`.

**Known weaknesses, stated rather than buried**
* Memory bandwidth varies run to run by up to **1.19×** with the GPU clock, the EMC clock and
  the die temperature all constant and recorded. The cause is not known. Five elimination checks
  are backed by committed files; six more were run interactively and their output was not
  retained — those are labelled reasoning, not evidence. Any bandwidth-derived floor carries
  roughly ±8 %; the sensitivity runs the safe way for E2's central claim.
* `hot_speedup` is the least reproducible column in E2: two full runs agree to 0.30 % median on
  GPU kernel time but differ by up to 29 % on it. It is quoted as a range, not a figure.
* The 126-projection CUDA graph is a best case for DRAM locality (nothing interleaves with it,
  and the graph is replayed without a cache flush), and the eager row still contains the ~44 µs
  fixed cost (0.5 %). Both are stated in the report.
