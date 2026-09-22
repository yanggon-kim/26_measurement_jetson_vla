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

**Result directories**
* `results/20260921_E1_peak/` — E1 peaks, `hw_constants.json`, bandwidth-mode and sustained-load probes
* `results/20260921_E2_ae_kernels/` — E2 per-projection and per-operator sweep, `ncu/sysmem_miss_validation.csv`, `raw/ncu_clock_check.csv`
* `results/20260921_E2_kernel_counts{,_N2,_N10}/` — kernel launches per denoising step, by slope over N
* `results/20260921_E7_rails_check/` — the four-rail double-count check

**Report:** `reports/E1_E2_microbench.md`

**`bench/` commit:** recorded in each `meta.json` (`bench_commit`, plus `bench_files_sha256`).

**Headline.** Measured peaks 185.59 GB/s (90.6 % of spec) and 33.55 TFLOP/s bf16 (79.9 %),
ridge 180.8 FLOP/B against the ≈ 208 the task expected. The two `[ASSUMPTION]` fractions in
`docs/00_context.md` are measured at 0.906 and 0.799 — the compute one was 2.00× pessimistic.
The expert's 126 projection GEMMs (622.85 MB, 97.6 % of the 638 MB T002's `ae_step` floor used)
run at **1.31× the memory floor** as a CUDA graph and 2.58× from Python; a denoising step
launches **969 kernels**, only 17 % of them GEMMs, and CPU dispatch is ~62 µs per `F.linear`
regardless of shape.

**Deviations from the task text**
1. **`clone()` is not a kernel.** Case (a) of the substitute validation asked for `clone()` of a
   256 MiB tensor; a contiguous device-to-device clone dispatches to `cudaMemcpyAsync`, so ncu
   reports "No kernels were profiled". An elementwise `z = 2·x` with identical traffic
   (read 256 MiB, write 256 MiB) was substituted.
2. **Case (d) does not show ≈ 0 misses.** ncu's kernel replay restores device state between
   passes, so an L2-resident kernel still reports full misses, and a two-launch control gives
   byte-identical counters. The L2-retention question is answered instead by E2's `graph_hot`
   measurements with no profiler attached.
3. **The write half of the substitute needed a different counter.** As the task anticipated,
   the write path does not appear in any `*_lookup_miss` counter — it is identically zero
   because write sectors register as L2 *hits*. `op_write.sum × 32 B` recovers it exactly
   (ratios 1.001–1.008). `docs/02_protocol.md` §3.3's definition needs this correction.
4. **Two extra result directories** beyond the task's list: the kernel-count runs, added because
   the report needed a measured number of operations per step rather than an assumed one.
5. **E2 measures the non-GEMM operators the plan lists** (SDPA, RMSNorm, RoPE, GeLU, softmax,
   elementwise add), but reports them as **L2 traffic**, not against a DRAM roofline: their
   working sets fit the 4 MB L2 and several exceed 100 % of DRAM peak, which is the proof.
6. **`docs/` corrected as owner** (both disclosed in the files themselves): the E2 table in
   `docs/01_measurement_plan.md` listed one `kv_proj 1024 → 512`, but the expert is multi-query
   so k and v are two separate 1024 → 256 GEMMs; and its floor definition said to divide by the
   204.8 spec, which protocol §8 forbids. `docs/00_context.md`'s ladder, launch-overhead,
   efficiency-fraction, peak and "S ≈ 16 crossover" rows are updated with measured values.
7. **90 `peak_source = measured` rows filled** across every existing `floor.csv`
   (`bench/fill_measured_floors.py`) — the deliverable T004/E6 depends on. Two consequences:
   `ae_step`'s floor becomes 3.438 ms (eager step 15.3× floor, not 16.9×), and `vlm_prefix` is
   **compute**-bound at the measured peaks, not memory-bound.

**Two errors found by the independent review and fixed before this block was written**
* The first per-kernel timing measured **Python dispatch, not kernels** — every projection came
  out at 0.10 ms regardless of shape. Redone with CUDA graphs.
* The corrected version still multiplied a **fixed ~45 µs timed-region cost** by 18 and 36 to
  build a per-step budget, the same class of error as T002's "+3.13 ms finding". The budget was
  replaced by measured kernel counts, and every operator is now also timed as a chain so the
  marginal cost is separated from the fixed one. Superseded runs are in `scratch/T005/quarantine/`.

**Known weakness, stated rather than buried.** Memory bandwidth on this device varies run to run
by up to 1.14× with the GPU clock, the EMC clock and the die temperature all constant and
recorded. The cause is not known. Five elimination tests are backed by committed files; six more
were run interactively and their output was not retained — those are reported as reasoning, not
evidence. Any bandwidth-derived floor carries roughly ±7 %.
