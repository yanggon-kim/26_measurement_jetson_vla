# E5b — GR00T N1.7's action head at instruction level

**Task T009 step 3. Binding: `docs/03_instruction_level_analysis.md` (§1 gate, §5 coverage,
§7 stall mapping, "S sweep (isolated kernels)"). Results:
`results/20260925_E5b_groot_n17_{eager,best}/` and `results/20260925_E5b_groot_n17_ssweep/`.
Full per-kernel stall tables: `reports/E5b_groot_analysis.md` (generated).**

## The answer

**GR00T N1.7's dominant action-head GEMM is memory-latency-bound, the first kernel of any
model in this campaign to meet §7's threshold.**

| mode | kernel | time in window | SM % | long-scoreboard, % of waiting | §7 verdict |
|---|---|---:|---:|---:|---|
| eager | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_64x3_tn` | 3304 µs | 37.7 | **61.6** | **memory-latency bound** |
| eager | `ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_32x5_tn` | 3881 µs | 65.6 | 6.4 | compute bound (math-pipe throttle 14.9 % > 10 %) |
| best | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_64x3_tn` | 2908 µs | 41.5 | **61.0** | **memory-latency bound** |
| best | `Kernel2` (CUTLASS) | 2564 µs | 50.6 | 15.6 | neither; largest wait is `wait` 24.6 % |

For contrast, `reports/E5b_summary.md` found the six π0 / π0.5 / SmolVLA configurations at
**19.5–47.6 %** long scoreboard, time-weighted, with **none** reaching the 60 % bar. GR00T's
128×64 GEMM clears it in both modes. This agrees with the corrected E6: eager-to-eager,
GR00T's action step sits at 33 % of peak bandwidth against π0's 6.5 %, and compiled it
reaches 83–87 % (`reports/E6_groot_roofline.md`, correction block).

## The S sweep: at the model's own S, the time is set by weight bytes, not tokens

> **CORRECTION (2026-09-25, late).** The "median µs", "× floor" and "GB/s" columns of the table
> below came from CUDA events with an event pair **and a synchronize around every single call**,
> which on Orin adds ~45–75 µs per call (found when the π0.5 / SmolVLA sweeps put every small GEMM
> at ~90 µs whatever its shape). They measured the timing harness, not the GEMM. The kernel time is
> in the ncu captures of the same sweep: `results/20260925_E5b_groot_n17_ssweep/s_sweep_ncu.csv`
> (`bench/sweep_ncu_table.py`). Corrected table:
>
> | target | projection (in → out, dtype) | S | kernel µs (ncu) | × floor | GB/s | % measured BW | kernel | events said µs (superseded) |
> |---|---|---:|---:|---:|---:|---:|---|---:|
> | `largest` | `ff.net.0.proj` (1536 → 6144, bfloat16) | 1 | **118.6** | 1.18 | 159.4 | 85.0 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f` | 166.8 |
> | `largest` | `ff.net.0.proj` (1536 → 6144, bfloat16) | 16 | **119.3** | 1.17 | 160.4 | 85.6 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f` | 166.1 |
> | `largest` | `ff.net.0.proj` (1536 → 6144, bfloat16) | 41 | **124.8** | 1.20 | 156.3 | 83.4 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f` | 170.2 |
> | `largest` | `ff.net.0.proj` (1536 → 6144, bfloat16) | 256 | **183.1** | 1.28 | 124.6 | 66.5 | `ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2` | 228.4 |
> | `q` | `attn1.to_q` (1536 → 1536, bfloat16) | 1 | **37.1** | 1.47 | 127.6 | 68.1 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8` | 93.9 |
> | `q` | `attn1.to_q` (1536 → 1536, bfloat16) | 16 | **37.1** | 1.44 | 129.9 | 69.3 | `ampere_bf16_s16816gemm_bf16_64x64_ldg8_relu_f2f_` | 92.8 |
> | `q` | `attn1.to_q` (1536 → 1536, bfloat16) | 41 | **38.1** | 1.44 | 130.6 | 69.7 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8` | 93.1 |
> | `q` | `attn1.to_q` (1536 → 1536, bfloat16) | 256 | **82.0** | 2.30 | 76.8 | 41.0 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f` | 128.9 |
>
> **Readings that change:** the largest projection is at **1.17–1.20× its memory floor** up to
> S = 41 (was quoted 1.6×) — 83–86 % of the measured bus bandwidth, i.e. essentially at the
> weight-streaming limit; the q projection is at **1.44–1.47×** (was 3.5–3.7×), so "too small to
> reach the bus's stride" overstated it. **Readings that stand:** the largest projection costs the
> same from S = 1 to S = 41 (118.6 / 119.3 / 124.8 µs); the kernel choices and stall columns come
> from ncu and are unaffected. The superseded table is kept below as it was written.

DiT block 0's projections were read from the loaded model; the largest is `ff.net.0.proj`
(1536 → 6144) and the q projection `attn1.to_q` (1536 → 1536), both bf16. Each was run alone
as `F.linear` at S ∈ {1, 16, 41, 256}. The table gives the median of 200 reps under CUDA
events, then one ncu capture per point.

| projection | S | median µs | × floor | GB/s | kernel cuBLAS chose | long-scoreboard % of waiting | SM % |
|---|---:|---:|---:|---:|---|---:|---:|
| `ff.net.0.proj` | 1 | 166.80 | 1.65 | 113.3 | `128x64 … stages_32x6` | 20.5 | 25.2 |
| | 16 | 166.08 | 1.63 | 115.2 | `128x64 … stages_32x6` | 20.2 | 25.2 |
| | **41** | **170.24** | **1.64** | 114.6 | **`128x64 … stages_64x3`** | **72.7** | 24.1 |
| | 256 | 228.42 | 1.60 | 99.9 | `128x128 … stages_32x5` | 10.1 | 66.5 |
| `attn1.to_q` | 1 | 93.86 | 3.72 | 50.4 | `64x64_sliced1x2 …` | 53.7 | 20.7 |
| | 16 | 92.83 | 3.61 | 51.9 | `64x64 … stages_64x5` | 51.2 | 21.1 |
| | **41** | **93.07** | **3.51** | 53.4 | `64x64_sliced1x2 …` | 54.0 | 20.1 |
| | 256 | 128.94 | 3.61 | 48.8 | `128x64 … stages_64x3` | 55.6 | 36.8 |

* **The largest projection costs the same from S = 1 to S = 41** (166–170 µs), at 1.6× its
  memory floor. Its time is the weight read; the 41 action tokens are nearly free. That is
  the precise sense in which this action head is a weight-streaming workload.
* **The isolated sweep reproduces the in-model kernel choice.** At S = 41 cuBLAS picks the
  same `128x64 … stages_64x3` kernel that E5b captured inside the model, and it is
  memory-latency bound there too (72.7 %). At S = 256 cuBLAS switches to the compute-bound
  `128x128` tile, the one the eager in-model capture found compute-bound. The sweep measures
  what the model runs, which is the check an isolated micro-benchmark has to pass.
* The small q projection runs at 3.5–3.7× floor at every S. A 1536×1536 bf16 GEMM
  (4.7 MB) is too small to reach the bus's stride.

## Two disagreements with the task text, recorded rather than assumed away

* **Block count.** T009 says "action head = 16 DiT blocks". The loaded N1.7 has **32**
  `transformer_blocks` (`shapes.json` → `n_blocks`). Block 0 is a cross-attention block
  (`attn1.to_k` takes 2048-wide backbone features), consistent with alternating blocks.
* **S.** T009 says S = 41; the loaded config's `action_horizon` is **40**. The sweep uses
  T009's 41. The 40-vs-41 difference falls inside the flat region of every curve above, so it
  changes no conclusion. The extra token is plausibly a state token; that is not verified.

Both are raised in `STATUS.md` under questions for the research side.

## Method, and what the window covers

* **Targets** were chosen by `bench/pick_e5b_targets.py --only-groot` from C1's committed
  capture (`results/20260923_E4g_c1_groot_n17_{eager,best}`), applying §1 (skip if SM < 5 %
  and per-launch duration < the 62 µs launch gap) and §5 (≥ 60 % coverage). Coverage came out
  at 86.8 % eager and 62.2 % best. **It is coverage of C1's captured window**, the first 40
  kernels of `ae_total`, not of the whole region. GR00T's denoising loop has no per-step wrap
  point, so its window is `ae_total` where the LeRobot models used `ae_step_0`.
* **Captures** used `bench/run_ncu_e5b_groot.sh`. It carries `run_ncu_e5b.sh`'s full metric set
  (per-pipe issue, LSU requests, 14 stall ratios, launch and occupancy limits, InstructionStats)
  with `run_ncu_c1_groot.sh`'s GR00T configuration copied verbatim, so the kernels captured are
  the ones C1 ranked. Each capture was capped at 20 launches, and pipe metrics were present in
  all four.
* **SASS** was exported from the reports (`ncu_sass_dump.py`), i.e. what actually executed.
  These cuBLAS/CUTLASS kernels live in `libcublasLt` and have no `.cubin` on disk. Inventories
  and static-vs-dynamic cross-checks were in each directory's `inventory/` (2026-09-28: moved to
  the docs/03 §0 paths `sass/<kernel>_sm87_*`, one per SYMBOL, by `bench/e5b_static.py`; see that
  directory's meta.json `e5b_static`). Each cross-check is
  paired with the details CSV of the **same** kernel. The older queue took the first CSV in the
  directory, which in a multi-target directory would pair a kernel with another's metrics.
* **Relation to T007's kernels.** GR00T's GEMMs are the `_relu_` epilogue variants, which are
  different symbols from π0's `…128x64_ldg8_f2f_stages_64x3_tn`, so the static part was done
  afresh rather than referenced. `Kernel2` shares a short name with a π0 target, but CUTLASS
  names many instantiations `Kernel2`, and identity is not claimed.

## Measured / derived / not measured

**Measured:** the four in-model ncu captures, the eight isolated sweep captures, and sweep
timings (n = 200 per point).
**Derived:** the §7 verdicts (thresholds applied to the measured stall shares), × floor
(against E1's measured 187.47 GB/s / 33.85 TFLOP/s), and coverage.
**Not measured:**
- E5b for GR00T N1.5 and N1.6; their C1 captures exist and the scripts take `n15` and `n16`.
- A sweep of the cross-attention k/v projections (2048-wide input).
- The S sweep for π0 (T007). π0.5 and SmolVLA were swept on 2026-09-25 (T008 step 6):
  `E5b_pi05.md`, `E5b_smolvla.md`.
