# E3 / E4 across models — regions, per-step time eager vs compiled, floor and gap (T006)

One table across the four models the campaign measured, as T006's acceptance asks (per-region ms,
ms per denoising step eager and compiled, floor and gap ratio per protocol §9). GR00T's own detail
is in `E3_E4_groot.md`; π0's in `E3_pi0_eager.md` / `E4_pi0_compile.md`; the ladders behind the
"best rung" choice in `E4i_best_combinations.md`.

Device: Jetson AGX Orin 64 GB, MAXN, GPU pinned 1300.5 MHz, EMC 3199 MHz, 16 SMs, batch 1.

## 1. Answer

Per-step time is protocol §2 definition (b): the least-squares slope of `ae_total` over
N ∈ {1, 2, 4, 6, 8, 10} (derived). Floor per step = bytes one denoising step must read / the
measured 187.47 GB/s (E6 `floor.csv`, derived). Gap = slope / floor (derived).

| model | checkpoint | eager ms/step | r² | best rung | compiled ms/step | r² | floor ms/step | gap eager | gap compiled | compile speed-up / step |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| π0 | `lerobot/pi0_libero_finetuned_v044` | **52.94** | 0.99999 | `reduce-overhead` ¹ | **16.30** | ¹ | 3.40 | **15.6×** | **4.8×** | 3.25× |
| π0.5 | `lerobot/pi05_libero_base` | **61.60** | 0.99974 | `bf16_both_graphs` | **18.40** | 0.99999 | 9.25 eager / 5.84 bf16 ² | **6.7×** | **3.2×** | 3.35× |
| SmolVLA | `lerobot/smolvla_base` | **61.46** | 0.99821 | `vlm_graphs` | **4.33** | 0.99993 | 1.08 | **56.6×** | **4.0×** | 14.2× |
| GR00T N1.7 | `nvidia/GR00T-N1.7-3B` | **51.75** | 0.99939 | `vlm_compile` | **18.13** | 0.99999 | 17.29 ³ | **3.0×** | **1.05×** | 2.85× |

¹ π0's compiled slope is from `E4_pi0_compile.md` (`reduce-overhead`, six N,
`results/20260921_E4_pi0_reduce-overhead_N*`); it is the slope of **e2e** over N (that report's
fit; for every other model here e2e and ae_total slopes agree to ≤ 1.3 ms, §3). That report
quotes 5.2× floor against an earlier floor figure; 4.8× here uses E6's 3.40 ms. π0's overall best rung, `bf16_both`, was measured at
N = 10 only; its ae_total/N is 16.87 ms (`results/20260925_E6_rungs/pi0_bf16_both`, gap 4.96×),
which includes the N-independent part, so a `bf16_both` slope is **not measured**.
² π0.5 ships fp32; its best rung casts to bf16, which halves the weight bytes, so the floor moves
with the rung (`results/20260925_E6_rungs/pi05_*`). Each gap uses its own rung's floor.
³ GR00T's byte count is a lower bound (the cross-attended context is not counted) and it has no
compute floor (FLOPs not counted); see `E3_E4_groot.md` §3.

**Reading.** Eager, all four models spend 52–62 ms per denoising step, whatever their size — the
eager step is launch- and dispatch-bound, not weight-bound (SmolVLA reads 1/16 of GR00T's bytes
per step yet takes longer). Compilation removes most of that: the compiled step tracks the
weight bytes. After compilation GR00T's DiT is at its memory floor (1.05×); π0, π0.5 and
SmolVLA are still 3–5× above theirs.

## 1b. Model inputs and geometry, read from the loaded models (T006 step 1)

| model | S (valid) | P processed (valid) | images | expert layers | heads / KV heads / head_dim | source |
|---|---:|---:|---:|---:|---|---|
| π0.5 | 50 (50) | 968 (574) | 3 × 224² (1 masked) | 18 (VLM 18) | 8 / 1 / 256 | `results/20260925_E6_pi05/floor_inputs.json`, `E6_pi05_roofline.md` |
| SmolVLA | 50 (50) | 177 (151) | 2 × 512² | 16 (VLM 16 — config says 32; truncated at load) | 15 / 5 / 64 | `results/20260925_E6_smolvla/floor_inputs.json`, `E6_smolvla_roofline.md` |

π0's are T002's (`E3_pi0_eager.md`); GR00T's in `E3_E4_groot.md` / `results/20260923_E4f_groot_S_n17`.
Checkpoints: `lerobot/pi05_libero_base` @ `a217bfd3`, `lerobot/smolvla_base` @ `d9f33c94` (HF snapshot sha).

## 2. Measured — per-region medians at N = 10 (ms)

| model | rung | vision | vlm_prefix | ae_total | e2e | source |
|---|---|---:|---:|---:|---:|---|
| π0 | eager (E3) | 239.60 | 232.93 | 533.03 | — ⁴ | `results/20260921_E3_pi0_eager_N10/latency_summary.csv` |
| π0 | eager_tf32 (ladder) | 241.01 | 234.57 | 586.91 | 1069.04 | `E4i_best_combinations.md` / `results/20260922_E4c_pi0_opt_ladder` |
| π0 | bf16_both | 56.74 | 192.86 | 168.68 | 424.46 | same |
| π0.5 | eager_tf32 | 103.28 | 559.11 | 612.27 | 1280.43 | `results/20260925_E3E4_pi05_Nsweep/opt_ladder.csv` |
| π0.5 | bf16_both_graphs | 56.37 | 227.43 | 184.09 | 474.01 | same |
| SmolVLA | eager_tf32 | 33.36 | 66.09 | 625.72 | 732.84 | `results/20260925_E3E4_smolvla_Nsweep/opt_ladder.csv` |
| SmolVLA | vlm_graphs | 31.03 | 7.60 | 43.36 | 89.11 | same |
| GR00T N1.7 | eager_tf32 | 69.80 | 85.03 | 511.01 | 667.97 | `results/20260925_E3E4_groot_n17_N10_eager_tf32/phase_split.csv` |
| GR00T N1.7 | vlm_compile | 70.27 | 84.48 | 188.71 | 345.80 | `results/20260925_E3E4_groot_n17_N10_vlm_compile/phase_split.csv` |

⁴ E3's e2e is in its own report (`E3_pi0_eager.md`); the E3 and ladder eager rows differ in
TF32 and in session, which is why ae_total differs by 10 %. The compare-across-models column is
the slope in §1, not these absolutes.

Vision and prefix do not move with N in any model (slopes |b| ≤ 1.3 ms/step against 30–560 ms
intercepts, r² ≤ 0.49 — flat), so only N = 10 is shown; every N is in the CSVs named above.

## 3. Derived — slope fits (least squares over the six N)

| model | rung | ae_total slope | intercept | r² | e2e slope | e2e intercept |
|---|---|---:|---:|---:|---:|---:|
| π0.5 | eager_tf32 | 61.598 | 1.12 | 0.99974 | 62.871 | 667.98 |
| π0.5 | bf16_both_graphs | 18.396 | 0.15 | 0.99999 | 18.385 | 289.96 |
| SmolVLA | eager_tf32 | 61.458 | −1.49 | 0.99821 | 61.431 | 104.01 |
| SmolVLA | vlm_graphs | 4.329 | 0.21 | 0.99993 | 4.330 | 46.03 |
| GR00T N1.7 | eager_tf32 | 51.751 | −1.96 | 0.99939 | 51.680 | 157.08 |
| GR00T N1.7 | vlm_compile | 18.130 | 7.35 | 0.99999 | 18.166 | 164.21 |
| π0 | eager (E3) | 52.943 | 3.69 | 0.99999 | | |

## 4. Conditions and anomalies

* π0.5 / SmolVLA sweeps: `bench/opt_ladder.py --model {pi05,smolvla} --steps 1,2,4,6,8,10
  --rungs eager_tf32,<best>`, 20 warm-up + 100 timed, one process per rung and N. **Their
  `meta.json` originally named pi0's checkpoint** — a writer bug (P.CHECKPOINT hard-coded), fixed
  in d3d4591 with the original values kept under `meta_correction`. The model that ran was the
  right one: `--model` reaches every worker, and the numbers are not pi0's.
* π0.5 at N = 10 in this sweep: 474.01 ms e2e vs 487.81 ms in the earlier E4i ladder (−2.8 %,
  different session, same rung). The slope is unaffected by a constant offset.
* π0.5's checkpoint is the **base** checkpoint (no LeRobot LIBERO fine-tune exists); latency does
  not depend on that, closed-loop success does (E8).
* GR00T: `bench/groot_phases.py`, 20 + 50, one process per point; its own
  `benchmark_inference.py` measured: eager action head agrees to 0.6 % (204.42 vs 205.72 ms at N = 4;
  `E3_E4_groot.md` §4).

## 5. Not measured

* π0 `bf16_both` N-sweep (footnote ¹).
* ae_step[i] instrumented per-step (§2 (a)) for π0.5, SmolVLA, GR00T: the ladders time regions,
  and §2 makes the slope primary under compilation.
* N-sweeps for GR00T N1.5 / N1.6 (N = 4 only).
* E8 closed-loop success — `E8_libero.md` (running).

## 6. Self-review (§11)

Every non-derived number is a committed CSV cell; every derived one is recomputed from those CSVs
by the least-squares fit above (script-free: a 6-point fit, reproducible from the CSV columns).
Floors are E6's, not recomputed here. The one metadata defect found while writing this (the
opt_ladder model identity) is fixed and disclosed; no measured value changed.
