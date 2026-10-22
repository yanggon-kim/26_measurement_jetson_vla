# PROGRESS — Jetson AGX Thor campaign

**The top tracker.** The on-device agent updates it and pushes at the end of every task, and after every finished
result directory inside a long task (`AGENTS.md`, "Close the task"). Three sections, in this order: the
"Needs the human" box, the task table, the deliverables checklist. States: `todo`, `claimed`, `done`,
`done-partial` (missing items named), `blocked` (what it waits for). Dates are UTC; "commit" is the commit that
closed the task.

## Needs the human
*(none open. H2 — Hugging Face login + Cosmos-Reason2 licence — resolved 2026-10-22, see the log in `STATUS.md`.)*

## Tasks
| task | what | priority | state | date | commit | headline (with its number) |
|---|---|---|---|---|---|---|
| [T001](tasks/T001_setup_and_availability.md) | setup and availability | P0 | done | 2026-10-22 | 181ee2d | MAXN, 20 SMs, sm_110, L2 32 MiB; torch 2.10.0+cu130 OK; 6 of 6 models pass smoke; no dram__ counters (L2 sysmem 1.003× analytic); nsys binaries NOT CLEAN |
| [T002](tasks/T002_platform_record.md) | platform record (E0) | P0 | done | 2026-10-22 | 3a5417b | MAXN, locked 1575 MHz under load, idle module power 33.27 W; VDD_GPU current-limited at ≈ 100 W: sustained bf16 GEMM −5.6 % (invisible to devfreq) |
| [T003](tasks/T003_peaks_and_kernel_sweep.md) | peaks → `hw_constants.json` (E1), expert kernel sweep (E2) | P0 | done | 2026-10-22 | 09bb687 | BW 258.2 GB/s, bf16 129.8 TFLOP/s (ridge 503 FLOP/B); counters validated to 0.1 %; π0 expert projections 3.29 ms/step = 1.37× floor; hidden fast memory-path state (×1.26) found |
| [T004](tasks/T004_baseline_latency.md) | baseline latency, all models (E3, E4 stock modes, floors) | P0 | done | 2026-10-22 | ce769e5 | eager AE share of e2e: π0 29.7 %, π0.5 28.2 %, SmolVLA 80.6 %, GR00T N1.7/1.6/1.5 47.8/54.2/37.1 %; GR00T DiT steps 2.1–2.6× their memory floor; shipped GR00T computes unused lm_head logits (3.5–19.9 ms); stock compile ≤ 15 % (π0) |
| [T005](tasks/T005_ladder_and_best_bf16.md) | ladder and best BF16 (E4 A/B, E4i, E4d) — deliverable 1 | P0 | todo | | | |
| [T006](tasks/T006_roofline.md) | roofline per part and per kernel (E5 captures, E6, C3) — deliverable 2 | P0 | todo | | | |
| [T007](tasks/T007_timelines_and_phase_bounds.md) | timelines and phase bounds (E5, E4g C1/C2) — deliverable 3a | P1 | todo | | | |
| [T008](tasks/T008_kernel_bottlenecks.md) | per-kernel bottlenecks (E5b, E4h, E5c) — deliverable 3b | P1 | todo | | | |
| [T009](tasks/T009_core_synthesis.md) | core synthesis, Thor vs Orin | P1 | todo | | | |
| [T010](tasks/T010_power_energy.md) | power and energy (E7, E7v) | P2 | todo | | | |
| [T011](tasks/T011_libero.md) | LIBERO closed loop (E8, E8s) | P2 | todo | | | |
| [T012](tasks/T012_hybrid_inputs.md) | hybrid-model inputs (E11) | P2 | todo | | | |
| [T013](tasks/T013_corun.md) | co-run (E12, E12b) | P2 | todo | | | |
| [T014](tasks/T014_prefix_bandwidth_and_savings.md) | prefix bandwidth and savings (E13, E14, E14b) | P2 | todo | | | |
| [T015](tasks/T015_full_synthesis.md) | full synthesis and repository check | P2 | todo | | | |

## Deliverables checklist
`eager_fp32` = baseline; `best_bf16` = the fastest BF16 configuration T005 finds (its rung name goes in the cell when
ticked). Tick a cell with `[x]` and the result directory that delivers it.

**Ceilings:** [x] `hw_constants.json` with measured bandwidth, TFLOP/s per dtype and ridge (T003): `results/20261022_E1_peak/` · [x] DRAM-byte
and FLOP counters validated (T003): `results/20261022_E1_peak/validation.csv` · [x] nsys binary secrets test (T001): **NOT CLEAN** — nsys binaries never committed; ncu reports clean (`results/20261022_E0_secrets_test/`)

| model | mode | latency: vision / prefix / expert (T005) | roofline: parts + every kernel (T006) | bottleneck: ncu per part (T007) + SASS of dominant kernels (T008) |
|---|---|---|---|---|
| π0 | `eager_fp32` | [ ] | [ ] | [ ] ncu · [ ] SASS |
| π0 | `best_bf16` | [ ] | [ ] | [ ] ncu · [ ] SASS |
| π0.5 | `eager_fp32` | [ ] | [ ] | [ ] ncu · [ ] SASS |
| π0.5 | `best_bf16` | [ ] | [ ] | [ ] ncu · [ ] SASS |
| SmolVLA | `eager_fp32` | [ ] | [ ] | [ ] ncu · [ ] SASS |
| SmolVLA | `best_bf16` | [ ] | [ ] | [ ] ncu · [ ] SASS |
| GR00T N1.7 | `eager_fp32` | [ ] | [ ] | [ ] ncu · [ ] SASS |
| GR00T N1.7 | `best_bf16` | [ ] | [ ] | [ ] ncu · [ ] SASS |
| GR00T N1.6 | `eager_fp32` | [ ] | [ ] | [ ] ncu · [ ] SASS |
| GR00T N1.6 | `best_bf16` | [ ] | [ ] | [ ] ncu · [ ] SASS |
| GR00T N1.5 | `eager_fp32` | [ ] | [ ] | [ ] ncu · [ ] SASS |
| GR00T N1.5 | `best_bf16` | [ ] | [ ] | [ ] ncu · [ ] SASS |

**Remaining Orin jobs:** [ ] E7 power per rail and part (T010) · [ ] E8 / E8s LIBERO (T011) · [ ] E11
`model_inputs.csv` × 6 (T012) · [ ] E12 / E12b co-run (T013) · [ ] E13 / E14 / E14b (T014) · [ ] `SYNTHESIS_thor.md`
(T009, T015)
