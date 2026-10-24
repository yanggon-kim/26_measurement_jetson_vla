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
| [T005](tasks/T005_ladder_and_best_bf16.md) | ladder and best BF16 (E4 A/B, E4i, E4d) — deliverable 1 | P0 | claimed (deliverable 1 done; Part A, π0 N=4 control pending) | 2026-10-25 | 03a983b | best_bf16 confirmed for all six: π0 129.5, π0.5 142.4, SmolVLA 48.9, N1.7 120.4, N1.6 102.9, N1.5 106.3 ms (2026-10-25: π0/π0.5 kv_shallow by the owner's decision, N1.7 vision compiled); expert largest part in 4 of 6 (π0: prefix 55.0 > expert 51.5; π0.5 level) |
| [T006](tasks/T006_roofline.md) | roofline per part and per kernel (E5 captures, E6, C3) — deliverable 2 | P0 | claimed (best_bf16 done for all six; eager_fp32 pending) | 2026-10-24 | fcccc1f | best_bf16: every part of every model on the memory side of the ridge except N1.5's lm_head (92 % of bf16 peak); expert steps 49–80 % of peak BW; coverage 100 %, 0 join mismatches |
| [T007](tasks/T007_timelines_and_phase_bounds.md) | timelines and phase bounds (E5, E4g C1/C2) — deliverable 3a | P1 | todo | | | |
| [T008](tasks/T008_kernel_bottlenecks.md) | per-kernel bottlenecks (E5b, E4h, E5c) — deliverable 3b | P1 | todo | | | |
| [T009](tasks/T009_core_synthesis.md) | core synthesis, Thor vs Orin | P1 | todo | | | |
| [T010](tasks/T010_power_energy.md) | power and energy (E7, E7v) | P2 | todo | | | |
| [T011](tasks/T011_libero.md) | LIBERO closed loop (E8, E8s) | P2 | todo | | | |
| [T012](tasks/T012_hybrid_inputs.md) | hybrid-model inputs (E11) | P2 | todo | | | |
| [T013](tasks/T013_corun.md) | co-run (E12, E12b) | P2 | todo | | | |
| [T014](tasks/T014_prefix_bandwidth_and_savings.md) | prefix bandwidth and savings (E13, E14, E14b) | P2 | todo | | | |
| [T015](tasks/T015_full_synthesis.md) | full synthesis and repository check | P2 | todo | | | |

### Resume point (updated 2026-10-24)
**The owner's order (2026-10-24):** (1) the strong optimised version and its latency breakdown — **done**
(deliverable 1, `results/20261024_E4i_summary`); (2) the ncu work for those optimised configurations (T006, T007
`best_bf16`) — **T006 done, T007 running**; (3) the rest, step by step.

**Done and pushed:** T005 deliverable 1 (six `best_bf16` confirmed; updated 2026-10-25: π0 / π0.5 `kv_shallow` by the
owner's decision, N1.7 vision compiled; `results/20261025_E4i_summary`); T006 `best_bf16` for SmolVLA, N1.6, N1.5
(`results/20261024_E6_*`; the GR00T floor fix applied); T007 timelines for the 2026-10-24 configurations and C1 for SmolVLA;
`results/20261025_E4_cpu_overhead_diag`.

**Next steps, in order:**
1. **Running (`scratch/run_t006_t007.sh`, log `scratch/t006_t007.log`):** T006 `best_bf16` for π0, π0.5, N1.7 at the new
   configurations (`--date 20261025`), their timelines, then T007 C1 for π0, π0.5, N1.7 (`--e5-date 20261025`) and
   N1.6, N1.5 (`--e5-date 20261024`). Re-run the same commands with `--skip-done` to resume. Then mark the superseded
   `20261024_E5/E6/timeline` directories of π0, π0.5, N1.7 (SUPERSEDED.md) and regenerate `make_e6_reports.py`,
   `make_e5_report.py`.
2. T007 C2 / root-cause chains / `reports/E4g_c1_c2_bottlenecks.md` (needs `eager_fp32` C1 too).
3. **Rest of T005:** the π0 N = 4 control (`bench/opt_ladder.py --model pi0 --n 4 --out results/<date>_E4c_pi0_opt_ladder_N4`,
   rungs from `bench/queues/t005_ladders.sh`), Part A (`bash bench/queues/t005_part_a.sh <date>`), the close.
4. **`eager_fp32`:** `bench/t006_run.py --date <date> --modes eager_fp32`, `bench/t007_run.py --what timeline` and C1
   with `--modes eager_fp32`, `--finalise-dims`, the reports; the T006 independent review; close T006.
5. **Then** T008 (best_bf16 first) → T015 (`~/.claude/plans/sunny-dazzling-cocoa.md`).

**Operating notes:**
- Gate before each GPU job (`bench/gate.py`; the device is shared with another user).
- Profiling reuses the Inductor / Triton caches of the run that timed the configuration (`t006_run.timed_cache`).
- ncu runs with `--target-processes application-only`; the ncu ↔ nsys join is launch order + base kernel name.
- Pause a queue by waiting on the parent's children (`while pgrep -P <parent>`).
- GR00T runs from `/tmp`; compiled GR00T heads use `fallback_random`.

## Deliverables checklist
`eager_fp32` = baseline; `best_bf16` = the fastest BF16 configuration T005 finds (its rung name goes in the cell when
ticked). Tick a cell with `[x]` and the result directory that delivers it.

**Ceilings:** [x] `hw_constants.json` with measured bandwidth, TFLOP/s per dtype and ridge (T003): `results/20261022_E1_peak/` · [x] DRAM-byte
and FLOP counters validated (T003): `results/20261022_E1_peak/validation.csv` · [x] nsys binary secrets test (T001): **NOT CLEAN** — nsys binaries never committed; ncu reports clean (`results/20261022_E0_secrets_test/`)

| model | mode | latency: vision / prefix / expert (T005) | roofline: parts + every kernel (T006) | bottleneck: ncu per part (T007) + SASS of dominant kernels (T008) |
|---|---|---|---|---|
| π0 | `eager_fp32` | [x] `20261022_E4c_pi0_opt_ladder` (+ `20261024_E4i_summary/phase_balance.csv`) | [ ] | [ ] ncu · [ ] SASS |
| π0 | `best_bf16` | [x] `bf16_mpad_graphs_ae_at_kvs`: `20261025_E4i_pi0_best_confirm` (+ `20261025_E4i_summary`) | [ ] re-capture at the new `best_bf16` running (`20261024_E6_*` was the stock-code configuration) | [ ] ncu · [ ] SASS |
| π0.5 | `eager_fp32` | [x] `20261022_E4c_pi05_opt_ladder` (+ `20261024_E4i_summary/phase_balance.csv`) | [ ] | [ ] ncu · [ ] SASS |
| π0.5 | `best_bf16` | [x] `bf16_mpad_graphs_ae_at_islands_kvs`: `20261025_E4i_pi05_best_confirm` (+ `20261025_E4i_summary`) | [ ] re-capture at the new `best_bf16` running (`20261024_E6_*` was the stock-code configuration) | [ ] ncu · [ ] SASS |
| SmolVLA | `eager_fp32` | [x] `20261022_E4c_smolvla_opt_ladder` (+ `20261024_E4i_summary/phase_balance.csv`) | [ ] | [ ] ncu · [ ] SASS |
| SmolVLA | `best_bf16` | [x] `bf16_mpad_graphs_ae_at_islands`: `20261024_E4i_smolvla_best_confirm` (+ `20261025_E4i_summary`) | [x] `20261024_E6_smolvla_best_bf16_v2` | [x] ncu `20261024_E4g_c1_smolvla_best_bf16` · [ ] SASS |
| GR00T N1.7 | `eager_fp32` | [x] `20261022_E4d_groot_n17_ladder` (+ `20261024_E4i_summary/phase_balance.csv`) | [ ] | [ ] ncu · [ ] SASS |
| GR00T N1.7 | `best_bf16` | [x] `vis_fa_at_ae_at`: `20261025_E4i_groot_n17_best_confirm` (+ `20261025_E4i_summary`) | [ ] re-capture at the new `best_bf16` running (`20261024_E6_*` was the stock-code configuration) | [ ] ncu · [ ] SASS |
| GR00T N1.6 | `eager_fp32` | [x] `20261022_E4d_groot_n16_ladder` (+ `20261024_E4i_summary/phase_balance.csv`) | [ ] | [ ] ncu · [ ] SASS |
| GR00T N1.6 | `best_bf16` | [x] `all_compile_ae_vis_at`: `20261024_E4i_groot_n16_best_confirm` (+ `20261024_E4i_summary`) | [x] `20261024_E6_groot_n16_best_bf16` | [ ] ncu · [ ] SASS |
| GR00T N1.5 | `eager_fp32` | [x] `20261022_E4d_groot_n15_ladder` (+ `20261024_E4i_summary/phase_balance.csv`) | [ ] | [ ] ncu · [ ] SASS |
| GR00T N1.5 | `best_bf16` | [x] `ae_at_mpad`: `20261024_E4i_groot_n15_best_confirm2` (+ `20261024_E4i_summary`) | [x] `20261024_E6_groot_n15_best_bf16` | [ ] ncu · [ ] SASS |

**Remaining Orin jobs:** [ ] E7 power per rail and part (T010) · [ ] E8 / E8s LIBERO (T011) · [ ] E11
`model_inputs.csv` × 6 (T012) · [ ] E12 / E12b co-run (T013) · [ ] E13 / E14 / E14b (T014) · [ ] `SYNTHESIS_thor.md`
(T009, T015)
