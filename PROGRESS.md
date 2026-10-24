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
| [T005](tasks/T005_ladder_and_best_bf16.md) | ladder and best BF16 (E4 A/B, E4i, E4d) — deliverable 1 | P0 | claimed (deliverable 1 done; Part A, π0 N=4 control pending) | 2026-10-24 | 595a899 | best_bf16 confirmed for all six: π0 149.3, π0.5 164.4, SmolVLA 48.9, N1.7 133.9, N1.6 102.9, N1.5 106.3 ms; expert largest part in 5 of 6 (π0: prefix/expert tie) |
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

### T005 resume point (paused by the owner 2026-10-24 ≈ 04:11 UTC; restart only when the owner says so)
**The owner's order (2026-10-24):**
1. the strong optimised version and its latency breakdown;
2. the ncu work for those optimised configurations (T006, T007 `best_bf16`);
3. the rest, step by step.

**Done and pushed:**
- six ladders;
- E4i stages 1 and 2 for all six models; the N1.7 re-run;
- the N1.5 divergence diagnosis;
- `bf16_islands`, per-part autotuning, the stability guard;
- **`best_bf16` chosen** (`bench/e4i_summary.py`; the selection is in the uncommitted local
  `results/20261024_E4i_summary/`; it is regenerated at the restart);
- **confirmation runs** for π0, π0.5, SmolVLA and N1.7: `results/20261024_E4i_<m>_best_confirm/`, each within 2 % of
  its search result;
- the T006 tooling and `bench/t007_c1.py`, not yet run on the GPU.

**`best_bf16` per model** (ms; confirm run where done):

| model | rung | e2e | vision | prefix | expert | confirm ÷ search |
|---|---|---|---|---|---|---|
| π0 | `bf16_mpad_graphs_ae_at` | 149.4 | 21.1 | 63.1 | 62.6 | 1.004 |
| π0.5 | `bf16_graphs_ae_at_islands` | 164.4 | 21.1 | 65.9 | 74.7 | 1.007 |
| SmolVLA | `bf16_mpad_graphs_ae_at_islands` | 48.9 | 15.2 | 4.3 | 26.0 | 0.982 |
| GR00T N1.7 | `ae_graphs_autotune` | 133.4 | 29.0 | 25.3 | 53.2 | 0.992 |
| GR00T N1.6 | `vlm_graphs_ae_at` | 98.4 (search) | 17.2 | 13.9 | 47.3 | pending |
| GR00T N1.5 | `llm_graphs_autotune` | 116.6 (search) | 19.0 | 24.1 | 35.5 | pending |

**Next steps, in order:**
1. **Gate.** Check `nvidia-smi` and other users first.
2. **Confirm N1.6 and N1.5:** `bash bench/queues/t005_confirm.sh 20261024 groot_n16 groot_n15 >
   scratch/t005/queue_confirm2.log 2>&1`.
   - Finalise each: `bench/t005_finalise.py --run results/20261024_E4i_<m>_best_confirm --ref-dir <ladder>`.
   - N1.6's interrupted directory is in `scratch/t005/interrupted/`; unused.
3. **Final summary:** `~/venvs/base/bin/python bench/e4i_summary.py --out results/20261024_E4i_summary --report
   reports/E4d_phase_balance.md`, then finalise its `meta.json`, commit, and tick deliverable-1 latency cells.
4. **Reports:**
   - `reports/E4i_best_combinations.md`, by hand, citing rows:
     - attention backends lost to Inductor's own fusion;
     - per-part autotuning;
     - `linear_mpad`;
     - `bf16_islands`;
     - N1.5's knife-edge seed; per-seed medians are in `candidates.csv`;
   - update `NOTE_gemm_shape_cliff_m_padding.md`.
5. **ncu for `best_bf16` (owner's priority 2):**
   - Smoke-test `bench/t006_run.py --date <date> --models smolvla --modes best_bf16`. Check the ncu `--nvtx-include`
     filter, the kernel counts vs nsys, the join, and the sizes.
   - Then all six models. Then T007 C1 (`bench/t007_c1.py`) per part for `best_bf16`.
6. **Rest of T005:** the π0 N = 4 control, Part A, the independent review, the close.
7. **Then:** the `eager_fp32` ncu work (T006 / T007), then T008 → T015 (`~/.claude/plans/sunny-dazzling-cocoa.md`).

**Operating notes:**
- Pause a queue by waiting on the parent's children (`while pgrep -P <parent>`).
- Rows carry `bench_provenance` and `timing_attempts`.
- GR00T runs from `/tmp`; compiled GR00T heads use `fallback_random`.
- Equivalence bound: max(1e-2, 1.5 × the model's compile-only `ae_graphs` relative L2).
- The device is shared; the stability guard re-times disturbed blocks.

## Deliverables checklist
`eager_fp32` = baseline; `best_bf16` = the fastest BF16 configuration T005 finds (its rung name goes in the cell when
ticked). Tick a cell with `[x]` and the result directory that delivers it.

**Ceilings:** [x] `hw_constants.json` with measured bandwidth, TFLOP/s per dtype and ridge (T003): `results/20261022_E1_peak/` · [x] DRAM-byte
and FLOP counters validated (T003): `results/20261022_E1_peak/validation.csv` · [x] nsys binary secrets test (T001): **NOT CLEAN** — nsys binaries never committed; ncu reports clean (`results/20261022_E0_secrets_test/`)

| model | mode | latency: vision / prefix / expert (T005) | roofline: parts + every kernel (T006) | bottleneck: ncu per part (T007) + SASS of dominant kernels (T008) |
|---|---|---|---|---|
| π0 | `eager_fp32` | [x] `20261022_E4c_pi0_opt_ladder` (+ `20261024_E4i_summary/phase_balance.csv`) | [ ] | [ ] ncu · [ ] SASS |
| π0 | `best_bf16` | [x] `bf16_mpad_graphs_ae_at`: `20261024_E4i_pi0_best_confirm` (+ `20261024_E4i_summary`) | [ ] | [ ] ncu · [ ] SASS |
| π0.5 | `eager_fp32` | [x] `20261022_E4c_pi05_opt_ladder` (+ `20261024_E4i_summary/phase_balance.csv`) | [ ] | [ ] ncu · [ ] SASS |
| π0.5 | `best_bf16` | [x] `bf16_graphs_ae_at_islands`: `20261024_E4i_pi05_best_confirm` (+ `20261024_E4i_summary`) | [ ] | [ ] ncu · [ ] SASS |
| SmolVLA | `eager_fp32` | [x] `20261022_E4c_smolvla_opt_ladder` (+ `20261024_E4i_summary/phase_balance.csv`) | [ ] | [ ] ncu · [ ] SASS |
| SmolVLA | `best_bf16` | [x] `bf16_mpad_graphs_ae_at_islands`: `20261024_E4i_smolvla_best_confirm` (+ `20261024_E4i_summary`) | [ ] | [ ] ncu · [ ] SASS |
| GR00T N1.7 | `eager_fp32` | [x] `20261022_E4d_groot_n17_ladder` (+ `20261024_E4i_summary/phase_balance.csv`) | [ ] | [ ] ncu · [ ] SASS |
| GR00T N1.7 | `best_bf16` | [x] `ae_at`: `20261024_E4i_groot_n17_best_confirm2` (+ `20261024_E4i_summary`) | [ ] | [ ] ncu · [ ] SASS |
| GR00T N1.6 | `eager_fp32` | [x] `20261022_E4d_groot_n16_ladder` (+ `20261024_E4i_summary/phase_balance.csv`) | [ ] | [ ] ncu · [ ] SASS |
| GR00T N1.6 | `best_bf16` | [x] `all_compile_ae_vis_at`: `20261024_E4i_groot_n16_best_confirm` (+ `20261024_E4i_summary`) | [ ] | [ ] ncu · [ ] SASS |
| GR00T N1.5 | `eager_fp32` | [x] `20261022_E4d_groot_n15_ladder` (+ `20261024_E4i_summary/phase_balance.csv`) | [ ] | [ ] ncu · [ ] SASS |
| GR00T N1.5 | `best_bf16` | [x] `ae_at_mpad`: `20261024_E4i_groot_n15_best_confirm2` (+ `20261024_E4i_summary`) | [ ] | [ ] ncu · [ ] SASS |

**Remaining Orin jobs:** [ ] E7 power per rail and part (T010) · [ ] E8 / E8s LIBERO (T011) · [ ] E11
`model_inputs.csv` × 6 (T012) · [ ] E12 / E12b co-run (T013) · [ ] E13 / E14 / E14b (T014) · [ ] `SYNTHESIS_thor.md`
(T009, T015)
