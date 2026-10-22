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
| [T005](tasks/T005_ladder_and_best_bf16.md) | ladder and best BF16 (E4 A/B, E4i, E4d) — deliverable 1 | P0 | claimed (paused by the owner) | 2026-10-22 | ad2ba6e | 6 ladders + E4i stage 1 done; provisional best: π0 157.8, π0.5 183.1, SmolVLA 49.2, N1.7 134.5, N1.6 106.4, N1.5 116.6 ms (resume point below) |
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

### T005 resume point (paused by the owner 2026-10-22 ≈ 17:20 UTC; restart only when the owner says so)
**Done and pushed:**
- six ladders, one process per rung:
  - `results/20261022_E4c_{pi0,pi05,smolvla}_opt_ladder/`;
  - `results/20261022_E4d_groot_n1{7,6,5}_ladder/`;
- two diagnostics:
  - `results/20261022_E4_prefix_pi0_vs_pi05_diag/`: π0's prefix slower than π0.5's because of a cuBLASLt kernel choice at M = 816;
  - `results/20261022_E4_gemm_m816_fix/`: the `linear_mpad` fix, bit-identical;
- the note for other device agents: `reports/NOTE_gemm_shape_cliff_m_padding.md`;
- E4i stage 1 for all six models: `results/20261022_E4i_<m>_best_search/`. N1.5 ran 9 of 10 candidates;
  `all_compile_autotune_mpad` was not run and is low priority, since `linear_mpad` slows GR00T.

**Provisional optimised breakdown.** Medians in ms (% of e2e); not yet `best_bf16`:

| model | configuration | e2e | vision | prefix | expert | other |
|---|---|---|---|---|---|---|
| π0 | `bf16_mpad_graphs_autotune` | 157.8 | 25.7 (16.3 %) | 66.5 (42.1 %) | 63.1 (40.0 %) | 2.5 (1.6 %) |
| π0.5 | `bf16_both_graphs` (ladder) | 183.1 | 21.1 (11.5 %) | 66.9 (36.5 %) | 92.4 (50.5 %) | 2.6 (1.4 %) |
| SmolVLA | `bf16_mpad_graphs_nofuse` | 49.2 | 15.5 (31.5 %) | 3.9 (7.9 %) | 26.5 (53.8 %) | 3.3 (6.8 %) |
| GR00T N1.7 | `ae_graphs_autotune` | 134.5 | 29.9 (22.2 %) | 25.8 (19.2 %) | 51.7 (38.5 %) | 27.0 (20.1 %) |
| GR00T N1.6 | `llm_compile_autotune` | 106.4 | 21.2 (20.0 %) | 16.4 (15.4 %) | 49.5 (46.5 %) | 19.4 (18.2 %) |
| GR00T N1.5 | `llm_graphs_autotune` | 116.6 | 19.0 (16.3 %) | 24.1 (20.7 %) | 35.5 (30.5 %) | 37.9 (32.5 %) |

**Next steps, in order:**
1. **Open checks, before stage 2.**
   - **(a) fp32 weights remain in two models.** π0.5's best keeps 5.6 % of parameter bytes fp32: the adaptive-norm
     `dense` Linears inside the expert, fed from fp32 time embeddings. SmolVLA keeps 1.4 %: the outer action / state /
     time projections. Add a prep that casts these remaining fp32 Linears to bf16, with input and output dtype casts,
     and add candidates with it. This is needed for "weights and activations BF16" (rule 3 in `bench/e4i_summary.py`),
     and it may speed π0.5's bandwidth-bound expert.
   - **(b) Two GR00T N1.5 candidates differ from eager far more than rounding.** `vlm_graphs_autotune` and
     `llm_graphs_mpad_autotune` show relative L2 0.031–0.138, against a bound of 0.026. Suspect overwritten CUDA-graph
     outputs. Investigate before trusting any N1.5 graphs-on-LLM / vision-compile row. Two N1.5 rows were also
     throttled; re-measure them.
   - **(c) Re-measure GR00T N1.6 / N1.5 `all_compile_autotune`** (ladder 101.9 / 108.4 ms) with `fallback_random`;
     the ladder rows drew different noise. Already in the stage-2 list.
2. **E4i stage 2** (autotuning per part):
   `bash bench/queues/t005_e4i_s2.sh 20261022 > scratch/t005/queue_e4i_s2.log 2>&1`
   writes `results/20261022_E4i_<m>_best_search_s2/`. After each model:
   `bench/t005_finalise.py --run <dir> --ref-dir <its ladder> ...`, then commit.
3. **Choose `best_bf16`:**
   - `~/venvs/base/bin/python bench/e4i_summary.py --out results/<date>_E4i_summary --report reports/E4d_phase_balance.md`;
   - then confirmation runs: `bench/opt_ladder.py --model <m> --rung best_bf16 --out results/<date>_E4i_<m>_best_confirm
     --slope-steps 2,6 --slope-rungs all`. It replays the recorded `linear_mpad` table, and the GR00T runs execute
     from `/tmp`;
   - then re-run `e4i_summary.py`.
4. **π0 N = 4 control:** `~/venvs/lerobot/bin/python bench/opt_ladder.py --model pi0 --n 4 --out
   results/<date>_E4c_pi0_opt_ladder_N4 --rungs <LR list in bench/queues/t005_ladders.sh> --slope-steps 2,6
   --slope-rungs whole_graphs`.
5. **Part A:** `bash bench/queues/t005_part_a.sh <date>`.
6. **Write up and close:**
   - write `reports/E4i_best_combinations.md`;
   - check the generated `reports/E4d_phase_balance.md`, including the Thor ÷ Orin table and the one-paragraph
     answer;
   - independent review (P0);
   - close T005 and tick the deliverable-1 cells below.

**Operating notes:**
- Pause a queue at a rung boundary by waiting on the parent's children (`while pgrep -P <parent>`). Never use a
  `pgrep -f` pattern that matches the waiting shell itself.
- Bench edits during a run are recorded per row (`bench_provenance`: git head + file hashes).
- GR00T children run with `cwd=/tmp`. Compiled GR00T heads use `fallback_random` (seeded noise as eager).
- Equivalence bound: max(1e-2, 1.5 × the model's compile-only `ae_graphs` relative L2).

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
