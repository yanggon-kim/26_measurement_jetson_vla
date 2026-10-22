# T005 — Optimisation ladder, the best BF16 configuration, and the phase balance (E4 Parts A and B, E4i, E4d) — deliverable 1 · priority P0 · depends on: T004

## Why
Deliverable 1: for every model, the latency of vision, VLM prefix and action expert in the baseline (`eager_fp32`)
and in the optimised configuration (`best_bf16`). On the Orin, optimisation shrank the expert faster than the rest
of the frame in five of six models, and no optimised expert dominated (`reference/orin/reports/E4i_best_combinations.md`).
On Thor the vision and prefix may shrink more than the bandwidth-bound expert, which would reverse that. The
ladder attributes each speedup to one factor; the search finds the best BF16 configuration on **this** device, which
may not be the Orin's rung. Every later task profiles exactly the configuration this task names `best_bf16`.

## Orin counterpart
`reference/orin/reports/E4_PARTS_ABC_GUIDE.md` (§3 Part A, §4 Part B, §5b the search), `E4d_phase_balance.md`,
`E4i_best_combinations.md`; scripts `reference/orin/bench/opt_ladder.py`, `paper_repro.py`, `groot_phases.py`,
`make_phase_report.py`, `make_best_combo_report.py`, `check_equivalence.py`.

## Configuration
Maximum-performance mode, `jetson_clocks`, gate passed. Each model at its own default N (π0, π0.5, SmolVLA 10;
GR00T 4); π0 also at N = 4 as a control, reported separately. A fresh policy is loaded for every rung
(`torch.compile` is not reversible). ≥ 20 warm-up (until no recompilation), ≥ 100 timed per rung.

## Steps
1. **Part A** for π0, π0.5 and SmolVLA: the published characterisation's configuration (eager + TF32, own N) and
   its fp32 control (`paper_repro.py`). Report against the Orin's reproduction (labelled Orin).
2. **Part B, the ladder** (plan E4): every rung for every model, one factor at a time, regions installed after
   compilation, per-region latency and e2e; GR00T's ladder **cumulative** (vision and LLM compiled together). A rung
   that does not apply is recorded `not applicable` with the reason (e.g. SmolVLA's fp32 islands). One raw row file
   per rung, written as it completes (`raw/rows/<rung>_N<N>.json`).
3. **E4i, the search for `best_bf16`**: combine the winning factors per model — bf16 for every part where the model
   allows it, compile and CUDA graphs per part, the whole-policy compile, and forced GEMM autotuning when Inductor's
   SM threshold blocks it (`all_compile_autotune`, a labelled override). Every candidate: finite actions, and its
   actions vs the eager bf16 path on ≥ 10 fixture inputs with seeded noise (max |Δ| and relative ‖Δ‖₂, recorded, not
   tuned). Keep the fastest candidate **whose weights and activations are BF16** (fp32 accumulation inside kernels is
   fine; say if any op stays fp32 and why). Also report the configuration that maximises the expert's share.
4. **Name it.** Write `results/<date>_E4i_summary/best_bf16.json`: per model the rung or combination, the exact
   flags and code path (which callables are compiled, mode, graphs, precision settings, autotune override), N, and
   the result directory it was timed in; and `bench/` must reproduce it from one option (`--rung best_bf16`). T006–T014
   read this file; they never re-derive it.
5. **E4d, the phase balance**: per model, vision / prefix / expert / other (ms and %) for `eager_fp32`, `best_bf16`
   and every rung; per-step time by slope for compiled rungs; the AE share eager vs best; the Thor ÷ Orin speedup per
   part per mode (Orin values from `reference/orin/reports/E4i_best_combinations.md` and `E4d_phase_balance.md`).
   Answer in one paragraph: **does the action expert dominate the optimised frame on Thor, for which models?**

## Result directories and reports
`results/<date>_E4a_<model>_paper_config/`, `results/<date>_E4c_<model>_opt_ladder/` (LeRobot models),
`results/<date>_E4d_<groot model>_ladder/`, `results/<date>_E4i_<model>_best_search/`, `results/<date>_E4i_summary/`
(`best_bf16.json`, `phase_balance.csv`: `model, N, mode, rung, vision_ms, prefix_ms, expert_ms, other_ms, e2e_ms,
p5/p95 per part, n, ae_share`); reports `reports/E4d_phase_balance.md` (generated from the rows, never typed),
`reports/E4i_best_combinations.md`.

## Acceptance
- Every model has `eager_fp32` and `best_bf16` with vision, prefix, expert and other as median, p5, p95 and n ≥ 100,
  from the same harness; the deliverable-1 cells of `PROGRESS.md` ticked with the result directory.
- `best_bf16.json` exists and `bench/` reproduces each entry from one option.
- Every ladder rung measured or `not applicable` with a reason; every search candidate's equivalence recorded.
- The Thor vs Orin per-part table, and the one-paragraph answer.
- Independent review pass (P0).

## Push points
After each model's ladder directory, after each model's search, after `E4i_summary`, at the close.

## Status (appended by the on-device agent)

status: claimed 2026-10-22
