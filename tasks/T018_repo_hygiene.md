# T018 — repository hygiene: trim STATUS.md and PROGRESS.md, delete superseded reports (no GPU)

*Issued 2026-10-08 by the research side. No measurement, no GPU, any power mode. Git keeps every deleted line.*

**Why.** Every research-side session starts by reading the head of `STATUS.md` (109 KB, 1,700 lines of dated
session logs) and pulls `PROGRESS.md` (a finished 2026-09-22 campaign). The research side has already trimmed its
own files (`docs/`, `HANDOFF.md`). Trim yours to the current state.

## Steps
1. **`STATUS.md` → current state only, ≤ 20 KB.** Keep, in this order: (a) *Now* — device state (power mode, SM
   count, clocks), what is running, the last bench commit; (b) open items and anything that needs the owner;
   (c) standing facts you rely on that are written nowhere else (sudo route, the software stack that works, the
   ncu metrics this Orin exposes — or a one-line pointer to the `env/` or `results/` file that holds them);
   (d) *Questions for the research side* — only the unanswered ones; these four stay open until answered: the empty
   camera (read results as-is with the caveat?), OD-1 (which variants and which N enter the GPU baseline — the
   user's decision), the `model_commit: null` back-fill, `owngraph_expert`; (e) the latest *Repository size* lines
   (protocol §6f). Delete the dated session logs, "DONE" blocks, superseded headlines and answered questions.
   Before deleting a **correction** block, check its content is in the report or result it corrects (`NOTE.md`,
   `SUPERSEDED.md`, a report section); if it is not, add a one-line pointer to where it now lives.
2. **`PROGRESS.md`.** The campaign it tracks is finished. Reduce it to a two-line header plus the
   `<!-- FAST:BEGIN/END -->` and `<!-- SLOW:BEGIN/END -->` blocks, or delete it — if you delete it,
   `bench/progress_monitor.py` must still run (create the file on demand or drop the write) and say which you did.
3. **`reports/` — delete only superseded material.** A file goes only when a later report fully contains it and
   `git grep -nF '<file name>'` finds no reference outside the `STATUS.md` text you removed in step 1. Never delete:
   a report named in a task's acceptance items or status lines; anything `bench/` reads (`reports/E5b_parts/`,
   `E5b_summary.md`, `SYNTHESIS_all_measurements.md`); and these, which the research side cites:
   `E0_platform.md`, `E1_E2_microbench.md`, `E3_pi0_eager.md`, `E4_pi0_compile.md`, `E4d_phase_balance.md`,
   `E4g_c1_c2_bottlenecks.md`, `E4i_best_combinations.md`, `E7_power_energy.md`, `E11_models.md`,
   `E11_pi0_hybrid_inputs.md`, `E11_groot_n17_hybrid_inputs.md`, `E12b_pi0_compiled_corun.md`,
   `SYNTHESIS_all_measurements.md`. Deleting nothing is an acceptable answer.

## Non-negotiables
- No file under `results/`, `env/`, `docs/`, `analysis/` or a task body is edited or deleted.
- No number is changed. A wrong number you notice is reported under (d), not fixed silently.

## Acceptance
- `STATUS.md` ≤ 20 KB and starts with the *Now* section; the four questions in step 1 (d) are present.
- For every deleted file, `git grep -nF '<name>'` returns no hit (paste the check in the status block).
- The status block lists the deleted files, the before/after bytes of `STATUS.md`, `PROGRESS.md` and `reports/`,
  and the commit hash; one commit per step, pushed.
