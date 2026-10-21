# T009 — Core synthesis: the three deliverables per model and mode, Thor vs Orin · priority P1 · depends on: T005, T006, T007, T008

## Why
One document the research side can read without opening a result directory: per model and mode, the latency
breakdown, the roofline position of each part and its heaviest kernels, and what each part is bound by — and how
each of these moved from the Orin to Thor. It also tests the research side's projection (`docs/00_context.md`),
which was recorded before any Thor measurement.

## Orin counterpart
`reference/orin/reports/SYNTHESIS_all_measurements.md`; `reference/orin/analysis/thor_projection.md`.

## Configuration
CPU-only: no new GPU measurement. Every number is restated from a committed Thor result file (cited) or from
`reference/orin/` (cited and labelled Orin). Generate the tables with a committed script
(`bench/make_synthesis.py`), never by hand.

## Steps
1. **Deliverable 1 table**: per model × {`eager_fp32`, `best_bf16`}: vision, prefix, expert, other, e2e (median,
   p5–p95, n), shares, ms per step; the `best_bf16` rung per model.
2. **Deliverable 2 table**: per model × mode × part: FLOPs, DRAM bytes (measured and analytic), intensity, achieved
   TFLOP/s and GB/s, % of the measured ceilings, floor and gap ratio; and the five heaviest kernels per part with
   their roofline point. Embed (link) the T006 plots.
3. **Deliverable 3 table**: per model × mode × part: the verdict, the top stall and its share, tensor-pipe use,
   occupancy, idle fraction, the root cause in one line, the E5b answer.
4. **Thor vs Orin**: per model, mode and part, the Thor ÷ Orin speedup and the change in share; which parts moved
   most and the measured reason (from deliverables 2 and 3). The Orin hardware ceilings beside Thor's (Orin from
   `reference/orin/reports/E1_E2_microbench.md`).
5. **The projection, tested**: measured vs projected per model (`docs/00_context.md`): e2e, AE share, the front and
   expert factors. Where it was wrong, say which assumption failed.
6. **The answer** to the campaign's question in one paragraph at the top: on Thor, does the action expert dominate
   the optimised frame, for which models, and is it memory-bound there?

## Result directories and reports
`results/<date>_SYN_core/` (`synthesis_tables.csv` and the inputs restated under `raw/restated/<run dir>/`, protocol
§13.8's copy rule); `reports/SYNTHESIS_thor.md`.

## Acceptance
- All six models (or each missing one with its reason) in all three tables for both modes; every number cites its
  file; Orin numbers labelled; the projection comparison; the answer paragraph first.
- `PROGRESS.md`: T009's row with the headline answer.

## Push points
At the close.

## Status (appended by the on-device agent)
