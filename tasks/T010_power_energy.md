# T010 — Power and energy per rail and per part (E7, E7v) at the current mode · priority P2 · depends on: T005

## Why
Energy per part tells whether the expert is the expensive phase in energy as well as in time. On the Orin it was
not: the expert's energy-to-time ratio was below 1 in every point (`reference/orin/reports/E7_power_energy.md`).
Capped power modes are **not** measured on Thor (they need a reboot; `docs/01_measurement_plan.md`, "Dropped").

## Orin counterpart
`reference/orin/tasks/T004_pi0_nsys_power.md` step 4 (not 4b), `reference/orin/reports/E7_power_energy.md`;
scripts `reference/orin/bench/power_core.py`, `power_phases.py`, `phase_loops.py`, `e7v_split.py`,
`e7r_reconcile.py`, `power_rails_check.py`, `common.py` (`stop_tegrastats`).

## Configuration
The maximum-performance mode only, `jetson_clocks`, gate passed. All six models, `eager_fp32` and `best_bf16`, default
N. Rails and module power as T001's rail table defines them (protocol §4).

## Steps
1. Port the phase loops: vision-only, prefix-only (vision + VLM prefix), expert-only (all N steps on a fixed KV /
   feature hand-over), full inference; each ≥ 60 s back to back; the sampler covers the **timed window only**.
2. 60 s idle baseline before each power run; all rails from sysfs at ≥ 10 Hz plus tegrastats; per rail mean,
   module power, the memory rail (if one exists) absolute and above idle.
3. Energy per inference, per part and per denoising step, total and above idle (protocol §4).
4. **Reconciliation**: the three part loops interleaved in one session under one idle baseline; parts vs the full
   loop; residual < 5 % of the full-frame energy, or the measured reason why not (Orin's E7r found ~30 % missing
   before the fix: `reference/orin/tasks/T016_prefix_bandwidth_and_followups.md` step 4).

## Result directories and reports
`results/<date>_E7_<model>_<mode>/` (`power.csv` tidy, `energy.csv`, raw sysfs and tegrastats logs),
`results/<date>_E7_reconcile/`; `reports/E7_power_energy.md` (W per rail per phase, J per inference and per part,
mJ per step, energy-to-time ratio per part, Thor vs Orin labelled).

## Acceptance
- 6 models × 2 modes with per-rail power for the four loops plus idle, energy per part, and the reconciliation
  residual; raw logs committed; no capped mode.

## Push points
After each model, at the close.

## Status (appended by the on-device agent)
